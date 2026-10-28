
#include "precomp.hpp"
#include "contours_common.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <map>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#if defined(_MSC_VER)
#include <intrin.h>   // __iso_volatile_load8 / __iso_volatile_store8
#endif

namespace{

// Tunable block size. 1024 points = 8KB (Fits easily in L1 Cache)
template <size_t BLOCK_SIZE = 2048>
class TRUCOPagedContour {
public:
    struct Block {
        cv::Point data[BLOCK_SIZE];
    };

    TRUCOPagedContour() {
        allocateBlock();
        // Initialize pointers to the start of the first block
        curr_ptr_ = all_blocks_[0]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    ~TRUCOPagedContour() {
        for (Block* b : all_blocks_) cv::fastFree(b);
    }

    // --- HOT PATH: Minimal instructions ---
    // No counter updates, just raw pointer arithmetic.
    inline void push_back(const cv::Point& pt) {
        if (curr_ptr_ == end_ptr_) {
            current_block_idx_++;
            if (current_block_idx_ == all_blocks_.size()) {
                allocateBlock();
            }
            curr_ptr_ = all_blocks_[current_block_idx_]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
        *curr_ptr_++ = pt;
    }

    inline void pop_back() {
        // Safety check: do nothing if completely empty
        if (current_block_idx_ == 0 && curr_ptr_ == all_blocks_[0]->data) return;

        // Check if we are at the start of the current block
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            // Move to the previous block
            current_block_idx_--;
            // Point to the end of the previous block
            curr_ptr_ = all_blocks_[current_block_idx_]->data + BLOCK_SIZE;
            end_ptr_  = curr_ptr_;
        }
        curr_ptr_--;
    }

    inline const cv::Point& back() const {
        // Handle case where back() crosses block boundary
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            return all_blocks_[current_block_idx_ - 1]->data[BLOCK_SIZE - 1];
        }
        return *(curr_ptr_ - 1);
    }

    inline const cv::Point& front() const {
        return all_blocks_[0]->data[0];
    }

    // Calculated on demand (O(1) arithmetic, but slightly more math than reading a variable)
    size_t size() const {
        size_t elements_in_last = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        return (current_block_idx_ * BLOCK_SIZE) + elements_in_last;
    }

    void clear() {
        current_block_idx_ = 0;
        if (!all_blocks_.empty()) {
            curr_ptr_ = all_blocks_[0]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
    }

    // Optimized Copy: Uses block-wise memcpy
    void copyTo(std::vector<cv::Point>& out) const {
        size_t total = size();
        out.resize(total);
        if (total == 0) return;

        cv::Point* dst = out.data();

        // 1. Copy full blocks
        for (size_t i = 0; i < current_block_idx_; ++i) {
            std::memcpy(dst, all_blocks_[i]->data, BLOCK_SIZE * sizeof(cv::Point));
            dst += BLOCK_SIZE;
        }

        // 2. Copy partial last block
        size_t last_block_count = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        if (last_block_count > 0) {
            std::memcpy(dst, all_blocks_[current_block_idx_]->data, last_block_count * sizeof(cv::Point));
        }
    }

private:
    void grow() {
        current_block_idx_++;
        if (current_block_idx_ == all_blocks_.size()) {
            allocateBlock();
        }
        curr_ptr_ = all_blocks_[current_block_idx_]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    void allocateBlock() {
        Block* b = (Block*)cv::fastMalloc(sizeof(Block));
        all_blocks_.push_back(b);
    }

    std::vector<Block*> all_blocks_;
    size_t current_block_idx_ = 0;

    // Fast pointers for the hot loop
    cv::Point* curr_ptr_ = nullptr;
    cv::Point* end_ptr_  = nullptr;
};


////IMPLEMENTATION

// Relaxed atomic access to the shared board: other threads only test background vs. non-background,
// which never changes, so no ordering is needed. std::atomic_ref is C++20, hence the fallbacks.
static inline uchar loadRelaxed(const uchar* p)
{
#if defined(__cpp_lib_atomic_ref)
    return std::atomic_ref<uchar>(*const_cast<uchar*>(p)).load(std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(p, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
    return (uchar)__iso_volatile_load8((const volatile char*)p);
#else
    return *(const volatile uchar*)p;
#endif
}
static inline void storeRelaxed(uchar* p, uchar v)
{
#if defined(__cpp_lib_atomic_ref)
    std::atomic_ref<uchar>(*p).store(v, std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
    __iso_volatile_store8((volatile char*)p, (char)v);
#else
    *(volatile uchar*)p = v;
#endif
}

// Per contour C: its Suzuki-Abe order and a neighbouring border D found on its start row
// (external: east side of the previous run end; hole: west side of its run start).
// parent(C) = type(D) != type(C) ? D : parent(D).
struct ContourInfo {
    int64_t order;   // start pixel (linear index in the source image) * 2 + isHole
    int64_t refPix;  // pixel of D, -1: root. 64-bit for images of more than 2^31 pixels
    uchar refSide;   // REF_WEST or REF_EAST
    bool isHole;
};
enum : uchar { REF_WEST = 1, REF_EAST = 2 };

struct AccumulatorT:public std::vector<std::vector<cv::Point>>{
    std::vector<ContourInfo> info;
};

// One task per stripe of rows [s_k, e_k). Task k writes marks only on its own rows (plus a private
// copy of row e_k), and accepts a contour only if its top row is in [s_k, e_k): every contour is
// found exactly once and the result does not depend on the schedule.
class TRUCOntourTracer : public cv::ParallelLoopBody
{
public:

    // Accumulator: Vector of (Vector of Contours), where Contour is Vector of Points
    using AccumulatorType = std::vector<AccumulatorT>;

    // img   : padded binary image (0 / 255), the shared label board
    // extra : one private row per stripe (copy of row e_k), extra.row(k)
    TRUCOntourTracer(cv::Mat& img,
                     const std::vector<cv::Range>& stripRanges,
                     AccumulatorType& accumulator,
                     cv::Mat& extra)
        : padded_(img), extra_(extra), ranges_(stripRanges), accumulator_(accumulator)
    {
        step_ = padded_.step;
        int istep = (int)step_;
        // 0: East (Right)
        offsets_[0] = 1;
        // 1: NE (Up-Right)
        offsets_[1] = -istep + 1;
        // 2: North (Up)
        offsets_[2] = -istep;
        // 3: NW (Up-Left)
        offsets_[3] = -istep - 1;
        // 4: West (Left)
        offsets_[4] = -1;
        // 5: SW (Down-Left)
        offsets_[5] = istep - 1;
        // 6: South (Down)
        offsets_[6] = istep;
        // 7: SE (Down-Right)
        offsets_[7] = istep + 1;

        memcpy(offsets_ + 8, offsets_, 8 * sizeof(int));
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        // Pre-allocate buffer to avoid re-allocation during moves
        TRUCOPagedContour<4096> buffer;

        const int cols = padded_.cols;

        for (int k = range.start; k < range.end; ++k)
        {
            const int s = ranges_[k].start, e = ranges_[k].end;   // scans rows [s, e]
            auto& local_contours = accumulator_[k];

            // Hint for result vector size
            local_contours.reserve(2048);
            local_contours.info.reserve(2048);

            // private copy of row e, normalized to FOREGROUND / BACKGROUND
            // (the owner of row e may be writing marks on it concurrently)
            uchar* ext = extra_.data + (size_t)k * extra_.step;
            const uchar* src = padded_.data + (size_t)e * step_;
            for (int x = 0; x < cols; ++x)
                ext[x] = loadRelaxed(src + x) ? FOREGROUND : BACKGROUND;
            const Labels labels = { padded_.data, ext, step_, e };

            for (int r = s; r <= e; ++r)
            {
                uchar* row_ptr = (r < e) ? padded_.data + (size_t)r * step_ : ext;
                // linear index in the source image of the padded column x of row r: rowBase + x
                const int64_t rowBase = (int64_t)(r - 1) * (cols - 2) - 1;
                int prevEnd = -1;   // padded column of the end of the previous run of this row

                // "c" is updated by the find* functions
                for (int c = 1; c < cols - 1; )
                {
                    // 1. FAST SCAN: Skip background pixels
                    if ((c = findStartContourPoint(row_ptr, cols, c)) == cols) break;
                    const int runStart = c;

                    // 2. External contour: start point not visited yet, rows [s, e)
                    if (r < e && row_ptr[c] == FOREGROUND)
                    {
                        const ContourInfo info = { (rowBase + c) * 2,
                                                   prevEnd < 0 ? -1 : rowBase + prevEnd, REF_EAST, false };
                        if (traceContour(&buffer, r, c, s, e, labels, true))
                            emit(buffer, local_contours, info);
                    }

                    // 3. FAST SCAN: Find end of current component to skip processing it again
                    c = findEndContourPoint(row_ptr, cols, c + 1);
                    if (c >= cols) break; // end of row
                    prevEnd = c - 1;

                    // 4. Internal contour: rows (s, e]; its NE neighbour (row r-1, an own row) is not background
                    if (r > s && row_ptr[c - 1] != VISITED_OUTER_RIGHT &&
                        padded_.data[(size_t)(r - 1) * step_ + c] != BACKGROUND)
                    {
                        const ContourInfo info = { (rowBase + c - 1) * 2 + 1, rowBase + runStart, REF_WEST, true };
                        if (traceContour(&buffer, r, c - 1, s, e, labels, false))
                            emit(buffer, local_contours, info);
                    }
                }
            }
        }
    }

    static inline int findStartContourPoint(const uchar* src_data, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes())
        {
            cv::v_uint8 vmask = (cv::v_ne(cv::vx_load(src_data + j), v_zero));
            if (cv::v_check_any(vmask))
            {
                j += cv::v_scan_forward(vmask);
                return j;
            }
        }
#endif
        for (; j < width && !src_data[j]; ++j)
            ;
        return j;
    }

    inline static int findEndContourPoint(const uchar* src_data,int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (j <  width && !src_data[j])
        {
            return j;
        }
        else
        {
            cv::v_uint8 v_zero = cv::vx_setzero_u8();
            for (; j <=  width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes())
            {
                cv::v_uint8 vmask = (cv::v_eq(cv::vx_load(src_data + j), v_zero));
                if (cv::v_check_any(vmask))
                {
                    j += cv::v_scan_forward(vmask);
                    return j;
                }
            }
        }
#endif
        for (; j < width && src_data[j]; ++j)
            ;

        return j;
    }

private:
    // where the marks of a pixel (given by its index in the padded image and its row) live for a task
    struct Labels {
        uchar* shared; uchar* ext; size_t step; int e;
        inline uchar* at(size_t idx, int row) const {     // only valid for row <= e
            return row < e ? shared + idx : ext + (idx - (size_t)e * step);
        }
    };

    // adds the traced contour to the output
    inline void emit(TRUCOPagedContour<4096>& buffer, AccumulatorT& local_contours, const ContourInfo& info) const
    {
        if (buffer.size() > 1 && buffer.back() == buffer.front())
            buffer.pop_back();
        local_contours.emplace_back();
        buffer.copyTo(local_contours.back());
        local_contours.info.push_back(info);
    }

    // Moore tracing from (r, c). Returns true iff the contour was traced completely AND this attempt
    // is its owner (external: started on its top row; internal: started one row below its top row).
    bool traceContour(TRUCOPagedContour<4096>* buffer, int r, int c, int s, int e,
                      const Labels& labels, bool isExternal) const
    {
        buffer->clear();
        const uchar* board = padded_.data;   // only background / non-background is read here
        int curr_x = c, curr_y = r, minrow = r;
        int start_dir = -1, dir = -1;
        int search_idx = isExternal ? 5 : 1;
        size_t curr = (size_t)r * step_ + c;
        const size_t start = curr;
        bool is_first_move = true;

        // TRACING LOOP
        while (true)
        {
            buffer->push_back({ (curr_x - 1), (curr_y - 1) });
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                // Use offset cache
                const size_t neighbor = curr + offsets_[idx];
                if (loadRelaxed(board + neighbor) == BACKGROUND) continue;

                dir = idx & 7;
                // --- EXECUTE MOVE ---
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                if (curr_y < s)
                    return false;   // moved into the stripe above: not ours
                if (curr_y < minrow)
                    minrow = curr_y;

                // mark the pixel we are leaving; only rows <= e are stored by this task
                const int prev_y = curr_y - dy_[dir];
                if (prev_y <= e)
                {
                    uchar* label = labels.at(curr, prev_y);
                    if ((search_idx <= 1) || (dir <= search_idx - 2))
                        storeRelaxed(label, VISITED_OUTER_RIGHT);
                    else if (*label == FOREGROUND)
                        storeRelaxed(label, VISITED_);
                }

                // Short-circuit Jacob's Check
                if (curr == start && !is_first_move && dir == start_dir)
                    return isExternal ? (minrow == r) : (minrow == r - 1);

                curr = neighbor; // move
                // Reset search index for Moore neighbor
                search_idx = (dir + 6) & 7;
                break;
            }
            if (is_first_move) {
                if (dir == -1) { // single pixel
                    storeRelaxed(labels.at(curr, r), VISITED_OUTER_RIGHT);
                    return isExternal;   // an isolated pixel is an external contour only
                }
                start_dir = dir;
                is_first_move = false;
            }
        }
    }

    cv::Mat& padded_;
    cv::Mat& extra_;
    const std::vector<cv::Range>& ranges_;
    AccumulatorType& accumulator_;
    size_t step_;
    int offsets_[16];

    // 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE (CCW Rotation)
    const int dx_[8] = {  1,  1,  0, -1, -1, -1,  0,  1 };
    const int dy_[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };
    // Constants defined once
    enum : uchar { FOREGROUND = 255, BACKGROUND = 0, VISITED_OUTER_RIGHT = 100, VISITED_ = 200 };
};


void approxContour(std::vector<cv::Point> &inout,cv::ContourApproximationModes contApprox_)   {
    size_t n = inout.size();
    if (n <= 1) return;

    if (contApprox_ == cv::CHAIN_APPROX_SIMPLE) {
        std::vector<cv::Point> result;
        result.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            size_t prev_i = (i == 0) ? n - 1 : i - 1;
            size_t next_i = (i == n - 1) ? 0 : i + 1;

            cv::Point v1 = inout[i] - inout[prev_i];
            cv::Point v2 = inout[next_i] - inout[i];

            if (v1 != v2) {
                result.push_back(inout[i]);
            }
        }
        if (result.empty() && n > 0) result.push_back(inout[0]);
        inout = std::move(result);
    } else if (contApprox_ == cv::CHAIN_APPROX_TC89_L1 || contApprox_ == cv::CHAIN_APPROX_TC89_KCOS) {
        auto getCode = [](cv::Point d) -> schar {
            if (d.x == 1) {
                if (d.y == 0) return 0;
                if (d.y == -1) return 1;
                if (d.y == 1) return 7;
            } else if (d.x == 0) {
                if (d.y == -1) return 2;
                if (d.y == 1) return 6;
            } else if (d.x == -1) {
                if (d.y == -1) return 3;
                if (d.y == 0) return 4;
                if (d.y == 1) return 5;
            }
            return 0;
        };

        cv::ContourCodesStorage::storage_t codesStorage;
        cv::ContourCodesStorage codes(&codesStorage);
        for (size_t i = 0; i < n; ++i) {
            cv::Point delta = inout[(i + 1) % n] - inout[i];
            codes.push_back(getCode(delta));
        }

        cv::ContourPointsStorage::storage_t pointsStorage;
        cv::ContourPointsStorage points(&pointsStorage);
        cv::approximateChainTC89(codes, inout[0], contApprox_, points);

        inout.clear();
        inout.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            inout.push_back(points.at(i));
        }
    }
}

// direction (0=E, 1=NE, ..., 7=SE, y down) of the move a -> b between 8-neighbours
static inline int moveDir(const cv::Point& a, const cv::Point& b)
{
    static const int lut[3][3] = { { 3, 2, 1 }, { 4, -1, 0 }, { 5, 6, 7 } };   // [dy+1][dx+1]
    return lut[b.y - a.y + 1][b.x - a.x + 1];
}

// Background sides (REF_WEST / REF_EAST) examined at a point whose search started at direction sc
// and found the next point at direction d: the cyclic range [sc-1, d)
static inline uchar examinedSides(int sc, int d)
{
    const int from = (sc + 7) & 7;
    const int len = ((d - sc) & 7) + 1;
    uchar sides = 0;
    if (((4 - from) & 7) < len) sides |= REF_WEST;
    if (((0 - from) & 7) < len) sides |= REF_EAST;
    return sides;
}

// Sorts (pixel * 2 + bit, index) pairs concatenated from the sorted stripe lists: consecutive lists
// only overlap on their shared row, so only those rows are sorted.
static void sortStripeConcat(std::vector<std::pair<int64_t, int>>& v, int width)
{
    auto row = [&](size_t i) { return (v[i].first >> 1) / width; };
    const size_t n = v.size();
    for (size_t i = 1; i < n; ++i)
    {
        if (!(v[i] < v[i - 1])) continue;
        const int64_t r = row(i);
        size_t l = i, h = i;
        while (l > 0 && row(l - 1) == r) --l;
        while (h < n && row(h) == r) ++h;
        std::sort(v.begin() + l, v.begin() + h);
        i = h;
    }
}

static inline int popcount64(uint64_t x)
{
#if CV_POPCNT && defined(CV_POPCNT_U64)   // POPCNT in the baseline
    return (int)CV_POPCNT_U64(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

// Finds the contour D of every reference (pixel, side): a (pixel, background side) pair lies on exactly
// one border. Walks the contour points in parallel chunks and looks them up in a bitmap of references;
// the rank of the word and a popcount give the position of the reference in the sorted list.
class ReferenceResolver : public cv::ParallelLoopBody
{
public:
    enum { CHUNK = 1 << 15 };   // points per task

    ReferenceResolver(const std::vector<std::vector<cv::Point>>& contours, const std::vector<ContourInfo>& info,
                      int width, const uint64_t* bits, const uint32_t* rank,
                      const std::vector<std::pair<int64_t, int>>& refs,
                      const std::vector<size_t>& firstPoint, std::vector<int>& target)
        : contours_(contours), info_(info), width_(width), bits_(bits), rank_(rank), refs_(refs),
          firstPoint_(firstPoint), target_(target)
    {
        for (int din = 0; din < 8; ++din)
            for (int dout = 0; dout < 8; ++dout)
                sides_[din][dout] = examinedSides((din + 6) & 7, dout);
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        const size_t p0 = (size_t)range.start * CHUNK;
        const size_t p1 = std::min((size_t)range.end * CHUNK, firstPoint_.back());
        // contour that contains the point p0
        size_t i = std::upper_bound(firstPoint_.begin(), firstPoint_.end(), p0) - firstPoint_.begin() - 1;
        for (size_t p = p0; p < p1; ++i)
        {
            const std::vector<cv::Point>& c = contours_[i];
            const size_t m = c.size();
            const size_t j0 = p - firstPoint_[i], j1 = std::min(m, j0 + (p1 - p));
            p += j1 - j0;
            if (m == 0) continue;
            if (m == 1)   // isolated pixel
            {
                answer(c[0], REF_WEST | REF_EAST, (int)i);
                continue;
            }
            if (j0 == 0)   // first visit of the start point (search started at SW for external, NE for internal)
                answer(c[0], examinedSides(info_[i].isHole ? 1 : 5, moveDir(c[0], c[1])), (int)i);
            // every visit, including the closing one of the start point
            int din = moveDir(c[j0 == 0 ? m - 1 : j0 - 1], c[j0]);
            for (size_t j = j0; j < j1; ++j)
            {
                const int dout = moveDir(c[j], c[j + 1 == m ? 0 : j + 1]);
                answer(c[j], sides_[din][dout], (int)i);
                din = dout;
            }
        }
    }

private:
    inline void answer(const cv::Point& pt, uchar sides, int contour) const
    {
        const uint64_t key0 = ((uint64_t)pt.y * width_ + pt.x) * 2;   // even: both sides in the same word
        const uint64_t word = bits_[key0 >> 6];
        const int sh = (int)(key0 & 63);
        const uchar hit = (uchar)((word >> sh) & 3) & sides;
        if (!hit) return;
        for (int side = REF_WEST; side <= REF_EAST; ++side)
        {
            if (!(hit & side)) continue;
            const int b = sh + side - 1;
            const uint32_t k = rank_[key0 >> 6] + (uint32_t)popcount64(word & ((uint64_t(1) << b) - 1));
            // a contour may visit a pixel several times, possibly in two chunks: same value
            storeRelaxedInt(&target_[refs_[k].second], contour);
        }
    }
    static inline void storeRelaxedInt(int* p, int v)
    {
#if defined(__cpp_lib_atomic_ref)
        std::atomic_ref<int>(*p).store(v, std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
        __atomic_store_n(p, v, __ATOMIC_RELAXED);
#else
        *(volatile int*)p = v;
#endif
    }

    const std::vector<std::vector<cv::Point>>& contours_;
    const std::vector<ContourInfo>& info_;
    const int width_;
    const uint64_t* bits_;
    const uint32_t* rank_;
    const std::vector<std::pair<int64_t, int>>& refs_;
    const std::vector<size_t>& firstPoint_;
    std::vector<int>& target_;
    uchar sides_[8][8];
};

// Contours and hierarchy in the layout of cv::findContours: each contour, in Suzuki-Abe order, is
// added as the first child of its parent, and the output is the pre-order traversal of the tree.
void findTRUContoursImpl(cv::Mat& padded,
                         std::vector<std::vector<cv::Point>>& outContours,
                         std::vector<cv::Vec4i>& outHierarchy, bool needHierarchy,
                         int mode, int contApprox)
{
    const int width = padded.cols - 2, height = padded.rows - 2;
    const int nstripes = std::max(1, std::min(cv::getNumThreads(), height));
    std::vector<cv::Range> ranges;
    for (int t = 0, cur = 1; t < nstripes; ++t)
    {
        const int end = cur + height / nstripes + (t < height % nstripes ? 1 : 0);
        ranges.emplace_back(cur, end);
        cur = end;
    }

    // 1. tracing (parallel)
    std::vector<AccumulatorT> acc(ranges.size());
    cv::Mat extraRows((int)ranges.size(), padded.cols, CV_8UC1);
    TRUCOntourTracer worker(padded, ranges, acc, extraRows);
    cv::parallel_for_(cv::Range(0, (int)ranges.size()), worker);

    size_t total = 0;
    for (auto& a : acc) total += a.size();
    const int n = (int)total;
    const int K = (int)acc.size();

    outContours.clear();
    outContours.reserve(n);
    if (mode == cv::RETR_LIST)
    {
        // 2. every contour is a root: the output is the reverse Suzuki-Abe order. Each stripe list is
        //    sorted, and two consecutive lists only overlap on their shared row (source row
        //    ranges[k].end - 1): the holes at the tail of list k, the externals at the head of list k + 1
        auto onRow = [&](const ContourInfo& c, int row) { return (c.order >> 1) / width == row; };
        std::vector<size_t> head(K, 0), tail(K);   // list k: head [0, head[k]), tail [tail[k], size)
        for (int k = 0; k < K; ++k)
        {
            const auto& info = acc[k].info;
            tail[k] = info.size();
            if (k + 1 < K)
                while (tail[k] > 0 && onRow(info[tail[k] - 1], ranges[k].end - 1)) --tail[k];
            if (k > 0)
                while (head[k] < info.size() && onRow(info[head[k]], ranges[k].start - 1)) ++head[k];
        }
        for (int k = K - 1; k >= 0; --k)
        {
            auto& a = acc[k];
            for (size_t j = tail[k]; j-- > head[k]; )
                outContours.push_back(std::move(a[j]));
            if (k == 0) break;
            // merge (descending) the shared row of lists k - 1 and k
            auto& b = acc[k - 1];
            size_t i = head[k], t = b.size();
            while (i > 0 || t > tail[k - 1])
            {
                if (t == tail[k - 1] || (i > 0 && a.info[i - 1].order > b.info[t - 1].order))
                    { --i; outContours.push_back(std::move(a[i])); }
                else
                    { --t; outContours.push_back(std::move(b[t])); }
            }
        }
        if (needHierarchy)
        {
            outHierarchy.resize(n);
            for (int i = 0; i < n; ++i)
                outHierarchy[i] = cv::Vec4i(i + 1 < n ? i + 1 : -1, i - 1, -1, -1);
        }
    }
    else
    {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<ContourInfo> info;
        contours.reserve(n);
        info.reserve(n);
        for (auto& a : acc)
        {
            contours.insert(contours.end(), std::make_move_iterator(a.begin()), std::make_move_iterator(a.end()));
            info.insert(info.end(), a.info.begin(), a.info.end());
        }
        acc.clear();

        // 2. Suzuki-Abe order
        std::vector<std::pair<int64_t, int>> sorted(n);
        for (int i = 0; i < n; ++i) sorted[i] = { info[i].order, i };
        sortStripeConcat(sorted, width);

        // 3. references (parallel walk)
        std::vector<int> ref(n, -1);
        {
            // bitmap of the referenced keys (pixel * 2 + (side == REF_EAST)), and for every word with a
            // key the index of its first key in the sorted refs (the rank). calloc gives zeroed pages
            // lazily: only the words of referenced pixels are touched
            const size_t nwords = ((size_t)width * height * 2 + 63) / 64;
            std::unique_ptr<uint64_t, void (*)(void*)> bitsPtr((uint64_t*)std::calloc(nwords, sizeof(uint64_t)),
                                                               &std::free);
            std::unique_ptr<uint32_t, void (*)(void*)> rankPtr((uint32_t*)std::calloc(nwords, sizeof(uint32_t)),
                                                               &std::free);
            CV_Assert(bitsPtr && rankPtr);
            uint64_t* bits = bitsPtr.get();
            uint32_t* rank = rankPtr.get();
            std::vector<std::pair<int64_t, int>> refs;   // (key, contour)
            refs.reserve(n);
            std::vector<size_t> firstPoint(n + 1, 0);
            for (int i = 0; i < n; ++i)
            {
                firstPoint[i + 1] = firstPoint[i] + contours[i].size();
                if (info[i].refPix < 0) continue;
                const int64_t key = info[i].refPix * 2 + (info[i].refSide - 1);
                bits[key >> 6] |= uint64_t(1) << (key & 63);
                refs.emplace_back(key, i);
            }
            sortStripeConcat(refs, width);
            for (size_t k = refs.size(); k-- > 0; )
                rank[refs[k].first >> 6] = (uint32_t)k;
            const int nchunks = (int)((firstPoint[n] + ReferenceResolver::CHUNK - 1) / ReferenceResolver::CHUNK);
            ReferenceResolver resolver(contours, info, width, bits, rank, refs, firstPoint, ref);
            cv::parallel_for_(cv::Range(0, nchunks), resolver);
        }

        // 4. parents (RETR_TREE). ref[i] precedes i in raster order but not always in stripe order,
        //    hence the explicit stack
        std::vector<int> parent(n, -2);   // -2: unresolved, -1: root
        {
            std::vector<int> stack;
            for (int i = 0; i < n; ++i)
            {
                int j = i;
                while (parent[j] == -2)
                {
                    const int d = ref[j];
                    if (d < 0) { parent[j] = -1; break; }
                    if (info[d].isHole != info[j].isHole) { parent[j] = d; break; }
                    stack.push_back(j);
                    j = d;
                }
                while (!stack.empty())
                {
                    const int k = stack.back(); stack.pop_back();
                    parent[k] = parent[ref[k]];
                }
            }
        }

        // 5. tree of the requested mode
        std::vector<int> firstChild(n + 1, -1), next(n, -1), treeParent(n, -1);   // n: root
        for (int k = 0; k < n; ++k)
        {
            const int i = sorted[k].second;
            const int p = (mode == cv::RETR_CCOMP && !info[i].isHole) ? -1 : parent[i];
            treeParent[i] = p;
            const int pp = p < 0 ? n : p;
            next[i] = firstChild[pp];
            firstChild[pp] = i;
        }

        // 6. pre-order traversal
        std::vector<int> newIdx(n, -1), preorder;
        preorder.reserve(n);
        {
            std::vector<int> stack;
            auto pushChildren = [&](int node) {
                const size_t mark = stack.size();
                for (int c = firstChild[node]; c >= 0; c = next[c]) stack.push_back(c);
                std::reverse(stack.begin() + mark, stack.end());
            };
            pushChildren(n);
            while (!stack.empty())
            {
                const int i = stack.back(); stack.pop_back();
                newIdx[i] = (int)preorder.size();
                preorder.push_back(i);
                pushChildren(i);
            }
        }

        outHierarchy.assign(n, cv::Vec4i(-1, -1, -1, -1));
        for (int ni = 0; ni < n; ++ni)
        {
            const int i = preorder[ni];
            cv::Vec4i& h = outHierarchy[ni];
            h[0] = next[i] < 0 ? -1 : newIdx[next[i]];
            if (h[0] >= 0) outHierarchy[h[0]][1] = ni;
            h[2] = firstChild[i] < 0 ? -1 : newIdx[firstChild[i]];
            h[3] = treeParent[i] < 0 ? -1 : newIdx[treeParent[i]];
            outContours.push_back(std::move(contours[i]));
        }
    }

    // 7. contour approximation (parallel)
    if (contApprox != cv::CHAIN_APPROX_NONE)
    {
        cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++)
                approxContour(outContours[i], (cv::ContourApproximationModes)contApprox);
        });
    }
}
}

namespace cv{

// TRUCO for findContours: 8-bit image (any non-zero value is foreground), mode RETR_LIST, RETR_CCOMP
// or RETR_TREE. Same contours, order and hierarchy as the Suzuki-Abe implementation.
void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, OutputArray _hierarchy, int mode, int method)
{
    CV_INSTRUMENT_REGION();
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(mode == RETR_LIST || mode == RETR_CCOMP || mode == RETR_TREE);

    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, 0);   // ISOLATED: ignore the pixels outside a ROI
    cv::threshold(padded, padded, 0, 255, cv::THRESH_BINARY);

    // a std::vector<std::vector<Point>> output is filled directly; any other element type goes
    // through create(), which rejects it as Suzuki-Abe does
    std::vector<std::vector<cv::Point>> tempContours;
    std::vector<std::vector<cv::Point>>* contours = &tempContours;
    if (_contours.kind() == _InputArray::STD_VECTOR_VECTOR && _contours.type() == CV_32SC2)
        contours = reinterpret_cast<std::vector<std::vector<cv::Point>>*>(_contours.getObj());
    std::vector<Vec4i> hierarchy;
    findTRUContoursImpl(padded, *contours, hierarchy, _hierarchy.needed(), mode, method);

    if (contours->empty())
    {
        _contours.clear();
        return;
    }
    if (contours == &tempContours)
    {
        _contours.create((int)tempContours.size(), 1, 0, -1, true);
        for (size_t i = 0; i < tempContours.size(); i++)
        {
            _contours.create((int)tempContours[i].size(), 1, CV_32SC2, (int)i, true);
            Mat m = _contours.getMat((int)i);
            std::memcpy(m.data, tempContours[i].data(), tempContours[i].size() * sizeof(cv::Point));
        }
    }
    if (_hierarchy.needed())
    {
        _hierarchy.create(1, (int)hierarchy.size(), CV_32SC4, -1, true);
        std::memcpy(_hierarchy.getMat().data, hierarchy.data(), hierarchy.size() * sizeof(Vec4i));
    }
}

}
