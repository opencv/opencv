
#include "precomp.hpp"
#include "contours_common.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <map>
#include <atomic>
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

// Relaxed atomic byte access to the shared label board (the padded image). The only bytes accessed
// by more than one thread are read by the other threads just to tell background from non-background,
// which never changes, so no ordering is needed: relaxed atomics make the algorithm data-race-free
// and compile to plain byte loads/stores. std::atomic_ref is C++20, hence the compiler-specific
// equivalents (also used by the standard libraries themselves to implement relaxed atomics).
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

struct AccumulatorT:public std::vector<std::vector<cv::Point>>{
    std::vector<int> idx_internal_lastLine,idx_external_firstLine;
};

// The image is split in horizontal stripes [s_k, e_k), one per task. Correctness does not depend
// on the schedule of the tasks (the output is a pure function of the image) because of two rules:
//  1. Task k writes VISITED / VISITED_OUTER_RIGHT marks only on its own rows [s_k, e_k) of the shared
//     board, plus a private copy of row e_k (the first row of the next stripe, scanned by task k for
//     internal contours only). Traces still run freely downwards, but do not write outside those rows,
//     and background pixels are never written. Hence every decision of task k (the external/internal
//     start tests) reads only marks written by task k itself, while the other tasks only test
//     background vs. non-background on its rows.
//  2. Ownership is geometric. Let minrow(C) be the top row of contour C. An external contour is
//     accepted only from an attempt on row minrow(C), an internal one only from an attempt on row
//     minrow(C)+1; both attempts are the Suzuki-Abe border following start points of C. External
//     attempts are made on rows [s_k, e_k) and internal ones on rows (s_k, e_k], so each contour is
//     accepted exactly once, by the task whose own rows contain minrow(C). A trace that moves above
//     s_k is aborted.
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
                     cv::Mat& extra,
                     size_t minSize)
        : padded_(img), extra_(extra), ranges_(stripRanges), accumulator_(accumulator), minSize_(minSize)
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

                // "c" is updated by the find* functions
                for (int c = 1; c < cols - 1; )
                {
                    // 1. FAST SCAN: Skip background pixels
                    if ((c = findStartContourPoint(row_ptr, cols, c)) == cols) break;

                    // 2. External contour: start point not visited yet, rows [s, e)
                    if (r < e && row_ptr[c] == FOREGROUND)
                    {
                        if (traceContour(&buffer, r, c, s, e, labels, true) &&
                            emit(buffer, local_contours) && r == s)
                            // register this so we can zip them as Suzuki&Abe method
                            local_contours.idx_external_firstLine.push_back((int)(local_contours.size() - 1));
                    }

                    // 3. FAST SCAN: Find end of current component to skip processing it again
                    c = findEndContourPoint(row_ptr, cols, c + 1);
                    if (c >= cols) break; // end of row

                    // 4. Internal contour: rows (s, e]; its start point always has a non-background NE
                    //    neighbour (the pixel above the first background pixel of the hole), on row r-1,
                    //    which is always an own row
                    if (r > s && row_ptr[c - 1] != VISITED_OUTER_RIGHT &&
                        padded_.data[(size_t)(r - 1) * step_ + c] != BACKGROUND)
                    {
                        if (traceContour(&buffer, r, c - 1, s, e, labels, false) &&
                            emit(buffer, local_contours) && r == e)
                            // register this so we can zip them as Suzuki&Abe method
                            local_contours.idx_internal_lastLine.push_back((int)(local_contours.size() - 1));
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

    // adds the traced contour to the output; returns false if it is discarded by minSize
    inline bool emit(TRUCOPagedContour<4096>& buffer, AccumulatorT& local_contours) const
    {
        if (buffer.size() > 1 && buffer.back() == buffer.front())
            buffer.pop_back();
        if (buffer.size() < minSize_)
            return false;
        local_contours.emplace_back();
        buffer.copyTo(local_contours.back());
        return true;
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
    size_t minSize_;
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

// ==========================================================
// 1. The Core Implementation (Operates on std::vector directly)
// ==========================================================
void findTRUContoursImpl(cv::Mat& padded,
                           std::vector<std::vector<cv::Point>>& outContours,
                           int minSize,int contApprox)
{
    //   Load Balancing Logic: one stripe of rows per thread (at least one row per stripe)
    const int innerRows = padded.rows - 2;
    const int nstripes = std::max(1, std::min(cv::getNumThreads(), innerRows));
    std::vector<cv::Range> balancedRanges;
    int rowsPerStripe = innerRows / nstripes;
    int remainingRows = innerRows % nstripes;
    int currentRow = 1;
    for (int t = 0; t < nstripes; ++t) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerStripe + (t < remainingRows ? 1 : 0);
        balancedRanges.emplace_back(startRow, endRow);
        currentRow = endRow;
    }
    //  Parallel Execution
    std::vector<AccumulatorT> threadAccumulators(balancedRanges.size());
    cv::Mat extraRows((int)balancedRanges.size(), padded.cols, CV_8UC1);   // one private row per stripe
    TRUCOntourTracer worker(padded, balancedRanges, threadAccumulators, extraRows, minSize);
    cv::parallel_for_(cv::Range(0, (int)balancedRanges.size()), worker);


    // REORG To Match Suzuki & Abe's Contour Ordering.
    // Every adjacent strip pair (t, t+1) shares a boundary row. On that row:
    //   * thread t   recorded INTERNAL fragments -> idx_internal_lastLine (tail of accT)
    //   * thread t+1 recorded EXTERNAL fragments -> idx_external_firstLine (head of accT1)
    // Both runs are already in X-ascending order (left-to-right raster scan),
    // so a two-pointer merge restores the sequential Suzuki & Abe ordering.
    for (size_t t = 0; t + 1 < threadAccumulators.size(); ++t) {
        auto& accT  = threadAccumulators[t];
        auto& accT1 = threadAccumulators[t + 1];

        const size_t kI = accT.idx_internal_lastLine.size();
        const size_t kE = accT1.idx_external_firstLine.size();
        if (kI == 0 && kE == 0) continue;

        const size_t tailStart = accT.size() - kI;

        // Merge by ascending X of each contour's first point. Contours are moved,
        // not copied — only std::vector handles (three pointers) change hands.
        std::vector<std::vector<cv::Point>> merged;
        merged.reserve(kI + kE);
        size_t i = tailStart, iEnd = accT.size();
        size_t j = 0,         jEnd = kE;
        while (i < iEnd && j < jEnd) {
            if (accT[i].front().x <= accT1[j].front().x)
                merged.emplace_back(std::move(accT[i++]));
            else
                merged.emplace_back(std::move(accT1[j++]));
        }
        while (i < iEnd) merged.emplace_back(std::move(accT[i++]));
        while (j < jEnd) merged.emplace_back(std::move(accT1[j++]));

        // Replace accT's tail with the merged run.
        accT.resize(tailStart);
        for (auto& c : merged) accT.emplace_back(std::move(c));

        // Drop the consumed externals from the head of accT1.
        accT1.erase(accT1.begin(), accT1.begin() + kE);

        // Any remaining bookkeeping in accT1 indexed absolute positions — fix them.
        for (auto& idx : accT1.idx_internal_lastLine)
            idx -= static_cast<int>(kE);

        // This boundary's bookkeeping is now consumed.
        accT.idx_internal_lastLine.clear();
        accT1.idx_external_firstLine.clear();
    }

    //   ZERO-COPY MERGE
    size_t totalContours = 0;
    for (auto& tVec : threadAccumulators)
        totalContours += tVec.size();


    outContours.clear();
    outContours.reserve(totalContours);
    // move the contours from thread accumulators to output without copying pixel data
    for (auto& tVec : threadAccumulators) {
        // move_iterator moves the vector internals (pointers) without copying pixel data
        outContours.insert(outContours.end(),
                           std::make_move_iterator(tVec.begin()),
                           std::make_move_iterator(tVec.end()));
    }
    //  reverse the order to match original findContours Suzuki&Abe
    std::reverse(outContours.begin(), outContours.end());


    //7. now, lets do the contour approximation if needed in parallel
    if(contApprox!=cv::CHAIN_APPROX_NONE){
        cv::parallel_for_(cv::Range(0, (int)outContours.size()), [&](const cv::Range& range){
            for(int i=range.start;i<range.end;i++){
                approxContour(outContours[i],(cv::ContourApproximationModes)contApprox);
            }
        });
    }

}
}

namespace cv{

// ==========================================================
//
//  Public API: Handles OutputArray and dispatches to the core implementation
//  This is a modified version of the original TRUCO parallel algorithm to produce the exact same output
//  as original findContours with RETR_LIST mode (no hierarchy, all contours are external). It also supports contour approximation.
//
// ==========================================================
void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, int minSize, bool binarize,int method)
{
    CV_INSTRUMENT_REGION();
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);

    // Buffer handling
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    if (binarize)
        cv::threshold(padded, padded, 0, 255, cv::THRESH_BINARY);


    // Fast path: caller passed std::vector<std::vector<cv::Point>> directly.
    // Write into it without any intermediate copy.
    if (_contours.kind() == _InputArray::STD_VECTOR_VECTOR) {
        auto* vec = reinterpret_cast<std::vector<std::vector<cv::Point>>*>(_contours.getObj());
        findTRUContoursImpl(padded, *vec, minSize,method);
    }
    else{ // Slow path: generic OutputArray — build in a temp vector then copy.
        std::vector<std::vector<cv::Point>> tempContours;
        findTRUContoursImpl(padded, tempContours, minSize,method);

        _contours.create((int)tempContours.size(), 1, 0, -1, true);
        for (size_t i = 0; i < tempContours.size(); i++) {
            _contours.create((int)tempContours[i].size(), 1, CV_32SC2, (int)i, true);
            Mat m = _contours.getMat((int)i);
            std::memcpy(m.data, tempContours[i].data(), tempContours[i].size() * sizeof(cv::Point));
        }
    }
}

}
