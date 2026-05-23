
#include "precomp.hpp"
#include "contours_common.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <map>

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

struct AccumulatorT:public std::vector<std::vector<cv::Point>>{
    std::vector<int> idx_internal_lastLine,idx_external_firstLine;
};


class TRUCOntourTracer : public cv::ParallelLoopBody
{
public:

    // We use a pointer to the accumulator to avoid passing huge objects
    // Accumulator: Vector of (Vector of Contours), where Contour is Vector of Points
    using AccumulatorType = std::vector<AccumulatorT>;

    TRUCOntourTracer(const cv::Mat& img,
                     const std::vector<cv::Range>& stripRanges,
                     AccumulatorType& accumulator,
                     size_t minSize)
        : padded_(img), ranges_(stripRanges), accumulator_(accumulator), minSize_(minSize)
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

    //trace external contour marking EAST pixels (VISITED_OUTER_RIGHT) only so that later the analysis of the internal contour is exactly as expected
    void traceExternalContourMock(  int r,int c,uchar *row_ptr, const cv::Range& rowRange)const{
        int curr_x = c , curr_y = r;
        int start_dir = -1 ;
        int search_idx = 5;
        uchar* curr_ptr = row_ptr + c , * start_ptr = curr_ptr;
        int dir=-1;
        bool is_first_move = true;
        // 3. TRACING LOOP
        while(true)
        {
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                // Use offset cache
                uchar* neighbor = curr_ptr + offsets_[idx];
                if (*neighbor == BACKGROUND) continue;
                dir = idx & 7;
                if (((search_idx <= 1)  || (dir <= search_idx - 2)) && (curr_x!=c && curr_y!=r))//do nt apply to first pixel in the way back
                    *curr_ptr = VISITED_OUTER_RIGHT;
                // --- EXECUTE MOVE ---
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                // Check bounds //we need to move out of the range  //if first line, and internal contour, we let it go, but no further from this line
                if( curr_y < rowRange.start )
                    return  ;
                // Short-circuit Jacob's Check
                if (curr_ptr == start_ptr) {
                    if (!is_first_move && dir == start_dir) {
                        return  ;//done
                    }
                }
                curr_ptr = neighbor;//move ptr
                // Reset search index for Moore neighbor
                search_idx = (dir +6) & 7;
                break;

            }
            if (is_first_move) {
                if(dir==-1){//single pixel
                    break;//not moved
                }
                start_dir = dir;
                is_first_move = false;
            }

        }
    }
    bool traceContour( TRUCOPagedContour<4096>* buffer,  int r,int c,uchar *row_ptr, const cv::Range& rowRange,bool isExternal)const{

        buffer->clear();

        int curr_x = c , curr_y = r;
        int start_dir = -1 ;
        int search_idx = isExternal ? 5 :1;
        uchar* curr_ptr = row_ptr + c , * start_ptr = curr_ptr;
        int dir=-1;
        // int sign=isExternal?1:-1;

        bool is_first_move = true;


        // QUick test to find the first element of an internal contour. We know it should be NE
        if(!isExternal){
            int n=0;
            for ( n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                if ( *(curr_ptr + offsets_[idx]) == BACKGROUND) continue;
                if(curr_x+ dx_[ idx & 7]!=c+1 || curr_y+ dy_[ idx & 7]!=r-1) return false;
                break;
            }
            if(n==8) return false;//isolated pixels must not be considered as internal
        }
        // 3. TRACING LOOP
        while(true)
        {
            buffer->push_back({ (curr_x - 1),  (curr_y - 1)});
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                // Use offset cache
                uchar* neighbor = curr_ptr + offsets_[idx];
                if (*neighbor == BACKGROUND) continue;

                dir = idx & 7;
                // --- EXECUTE MOVE ---
                curr_y += dy_[dir];
                curr_x += dx_[dir];


                // Check bounds //we need to move out of the range  //if first line, and internal contour, we let it go, but no further from this line
                if( curr_y < rowRange.start ){
                    return false;
                }
                if ((search_idx <= 1)  || (dir <= search_idx - 2))
                {
                    *curr_ptr = VISITED_OUTER_RIGHT;
                }
                else if (*curr_ptr == FOREGROUND)
                {
                    *curr_ptr = VISITED_;
                }

                // Short-circuit Jacob's Check
                if (curr_ptr == start_ptr) {
                    if (!is_first_move && dir == start_dir) {
                        return true;//done
                    }
                }

                curr_ptr = neighbor;//move ptr

                // Reset search index for Moore neighbor
                search_idx = (dir +6) & 7;
                break;

            }
            if (is_first_move) {
                if(dir==-1){//single pixel
                    *curr_ptr = VISITED_OUTER_RIGHT;
                    break;//not moved
                }
                start_dir = dir;
                is_first_move = false;
            }

        }
        return true;
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        // Pre-allocate buffer to avoid re-allocation during moves
        TRUCOPagedContour<4096> buffer;

        int cols = padded_.cols;

        for (int i = range.start; i < range.end; ++i)
        {
            const cv::Range& rowRange = ranges_[i];
            auto& local_contours = accumulator_[i];

            // Hint for result vector size
            local_contours.reserve(2048);

            for (int r = rowRange.start; r <= rowRange.end; ++r)
            {
                uchar* row_ptr = padded_.data + r * step_;

                // "c" is updated by the find* functions
                for (int c = 1; c < cols - 1; )
                {
                    // 1. FAST SCAN: Skip background pixels
                    if ((c = findStartContourPoint(row_ptr, cols, c)) == cols) break;

                    // 2. CHECK: Only process if actually FOREGROUND (redundancy check)
                    if (row_ptr[c] == FOREGROUND && r<rowRange.end )
                    {
                        if( traceContour(&buffer,r,c,row_ptr,rowRange,true)){
                            // Post-processing
                            if (buffer.size() > 1 && buffer.back() == buffer.front()) {
                                buffer.pop_back();
                            }
                            if (buffer.size() >= minSize_) {
                                // Instead of copying the vector, we move it.
                                local_contours.emplace_back();
                                buffer.copyTo(local_contours.back());
                                if( r==rowRange.start){//register this so we can zip them as Suzuki&Abe method
                                    local_contours.idx_external_firstLine.push_back((int)(local_contours.size()-1) );
                                }
                            }
                        }
                    }
                    else if( row_ptr[c] == FOREGROUND && r==rowRange.end ){//extra step to mark east pixels only so that internal contours can be correctly extracted in this line
                        traceExternalContourMock(r,c,row_ptr,rowRange);
                    }

                    // 3. FAST SCAN: Find end of current component to skip processing it again
                    c = findEndContourPoint(row_ptr, cols, c + 1);
                    if(c>=cols)break;//end of row
                    //internal contour
                    if(row_ptr[c-1]>VISITED_OUTER_RIGHT && r>rowRange.start){//inner contours of first line are handled by the thread above

                        if(traceContour(&buffer,r,c-1,row_ptr,rowRange,false)){
                            // Post-processing
                            if (buffer.size() > 1 && buffer.back() == buffer.front()) {
                                buffer.pop_back();
                            }
                            if (buffer.size() >= minSize_) {
                                local_contours.emplace_back();
                                buffer.copyTo(local_contours.back());
                                if( r==rowRange.end){//register this so we can zip them as Suzuki&Abe method
                                    local_contours.idx_internal_lastLine.push_back((int)(local_contours.size() -1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static inline int findStartContourPoint(uchar* src_data, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes())
        {
            cv::v_uint8 vmask = (cv::v_ne(cv::vx_load((uchar*)(src_data + j)), v_zero));
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

    inline static int findEndContourPoint(uchar* src_data,int width, int j)
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
                cv::v_uint8 vmask = (cv::v_eq(cv::vx_load((uchar*)(src_data + j)), v_zero));
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
    cv::Mat padded_;
    const std::vector<cv::Range>& ranges_;
    AccumulatorType& accumulator_;
    size_t minSize_;
    size_t step_;
    int offsets_[16];

    // 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE (CCW Rotation)
    const int dx_[8] = {  1,  1,  0, -1, -1, -1,  0,  1 };
    const int dy_[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };
    // Constants defined once
    const uchar FOREGROUND = 255;
    const uchar BACKGROUND = 0;
    const uchar VISITED_OUTER_RIGHT    = 100;
    const uchar VISITED_    = 200;
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
    //   Load Balancing Logic
    const int nstripes = cv::getNumThreads();
    std::vector<cv::Range> balancedRanges;
    if (nstripes > 1) {
        int rowsPerStripe = (padded.rows - 2) / nstripes;
        int remainingRows = (padded.rows - 2) % nstripes;
        int currentRow = 1;
        for (int t = 0; t < nstripes; ++t) {
            int startRow = currentRow;
            int endRow = startRow + rowsPerStripe + (t < remainingRows ? 1 : 0);
            balancedRanges.emplace_back(startRow, endRow);
            currentRow = endRow;
        }
    }
    else {
        balancedRanges.emplace_back(1, padded.rows - 1);
    }
    //  Parallel Execution
    std::vector<AccumulatorT> threadAccumulators(balancedRanges.size());
    TRUCOntourTracer worker(padded, balancedRanges, threadAccumulators, minSize);
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
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

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
