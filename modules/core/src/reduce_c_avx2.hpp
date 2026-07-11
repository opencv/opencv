// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace reduce_c_avx2
{

template<bool isMax>
static void minMax8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            uchar result = isMax ? 0 : UCHAR_MAX;
            int x = 0;
            __m256i acc = _mm256_set1_epi8((char)result);
            for (; x <= cols - 32; x += 32)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x));
                acc = isMax ? _mm256_max_epu8(acc, v) : _mm256_min_epu8(acc, v);
            }
            uchar lanes[32];
            _mm256_storeu_si256((__m256i*)lanes, acc);
            for (int i = 0; i < 32; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dst[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const __m256i accInit = _mm256_set1_epi8((char)initial);
    const __m256i mask0 = _mm256_setr_epi8(
            0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
            1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
            2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i invalid0 = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i invalid12 = _mm256_setr_epi8(
            0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            __m256i acc0 = accInit, acc1 = accInit, acc2 = accInit;
            int x = 0;
            for (; x <= cols - 11; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 3));
                __m256i v0 = _mm256_shuffle_epi8(v, mask0);
                __m256i v1 = _mm256_shuffle_epi8(v, mask1);
                __m256i v2 = _mm256_shuffle_epi8(v, mask2);
                if (!isMax)
                {
                    v0 = _mm256_or_si256(v0, invalid0);
                    v1 = _mm256_or_si256(v1, invalid12);
                    v2 = _mm256_or_si256(v2, invalid12);
                }
                acc0 = isMax ? _mm256_max_epu8(acc0, v0) : _mm256_min_epu8(acc0, v0);
                acc1 = isMax ? _mm256_max_epu8(acc1, v1) : _mm256_min_epu8(acc1, v1);
                acc2 = isMax ? _mm256_max_epu8(acc2, v2) : _mm256_min_epu8(acc2, v2);
            }

            __m256i accs[3] = {acc0, acc1, acc2};
            for (int c = 0; c < 3; c++)
            {
                uchar lanes[32];
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                uchar result = initial;
                for (int i = 0; i < 32; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const __m256i accInit = _mm256_set1_epi8((char)initial);
    const __m256i maskInvalid = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 0, 0, 0, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask0 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask1 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask2 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask3 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            __m256i acc0 = accInit, acc1 = accInit, acc2 = accInit, acc3 = accInit;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                __m256i v0 = _mm256_shuffle_epi8(v, mask0);
                __m256i v1 = _mm256_shuffle_epi8(v, mask1);
                __m256i v2 = _mm256_shuffle_epi8(v, mask2);
                __m256i v3 = _mm256_shuffle_epi8(v, mask3);
                if (!isMax)
                {
                    v0 = _mm256_or_si256(v0, maskInvalid);
                    v1 = _mm256_or_si256(v1, maskInvalid);
                    v2 = _mm256_or_si256(v2, maskInvalid);
                    v3 = _mm256_or_si256(v3, maskInvalid);
                }
                acc0 = isMax ? _mm256_max_epu8(acc0, v0) : _mm256_min_epu8(acc0, v0);
                acc1 = isMax ? _mm256_max_epu8(acc1, v1) : _mm256_min_epu8(acc1, v1);
                acc2 = isMax ? _mm256_max_epu8(acc2, v2) : _mm256_min_epu8(acc2, v2);
                acc3 = isMax ? _mm256_max_epu8(acc3, v3) : _mm256_min_epu8(acc3, v3);
            }

            __m256i accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                uchar lanes[32];
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                uchar result = initial;
                for (int i = 0; i < 32; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = src[0];
            int x = 0;
            __m256 acc = _mm256_set1_ps(result);
            for (; x <= cols - 8; x += 8)
            {
                __m256 v = _mm256_loadu_ps(src + x);
                acc = isMax ? _mm256_max_ps(acc, v) : _mm256_min_ps(acc, v);
            }
            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            for (int i = 0; i < 8; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const __m256 accInit = _mm256_set1_ps(initial);
    const __m256 validMask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    const __m256i idx0 = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    const __m256i idx1 = _mm256_setr_epi32(1, 5, 0, 0, 0, 0, 0, 0);
    const __m256i idx2 = _mm256_setr_epi32(2, 6, 0, 0, 0, 0, 0, 0);
    const __m256i idx3 = _mm256_setr_epi32(3, 7, 0, 0, 0, 0, 0, 0);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            __m256 acc0 = accInit, acc1 = accInit, acc2 = accInit, acc3 = accInit;
            int x = 0;
            for (; x <= cols - 2; x += 2)
            {
                __m256 v = _mm256_loadu_ps(src + x * 4);
                __m256 v0 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx0), validMask);
                __m256 v1 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx1), validMask);
                __m256 v2 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx2), validMask);
                __m256 v3 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx3), validMask);
                acc0 = isMax ? _mm256_max_ps(acc0, v0) : _mm256_min_ps(acc0, v0);
                acc1 = isMax ? _mm256_max_ps(acc1, v1) : _mm256_min_ps(acc1, v1);
                acc2 = isMax ? _mm256_max_ps(acc2, v2) : _mm256_min_ps(acc2, v2);
                acc3 = isMax ? _mm256_max_ps(acc3, v3) : _mm256_min_ps(acc3, v3);
            }

            __m256 accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                float lanes[8];
                _mm256_storeu_ps(lanes, accs[c]);
                float result = initial;
                for (int i = 0; i < 8; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t result = 0;
            int x = 0;
            __m256i acc = _mm256_setzero_si256();
            for (; x <= cols - 32; x += 32)
            {
                __m256i bytes = _mm256_loadu_si256((const __m256i*)(src + x));
                __m128i lo = _mm256_castsi256_si128(bytes);
                __m128i hi = _mm256_extracti128_si256(bytes, 1);
                __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo16, lo16));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi16, hi16));
            }
            uint32_t lanes[8];
            _mm256_storeu_si256((__m256i*)lanes, acc);
            for (int i = 0; i < 8; i++)
                result += lanes[i];
            for (; x < cols; x++)
                result += (uint32_t)src[x] * src[x];
            dst[0] = (DT)(int32_t)result;
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const __m256i mask0 = _mm256_setr_epi8(
            0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
            1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
            2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            int x = 0;
            for (; x <= cols - 11; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 3));
                __m256i channels[3] = {
                    _mm256_shuffle_epi8(v, mask0),
                    _mm256_shuffle_epi8(v, mask1),
                    _mm256_shuffle_epi8(v, mask2)
                };
                __m256i* accs[3] = {&acc0, &acc1, &acc2};
                for (int c = 0; c < 3; c++)
                {
                    __m128i lo = _mm256_castsi256_si128(channels[c]);
                    __m128i hi = _mm256_extracti128_si256(channels[c], 1);
                    __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                    __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(lo16, lo16));
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(hi16, hi16));
                }
            }

            __m256i accs[3] = {acc0, acc1, acc2};
            for (int c = 0; c < 3; c++)
            {
                uint32_t lanes[8];
                uint32_t result = 0;
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 3 + c];
                    result += value * value;
                }
                dst[c] = (DT)(int32_t)result;
            }
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const __m256i mask0 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask1 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask2 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask3 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                __m256i channels[4] = {
                    _mm256_shuffle_epi8(v, mask0),
                    _mm256_shuffle_epi8(v, mask1),
                    _mm256_shuffle_epi8(v, mask2),
                    _mm256_shuffle_epi8(v, mask3)
                };
                __m256i* accs[4] = {&acc0, &acc1, &acc2, &acc3};
                for (int c = 0; c < 4; c++)
                {
                    __m128i lo = _mm256_castsi256_si128(channels[c]);
                    __m128i hi = _mm256_extracti128_si256(channels[c], 1);
                    __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                    __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(lo16, lo16));
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(hi16, hi16));
                }
            }

            __m256i accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                uint32_t lanes[8];
                uint32_t result = 0;
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 4 + c];
                    result += value * value;
                }
                dst[c] = (DT)(int32_t)result;
            }
        }
    });
    v_cleanup();
}

static void sum2_32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = 0;
            int x = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; x <= cols - 8; x += 8)
            {
                __m256 v = _mm256_loadu_ps(src + x);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
            }
            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            for (int i = 0; i < 8; i++)
                result += lanes[i];
            for (; x < cols; x++)
                result += src[x] * src[x];
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 validMask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    const __m256i idx0 = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    const __m256i idx1 = _mm256_setr_epi32(1, 5, 0, 0, 0, 0, 0, 0);
    const __m256i idx2 = _mm256_setr_epi32(2, 6, 0, 0, 0, 0, 0, 0);
    const __m256i idx3 = _mm256_setr_epi32(3, 7, 0, 0, 0, 0, 0, 0);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            __m256 acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;
            int x = 0;
            for (; x <= cols - 2; x += 2)
            {
                __m256 v = _mm256_loadu_ps(src + x * 4);
                __m256 v0 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx0), validMask);
                __m256 v1 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx1), validMask);
                __m256 v2 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx2), validMask);
                __m256 v3 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx3), validMask);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(v1, v1));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(v2, v2));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(v3, v3));
            }

            __m256 accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                float lanes[8];
                float result = 0;
                _mm256_storeu_ps(lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                    result += src[i * 4 + c] * src[i * 4 + c];
                dst[c] = result;
            }
        }
    });
    v_cleanup();
}

} // namespace reduce_c_avx2
