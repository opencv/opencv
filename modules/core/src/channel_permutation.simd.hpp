// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include <numeric>

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

bool permuteByteChannels(NAryMatIterator& it, const int* order, int channels);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#if CV_NEON && (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))
#define CV_CHANNEL_PERMUTATION_NEON_AARCH64 1
#else
#define CV_CHANNEL_PERMUTATION_NEON_AARCH64 0
#endif

#if CV_SSSE3 || CV_CHANNEL_PERMUTATION_NEON_AARCH64
namespace {

constexpr int vectorBytes = 16;

struct ByteShuffle
{
    int offset;
#if CV_SSSE3
    v_uint8x16 masks[4];
#else
    v_uint8x16 mask;
#endif
};

template<int vectors>
inline v_uint8x16 shuffleBytes(const uchar* src, const ByteShuffle& shuffle)
{
#if CV_SSSE3
    __m128i result = _mm_shuffle_epi8(v_load(src).val, shuffle.masks[0].val);
    for (int index = 1; index < vectors; ++index)
        result = _mm_or_si128(result, _mm_shuffle_epi8(v_load(src + index * vectorBytes).val,
                                                       shuffle.masks[index].val));
    return v_uint8x16(result);
#elif CV_CHANNEL_PERMUTATION_NEON_AARCH64
    if constexpr (vectors == 1)
        return v_uint8x16(vqtbl1q_u8(vld1q_u8(src), shuffle.mask.val));
    else if constexpr (vectors == 2)
        return v_uint8x16(vqtbl2q_u8(vld1q_u8_x2(src), shuffle.mask.val));
    else if constexpr (vectors == 3)
        return v_uint8x16(vqtbl3q_u8(vld1q_u8_x3(src), shuffle.mask.val));
    else
        return v_uint8x16(vqtbl4q_u8(vld1q_u8_x4(src), shuffle.mask.val));
#endif
}

template<int vectors>
void shufflePlanes(NAryMatIterator& it, const int* order, int channels,
                   const ByteShuffle* shuffles, int blockBytes)
{
    const size_t rowBytes = it.size * channels;
    const size_t bulk = rowBytes / blockBytes * blockBytes;
    for (size_t plane = 0; plane < it.nplanes; ++plane, ++it)
    {
        const uchar* src = it.ptrs[0];
        uchar* dst = it.ptrs[1];
        if (vectors == 1 && blockBytes == vectorBytes)
        {
            for (size_t out = 0; out < bulk; out += vectorBytes)
                v_store(dst + out, shuffleBytes<1>(src + out, shuffles[0]));
        }
        else
        {
            for (size_t out = 0; out < bulk; out += blockBytes)
                for (int step = 0; step < blockBytes / vectorBytes; ++step)
                    v_store(dst + out + step * vectorBytes,
                            shuffleBytes<vectors>(src + out + shuffles[step].offset, shuffles[step]));
        }
        for (size_t out = bulk; out < rowBytes; out += channels)
            for (int channel = 0; channel < channels; ++channel)
                dst[out + channel] = src[out + order[channel]];
    }
}

} // namespace
#endif

bool permuteByteChannels(NAryMatIterator& it, const int* order, int channels)
{
#if CV_SSSE3 || CV_CHANNEL_PERMUTATION_NEON_AARCH64
#if CV_CHANNEL_PERMUTATION_NEON_AARCH64
    if (channels >= 9 && channels <= 12)
        return false;
#endif

    // Align each block to both a pixel boundary and a vector boundary.
    const int blockBytes = std::lcm(channels, vectorBytes);
    const int steps = blockBytes / vectorBytes;

    int indices[vectorBytes][vectorBytes];
    int bases[vectorBytes];
    int vectors = 1;
    for (int step = 0; step < steps; ++step)
    {
        for (int lane = 0; lane < vectorBytes; ++lane)
        {
            const int out = step * vectorBytes + lane;
            indices[step][lane] = out / channels * channels + order[out % channels];
        }
        const auto range = std::minmax_element(indices[step], indices[step] + vectorBytes);
        bases[step] = *range.first / vectorBytes * vectorBytes;
        vectors = std::max(vectors, (*range.second - bases[step]) / vectorBytes + 1);
    }

    CV_DbgAssert(vectors <= 4);
    ByteShuffle shuffles[vectorBytes];
    for (int step = 0; step < steps; ++step)
    {
        ByteShuffle& shuffle = shuffles[step];
        shuffle.offset = std::min(bases[step], blockBytes - vectors * vectorBytes);
#if CV_SSSE3
        for (int index = 0; index < vectors; ++index)
        {
            uchar mask[vectorBytes];
            for (int lane = 0; lane < vectorBytes; ++lane)
            {
                const int local = indices[step][lane] - shuffle.offset - index * vectorBytes;
                mask[lane] = static_cast<uchar>(local >= 0 && local < vectorBytes ? local : 255);
            }
            shuffle.masks[index] = v_load(mask);
        }
#else
        uchar mask[vectorBytes];
        for (int lane = 0; lane < vectorBytes; ++lane)
            mask[lane] = static_cast<uchar>(indices[step][lane] - shuffle.offset);
        shuffle.mask = v_load(mask);
#endif
    }

    switch (vectors)
    {
    case 1: shufflePlanes<1>(it, order, channels, shuffles, blockBytes); return true;
    case 2: shufflePlanes<2>(it, order, channels, shuffles, blockBytes); return true;
    case 3: shufflePlanes<3>(it, order, channels, shuffles, blockBytes); return true;
    case 4: shufflePlanes<4>(it, order, channels, shuffles, blockBytes); return true;
    default: return false;
    }
#else
    CV_UNUSED(it);
    CV_UNUSED(order);
    CV_UNUSED(channels);
    return false;
#endif
}
#undef CV_CHANNEL_PERMUTATION_NEON_AARCH64
#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
