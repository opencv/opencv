// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"


namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void write_raw_to_xrgb8888( const uint8_t* src, uint8_t* dst, const int len, const int scn );

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if (CV_SIMD || CV_SIMD_SCALABLE)
// see the comments for vecmerge_ in merge.cpp
template<typename T, typename VecT> static void
vecwrite_T_to_xrgb8888_( const T* src, T* dst, int len, int scn )
{
    const int VECSZ = VTraits<VecT>::vlanes();
    int i, i0 = 0;
    const int dcn = 4; // XRGB

    int r0 = (int)((size_t)(void*)dst % (VECSZ*sizeof(T)));

    hal::StoreMode mode = hal::STORE_ALIGNED_NOCACHE;
    if( r0 != 0 )
    {
        mode = hal::STORE_UNALIGNED;
        if( r0 % sizeof(T) == 0 && len > VECSZ*2 )
            i0 = VECSZ - (r0 / sizeof(T));
    }

    if( scn == 1 )
    {
        for( i = 0; i < len; i += VECSZ )
        {
            if( i > len - VECSZ )
            {
                i = len - VECSZ;
                mode = hal::STORE_UNALIGNED;
            }
            const VecT g = vx_load(src + i*scn);                // Gray
            v_store_interleave (dst + i*dcn, g, g, g, g, mode); // BGRx(x is any)
            if( i < i0 )
            {
                i = i0 - VECSZ;
                mode = hal::STORE_ALIGNED_NOCACHE;
            }
        }
    }
    else
    {
        CV_CheckEQ(scn, 3, "invalid scn");
        for( i = 0; i < len; i += VECSZ )
        {
            if( i > len - VECSZ )
            {
                i = len - VECSZ;
                mode = hal::STORE_UNALIGNED;
            }
            VecT b,g,r;
            v_load_deinterleave(src + i*scn, b, g, r);          // BGR
            v_store_interleave (dst + i*dcn, b, g, r, r, mode); // BGRx(x is any)
            if( i < i0 )
            {
                i = i0 - VECSZ;
                mode = hal::STORE_ALIGNED_NOCACHE;
            }
        }
    }
}
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

template<typename T> static void
write_T_to_xrgb8888_( const T* src, T* dst, const int len, const int scn )
{
    const int dcn = 4;
    const int scn_step = scn * sizeof(T);
    const int dcn_step = dcn * sizeof(T);

    if( scn == 1 )
    {
        for (int x = 0; x < len; x++, src+=scn_step, dst+=dcn_step)
        {
            const T g = src[0];
            dst[0] = g; // B
            dst[1] = g; // G
            dst[2] = g; // R
                        // x
        }
    }
    else if( scn == 3 )
    {
        for (int x = 0; x < len; x++, src+=scn_step, dst+=dcn_step)
        {
            dst[0] = src[0]; // B
            dst[1] = src[1]; // G
            dst[2] = src[2]; // R
                             // x
        }
    }
    else
    {
        CV_CheckEQ(scn, 4, "invalid scn");
        memcpy(dst, src, len * scn_step );
    }
}

// Convert from [g8] / [b8:g8:r8] / [b8/g8/r8:a8] to [b8:g8:r8:x8]
void write_raw_to_xrgb8888(const uint8_t* src, uint8_t *dst, const int len, const int scn)
{
    CV_INSTRUMENT_REGION();
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if( len >= VTraits<v_uint8>::vlanes() && ( (scn == 1) || (scn == 3) ) )
    {
        vecwrite_T_to_xrgb8888_<uint8_t, v_uint8>(src, dst, len, scn);
    }
    else
#endif // (CV_SIMD || CV_SIMD_SCALABLE)
    {
        write_T_to_xrgb8888_<uint8_t>(src, dst, len, scn);
    }
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
