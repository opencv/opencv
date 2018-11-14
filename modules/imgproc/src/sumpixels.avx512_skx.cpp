/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019, Intel Corporation, all rights reserved.
//M*/
#include "precomp.hpp"
#include "sumpixels.hpp"

namespace cv {
namespace opt_AVX512_SKX {

//
// NOTE: Look at the bottom of the file for the entry-point function for external callers
//

// At the moment only 3 channel support untilted is supported
// More channel support coming soon.
// TODO: Add support for sqsum and 1,2, and 4 channels
class IntegralCalculator_3Channel {
public:
    IntegralCalculator_3Channel() {};


    void calculate_integral_avx512(const uchar *src, size_t _srcstep,
                                   double *sum,      size_t _sumstep,
                                   double *sqsum,    size_t _sqsumstep,
                                   int width, int height, int cn)
    {
        const int srcstep = (int)(_srcstep/sizeof(uchar));
        const int sumstep = (int)(_sumstep/sizeof(double));
        const int sqsumstep = (int)(_sqsumstep/sizeof(double));
        const int ops_per_line = width * cn;

        // Clear the first line of the sum as per spec (see integral documentation)
        // Also adjust the index of sum and sqsum to be at the real 0th element
        // and not point to the border pixel so it stays in sync with the src pointer
        memset( sum, 0, (ops_per_line+cn)*sizeof(double));
        sum += cn;

        if (sqsum) {
            memset( sqsum, 0, (ops_per_line+cn)*sizeof(double));
            sqsum += cn;
        }

        // Now calculate the integral over the whole image one line at a time
        for(int y = 0; y < height; y++) {
            const uchar * src_line    = &src[y*srcstep];
            double      * sum_above   = &sum[(y)*sumstep];
            double      * sum_line    = &sum_above[sumstep];
            double      * sqsum_above = (sqsum) ? &sqsum[y*sqsumstep]     : NULL;
            double      * sqsum_line  = (sqsum) ? &sqsum_above[sqsumstep] : NULL;

            integral_line_3channel_avx512(src_line, sum_line, sum_above, sqsum_line, sqsum_above, ops_per_line);

        }
    }

    __always_inline
    void integral_line_3channel_avx512(const uchar *srcs,
                                       double *sums,   double *sums_above,
                                       double *sqsums, double *sqsums_above,
                                       int num_ops_in_line) const
    {
        __m512i sum_accumulator   = _mm512_set_epi64(0,0,0,0,0,0,0,0);  // holds rolling sums for the line
        __m512i sqsum_accumulator = _mm512_set_epi64(0,0,0,0,0,0,0,0);  // holds rolling sqsums for the line

        // The first element on each line must be zeroes as per spec (see integral documentation)
        set_border_pixel_value(sums, sqsums);

        // Do all 64 byte chunk operations then do the last bits that don't fit in a 64 byte chunk
        aligned_integral(     srcs, sums, sums_above, sqsums, sqsums_above, sum_accumulator, sqsum_accumulator, num_ops_in_line);
        post_aligned_integral(srcs, sums, sums_above, sqsums, sqsums_above, sum_accumulator, sqsum_accumulator, num_ops_in_line);

    }


    __always_inline
    void set_border_pixel_value(double *sums, double *sqsums) const
    {
        // Sets the border pixel value to 0s.
        // Note the hard coded -3 and the 0x7 mask is because we only support 3 channel right now
        _mm512_mask_storeu_epi64(&sums[-3], 0x7, (__m512i ){0, 0, 0, 0, 0, 0, 0, 0});
        if (sqsums)
            _mm512_mask_storeu_epi64(&sqsums[-3], 0x7, (__m512i ){0,0,0,0,0,0,0,0});
    }


    __always_inline
    void aligned_integral(const uchar *&srcs,
                          double *&sums,  double *&sums_above,
                          double *&sqsum, double *&sqsum_above,
                          __m512i &sum_accumulator, __m512i &sqsum_accumulator,
                          int num_ops_in_line) const
    {
        // This function handles full 64 byte chunks of the source data at a time until it gets to the part of
        // the line that no longer contains a full 64 byte chunk.  Other code will handle the last part.
        const int num_chunks = num_ops_in_line / 64;
        for (int index_64byte_chunk = 0; index_64byte_chunk < num_chunks; index_64byte_chunk++){
            integral_64_operations_avx512((__m512i *) srcs,
                                          (__m512i *) sums,  (__m512i *) sums_above,
                                          (__m512i *) sqsum, (__m512i *) sqsum_above,
                                          0xFFFFFFFFFFFFFFFF, sum_accumulator, sqsum_accumulator);
            srcs+=64; sums+=64; sums_above+=64;
            if (sqsum){ sqsum+= 64; sqsum_above+=64; }
        }
    }


    __always_inline
    void post_aligned_integral(const uchar *srcs,
                               const double *sums,   const double *sums_above,
                               const double *sqsum,  const double *sqsum_above,
                               __m512i &sum_accumulator, __m512i &sqsum_accumulator,
                               int num_ops_in_line) const
    {
        // This function handles the last few straggling operations that are not a full chunk of 64 operations
        // We use the same algorithm, but we calculate a different operation mask.
        const ulong num_operations = (ulong) num_ops_in_line - (num_ops_in_line / 64 * 64);
        __mmask64 operation_mask = (1ULL<<num_operations) - 1ULL;

        integral_64_operations_avx512((__m512i *) srcs, (__m512i *) sums, (__m512i *) sums_above,
                                      (__m512i *) sqsum, (__m512i *) sqsum_above,
                                      operation_mask, sum_accumulator, sqsum_accumulator);
    }




    void integral_64_operations_avx512(const __m512i *srcs,
                                       __m512i *sums,       const __m512i *sums_above,
                                       __m512i *sqsums,     const __m512i *sqsums_above,
                                       __mmask64 data_mask,
                                       __m512i &sum_accumulator, __m512i &sqsum_accumulator) const
    {
        __m512i src_64byte_chunk = read_64_bytes(srcs, data_mask);

        for(int num_16byte_chunks=0; num_16byte_chunks<4; num_16byte_chunks++) {
            __m128i src_16bytes = _mm512_extracti64x2_epi64(src_64byte_chunk, 0x0); // Get lower 16 bytes of data

            for (int num_8byte_chunks = 0; num_8byte_chunks < 2; num_8byte_chunks++) {

                __m512i src_longs = convert_lower_8bytes_to_longs(src_16bytes);

                // Calculate integral for the sum on the 8 entries
                integral_8_operations(src_longs, sums_above, data_mask, sums, sum_accumulator);
                sums++; sums_above++;

                if (sqsums){ // Calculate integral for the sum on the 8 entries
                    __m512i squared_source = _mm512_mullo_epi64(src_longs, src_longs);

                    integral_8_operations(squared_source, sqsums_above, data_mask, sqsums, sqsum_accumulator);
                    sqsums++; sqsums_above++;
                }

                // Prepare for next iteration of inner loop
                // shift source to align next 8 bytes to lane 0 and shift the mask
                src_16bytes = shift_right_8_bytes(src_16bytes);
                data_mask = data_mask >> 8;

            }

            // Prepare for next iteration of outer loop
            src_64byte_chunk = shift_right_16_bytes(src_64byte_chunk);
        }
    }



    __always_inline
    void integral_8_operations(const __m512i src_longs, const __m512i *above_values_ptr, __mmask64 data_mask,
                               __m512i *results_ptr, __m512i &accumulator) const
     {
        _mm512_mask_storeu_pd(
                results_ptr,   // Store the result here
                data_mask,     // Using the data mask to avoid overrunning the line
                calculate_integral( // Writing the value of the integral derived from:
                        src_longs,                                           // input data
                        _mm512_maskz_loadu_pd(data_mask, above_values_ptr),  // and the results from line above
                        accumulator                                          // keeping track of the accumulator
                )
        );
    }


    __always_inline
    __m512d calculate_integral(__m512i src_longs, const __m512d above_values, __m512i &accumulator) const
    {
        __m512i carryover_idxs = _mm512_set_epi64(6, 5, 7, 6, 5, 7, 6, 5);

        // Align data to prepare for the adds:
        //    shifts data left by 3 and 6 qwords(lanes) and gets rolling sum in all lanes
        //   Vertical LANES:     76543210
        //   src_longs       :   HGFEDCBA
        //   shited3lanes    : + EDCBA
        //   shifted6lanes   : + BA
        //   carry_over_idxs : + 65765765  (index position of result from previous iteration)
        //                     = integral
        __m512i shifted3lanes = _mm512_maskz_expand_epi64(0xF8, src_longs);
        __m512i shifted6lanes = _mm512_maskz_expand_epi64(0xC0, src_longs);
        __m512i carry_over    = _mm512_permutex2var_epi64(accumulator, carryover_idxs, accumulator);

        // Do the adds in tree form (shift3 + shift 6) + (current_source_values + accumulator)
        __m512i sum_shift3and6 = _mm512_add_epi64(shifted3lanes, shifted6lanes);
        __m512i sum_src_carry  = _mm512_add_epi64(src_longs, carry_over);
        accumulator            = _mm512_add_epi64(sum_shift3and6, sum_src_carry);

        // Convert to packed double and add to the line above to get the true integral value
        __m512d accumulator_pd = _mm512_cvtepu64_pd(accumulator);
        __m512d integral_pd    = _mm512_add_pd(accumulator_pd, above_values);
        return integral_pd;
    }


    __always_inline
    __m512i read_64_bytes(const __m512i *srcs, __mmask64 data_mask) const {
        return _mm512_maskz_loadu_epi8(data_mask, srcs);
    }


    __always_inline
    __m512i convert_lower_8bytes_to_longs(__m128i src_16bytes) const {
        return _mm512_cvtepu8_epi64(src_16bytes);
    }


    __m128i shift_right_8_bytes(__m128i src_16bytes) const {
        return _mm_maskz_compress_epi64(2, src_16bytes);
    }


    __m512i shift_right_16_bytes(__m512i src_64byte_chunk) const {
        return _mm512_maskz_compress_epi64(0xFC, src_64byte_chunk);
    }

};


// This is the implementation for the external callers interface entry point.
// It should be the only function called into this file from outside
// Any new implementations should be directed from here
void calculate_integral_avx512(const uchar *src,   size_t _srcstep,
                               double      *sum,   size_t _sumstep,
                               double      *sqsum, size_t _sqsumstep,
                               int width, int height, int cn)
{
    IntegralCalculator_3Channel  calculator;
    calculator.calculate_integral_avx512(src, _srcstep, sum, _sumstep, sqsum, _sqsumstep, width, height, cn);
}


} // end namespace opt_AVX_512F
} // end namespace cv

