/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Copyright (c) 2010, Paul Furgale, Chi Hay Tong
//
// The original code was written by Paul Furgale and Chi Hay Tong 
// and later optimized and prepared for integration into OpenCV by Itseez.
//
//M*/

#include "internal_shared.hpp"
#include "opencv2/gpu/device/limits_gpu.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

#define CV_PI 3.1415926535897932384626433832795f

namespace cv { namespace gpu { namespace surf
{
    ////////////////////////////////////////////////////////////////////////
    // Help funcs

    // Wrapper for host reference to pass into kernel
    template <typename T> 
    class DeviceReference
    {
    public:
        explicit DeviceReference(T& host_val) : d_ptr(0), h_ptr(&host_val)
        {
            cudaSafeCall( cudaMalloc((void**)&d_ptr, sizeof(T)) );
            cudaSafeCall( cudaMemcpy(d_ptr, h_ptr, sizeof(T), cudaMemcpyHostToDevice) );
        }

        ~DeviceReference()
        {            
            cudaSafeCall( cudaMemcpy(h_ptr, d_ptr, sizeof(T), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaFree(d_ptr) );
        }

        // Casting to device pointer
        operator T*() {return d_ptr;}
        operator const T*() const {return d_ptr;}
    private:
        T* d_ptr;
        T* h_ptr;
    };

    __device__ void clearLastBit(int* f)
    {
        *f &= ~0x1;
    }
    __device__ void clearLastBit(float& f)
    {
        clearLastBit((int*)&f);
    }

    __device__ void setLastBit(int* f)
    {
        *f |= 0x1;
    }
    __device__ void setLastBit(float& f)
    {
        setLastBit((int*)&f);
    }

    ////////////////////////////////////////////////////////////////////////
    // Global parameters

    // The maximum number of features (before subpixel interpolation) that memory is reserved for.
    __constant__ int c_max_candidates;
    // The maximum number of features that memory is reserved for.
    __constant__ int c_max_features;
    // The number of intervals in the octave.
    __constant__ int c_nIntervals;
    // Mask sizes derived from the mask parameters
    __constant__ float c_mask_width;
    // Mask sizes derived from the mask parameters
    __constant__ float c_mask_height;
    // Mask sizes derived from the mask parameters
    __constant__ float c_dxy_center_offset;
    // Mask sizes derived from the mask parameters
    __constant__ float c_dxy_half_width;
    // Mask sizes derived from the mask parameters
    __constant__ float c_dxy_scale;
    // The scale associated with the first interval of the first octave
    __constant__ float c_initialScale;
    //! The interest operator threshold
    __constant__ float c_threshold;

    // Ther octave
    __constant__ int c_octave;
    // The width of the octave buffer.
    __constant__ int c_x_size;
    // The height of the octave buffer.
    __constant__ int c_y_size;
    // The size of the octave border in pixels.
    __constant__ int c_border;
    // The step size used in this octave in pixels.
    __constant__ int c_step;

    ////////////////////////////////////////////////////////////////////////
    // Integral image texture

    texture<float, 2, cudaReadModeElementType> sumTex(0, cudaFilterModeLinear, cudaAddressModeClamp);

    __device__ float iiAreaLookupCDHalfWH(float cx, float cy, float halfWidth, float halfHeight)
    {
        float result = 0.f;

        result += tex2D(sumTex, cx - halfWidth, cy - halfHeight);
        result -= tex2D(sumTex, cx + halfWidth, cy - halfHeight);
        result -= tex2D(sumTex, cx - halfWidth, cy + halfHeight);
        result += tex2D(sumTex, cx + halfWidth, cy + halfHeight);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    // Hessian

    __device__ float evalDyy(float x, float y, float t, float mask_width, float mask_height, float fscale)
    {
        float Dyy = 0.f;

        Dyy +=     iiAreaLookupCDHalfWH(x, y, mask_width, mask_height);
        Dyy -= t * iiAreaLookupCDHalfWH(x, y, mask_width, fscale);

        Dyy *=  1.0f / (fscale * fscale);

        return Dyy;
    }

    __device__ float evalDxx(float x, float y, float t, float mask_width, float mask_height, float fscale)
    {
    	float Dxx = 0.f;
	
	    Dxx +=     iiAreaLookupCDHalfWH(x, y, mask_height, mask_width);
	    Dxx -= t * iiAreaLookupCDHalfWH(x, y, fscale     , mask_width);

	    Dxx *=  1.0f / (fscale * fscale);

	    return Dxx;
    }
    
    __device__ float evalDxy(float x, float y, float fscale)
    {
    	float center_offset =  c_dxy_center_offset  * fscale;
	    float half_width    =  c_dxy_half_width     * fscale;

	    float Dxy = 0.f;

	    Dxy += iiAreaLookupCDHalfWH(x - center_offset, y - center_offset, half_width, half_width);
	    Dxy -= iiAreaLookupCDHalfWH(x - center_offset, y + center_offset, half_width, half_width);
	    Dxy += iiAreaLookupCDHalfWH(x + center_offset, y + center_offset, half_width, half_width);
	    Dxy -= iiAreaLookupCDHalfWH(x + center_offset, y - center_offset, half_width, half_width);
	
	    Dxy *= 1.0f / (fscale * fscale);

	    return Dxy;
    }

    __device__ float calcScale(int hidx_z)
    {
        float d = (c_initialScale * (1 << c_octave)) / (c_nIntervals - 2);
        return c_initialScale * (1 << c_octave) + d * (hidx_z - 1.0f) + 0.5f;
    }
    
    __global__ void fasthessian(PtrStepf hessianBuffer)
    {
  	    // Determine the indices in the Hessian buffer
        int hidx_x = threadIdx.x + blockIdx.x * blockDim.x;
        int hidx_y = threadIdx.y + blockIdx.y * blockDim.y;
        int hidx_z = threadIdx.z;

        float fscale = calcScale(hidx_z);

	    // Compute the lookup location of the mask center
        float x = hidx_x * c_step + c_border;
        float y = hidx_y * c_step + c_border;

	    // Scale the mask dimensions according to the scale
        if (hidx_x < c_x_size && hidx_y < c_y_size && hidx_z < c_nIntervals)
        {
	        float mask_width =  c_mask_width  * fscale;
	        float mask_height = c_mask_height * fscale;

	        // Compute the filter responses
	        float Dyy = evalDyy(x, y, c_mask_height, mask_width, mask_height, fscale);
	        float Dxx = evalDxx(x, y, c_mask_height, mask_width, mask_height, fscale);
	        float Dxy = evalDxy(x, y, fscale);
	
	        // Combine the responses and store the Laplacian sign
	        float result = (Dxx * Dyy) - c_dxy_scale * (Dxy * Dxy);

	        if (Dxx + Dyy > 0.f)
	            setLastBit(result);
	        else
	            clearLastBit(result);

	        hessianBuffer.ptr(c_y_size * hidx_z + hidx_y)[hidx_x] = result;
        }
    }   

    void fasthessian_gpu(PtrStepf hessianBuffer, int nIntervals, int x_size, int y_size)
    {
        dim3 threads;
        threads.x = 16;
        threads.y = 8;
        threads.z = nIntervals;

        dim3 grid;
        grid.x = divUp(x_size, threads.x);
        grid.y = divUp(y_size, threads.y);
        grid.z = 1;

  	    fasthessian<<<grid, threads>>>(hessianBuffer);

        cudaSafeCall( cudaThreadSynchronize() );
	}

    ////////////////////////////////////////////////////////////////////////
    // NONMAX
    
    texture<int, 2, cudaReadModeElementType> maskSumTex(0, cudaFilterModePoint, cudaAddressModeClamp);

    struct WithOutMask
    {
        static __device__ bool check(float, float, float)
        {
            return true;
        }
    };
    struct WithMask
    {
        static __device__ bool check(float x, float y, float fscale)
        {
	        float half_width = fscale / 2;
    
	        float result = 0.f;

            result += tex2D(maskSumTex, x - half_width, y - half_width);
            result -= tex2D(maskSumTex, x + half_width, y - half_width);
            result -= tex2D(maskSumTex, x - half_width, y + half_width);
            result += tex2D(maskSumTex, x + half_width, y + half_width);
	
	        result /= (fscale * fscale);

            return (result >= 0.5f);
        }
    };

    template <typename Mask>
    __global__ void nonmaxonly(PtrStepf hessianBuffer, int4* maxPosBuffer, unsigned int* maxCounter)
    {        
        #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 110

        extern __shared__ float fh_vals[];

        // The hidx variables are the indices to the hessian buffer.
        int hidx_x = threadIdx.x + blockIdx.x * (blockDim.x - 2);
        int hidx_y = threadIdx.y + blockIdx.y * (blockDim.y - 2);
        int hidx_z = threadIdx.z;
        int localLin = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

        // Is this thread within the hessian buffer?
        if (hidx_x < c_x_size && hidx_y < c_y_size && hidx_z < c_nIntervals)
        {
            fh_vals[localLin] = hessianBuffer.ptr(c_y_size * hidx_z + hidx_y)[hidx_x];
        }
        __syncthreads();
    
        // Is this location one of the ones being processed for nonmax suppression.
        // Blocks overlap by one so we don't process the border threads.
        bool inBounds2 = threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && hidx_x < c_x_size - 1 
            &&           threadIdx.y > 0 && threadIdx.y < blockDim.y-1 && hidx_y < c_y_size - 1
            &&           threadIdx.z > 0 && threadIdx.z < blockDim.z-1;

        float val = fh_vals[localLin];

	    // Compute the lookup location of the mask center
        float x = hidx_x * c_step + c_border;
        float y = hidx_y * c_step + c_border;
        float fscale = calcScale(hidx_z);

        if (inBounds2 && val >= c_threshold && Mask::check(x, y, fscale))
        {
            // Check to see if we have a max (in its 26 neighbours)
            int zoff = blockDim.x * blockDim.y;
            bool condmax  =  val > fh_vals[localLin                     + 1]
            &&               val > fh_vals[localLin                     - 1]
            &&               val > fh_vals[localLin        - blockDim.x + 1]
            &&               val > fh_vals[localLin        - blockDim.x    ]
            &&               val > fh_vals[localLin        - blockDim.x - 1]
            &&               val > fh_vals[localLin        + blockDim.x + 1]
            &&               val > fh_vals[localLin        + blockDim.x    ]
            &&               val > fh_vals[localLin        + blockDim.x - 1]
      
            &&               val > fh_vals[localLin - zoff              + 1]
            &&               val > fh_vals[localLin - zoff                 ]
            &&               val > fh_vals[localLin - zoff              - 1]
            &&               val > fh_vals[localLin - zoff - blockDim.x + 1]
            &&               val > fh_vals[localLin - zoff - blockDim.x    ]
            &&               val > fh_vals[localLin - zoff - blockDim.x - 1]
            &&               val > fh_vals[localLin - zoff + blockDim.x + 1]
            &&               val > fh_vals[localLin - zoff + blockDim.x    ]
            &&               val > fh_vals[localLin - zoff + blockDim.x - 1]
      
            &&               val > fh_vals[localLin + zoff              + 1]
            &&               val > fh_vals[localLin + zoff                 ]
            &&               val > fh_vals[localLin + zoff              - 1]
            &&               val > fh_vals[localLin + zoff - blockDim.x + 1]
            &&               val > fh_vals[localLin + zoff - blockDim.x    ]
            &&               val > fh_vals[localLin + zoff - blockDim.x - 1]
            &&               val > fh_vals[localLin + zoff + blockDim.x + 1]
            &&               val > fh_vals[localLin + zoff + blockDim.x    ]
            &&               val > fh_vals[localLin + zoff + blockDim.x - 1]
            ;

            if(condmax) 
            {
                unsigned int i = atomicInc(maxCounter,(unsigned int) -1);
      
                if (i < c_max_candidates) 
                {
	                int4 f = {hidx_x, hidx_y, threadIdx.z, c_octave};
	                maxPosBuffer[i] = f;	
                }
            }
        }  

        #endif
    }

    void nonmaxonly_gpu(PtrStepf hessianBuffer, int4* maxPosBuffer, unsigned int& maxCounter, 
        int nIntervals, int x_size, int y_size, bool use_mask)
    {
        dim3 threads;
        threads.x = 16;
        threads.y = 8;
        threads.z = nIntervals;

        dim3 grid;
        grid.x = divUp(x_size, threads.x - 2);
        grid.y = divUp(y_size, threads.y - 2);
        grid.z = 1;

        const size_t smem_size = threads.x * threads.y * threads.z * sizeof(float);

        DeviceReference<unsigned int> maxCounterWrapper(maxCounter);

        if (use_mask)
            nonmaxonly<WithMask><<<grid, threads, smem_size>>>(hessianBuffer, maxPosBuffer, maxCounterWrapper);
        else
            nonmaxonly<WithOutMask><<<grid, threads, smem_size>>>(hessianBuffer, maxPosBuffer, maxCounterWrapper);

        cudaSafeCall( cudaThreadSynchronize() );
    }

    ////////////////////////////////////////////////////////////////////////
    // INTERPOLATION
    
    #define MID_IDX 1
    __global__ void fh_interp_extremum(PtrStepf hessianBuffer, const int4* maxPosBuffer, 
        KeyPoint_GPU* featuresBuffer, unsigned int* featureCounter)
    {        
        #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 110

        int hidx_x = maxPosBuffer[blockIdx.x].x - 1 + threadIdx.x;
        int hidx_y = maxPosBuffer[blockIdx.x].y - 1 + threadIdx.y;
        int hidx_z = maxPosBuffer[blockIdx.x].z - 1 + threadIdx.z;

        __shared__ float fh_vals[3][3][3];
        __shared__ KeyPoint_GPU p;

        fh_vals[threadIdx.z][threadIdx.y][threadIdx.x] = hessianBuffer.ptr(c_y_size * hidx_z + hidx_y)[hidx_x];
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            __shared__ float H[3][3];

            //dxx
            H[0][0] =    fh_vals[MID_IDX    ][MID_IDX + 1][MID_IDX    ] 
	        -       2.0f*fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX    ]
	        +            fh_vals[MID_IDX    ][MID_IDX - 1][MID_IDX    ];

            //dyy
            H[1][1] =    fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX + 1] 
	        -       2.0f*fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX    ]
	        +            fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX - 1];
      
            //dss
            H[2][2] =    fh_vals[MID_IDX + 1][MID_IDX    ][MID_IDX    ] 
	        -       2.0f*fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX    ]
	        +            fh_vals[MID_IDX - 1][MID_IDX    ][MID_IDX    ];

            //dxy
            H[0][1]= 0.25f*
                (fh_vals[MID_IDX    ][MID_IDX + 1][MID_IDX + 1] -
		         fh_vals[MID_IDX    ][MID_IDX - 1][MID_IDX + 1] -
		         fh_vals[MID_IDX    ][MID_IDX + 1][MID_IDX - 1] + 
		         fh_vals[MID_IDX    ][MID_IDX - 1][MID_IDX - 1]);
      
            //dxs
            H[0][2]= 0.25f*
                (fh_vals[MID_IDX + 1][MID_IDX + 1][MID_IDX    ] -
		         fh_vals[MID_IDX + 1][MID_IDX - 1][MID_IDX    ] -
		         fh_vals[MID_IDX - 1][MID_IDX + 1][MID_IDX    ] + 
		         fh_vals[MID_IDX - 1][MID_IDX - 1][MID_IDX    ]);

            //dys
            H[1][2]= 0.25f*
                (fh_vals[MID_IDX + 1][MID_IDX    ][MID_IDX + 1] -
		         fh_vals[MID_IDX + 1][MID_IDX    ][MID_IDX - 1] -
		         fh_vals[MID_IDX - 1][MID_IDX    ][MID_IDX + 1] + 
		         fh_vals[MID_IDX - 1][MID_IDX    ][MID_IDX - 1]);

            //dyx = dxy
            H[1][0] = H[0][1];

            //dsx = dxs
            H[2][0] = H[0][2];

            //dsy = dys
            H[2][1] = H[1][2];

            __shared__ float dD[3];

            //dx
            dD[0] = 0.5f*(fh_vals[MID_IDX    ][MID_IDX + 1][MID_IDX    ] -
	 	        fh_vals[MID_IDX    ][MID_IDX - 1][MID_IDX    ]);
            //dy
            dD[1] = 0.5f*(fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX + 1] -
		        fh_vals[MID_IDX    ][MID_IDX    ][MID_IDX - 1]);
            //ds
            dD[2] = 0.5f*(fh_vals[MID_IDX + 1][MID_IDX    ][MID_IDX    ] -
    		    fh_vals[MID_IDX - 1][MID_IDX    ][MID_IDX    ]);

            __shared__ float invdet;
            invdet = 1.f /
                (
                H[0][0]*H[1][1]*H[2][2] 
                +   H[0][1]*H[1][2]*H[2][0]
                +   H[0][2]*H[1][0]*H[2][1]
                -   H[0][0]*H[1][2]*H[2][1]
                -   H[0][1]*H[1][0]*H[2][2]
                -   H[0][2]*H[1][1]*H[2][0]
                );

            //   // 1-based entries of a 3x3 inverse
            //   /*             [ |a22 a23|   |a12 a13|  |a12 a13|]     */
            //   /*             [ |a32 a33|  -|a32 a33|  |a22 a23|]     */
            //   /*             [                                 ]     */
            //   /*             [ |a21 a23|   |a11 a13|  |a11 a13|]     */
            //   /*    A^(-1) = [-|a31 a33|   |a31 a33| -|a21 a23|] / d */
            //   /*             [                                 ]     */
            //   /*             [ |a21 a22|   |a11 a12|  |a11 a12|]     */
            //   /*             [ |a31 a32|  -|a31 a32|  |a21 a22|]     */

            __shared__ float Hinv[3][3];
            Hinv[0][0] =  invdet*(H[1][1]*H[2][2]-H[1][2]*H[2][1]);
            Hinv[0][1] = -invdet*(H[0][1]*H[2][2]-H[0][2]*H[2][1]);
            Hinv[0][2] =  invdet*(H[0][1]*H[1][2]-H[0][2]*H[1][1]);

            Hinv[1][0] = -invdet*(H[1][0]*H[2][2]-H[1][2]*H[2][0]);
            Hinv[1][1] =  invdet*(H[0][0]*H[2][2]-H[0][2]*H[2][0]);
            Hinv[1][2] = -invdet*(H[0][0]*H[1][2]-H[0][2]*H[1][0]);

            Hinv[2][0] =  invdet*(H[1][0]*H[2][1]-H[1][1]*H[2][0]);
            Hinv[2][1] = -invdet*(H[0][0]*H[2][1]-H[0][1]*H[2][0]);
            Hinv[2][2] =  invdet*(H[0][0]*H[1][1]-H[0][1]*H[1][0]);

            __shared__ float x[3];

            x[0] = -(Hinv[0][0]*(dD[0]) + Hinv[0][1]*(dD[1]) + Hinv[0][2]*(dD[2]));
            x[1] = -(Hinv[1][0]*(dD[0]) + Hinv[1][1]*(dD[1]) + Hinv[1][2]*(dD[2]));
            x[2] = -(Hinv[2][0]*(dD[0]) + Hinv[2][1]*(dD[1]) + Hinv[2][2]*(dD[2]));

            if (fabs(x[0]) < 1.f && fabs(x[1]) < 1.f && fabs(x[2]) < 1.f) 
            { 
                // if the step is within the interpolation region, perform it
	
	            // Get a new feature index.
	            unsigned int i = atomicInc(featureCounter, (unsigned int)-1);

 	            if (i < c_max_features) 
                {	  
	                p.x = ((float)maxPosBuffer[blockIdx.x].x + x[1]) * (float)c_step + c_border;
	                p.y = ((float)maxPosBuffer[blockIdx.x].y + x[0]) * (float)c_step + c_border;

 	                if (x[2] > 0)
 	                {
                        float a = calcScale(maxPosBuffer[blockIdx.x].z);
                        float b = calcScale(maxPosBuffer[blockIdx.x].z + 1);

	                    p.size = (1.f - x[2]) * a + x[2] * b;
 	                } 
 	                else
 	                {
                        float a = calcScale(maxPosBuffer[blockIdx.x].z);
                        float b = calcScale(maxPosBuffer[blockIdx.x].z - 1);

	                    p.size = (1.f + x[2]) * a - x[2] * b;
 	                }

	                p.octave = c_octave;
			
	                p.response = fh_vals[MID_IDX][MID_IDX][MID_IDX];

	                // Should we split up this transfer over many threads?
	                featuresBuffer[i] = p;
	            }
            } // If the subpixel interpolation worked
        } // If this is thread 0.

        #endif
    }
    #undef MID_IDX

    void fh_interp_extremum_gpu(PtrStepf hessianBuffer, const int4* maxPosBuffer, unsigned int maxCounter, 
        KeyPoint_GPU* featuresBuffer, unsigned int& featureCounter)
    {
        dim3 threads;
        threads.x = 3;
        threads.y = 3;
        threads.z = 3;
    
        dim3 grid;
        grid.x = maxCounter;
        grid.y = 1; 
        grid.z = 1;

        DeviceReference<unsigned int> featureCounterWrapper(featureCounter);
    
        fh_interp_extremum<<<grid, threads>>>(hessianBuffer, maxPosBuffer, featuresBuffer, featureCounterWrapper);

        cudaSafeCall( cudaThreadSynchronize() );
    }

    ////////////////////////////////////////////////////////////////////////
    // Orientation

    // precomputed values for a Gaussian with a standard deviation of 2
    __constant__ float c_gauss1D[13] = 
    {
        0.002215924206f, 0.008764150247f, 0.026995483257f, 0.064758797833f, 
        0.120985362260f, 0.176032663382f, 0.199471140201f, 0.176032663382f, 
        0.120985362260f, 0.064758797833f, 0.026995483257f, 0.008764150247f, 
        0.002215924206f
    };

    __global__ void find_orientation(KeyPoint_GPU* features)
    {
        int tid = threadIdx.y * 17 + threadIdx.x;
        int tid2 = numeric_limits_gpu<int>::max();

        if (threadIdx.x < 13 && threadIdx.y < 13) 
        {
            tid2 = threadIdx.y * 13 + threadIdx.x;
        }

        __shared__ float texLookups[17][17];
    
        __shared__ float Edx[13*13];
        __shared__ float Edy[13*13];
        __shared__ float xys[3];

        // Read my x, y, size.
        if (tid < 3)
        {
	        xys[tid] = ((float*)(&features[blockIdx.x]))[tid];
        }
        __syncthreads();

        // Read all texture locations into memory
        // Maybe I should use __mul24 here?
        texLookups[threadIdx.x][threadIdx.y] = tex2D(sumTex, xys[SF_X] + ((int)threadIdx.x - 8) * xys[SF_SIZE], 
                  xys[SF_Y] + ((int)threadIdx.y - 8) * xys[SF_SIZE]);

        __syncthreads();

        float dx = 0.f;
        float dy = 0.f;
	 
	    // Computes lookups for all points in a 13x13 lattice.
	    // - SURF says to only use a circle, but the branching logic would slow it down
	    // - Gaussian weighting should reduce the effects of the outer points anyway
        if (tid2 < 169)
        {
	        dx -=     texLookups[threadIdx.x    ][threadIdx.y    ];
	        dx += 2.f*texLookups[threadIdx.x + 2][threadIdx.y    ];
	        dx -=     texLookups[threadIdx.x + 4][threadIdx.y    ];
	        dx +=     texLookups[threadIdx.x    ][threadIdx.y + 4];
	        dx -= 2.f*texLookups[threadIdx.x + 2][threadIdx.y + 4];
	        dx +=     texLookups[threadIdx.x + 4][threadIdx.y + 4];

	        dy -=     texLookups[threadIdx.x    ][threadIdx.y    ];
	        dy += 2.f*texLookups[threadIdx.x    ][threadIdx.y + 2];
	        dy -=     texLookups[threadIdx.x    ][threadIdx.y + 4];
	        dy +=     texLookups[threadIdx.x + 4][threadIdx.y    ];
	        dy -= 2.f*texLookups[threadIdx.x + 4][threadIdx.y + 2];
	        dy +=     texLookups[threadIdx.x + 4][threadIdx.y + 4];

	        float g = c_gauss1D[threadIdx.x] * c_gauss1D[threadIdx.y];

	        Edx[tid2] = dx * g;
	        Edy[tid2] = dy * g;
        }

        __syncthreads();

        // This is a scan to get the summed dx, dy values.
        // Gets 128-168
        if (tid < 41)
        {
            Edx[tid] += Edx[tid + 128]; 
        } 
        __syncthreads(); 
        if (tid < 64) 
        {
            Edx[tid] += Edx[tid + 64]; 
        } 
        __syncthreads(); 
        if (tid < 32) 
        {
            volatile float* smem = Edx;

            smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }

        // Gets 128-168
        if (tid < 41) 
        {
            Edy[tid] += Edy[tid + 128]; 
        } 
        __syncthreads(); 
        if (tid < 64) 
        {
            Edy[tid] += Edy[tid + 64]; 
        } 
        __syncthreads(); 
        if (tid < 32) 
        {
            volatile float* smem = Edy;

            smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }
 
        // Thread 0 saves back the result.
        if (tid == 0)
        {
	        features[blockIdx.x].angle = -atan2(Edy[0], Edx[0]) * (180.0f / CV_PI);
        }
    }

    void find_orientation_gpu(KeyPoint_GPU* features, int nFeatures) 
    {
        dim3 threads;
        threads.x = 17;
        threads.y = 17;

        dim3 grid;
        grid.x = nFeatures;
        grid.y = 1;
        grid.z = 1;

        find_orientation<<<grid, threads>>>(features);
        cudaSafeCall( cudaThreadSynchronize() );
    }

    ////////////////////////////////////////////////////////////////////////
    // Descriptors

    // precomputed values for a Gaussian with a standard deviation of 3.3
    // - it appears SURF uses a different value, but not sure what it is
    __constant__ float c_3p3gauss1D[20] = 
    {
        0.001917811039f, 0.004382549939f, 0.009136246641f, 0.017375153068f, 0.030144587513f,
		0.047710056854f, 0.068885910797f, 0.090734146446f, 0.109026229640f, 0.119511889092f,
		0.119511889092f, 0.109026229640f, 0.090734146446f, 0.068885910797f, 0.047710056854f,
		0.030144587513f, 0.017375153068f, 0.009136246641f, 0.004382549939f, 0.001917811039f
    };   

    template <int BLOCK_DIM_X>
    __global__ void normalize_descriptors(PtrStepf descriptors)
    {
        // no need for thread ID
        float* descriptor_base = descriptors.ptr(blockIdx.x);

        // read in the unnormalized descriptor values (squared)
        __shared__ float sqDesc[BLOCK_DIM_X];
        const float lookup = descriptor_base[threadIdx.x];
        sqDesc[threadIdx.x] = lookup * lookup;
        __syncthreads();

        if (BLOCK_DIM_X >= 128)
        {
            if (threadIdx.x < 64)
	            sqDesc[threadIdx.x] += sqDesc[threadIdx.x + 64];
            __syncthreads();
        }

        // reduction to get total
        if (threadIdx.x < 32)
        {
            volatile float* smem = sqDesc;

	        smem[threadIdx.x] += smem[threadIdx.x + 32];
	        smem[threadIdx.x] += smem[threadIdx.x + 16];
	        smem[threadIdx.x] += smem[threadIdx.x + 8];
	        smem[threadIdx.x] += smem[threadIdx.x + 4];
	        smem[threadIdx.x] += smem[threadIdx.x + 2];
	        smem[threadIdx.x] += smem[threadIdx.x + 1];
        }

        // compute length (square root)
        __shared__ float len;
        if (threadIdx.x == 0)
        {
	        len = sqrtf(sqDesc[0]);
        }
        __syncthreads();

        // normalize and store in output
        descriptor_base[threadIdx.x] = lookup / len;	
    }

    __device__ void calc_dx_dy(float sdx[4][4][25], float sdy[4][4][25], const KeyPoint_GPU* features)
    {
        // get the interest point parameters (x, y, size, response, angle)
        __shared__ float ipt[5];
        if (threadIdx.x < 5 && threadIdx.y == 0 && threadIdx.z == 0)
        {
	        ipt[threadIdx.x] = ((float*)(&features[blockIdx.x]))[threadIdx.x];
        }
        __syncthreads();

        float sin_theta, cos_theta;
        sincosf(ipt[SF_ANGLE] * (CV_PI / 180.0f), &sin_theta, &cos_theta);

        // Compute sampling points
        // since grids are 2D, need to compute xBlock and yBlock indices
        const int xIndex = threadIdx.y * 5 + threadIdx.x % 5;
        const int yIndex = threadIdx.z * 5 + threadIdx.x / 5;

        // Compute rotated sampling points
        // (clockwise rotation since we are rotating the lattice)
        // (subtract 9.5f to start sampling at the top left of the lattice, 0.5f is to space points out properly - there is no center pixel)
        const float sample_x = ipt[SF_X] + (cos_theta * ((float) (xIndex-9.5f)) * ipt[SF_SIZE] 
            + sin_theta * ((float) (yIndex-9.5f)) * ipt[SF_SIZE]);
        const float sample_y = ipt[SF_Y] + (-sin_theta * ((float) (xIndex-9.5f)) * ipt[SF_SIZE] 
            + cos_theta * ((float) (yIndex-9.5f)) * ipt[SF_SIZE]);

        // gather integral image lookups for Haar wavelets at each point (some lookups are shared between dx and dy)
        //	a b c
        //	d	f
        //	g h i

        const float a = tex2D(sumTex, sample_x - ipt[SF_SIZE], sample_y - ipt[SF_SIZE]);
        const float b = tex2D(sumTex, sample_x,                sample_y - ipt[SF_SIZE]);
        const float c = tex2D(sumTex, sample_x + ipt[SF_SIZE], sample_y - ipt[SF_SIZE]);
        const float d = tex2D(sumTex, sample_x - ipt[SF_SIZE], sample_y);
        const float f = tex2D(sumTex, sample_x + ipt[SF_SIZE], sample_y);
        const float g = tex2D(sumTex, sample_x - ipt[SF_SIZE], sample_y + ipt[SF_SIZE]);
        const float h = tex2D(sumTex, sample_x,                sample_y + ipt[SF_SIZE]);
        const float i = tex2D(sumTex, sample_x + ipt[SF_SIZE], sample_y + ipt[SF_SIZE]);

        // compute axis-aligned HaarX, HaarY
        // (could group the additions together into multiplications)
        const float gauss = c_3p3gauss1D[xIndex] * c_3p3gauss1D[yIndex]; // separable because independent (circular)
        const float aa_dx = gauss * (-(a-b-g+h) + (b-c-h+i));            // unrotated dx
        const float aa_dy = gauss * (-(a-c-d+f) + (d-f-g+i));            // unrotated dy

        // rotate responses (store all dxs then all dys)
        // - counterclockwise rotation to rotate back to zero orientation
        sdx[threadIdx.z][threadIdx.y][threadIdx.x] = aa_dx * cos_theta - aa_dy * sin_theta; // rotated dx
        sdy[threadIdx.z][threadIdx.y][threadIdx.x] = aa_dx * sin_theta + aa_dy * cos_theta; // rotated dy
    }

    __device__ void reduce_sum(float sdata1[4][4][25], float sdata2[4][4][25], float sdata3[4][4][25],
        float sdata4[4][4][25])
    {
        // first step is to reduce from 25 to 16
        if (threadIdx.x < 9) // use 9 threads
        {
	        sdata1[threadIdx.z][threadIdx.y][threadIdx.x] += sdata1[threadIdx.z][threadIdx.y][threadIdx.x + 16];
	        sdata2[threadIdx.z][threadIdx.y][threadIdx.x] += sdata2[threadIdx.z][threadIdx.y][threadIdx.x + 16];
	        sdata3[threadIdx.z][threadIdx.y][threadIdx.x] += sdata3[threadIdx.z][threadIdx.y][threadIdx.x + 16];
	        sdata4[threadIdx.z][threadIdx.y][threadIdx.x] += sdata4[threadIdx.z][threadIdx.y][threadIdx.x + 16];
        }
        __syncthreads();

        // sum (reduce) from 16 to 1 (unrolled - aligned to a half-warp)
        if (threadIdx.x < 16)
        {
            volatile float* smem = sdata1[threadIdx.z][threadIdx.y];

	        smem[threadIdx.x] += smem[threadIdx.x + 8];
	        smem[threadIdx.x] += smem[threadIdx.x + 4];
	        smem[threadIdx.x] += smem[threadIdx.x + 2];
	        smem[threadIdx.x] += smem[threadIdx.x + 1];

            smem = sdata2[threadIdx.z][threadIdx.y];

	        smem[threadIdx.x] += smem[threadIdx.x + 8];
	        smem[threadIdx.x] += smem[threadIdx.x + 4];
	        smem[threadIdx.x] += smem[threadIdx.x + 2];
	        smem[threadIdx.x] += smem[threadIdx.x + 1];

            smem = sdata3[threadIdx.z][threadIdx.y];

	        smem[threadIdx.x] += smem[threadIdx.x + 8];
	        smem[threadIdx.x] += smem[threadIdx.x + 4];
	        smem[threadIdx.x] += smem[threadIdx.x + 2];
	        smem[threadIdx.x] += smem[threadIdx.x + 1];

            smem = sdata4[threadIdx.z][threadIdx.y];

	        smem[threadIdx.x] += smem[threadIdx.x + 8];
	        smem[threadIdx.x] += smem[threadIdx.x + 4];
	        smem[threadIdx.x] += smem[threadIdx.x + 2];
	        smem[threadIdx.x] += smem[threadIdx.x + 1];
        }
    }

    // Spawn 16 blocks per interest point
    // - computes unnormalized 64 dimensional descriptor, puts it into d_descriptors in the correct location
    __global__ void compute_descriptors64(PtrStepf descriptors, const KeyPoint_GPU* features)
    {        
        // 2 floats (dx, dy) for each thread (5x5 sample points in each sub-region)
        __shared__ float sdx[4][4][25]; 
        __shared__ float sdy[4][4][25];

        calc_dx_dy(sdx, sdy, features);
        __syncthreads();

        __shared__ float sdxabs[4][4][25];
        __shared__ float sdyabs[4][4][25];
        
        sdxabs[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdx[threadIdx.z][threadIdx.y][threadIdx.x]); // |dx| array
        sdyabs[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdy[threadIdx.z][threadIdx.y][threadIdx.x]); // |dy| array
        __syncthreads();

        reduce_sum(sdx, sdy, sdxabs, sdyabs);

        float* descriptors_block = descriptors.ptr(blockIdx.x) + threadIdx.z * 16 + threadIdx.y * 4;

        // write dx, dy, |dx|, |dy|
        if (threadIdx.x == 0)
        {
            descriptors_block[0] = sdx[threadIdx.z][threadIdx.y][0];
            descriptors_block[1] = sdy[threadIdx.z][threadIdx.y][0];
            descriptors_block[2] = sdxabs[threadIdx.z][threadIdx.y][0];
            descriptors_block[3] = sdyabs[threadIdx.z][threadIdx.y][0];
        }
    }    

    // Spawn 16 blocks per interest point
    // - computes unnormalized 128 dimensional descriptor, puts it into d_descriptors in the correct location
    __global__ void compute_descriptors128(PtrStepf descriptors, const KeyPoint_GPU* features)
    {        
        // 2 floats (dx,dy) for each thread (5x5 sample points in each sub-region)
        __shared__ float sdx[4][4][25]; 
        __shared__ float sdy[4][4][25];
        
        calc_dx_dy(sdx, sdy, features);
        __syncthreads();

        // sum (reduce) 5x5 area response
        __shared__ float sd1[4][4][25];
        __shared__ float sd2[4][4][25];
        __shared__ float sdabs1[4][4][25]; 
        __shared__ float sdabs2[4][4][25];

        if (sdy[threadIdx.z][threadIdx.y][threadIdx.x] >= 0)
        {
            sd1[threadIdx.z][threadIdx.y][threadIdx.x] = sdx[threadIdx.z][threadIdx.y][threadIdx.x];
            sdabs1[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdx[threadIdx.z][threadIdx.y][threadIdx.x]);
            sd2[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sdabs2[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sd1[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sdabs1[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sd2[threadIdx.z][threadIdx.y][threadIdx.x] = sdx[threadIdx.z][threadIdx.y][threadIdx.x];
            sdabs2[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdx[threadIdx.z][threadIdx.y][threadIdx.x]);
        }
        __syncthreads();
        
        reduce_sum(sd1, sd2, sdabs1, sdabs2);
        
        float* descriptors_block = descriptors.ptr(blockIdx.x) + threadIdx.z * 32 + threadIdx.y * 8;

        // write dx (dy >= 0), |dx| (dy >= 0), dx (dy < 0), |dx| (dy < 0)
        if (threadIdx.x == 0)
        {
	        descriptors_block[0] = sd1[threadIdx.z][threadIdx.y][0];
	        descriptors_block[1] = sdabs1[threadIdx.z][threadIdx.y][0];
	        descriptors_block[2] = sd2[threadIdx.z][threadIdx.y][0];
	        descriptors_block[3] = sdabs2[threadIdx.z][threadIdx.y][0];
        }
        __syncthreads();

        if (sdx[threadIdx.z][threadIdx.y][threadIdx.x] >= 0)
        {
            sd1[threadIdx.z][threadIdx.y][threadIdx.x] = sdy[threadIdx.z][threadIdx.y][threadIdx.x];
            sdabs1[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdy[threadIdx.z][threadIdx.y][threadIdx.x]);
            sd2[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sdabs2[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sd1[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sdabs1[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
            sd2[threadIdx.z][threadIdx.y][threadIdx.x] = sdy[threadIdx.z][threadIdx.y][threadIdx.x];
            sdabs2[threadIdx.z][threadIdx.y][threadIdx.x] = fabs(sdy[threadIdx.z][threadIdx.y][threadIdx.x]);
        }
        __syncthreads();

        reduce_sum(sd1, sd2, sdabs1, sdabs2);

        // write dy (dx >= 0), |dy| (dx >= 0), dy (dx < 0), |dy| (dx < 0)
        if (threadIdx.x == 0)
        {
	        descriptors_block[4] = sd1[threadIdx.z][threadIdx.y][0];
	        descriptors_block[5] = sdabs1[threadIdx.z][threadIdx.y][0];
	        descriptors_block[6] = sd2[threadIdx.z][threadIdx.y][0];
	        descriptors_block[7] = sdabs2[threadIdx.z][threadIdx.y][0];
        }
    }

    void compute_descriptors_gpu(const DevMem2Df& descriptors, const KeyPoint_GPU* features, int nFeatures)
    {
        // compute unnormalized descriptors, then normalize them - odd indexing since grid must be 2D
        
        if (descriptors.cols == 64)
        {
            compute_descriptors64<<<dim3(nFeatures, 1, 1), dim3(25, 4, 4)>>>(descriptors, features);
            cudaSafeCall( cudaThreadSynchronize() );

            normalize_descriptors<64><<<dim3(nFeatures, 1, 1), dim3(64, 1, 1)>>>(descriptors);
            cudaSafeCall( cudaThreadSynchronize() );
        }
        else
        {
            compute_descriptors128<<<dim3(nFeatures, 1, 1), dim3(25, 4, 4)>>>(descriptors, features);
            cudaSafeCall( cudaThreadSynchronize() );

            normalize_descriptors<128><<<dim3(nFeatures, 1, 1), dim3(128, 1, 1)>>>(descriptors);
            cudaSafeCall( cudaThreadSynchronize() );
        }
    }
}}}
