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
//M*/

#if !defined CUDA_DISABLER

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

namespace cv { namespace cuda { namespace device
{
    // // namespace imgproc
    // {

        __device__ void histogramAddAndSub8(int* H, const int * hist_colAdd,const int * hist_colSub){
            int tx = threadIdx.x;
            if (tx<8){
                H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
            }
        }

        __device__ void histogramMultipleAdd8(int* H, const int * hist_col,int histCount){
            int tx = threadIdx.x;
            if (tx<8){
                int temp=H[tx];
                for(int i=0; i<histCount; i++)
                    temp+=hist_col[(i<<3)+tx];
                H[tx]=temp;
            }
        }

        __device__ void histogramClear8(int* H){
            int tx = threadIdx.x;
            if (tx<8){
                H[tx]=0;
            }
        }

        __device__ void histogramAdd8(int* H, const int * hist_col){
            int tx = threadIdx.x;
            if (tx<8){
                H[tx]+=hist_col[tx];
            }
        }

        __device__ void histogramSub8(int* H, const int * hist_col){
            int tx = threadIdx.x;
            if (tx<8){
                H[tx]-=hist_col[tx];
            }
        }


        __device__ void histogramAdd32(int* H, const int * hist_col){
            int tx = threadIdx.x;
            if (tx<32){
                H[tx]+=hist_col[tx];
            }
        }

        __device__ void histogramAddAndSub32(int* H, const int * hist_colAdd,const int * hist_colSub){
            int tx = threadIdx.x;
            if (tx<32){
                H[tx]+=hist_colAdd[tx]-hist_colSub[tx];
            }
        }


        __device__ void histogramClear32(int* H){
            int tx = threadIdx.x;
            if (tx<32){
                H[tx]=0;
            }
        }

        __device__ void lucClear8(int* luc){
            int tx = threadIdx.x;
            if (tx<8)
                luc[tx]=0;
        }

        __device__ void histogramMedianPar8LookupOnly(int* H,int* Hscan, const int medPos,int* retval, int* countAtMed){
            int tx=threadIdx.x;
            *retval=*countAtMed=0;
            if(tx<8){
                Hscan[tx]=H[tx];
            }
            __syncthreads();
            if(tx<8){
                if(tx>=1 )
                  Hscan[tx]+=Hscan[tx-1];
                if(tx>=2)
                  Hscan[tx]+=Hscan[tx-2];
                if(tx>=4)
                  Hscan[tx]+=Hscan[tx-4];
            }
            __syncthreads();

            if(tx<7){
                if(Hscan[tx+1] > medPos && Hscan[tx] < medPos){
                    *retval=tx+1;
                    *countAtMed=Hscan[tx];
                }
                else if(Hscan[tx]==medPos){
                  if(Hscan[tx+1]>medPos){
                     *retval=tx+1;
                     *countAtMed=Hscan[tx];
                  }
                }
            }
        }

        __device__ void histogramMedianPar32LookupOnly(int* H,int* Hscan, const int medPos,int* retval, int* countAtMed){
            int tx=threadIdx.x;
            *retval=*countAtMed=0;
            if(tx<32){
                Hscan[tx]=H[tx];
            }
            __syncthreads();
            if(tx<32){
                if(tx>=1)
                  Hscan[tx]+=Hscan[tx-1];
                if(tx>=2)
                  Hscan[tx]+=Hscan[tx-2];
                if(tx>=4)
                  Hscan[tx]+=Hscan[tx-4];
                if(tx>=8)
                  Hscan[tx]+=Hscan[tx-8];
                if(tx>=16)
                  Hscan[tx]+=Hscan[tx-16];
            }
            __syncthreads();
            if(tx<31){
                if(Hscan[tx+1] > medPos && Hscan[tx] < medPos){
                    *retval=tx+1;
                    *countAtMed=Hscan[tx];
                }
                else if(Hscan[tx]==medPos){
                  if(Hscan[tx+1]>medPos){
                      *retval=tx+1;
                      *countAtMed=Hscan[tx];
                  }
                }
            }
         }

    __global__ void cuMedianFilterMultiBlock(PtrStepSzb src, PtrStepSzb  dest, PtrStepSzi histPar, PtrStepSzi coarseHistGrid,int r, int medPos_)
    {
        __shared__ int HCoarse[8];
        __shared__ int HCoarseScan[32];
        __shared__ int HFine[8][32];

        __shared__ int luc[8];

        __shared__ int firstBin,countAtMed, retval;

        int rows = src.rows, cols=src.cols;

        int extraRowThread=rows%gridDim.x;
        int doExtraRow=blockIdx.x<extraRowThread;
        int startRow=0, stopRow=0;
        int rowsPerBlock= rows/gridDim.x+doExtraRow;


        // The following code partitions the work to the blocks. Some blocks will do one row more
        // than other blocks. This code is responsible for doing that balancing
        if(doExtraRow){
            startRow=rowsPerBlock*blockIdx.x;
            stopRow=::min(rows, startRow+rowsPerBlock);
        }
        else{
            startRow=(rowsPerBlock+1)*extraRowThread+(rowsPerBlock)*(blockIdx.x-extraRowThread);
            stopRow=::min(rows, startRow+rowsPerBlock);
        }

        int* hist= histPar.data+cols*256*blockIdx.x;
        int* histCoarse=coarseHistGrid.data +cols*8*blockIdx.x;

        if (blockIdx.x==(gridDim.x-1))
            stopRow=rows;
        __syncthreads();
        int initNeeded=0, initVal, initStartRow, initStopRow;

        if(blockIdx.x==0){
            initNeeded=1; initVal=r+2; initStartRow=1;  initStopRow=r;
        }
        else if (startRow<(r+2)){
            initNeeded=1; initVal=r+2-startRow; initStartRow=1; initStopRow=r+startRow;
        }
        else{
            initNeeded=0; initVal=0; initStartRow=startRow-(r+1);   initStopRow=r+startRow;
        }
       __syncthreads();


        // In the original algorithm an initialization phase was required as part of the window was outside the
        // image. In this parallel version, the initializtion is required for all thread blocks that part
        // of the median filter is outside the window.
        // For all threads in the block the same code will be executed.
        if (initNeeded){
            for (int j=threadIdx.x; j<(cols); j+=blockDim.x){
                hist[j*256+src.ptr(0)[j]]=initVal;
                histCoarse[j*8+(src.ptr(0)[j]>>5)]=initVal;
            }
        }
        __syncthreads();

        // Fot all remaining rows in the median filter, add the values to the the histogram
        for (int j=threadIdx.x; j<cols; j+=blockDim.x){
            for(int i=initStartRow; i<initStopRow; i++){
                    int pos=::min(i,rows-1);
                    hist[j*256+src.ptr(pos)[j]]++;
                    histCoarse[j*8+(src.ptr(pos)[j]>>5)]++;
                }
        }
        __syncthreads();
         // Going through all the rows that the block is responsible for.
         int inc=blockDim.x*256;
         int incCoarse=blockDim.x*8;
         for(int i=startRow; i< stopRow; i++){
             // For every new row that is started the global histogram for the entire window is restarted.

             histogramClear8(HCoarse);
             lucClear8(luc);
             // Computing some necessary indices
             int possub=::max(0,i-r-1),posadd=::min(rows-1,i+r);
             int histPos=threadIdx.x*256;
             int histCoarsePos=threadIdx.x*8;
             // Going through all the elements of a specific row. Foeach histogram, a value is taken out and
             // one value is added.
             for (int j=threadIdx.x; j<cols; j+=blockDim.x){
                hist[histPos+ src.ptr(possub)[j] ]--;
                hist[histPos+ src.ptr(posadd)[j] ]++;
                histCoarse[histCoarsePos+ (src.ptr(possub)[j]>>5) ]--;
                histCoarse[histCoarsePos+ (src.ptr(posadd)[j]>>5) ]++;

                histPos+=inc;
                histCoarsePos+=incCoarse;
             }
            __syncthreads();

            histogramMultipleAdd8(HCoarse,histCoarse, 2*r+1);
//            __syncthreads();
            int cols_m_1=cols-1;

             for(int j=r;j<cols-r;j++){
                int possub=::max(j-r,0);
                int posadd=::min(j+1+r,cols_m_1);
                int medPos=medPos_;
                __syncthreads();

                histogramMedianPar8LookupOnly(HCoarse,HCoarseScan,medPos, &firstBin,&countAtMed);
                __syncthreads();

                if ( luc[firstBin] <= (j-r))
                {
                    histogramClear32(HFine[firstBin]);
                    for ( luc[firstBin] = j-r; luc[firstBin] < ::min(j+r+1,cols); luc[firstBin]++ ){
                        histogramAdd32(HFine[firstBin], hist+(luc[firstBin]*256+(firstBin<<5) ) );
                    }
                }
                else{
                    for ( ; luc[firstBin] < (j+r+1);luc[firstBin]++ ) {
                        histogramAddAndSub32(HFine[firstBin],
                        hist+(::min(luc[firstBin],cols_m_1)*256+(firstBin<<5) ),
                        hist+(::max(luc[firstBin]-2*r-1,0)*256+(firstBin<<5) ) );
                        __syncthreads();

                    }
                }
                __syncthreads();

                int leftOver=medPos-countAtMed;
                if(leftOver>=0){
                    histogramMedianPar32LookupOnly(HFine[firstBin],HCoarseScan,leftOver,&retval,&countAtMed);
                }
                else retval=0;
                __syncthreads();

                if (threadIdx.x==0){
                    dest.ptr(i)[j]=(firstBin<<5) + retval;
                }
                histogramAddAndSub8(HCoarse, histCoarse+(int)(posadd<<3),histCoarse+(int)(possub<<3));

                __syncthreads();
            }
             __syncthreads();
        }
    }

    void medianFiltering_gpu(const PtrStepSzb src, PtrStepSzb dst, PtrStepSzi devHist, PtrStepSzi devCoarseHist,int kernel, int partitions,cudaStream_t stream){
        int medPos=2*kernel*kernel+2*kernel;
        dim3 gridDim; gridDim.x=partitions;
        dim3 blockDim; blockDim.x=32;
        cuMedianFilterMultiBlock<<<gridDim,blockDim,0, stream>>>(src, dst, devHist,devCoarseHist, kernel, medPos);
        if (!stream)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

}}}

#endif
