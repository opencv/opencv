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

#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>
#include <opencv2/gpu/device/vec_math.hpp>
#include <opencv2/gpu/device/emulation.hpp>

#include <iostream>
#include <stdio.h>

namespace cv { namespace gpu { namespace device
{
    namespace ccl
    {
        enum
        {
            WARP_SIZE  = 32,
            WARP_LOG   = 5,

            CTA_SIZE_X = 32,
            CTA_SIZE_Y = 8,

            STA_SIZE_MERGE_Y = 4,
            STA_SIZE_MERGE_X = 32,

            TPB_X = 1,
            TPB_Y = 4,

            TILE_COLS = CTA_SIZE_X * TPB_X,
            TILE_ROWS = CTA_SIZE_Y * TPB_Y
        };

        template<typename T> struct IntervalsTraits
        {
            typedef T elem_type;
        };

        template<> struct IntervalsTraits<unsigned char>
        {
            typedef int dist_type;
            enum {ch = 1};
        };

        template<> struct IntervalsTraits<uchar3>
        {
            typedef int3 dist_type;
            enum {ch = 3};
        };

        template<> struct IntervalsTraits<uchar4>
        {
            typedef int4 dist_type;
            enum {ch = 4};
        };

        template<> struct IntervalsTraits<unsigned short>
        {
            typedef int dist_type;
            enum {ch = 1};
        };

        template<> struct IntervalsTraits<ushort3>
        {
            typedef int3 dist_type;
            enum {ch = 3};
        };

        template<> struct IntervalsTraits<ushort4>
        {
            typedef int4 dist_type;
            enum {ch = 4};
        };

        template<> struct IntervalsTraits<float>
        {
            typedef float dist_type;
            enum {ch = 1};
        };

        template<> struct IntervalsTraits<int>
        {
            typedef int dist_type;
            enum {ch = 1};
        };

        typedef unsigned char component;
        enum Edges { UP = 1, DOWN = 2, LEFT = 4, RIGHT = 8, EMPTY = 0xF0 };

        template<typename T, int CH> struct InInterval {};

        template<typename T> struct InInterval<T, 1>
        {
            typedef typename VecTraits<T>::elem_type E;
            __host__ __device__ __forceinline__ InInterval(const float4& _lo, const float4& _hi) : lo((E)(-_lo.x)), hi((E)_hi.x) {};
            T lo, hi;

            template<typename I> __device__ __forceinline__ bool operator() (const I& a, const I& b) const
            {
                I d = a - b;
                return lo <= d && d <= hi;
            }
        };


        template<typename T> struct InInterval<T, 3>
        {
            typedef typename VecTraits<T>::elem_type E;
            __host__ __device__ __forceinline__ InInterval(const float4& _lo, const float4& _hi)
            : lo (VecTraits<T>::make((E)(-_lo.x), (E)(-_lo.y), (E)(-_lo.z))), hi (VecTraits<T>::make((E)_hi.x, (E)_hi.y, (E)_hi.z)){};
            T lo, hi;

            template<typename I> __device__ __forceinline__ bool operator() (const I& a, const I& b) const
            {
                I d = saturate_cast<I>(a - b);
                return lo.x <= d.x && d.x <= hi.x &&
                       lo.y <= d.y && d.y <= hi.y &&
                       lo.z <= d.z && d.z <= hi.z;
            }
        };

        template<typename T> struct InInterval<T, 4>
        {
            typedef typename VecTraits<T>::elem_type E;
            __host__ __device__ __forceinline__ InInterval(const float4& _lo, const float4& _hi)
            : lo (VecTraits<T>::make((E)(-_lo.x), (E)(-_lo.y), (E)(-_lo.z), (E)(-_lo.w))), hi (VecTraits<T>::make((E)_hi.x, (E)_hi.y, (E)_hi.z, (E)_hi.w)){};
            T lo, hi;

            template<typename I> __device__ __forceinline__ bool operator() (const I& a, const I& b) const
            {
                I d = saturate_cast<I>(a - b);
                return lo.x <= d.x && d.x <= hi.x &&
                       lo.y <= d.y && d.y <= hi.y &&
                       lo.z <= d.z && d.z <= hi.z &&
                       lo.w <= d.w && d.w <= hi.w;
            }
        };


        template<typename T, typename F>
        __global__ void computeConnectivity(const PtrStepSz<T> image, PtrStepSzb components, F connected)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= image.cols || y >= image.rows) return;

            T intensity = image(y, x);
            component c = 0;

            if ( x > 0 && connected(intensity, image(y, x - 1)))
                c |= LEFT;

            if ( y > 0 && connected(intensity, image(y - 1, x)))
                c |= UP;

            if ( x + 1 < image.cols && connected(intensity, image(y, x + 1)))
                c |= RIGHT;

            if ( y + 1 < image.rows && connected(intensity, image(y + 1, x)))
                c |= DOWN;

            components(y, x) = c;
        }

        template< typename T>
        void computeEdges(const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream)
        {
            dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
            dim3 grid(divUp(image.cols, block.x), divUp(image.rows, block.y));

            typedef InInterval<typename IntervalsTraits<T>::dist_type, IntervalsTraits<T>::ch> Int_t;

            Int_t inInt(lo, hi);
            computeConnectivity<T, Int_t><<<grid, block, 0, stream>>>(static_cast<const PtrStepSz<T> >(image), edges, inInt);

            cudaSafeCall( cudaGetLastError() );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void computeEdges<uchar>  (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<uchar3> (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<uchar4> (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<ushort> (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<ushort3>(const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<ushort4>(const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<int>    (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);
        template void computeEdges<float>  (const PtrStepSzb& image, PtrStepSzb edges, const float4& lo, const float4& hi, cudaStream_t stream);

        __global__ void lableTiles(const PtrStepSzb edges, PtrStepSzi comps)
        {
            int x = threadIdx.x + blockIdx.x * TILE_COLS;
            int y = threadIdx.y + blockIdx.y * TILE_ROWS;

            if (x >= edges.cols || y >= edges.rows) return;

            //currently x is 1
            int bounds = ((y + TPB_Y) < edges.rows);

            __shared__ int labelsTile[TILE_ROWS][TILE_COLS];
            __shared__ int  edgesTile[TILE_ROWS][TILE_COLS];

            int new_labels[TPB_Y][TPB_X];
            int old_labels[TPB_Y][TPB_X];

            #pragma unroll
            for (int i = 0; i < TPB_Y; ++i)
                #pragma unroll
                for (int j = 0; j < TPB_X; ++j)
                {
                    int yloc = threadIdx.y + CTA_SIZE_Y * i;
                    int xloc = threadIdx.x + CTA_SIZE_X * j;
                    component c = edges(bounds * (y + CTA_SIZE_Y * i), x + CTA_SIZE_X * j);

                    if (!xloc) c &= ~LEFT;
                    if (!yloc) c &= ~UP;

                    if (xloc == TILE_COLS -1) c &= ~RIGHT;
                    if (yloc == TILE_ROWS -1) c &= ~DOWN;

                    new_labels[i][j] = yloc * TILE_COLS + xloc;
                    edgesTile[yloc][xloc] = c;
                }

            for (int k = 0; ;++k)
            {
                //1. backup
                #pragma unroll
                for (int i = 0; i < TPB_Y; ++i)
                    #pragma unroll
                    for (int j = 0; j < TPB_X; ++j)
                    {
                        int yloc = threadIdx.y + CTA_SIZE_Y * i;
                        int xloc = threadIdx.x + CTA_SIZE_X * j;

                        old_labels[i][j]       = new_labels[i][j];
                        labelsTile[yloc][xloc] = new_labels[i][j];
                    }

                __syncthreads();

                //2. compare local arrays
                #pragma unroll
                for (int i = 0; i < TPB_Y; ++i)
                    #pragma unroll
                    for (int j = 0; j < TPB_X; ++j)
                    {
                        int yloc = threadIdx.y + CTA_SIZE_Y * i;
                        int xloc = threadIdx.x + CTA_SIZE_X * j;

                        component c = edgesTile[yloc][xloc];
                        int label = new_labels[i][j];

                        if (c & UP)
                           label = ::min(label, labelsTile[yloc - 1][xloc]);

                        if (c &  DOWN)
                           label = ::min(label, labelsTile[yloc + 1][xloc]);

                        if (c & LEFT)
                           label = ::min(label, labelsTile[yloc][xloc - 1]);

                        if (c & RIGHT)
                           label = ::min(label, labelsTile[yloc][xloc + 1]);

                       new_labels[i][j] = label;
                    }

                __syncthreads();

                //3. determine: Is any value changed?
                int changed = 0;
                #pragma unroll
                for (int i = 0; i < TPB_Y; ++i)
                    #pragma unroll
                    for (int j = 0; j < TPB_X; ++j)
                    {
                        if (new_labels[i][j] < old_labels[i][j])
                        {
                            changed = 1;
                            Emulation::smem::atomicMin(&labelsTile[0][0] + old_labels[i][j], new_labels[i][j]);
                        }
                    }

                changed = Emulation::syncthreadsOr(changed);

                if (!changed)
                    break;

                //4. Compact paths
                const int *labels = &labelsTile[0][0];
                #pragma unroll
                for (int i = 0; i < TPB_Y; ++i)
                    #pragma unroll
                    for (int j = 0; j < TPB_X; ++j)
                    {
                        int label = new_labels[i][j];

                        while( labels[label] < label ) label = labels[label];

                        new_labels[i][j] = label;
                    }
                __syncthreads();
            }

            #pragma unroll
            for (int i = 0; i < TPB_Y; ++i)
            #pragma unroll
                for (int j = 0; j < TPB_X; ++j)
                {
                    int label = new_labels[i][j];
                    int yloc = label / TILE_COLS;
                    int xloc = label - yloc * TILE_COLS;

                    xloc += blockIdx.x * TILE_COLS;
                    yloc += blockIdx.y * TILE_ROWS;

                    label = yloc * edges.cols + xloc;
                    // do it for x too.
                    if (y + CTA_SIZE_Y * i < comps.rows) comps(y + CTA_SIZE_Y * i, x + CTA_SIZE_X * j) = label;
                }
        }

        __device__ __forceinline__ int root(const PtrStepSzi& comps, int label)
        {
            while(1)
            {
                int y = label / comps.cols;
                int x = label - y * comps.cols;

                int parent = comps(y, x);

                if (label == parent) break;

                label = parent;
            }
            return label;
        }

        __device__ __forceinline__ void isConnected(PtrStepSzi& comps, int l1, int l2, bool& changed)
        {
            int r1 = root(comps, l1);
            int r2 = root(comps, l2);

            if (r1 == r2) return;

            int mi = ::min(r1, r2);
            int ma = ::max(r1, r2);

            int y = ma / comps.cols;
            int x = ma - y * comps.cols;

            atomicMin(&comps.ptr(y)[x], mi);
            changed = true;
        }

        __global__ void crossMerge(const int tilesNumY, const int tilesNumX, int tileSizeY, int tileSizeX,
            const PtrStepSzb edges, PtrStepSzi comps, const int yIncomplete, int xIncomplete)
        {
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int stride = blockDim.y * blockDim.x;

            int ybegin = blockIdx.y * (tilesNumY * tileSizeY);
            int yend   = ybegin + tilesNumY * tileSizeY;

            if (blockIdx.y == gridDim.y - 1)
            {
                yend -= yIncomplete * tileSizeY;
                yend -= tileSizeY;
                tileSizeY = (edges.rows % tileSizeY);

                yend += tileSizeY;
            }

            int xbegin = blockIdx.x * tilesNumX * tileSizeX;
            int xend   = xbegin + tilesNumX * tileSizeX;

            if (blockIdx.x == gridDim.x - 1)
            {
                if (xIncomplete) yend = ybegin;
                xend -= xIncomplete * tileSizeX;
                xend -= tileSizeX;
                tileSizeX = (edges.cols % tileSizeX);

                xend += tileSizeX;
            }

            if (blockIdx.y == (gridDim.y - 1) && yIncomplete)
            {
                xend = xbegin;
            }

            int tasksV = (tilesNumX - 1) * (yend - ybegin);
            int tasksH = (tilesNumY - 1) * (xend - xbegin);

            int total = tasksH + tasksV;

            bool changed;
            do
            {
                changed = false;
                for (int taskIdx = tid; taskIdx < total; taskIdx += stride)
                {
                    if (taskIdx < tasksH)
                    {
                        int indexH = taskIdx;

                        int row = indexH / (xend - xbegin);
                        int col = indexH - row * (xend - xbegin);

                        int y = ybegin + (row + 1) * tileSizeY;
                        int x = xbegin + col;

                        component e = edges( x, y);
                        if (e & UP)
                        {
                            int lc = comps(y,x);
                            int lu = comps(y - 1, x);

                            isConnected(comps, lc, lu, changed);
                        }
                    }
                    else
                    {
                        int indexV = taskIdx - tasksH;

                        int col = indexV / (yend - ybegin);
                        int row = indexV - col * (yend - ybegin);

                        int x = xbegin + (col + 1) * tileSizeX;
                        int y = ybegin + row;

                        component e = edges(x, y);
                        if (e & LEFT)
                        {
                            int lc = comps(y, x);
                            int ll = comps(y, x - 1);

                            isConnected(comps, lc, ll, changed);
                        }
                    }
                }
            } while (Emulation::syncthreadsOr(changed));
        }

        __global__ void flatten(const PtrStepSzb edges, PtrStepSzi comps)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if( x < comps.cols && y < comps.rows)
                comps(y, x) = root(comps, comps(y, x));
        }

        enum {CC_NO_COMPACT = 0, CC_COMPACT_LABELS = 1};

        void labelComponents(const PtrStepSzb& edges, PtrStepSzi comps, int flags, cudaStream_t stream)
        {
            (void) flags;
            dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
            dim3 grid(divUp(edges.cols, TILE_COLS), divUp(edges.rows, TILE_ROWS));

            lableTiles<<<grid, block, 0, stream>>>(edges, comps);
            cudaSafeCall( cudaGetLastError() );

            int tileSizeX = TILE_COLS, tileSizeY = TILE_ROWS;
            while (grid.x > 1 || grid.y > 1)
            {
                dim3 mergeGrid((int)ceilf(grid.x / 2.f), (int)ceilf(grid.y / 2.f));
                dim3 mergeBlock(STA_SIZE_MERGE_X, STA_SIZE_MERGE_Y);
                // debug log
                // std::cout << "merging: " << grid.y  << " x " << grid.x << " ---> " << mergeGrid.y <<  " x " << mergeGrid.x << " for tiles: " << tileSizeY << " x " << tileSizeX << std::endl;
                crossMerge<<<mergeGrid, mergeBlock, 0, stream>>>(2, 2, tileSizeY, tileSizeX, edges, comps, (int)ceilf(grid.y / 2.f) - grid.y / 2, (int)ceilf(grid.x / 2.f) - grid.x / 2);
                tileSizeX <<= 1;
                tileSizeY <<= 1;
                grid = mergeGrid;

                cudaSafeCall( cudaGetLastError() );
            }

            grid.x = divUp(edges.cols, block.x);
            grid.y = divUp(edges.rows, block.y);
            flatten<<<grid, block, 0, stream>>>(edges, comps);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
} } }

#endif /* CUDA_DISABLER */
