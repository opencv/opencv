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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@outlook.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include <iomanip>
#include "precomp.hpp"

namespace cv
{
namespace ocl
{

extern const char * kernel_sort_by_key;
extern const char * kernel_stablesort_by_key;
extern const char * kernel_radix_sort_by_key;

void sortByKey(oclMat& keys, oclMat& vals, size_t vecSize, int method, bool isGreaterThan);

//TODO(pengx17): change this value depending on device other than a constant
const static unsigned int GROUP_SIZE = 256;

const char * depth_strings[] =
{
    "uchar",   //CV_8U
    "char",    //CV_8S
    "ushort",  //CV_16U
    "short",   //CV_16S
    "int",     //CV_32S
    "float",   //CV_32F
    "double"   //CV_64F
};

void static genSortBuildOption(const oclMat& keys, const oclMat& vals, bool isGreaterThan, char * build_opt_buf)
{
    sprintf(build_opt_buf, "-D IS_GT=%d -D K_T=%s -D V_T=%s",
            isGreaterThan?1:0, depth_strings[keys.depth()], depth_strings[vals.depth()]);
    if(vals.oclchannels() > 1)
    {
        sprintf( build_opt_buf + strlen(build_opt_buf), "%d", vals.oclchannels());
    }
}
inline bool isSizePowerOf2(size_t size)
{
    return ((size - 1) & (size)) == 0;
}

namespace bitonic_sort
{
static void sortByKey(oclMat& keys, oclMat& vals, size_t vecSize, bool isGreaterThan)
{
    CV_Assert(isSizePowerOf2(vecSize));

    Context * cxt = Context::getContext();
    size_t globalThreads[3] = {vecSize / 2, 1, 1};
    size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

    // 2^numStages should be equal to vecSize or the output is invalid
    int numStages = 0;
    for(int i = vecSize; i > 1; i >>= 1)
    {
        ++numStages;
    }
    char build_opt_buf [100];
    genSortBuildOption(keys, vals, isGreaterThan, build_opt_buf);
    const int argc = 5;
    std::vector< std::pair<size_t, const void *> > args(argc);
    String kernelname = "bitonicSort";

    args[0] = std::make_pair(sizeof(cl_mem), (void *)&keys.data);
    args[1] = std::make_pair(sizeof(cl_mem), (void *)&vals.data);
    args[2] = std::make_pair(sizeof(cl_int), (void *)&vecSize);

    for(int stage = 0; stage < numStages; ++stage)
    {
        args[3] = std::make_pair(sizeof(cl_int), (void *)&stage);
        for(int passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {
            args[4] = std::make_pair(sizeof(cl_int), (void *)&passOfStage);
            openCLExecuteKernel(cxt, &kernel_sort_by_key, kernelname, globalThreads, localThreads, args, -1, -1, build_opt_buf);
        }
    }
}
}  /* bitonic_sort */

namespace selection_sort
{
// FIXME:
// This function cannot sort arrays with duplicated keys
static void sortByKey(oclMat& keys, oclMat& vals, size_t vecSize, bool isGreaterThan)
{
    CV_Error(-1, "This function is incorrect at the moment.");
    Context * cxt = Context::getContext();

    size_t globalThreads[3] = {vecSize, 1, 1};
    size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

    std::vector< std::pair<size_t, const void *> > args;
    char build_opt_buf [100];
    genSortBuildOption(keys, vals, isGreaterThan, build_opt_buf);

    //local
    String kernelname = "selectionSortLocal";
    int lds_size = GROUP_SIZE * keys.elemSize();
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&keys.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&vals.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&vecSize));
    args.push_back(std::make_pair(lds_size,       (void*)NULL));

    openCLExecuteKernel(cxt, &kernel_sort_by_key, kernelname, globalThreads, localThreads, args, -1, -1, build_opt_buf);

    //final
    kernelname = "selectionSortFinal";
    args.pop_back();
    openCLExecuteKernel(cxt, &kernel_sort_by_key, kernelname, globalThreads, localThreads, args, -1, -1, build_opt_buf);
}

}  /* selection_sort */


namespace radix_sort
{
//FIXME(pengx17): 
// exclusive scan, need to be optimized as this is too naive...
//void naive_scan_addition(oclMat& input, oclMat& output)
//{
//    Context * cxt = Context::getContext();
//    size_t vecSize = input.cols;
//    size_t globalThreads[3] = {1, 1, 1};
//    size_t localThreads[3]  = {1, 1, 1};
//
//    String kernelname = "naiveScanAddition";
//
//    std::vector< std::pair<size_t, const void *> > args;
//    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&input.data));
//    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&output.data));
//    args.push_back(std::make_pair(sizeof(cl_int), (void *)&vecSize));
//    openCLExecuteKernel(cxt, &kernel_radix_sort_by_key, kernelname, globalThreads, localThreads, args, -1, -1);
//}

void static naive_scan_addition_cpu(oclMat& input, oclMat& output)
{
    Mat m_input = input, m_output(output.size(), output.type());
    MatIterator_<int> i_mit = m_input.begin<int>();
    MatIterator_<int> o_mit = m_output.begin<int>();
    *o_mit = 0;
    ++i_mit;
    ++o_mit;
    for(; i_mit != m_input.end<int>(); ++i_mit, ++o_mit)
    {
        *o_mit = *(o_mit - 1) + *(i_mit - 1);
    }
    output = m_output;
}


//radix sort ported from Bolt
static void sortByKey(oclMat& keys, oclMat& vals, size_t origVecSize, bool isGreaterThan)
{
    CV_Assert(keys.depth() == CV_32S || keys.depth() == CV_32F); // we assume keys are 4 bytes

    bool isKeyFloat = keys.type() == CV_32F;

    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.
    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    bool  newBuffer = false;
    size_t vecSize = origVecSize;

    unsigned int groupSize  = RADICES;

    size_t mulFactor = groupSize * RADICES;

    oclMat buffer_keys, buffer_vals;

    if(origVecSize % mulFactor != 0)
    {
        vecSize = ((vecSize + mulFactor) / mulFactor) * mulFactor;
        buffer_keys.create(1, vecSize, keys.type());
        buffer_vals.create(1, vecSize, vals.type());
        Scalar padding_value;
        oclMat roi_buffer_vals = buffer_vals(Rect(0,0,origVecSize,1));

        if(isGreaterThan)
        {
            switch(buffer_keys.depth())
            {
            case CV_32F:
                padding_value = Scalar::all(-FLT_MAX);
                break;
            case CV_32S:
                padding_value = Scalar::all(INT_MIN);
                break;
            }
        }
        else
        {
            switch(buffer_keys.depth())
            {
            case CV_32F:
                padding_value = Scalar::all(FLT_MAX);
                break;
            case CV_32S:
                padding_value = Scalar::all(INT_MAX);
                break;
            }
        }
        ocl::copyMakeBorder(
            keys(Rect(0,0,origVecSize,1)), buffer_keys, 
            0, 0, 0, vecSize - origVecSize, 
            BORDER_CONSTANT, padding_value);
        vals(Rect(0,0,origVecSize,1)).copyTo(roi_buffer_vals);
        newBuffer = true;
    }
    else
    {
        buffer_keys = keys;
        buffer_vals = vals;
        newBuffer = false;
    }
    oclMat swap_input_keys(1, vecSize, keys.type());
    oclMat swap_input_vals(1, vecSize, vals.type());
    oclMat hist_bin_keys(1, vecSize, CV_32SC1);
    oclMat hist_bin_dest_keys(1, vecSize, CV_32SC1);

    Context * cxt = Context::getContext();

    size_t globalThreads[3] = {vecSize / RADICES, 1, 1};
    size_t localThreads[3]  = {groupSize, 1, 1};

    std::vector< std::pair<size_t, const void *> > args;
    char build_opt_buf [100];
    genSortBuildOption(keys, vals, isGreaterThan, build_opt_buf);

    //additional build option for radix sort
    sprintf(build_opt_buf + strlen(build_opt_buf), " -D K_%s", isKeyFloat?"FLT":"INT"); 

    String kernelnames[2] = {String("histogramRadixN"), String("permuteRadixN")};

    int swap = 0;
    for(int bits = 0; bits < (static_cast<int>(keys.elemSize()) * 8); bits += RADIX)
    {
        args.clear();
        //Do a histogram pass locally
        if(swap == 0)
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&buffer_keys.data));
        }
        else
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&swap_input_keys.data));
        }
        args.push_back(std::make_pair(sizeof(cl_mem), (void *)&hist_bin_keys.data));
        args.push_back(std::make_pair(sizeof(cl_int), (void *)&bits));
        openCLExecuteKernel(cxt, &kernel_radix_sort_by_key, kernelnames[0], globalThreads, localThreads,
            args, -1, -1, build_opt_buf);

        args.clear();
        //Perform a global scan
        naive_scan_addition_cpu(hist_bin_keys, hist_bin_dest_keys);
        // end of scan
        if(swap == 0)
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&buffer_keys.data));
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&buffer_vals.data));
        }
        else
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&swap_input_keys.data));
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&swap_input_vals.data));
        }
        args.push_back(std::make_pair(sizeof(cl_mem), (void *)&hist_bin_dest_keys.data));
        args.push_back(std::make_pair(sizeof(cl_int), (void *)&bits));

        if(swap == 0)
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&swap_input_keys.data));
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&swap_input_vals.data));
        }
        else
        {
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&buffer_keys.data));
            args.push_back(std::make_pair(sizeof(cl_mem), (void *)&buffer_vals.data));
        }
        openCLExecuteKernel(cxt, &kernel_radix_sort_by_key, kernelnames[1], globalThreads, localThreads,
            args, -1, -1, build_opt_buf);
        swap = swap ? 0 : 1;
    }
    if(newBuffer)
    {
        buffer_keys(Rect(0,0,origVecSize,1)).copyTo(keys);
        buffer_vals(Rect(0,0,origVecSize,1)).copyTo(vals);
    }
}

}  /* radix_sort */

namespace merge_sort
{
static void sortByKey(oclMat& keys, oclMat& vals, size_t vecSize, bool isGreaterThan)
{
    Context * cxt = Context::getContext();

    size_t globalThreads[3] = {vecSize, 1, 1};
    size_t localThreads[3]  = {GROUP_SIZE, 1, 1};

    std::vector< std::pair<size_t, const void *> > args;
    char build_opt_buf [100];
    genSortBuildOption(keys, vals, isGreaterThan, build_opt_buf);

    String kernelname[] = {String("blockInsertionSort"), String("merge")};
    int keylds_size = GROUP_SIZE * keys.elemSize();
    int vallds_size = GROUP_SIZE * vals.elemSize();
    args.push_back(std::make_pair(sizeof(cl_mem),  (void *)&keys.data));
    args.push_back(std::make_pair(sizeof(cl_mem),  (void *)&vals.data));
    args.push_back(std::make_pair(sizeof(cl_uint), (void *)&vecSize));
    args.push_back(std::make_pair(keylds_size,     (void*)NULL));
    args.push_back(std::make_pair(vallds_size,     (void*)NULL));

    openCLExecuteKernel(cxt, &kernel_stablesort_by_key, kernelname[0], globalThreads, localThreads, args, -1, -1, build_opt_buf);

    //  Early exit for the case of no merge passes, values are already in destination vector
    if(vecSize <= GROUP_SIZE)
    {
        return;
    }

    //  An odd number of elements requires an extra merge pass to sort
    size_t numMerges = 0;
    //  Calculate the log2 of vecSize, taking into acvecSize our block size from kernel 1 is 64
    //  this is how many merge passes we want
    size_t log2BlockSize = vecSize >> 6;
    for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
    {
        ++numMerges;
    }
    //  Check to see if the input vector size is a power of 2, if not we will need last merge pass
    numMerges += isSizePowerOf2(vecSize)? 1: 0;

    //  Allocate a flipflop buffer because the merge passes are out of place
    oclMat tmpKeyBuffer(keys.size(), keys.type());
    oclMat tmpValBuffer(vals.size(), vals.type());
    args.resize(8);

    args[4] = std::make_pair(sizeof(cl_uint), (void *)&vecSize);
    args[6] = std::make_pair(keylds_size,    (void*)NULL);
    args[7] = std::make_pair(vallds_size,    (void*)NULL);

    for(size_t pass = 1; pass <= numMerges; ++pass )
    {
        //  For each pass, flip the input-output buffers
        if( pass & 0x1 )
        {
            args[0] = std::make_pair(sizeof(cl_mem), (void *)&keys.data);
            args[1] = std::make_pair(sizeof(cl_mem), (void *)&vals.data);
            args[2] = std::make_pair(sizeof(cl_mem), (void *)&tmpKeyBuffer.data);
            args[3] = std::make_pair(sizeof(cl_mem), (void *)&tmpValBuffer.data);
        }
        else
        {
            args[0] = std::make_pair(sizeof(cl_mem), (void *)&tmpKeyBuffer.data);
            args[1] = std::make_pair(sizeof(cl_mem), (void *)&tmpValBuffer.data);
            args[2] = std::make_pair(sizeof(cl_mem), (void *)&keys.data);
            args[3] = std::make_pair(sizeof(cl_mem), (void *)&vals.data);
        }
        //  For each pass, the merge window doubles
        unsigned int srcLogicalBlockSize = static_cast<unsigned int>( localThreads[0] << (pass-1) );
        args[5] = std::make_pair(sizeof(cl_uint), (void *)&srcLogicalBlockSize);
        openCLExecuteKernel(cxt, &kernel_stablesort_by_key, kernelname[1], globalThreads, localThreads, args, -1, -1, build_opt_buf);
    }
    //  If there are an odd number of merges, then the output data is sitting in the temp buffer.  We need to copy
    //  the results back into the input array
    if( numMerges & 1 )
    {
        tmpKeyBuffer.copyTo(keys);
        tmpValBuffer.copyTo(vals);
    }
}
}  /* merge_sort */

}
} /* namespace cv { namespace ocl */


void cv::ocl::sortByKey(oclMat& keys, oclMat& vals, size_t vecSize, int method, bool isGreaterThan)
{
    CV_Assert( keys.rows == 1 ); // we only allow one dimensional input
    CV_Assert( keys.channels() == 1 ); // we only allow one channel keys
    CV_Assert( vecSize <= static_cast<size_t>(keys.cols) );
    switch(method)
    {
    case SORT_BITONIC:
        bitonic_sort::sortByKey(keys, vals, vecSize, isGreaterThan);
        break;
    case SORT_SELECTION:
        selection_sort::sortByKey(keys, vals, vecSize, isGreaterThan);
        break;
    case SORT_RADIX:
        radix_sort::sortByKey(keys, vals, vecSize, isGreaterThan);
        break;
    case SORT_MERGE:
        merge_sort::sortByKey(keys, vals, vecSize, isGreaterThan);
        break;
    }
}

void cv::ocl::sortByKey(oclMat& keys, oclMat& vals, int method, bool isGreaterThan)
{
    CV_Assert( keys.size() == vals.size() );
    CV_Assert( keys.rows == 1 ); // we only allow one dimensional input
    size_t vecSize = static_cast<size_t>(keys.cols);
    sortByKey(keys, vals, vecSize, method, isGreaterThan);
}
