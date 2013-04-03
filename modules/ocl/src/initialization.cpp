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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Guoping Long, longguoping@gmail.com
//    Niko Li, newlife20080214@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
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

#include "precomp.hpp"
#include <iomanip>
#include <fstream>
#include "binarycaching.hpp"

using namespace cv;
using namespace cv::ocl;
using namespace std;
using std::cout;
using std::endl;

//#define PRINT_KERNEL_RUN_TIME
#define RUN_TIMES 100
#ifndef CL_MEM_USE_PERSISTENT_MEM_AMD
#define CL_MEM_USE_PERSISTENT_MEM_AMD 0
#endif
//#define AMD_DOUBLE_DIFFER

namespace cv
{
    namespace ocl
    {
        extern void fft_teardown();
        /*
         * The binary caching system to eliminate redundant program source compilation.
         * Strictly, this is not a cache because we do not implement evictions right now.
         * We shall add such features to trade-off memory consumption and performance when necessary.
         */
        auto_ptr<ProgramCache> ProgramCache::programCache;
        ProgramCache *programCache = NULL;
        DevMemType gDeviceMemType = DEVICE_MEM_DEFAULT;
        DevMemRW gDeviceMemRW = DEVICE_MEM_R_W;
        int gDevMemTypeValueMap[5] = {0,
                                      CL_MEM_ALLOC_HOST_PTR,
                                      CL_MEM_USE_HOST_PTR,
                                      CL_MEM_COPY_HOST_PTR,
                                      CL_MEM_USE_PERSISTENT_MEM_AMD};
        int gDevMemRWValueMap[3] = {CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};

        ProgramCache::ProgramCache()
        {
            codeCache.clear();
            cacheSize = 0;
        }

        ProgramCache::~ProgramCache()
        {
            releaseProgram();
        }

        cl_program ProgramCache::progLookup(string srcsign)
        {
            map<string, cl_program>::iterator iter;
            iter = codeCache.find(srcsign);
            if(iter != codeCache.end())
                return iter->second;
            else
                return NULL;
        }

        void ProgramCache::addProgram(string srcsign , cl_program program)
        {
            if(!progLookup(srcsign))
            {
                codeCache.insert(map<string, cl_program>::value_type(srcsign, program));
            }
        }

        void ProgramCache::releaseProgram()
        {
            map<string, cl_program>::iterator iter;
            for(iter = codeCache.begin(); iter != codeCache.end(); iter++)
            {
                openCLSafeCall(clReleaseProgram(iter->second));
            }
            codeCache.clear();
            cacheSize = 0;
        }


        struct Info::Impl
        {
            cl_platform_id oclplatform;
            std::vector<cl_device_id> devices;
            std::vector<std::string> devName;

            cl_context oclcontext;
            cl_command_queue clCmdQueue;
            int devnum;
            size_t maxWorkGroupSize;
            cl_uint maxDimensions; // == maxWorkItemSizes.size()
            std::vector<size_t> maxWorkItemSizes;
            cl_uint maxComputeUnits;
            char extra_options[512];
            int  double_support;
            int unified_memory; //1 means integrated GPU, otherwise this value is 0
            string binpath;
            int refcounter;

            Impl()
            {
                refcounter = 1;
                oclplatform = 0;
                oclcontext = 0;
                clCmdQueue = 0;
                devnum = -1;
                maxComputeUnits = 0;
                maxWorkGroupSize = 0;
                memset(extra_options, 0, 512);
                double_support = 0;
                unified_memory = 0;
            }

            void setDevice(void *ctx, void *q, int devnum);

            void release()
            {
                if(1 == CV_XADD(&refcounter, -1))
                {
                    releaseResources();
                    delete this;
                }
            }

            Impl* copy()
            {
                CV_XADD(&refcounter, 1);
                return this;
            }

        private:
            Impl(const Impl&);
            Impl& operator=(const Impl&);
            void releaseResources();
        };

        void Info::Impl::releaseResources()
        {
            devnum = -1;

            if(clCmdQueue)
            {
                openCLSafeCall(clReleaseCommandQueue(clCmdQueue));
                clCmdQueue = 0;
            }

            if(oclcontext)
            {
                openCLSafeCall(clReleaseContext(oclcontext));
                oclcontext = 0;
            }
        }

        void Info::Impl::setDevice(void *ctx, void *q, int dnum)
        {
            if((ctx && q) || devnum != dnum)
                releaseResources();

            CV_Assert(dnum >= 0 && dnum < (int)devices.size());
            devnum = dnum;
            if(ctx && q)
            {
                oclcontext = (cl_context)ctx;
                clCmdQueue = (cl_command_queue)q;
                clRetainContext(oclcontext);
                clRetainCommandQueue(clCmdQueue);
            }
            else
            {
                cl_int status = 0;
                cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(oclplatform), 0 };
                oclcontext = clCreateContext(cps, 1, &devices[devnum], 0, 0, &status);
                openCLVerifyCall(status);
                clCmdQueue = clCreateCommandQueue(oclcontext, devices[devnum], CL_QUEUE_PROFILING_ENABLE, &status);
                openCLVerifyCall(status);
            }

            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&maxWorkGroupSize, 0));
            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), (void *)&maxDimensions, 0));
            maxWorkItemSizes.resize(maxDimensions);
            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxDimensions, (void *)&maxWorkItemSizes[0], 0));
            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), (void *)&maxComputeUnits, 0));

            cl_bool unfymem = false;
            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), (void *)&unfymem, 0));
            unified_memory = unfymem ? 1 : 0;

            //initialize extra options for compilation. Currently only fp64 is included.
            //Assume 4KB is enough to store all possible extensions.
            const int EXT_LEN = 4096 + 1 ;
            char extends_set[EXT_LEN];
            size_t extends_size;
            openCLSafeCall(clGetDeviceInfo(devices[devnum], CL_DEVICE_EXTENSIONS, EXT_LEN, (void *)extends_set, &extends_size));
            extends_set[EXT_LEN - 1] = 0;
            size_t fp64_khr = std::string(extends_set).find("cl_khr_fp64");

            if(fp64_khr != std::string::npos)
            {
                sprintf(extra_options, "-D DOUBLE_SUPPORT");
                double_support = 1;
            }
            else
            {
                memset(extra_options, 0, 512);
                double_support = 0;
            }
        }

        ////////////////////////Common OpenCL specific calls///////////////
        int getDevMemType(DevMemRW& rw_type, DevMemType& mem_type)
        {
            rw_type = gDeviceMemRW;
            mem_type = gDeviceMemType;
            return Context::getContext()->impl->unified_memory;
        }

        int setDevMemType(DevMemRW rw_type, DevMemType mem_type)
        {
            if( (mem_type == DEVICE_MEM_PM && Context::getContext()->impl->unified_memory == 0) ||
                 mem_type == DEVICE_MEM_UHP ||
                 mem_type == DEVICE_MEM_CHP )
                return -1;
            gDeviceMemRW = rw_type;
            gDeviceMemType = mem_type;
            return 0;
        }

        inline int divUp(int total, int grain)
        {
            return (total + grain - 1) / grain;
        }

        int getDevice(std::vector<Info> &oclinfo, int devicetype)
        {
            //TODO: cache oclinfo vector
            oclinfo.clear();

            switch(devicetype)
            {
            case CVCL_DEVICE_TYPE_DEFAULT:
            case CVCL_DEVICE_TYPE_CPU:
            case CVCL_DEVICE_TYPE_GPU:
            case CVCL_DEVICE_TYPE_ACCELERATOR:
            case CVCL_DEVICE_TYPE_ALL:
                break;
            default:
                return 0;
            }

            // Platform info
            cl_uint numPlatforms;
            openCLSafeCall(clGetPlatformIDs(0, 0, &numPlatforms));
            if(numPlatforms < 1) return 0;

            std::vector<cl_platform_id> platforms(numPlatforms);
            openCLSafeCall(clGetPlatformIDs(numPlatforms, &platforms[0], 0));

            char deviceName[256];
            int devcienums = 0;
            for (unsigned i = 0; i < numPlatforms; ++i)
            {
                cl_uint numsdev;
                cl_int status = clGetDeviceIDs(platforms[i], devicetype, 0, NULL, &numsdev);
                if(status != CL_DEVICE_NOT_FOUND)
                    openCLVerifyCall(status);

                if(numsdev > 0)
                {
                    devcienums += numsdev;
                    std::vector<cl_device_id> devices(numsdev);
                    openCLSafeCall(clGetDeviceIDs(platforms[i], devicetype, numsdev, &devices[0], 0));

                    Info ocltmpinfo;
                    ocltmpinfo.impl->oclplatform = platforms[i];
                    for(unsigned j = 0; j < numsdev; ++j)
                    {
                        ocltmpinfo.impl->devices.push_back(devices[j]);
                        openCLSafeCall(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, 0));
                        ocltmpinfo.impl->devName.push_back(deviceName);
                        ocltmpinfo.DeviceName.push_back(deviceName);
                    }
                    oclinfo.push_back(ocltmpinfo);
                }
            }
            return devcienums;
        }

        void setDevice(Info &oclinfo, int devnum)
        {
            oclinfo.impl->setDevice(0, 0, devnum);
            Context::setContext(oclinfo);
        }

        void setDeviceEx(Info &oclinfo, void *ctx, void *q, int devnum)
        {
            oclinfo.impl->setDevice(ctx, q, devnum);
            Context::setContext(oclinfo);
         }

        void *getoclContext()
        {
            return &(Context::getContext()->impl->oclcontext);
        }

        void *getoclCommandQueue()
        {
            return &(Context::getContext()->impl->clCmdQueue);
        }

        void finish()
        {
            clFinish(Context::getContext()->impl->clCmdQueue);
        }

        void queryDeviceInfo(DEVICE_INFO info_type, void* info)
        {
            static Info::Impl* impl = Context::getContext()->impl;
            switch(info_type)
            {
            case WAVEFRONT_SIZE:
                {
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
                    try
                    {
                        openCLSafeCall(clGetDeviceInfo(Context::getContext()->impl->devices[0], 
                            CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(size_t), info, 0));
                    }
                    catch(const cv::Exception&)
#elif defined (CL_DEVICE_WARP_SIZE_NV)
                    const int EXT_LEN = 4096 + 1 ;
                    char extends_set[EXT_LEN];
                    size_t extends_size;
                    openCLSafeCall(clGetDeviceInfo(impl->devices[impl->devnum], CL_DEVICE_EXTENSIONS, EXT_LEN, (void *)extends_set, &extends_size));
                    extends_set[EXT_LEN - 1] = 0;
                    if(std::string(extends_set).find("cl_nv_device_attribute_query") != std::string::npos)
                    {
                        openCLSafeCall(clGetDeviceInfo(Context::getContext()->impl->devices[0], 
                            CL_DEVICE_WARP_SIZE_NV, sizeof(size_t), info, 0));
                    }
                    else
#endif
                    {
                        // if no way left for us to query the warp size, we can get it from kernel group info
                        static const char * _kernel_string = "__kernel void test_func() {}";
                        cl_kernel kernel;
                        kernel = openCLGetKernelFromSource(Context::getContext(), &_kernel_string, "test_func");
                        openCLSafeCall(clGetKernelWorkGroupInfo(kernel, impl->devices[impl->devnum],
                            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), info, NULL));
                    }

                }
                break;
            default:
                CV_Error(-1, "Invalid device info type");
                break;
            }
        }

        void openCLReadBuffer(Context *clCxt, cl_mem dst_buffer, void *host_buffer, size_t size)
        {
            cl_int status;
            status = clEnqueueReadBuffer(clCxt->impl->clCmdQueue, dst_buffer, CL_TRUE, 0,
                                         size, host_buffer, 0, NULL, NULL);
            openCLVerifyCall(status);
        }

        cl_mem openCLCreateBuffer(Context *clCxt, size_t flag , size_t size)
        {
            cl_int status;
            cl_mem buffer = clCreateBuffer(clCxt->impl->oclcontext, (cl_mem_flags)flag, size, NULL, &status);
            openCLVerifyCall(status);
            return buffer;
        }

        void openCLMallocPitch(Context *clCxt, void **dev_ptr, size_t *pitch,
                               size_t widthInBytes, size_t height)
        {
            openCLMallocPitchEx(clCxt, dev_ptr, pitch, widthInBytes, height, gDeviceMemRW, gDeviceMemType);
        }

        void openCLMallocPitchEx(Context *clCxt, void **dev_ptr, size_t *pitch,
                               size_t widthInBytes, size_t height, DevMemRW rw_type, DevMemType mem_type)
        {
            cl_int status;
            *dev_ptr = clCreateBuffer(clCxt->impl->oclcontext, gDevMemRWValueMap[rw_type]|gDevMemTypeValueMap[mem_type],
                                      widthInBytes * height, 0, &status);
            openCLVerifyCall(status);
            *pitch = widthInBytes;
        }

        void openCLMemcpy2D(Context *clCxt, void *dst, size_t dpitch,
                            const void *src, size_t spitch,
                            size_t width, size_t height, openCLMemcpyKind kind, int channels)
        {
            size_t buffer_origin[3] = {0, 0, 0};
            size_t host_origin[3] = {0, 0, 0};
            size_t region[3] = {width, height, 1};
            if(kind == clMemcpyHostToDevice)
            {
                if(dpitch == width || channels == 3 || height == 1)
                {
                    openCLSafeCall(clEnqueueWriteBuffer(clCxt->impl->clCmdQueue, (cl_mem)dst, CL_TRUE,
                                                        0, width * height, src, 0, NULL, NULL));
                }
                else
                {
                    openCLSafeCall(clEnqueueWriteBufferRect(clCxt->impl->clCmdQueue, (cl_mem)dst, CL_TRUE,
                                                            buffer_origin, host_origin, region, dpitch, 0, spitch, 0, src, 0, 0, 0));
                }
            }
            else if(kind == clMemcpyDeviceToHost)
            {
                if(spitch == width || channels == 3 || height == 1)
                {
                    openCLSafeCall(clEnqueueReadBuffer(clCxt->impl->clCmdQueue, (cl_mem)src, CL_TRUE,
                                                       0, width * height, dst, 0, NULL, NULL));
                }
                else
                {
                    openCLSafeCall(clEnqueueReadBufferRect(clCxt->impl->clCmdQueue, (cl_mem)src, CL_TRUE,
                                                           buffer_origin, host_origin, region, spitch, 0, dpitch, 0, dst, 0, 0, 0));
                }
            }
        }

        void openCLCopyBuffer2D(Context *clCxt, void *dst, size_t dpitch, int dst_offset,
                                const void *src, size_t spitch,
                                size_t width, size_t height, int src_offset)
        {
            size_t src_origin[3] = {src_offset % spitch, src_offset / spitch, 0};
            size_t dst_origin[3] = {dst_offset % dpitch, dst_offset / dpitch, 0};
            size_t region[3] = {width, height, 1};

            openCLSafeCall(clEnqueueCopyBufferRect(clCxt->impl->clCmdQueue, (cl_mem)src, (cl_mem)dst, src_origin, dst_origin,
                                                   region, spitch, 0, dpitch, 0, 0, 0, 0));
        }

        void openCLFree(void *devPtr)
        {
            openCLSafeCall(clReleaseMemObject((cl_mem)devPtr));
        }
        cl_kernel openCLGetKernelFromSource(const Context *clCxt, const char **source, string kernelName)
        {
            return openCLGetKernelFromSource(clCxt, source, kernelName, NULL);
        }


        void setBinpath(const char *path)
        {
            Context *clcxt = Context::getContext();
            clcxt->impl->binpath = path;
        }

        int savetofile(const Context*,  cl_program &program, const char *fileName)
        {
            size_t binarySize;
            openCLSafeCall(clGetProgramInfo(program,
                                    CL_PROGRAM_BINARY_SIZES,
                                    sizeof(size_t),
                                    &binarySize, NULL));
            char* binary = (char*)malloc(binarySize);
            if(binary == NULL)
            {
                CV_Error(CV_StsNoMem, "Failed to allocate host memory.");
            }
            openCLSafeCall(clGetProgramInfo(program,
                                    CL_PROGRAM_BINARIES,
                                    sizeof(char *),
                                    &binary,
                                    NULL));

            FILE *fp = fopen(fileName, "wb+");
            if(fp != NULL)
            {
                fwrite(binary, binarySize, 1, fp);
                free(binary);
                fclose(fp);
            }
            return 1;
        }

        cl_kernel openCLGetKernelFromSource(const Context *clCxt, const char **source, string kernelName,
                                            const char *build_options)
        {
            cl_kernel kernel;
            cl_program program ;
            cl_int status = 0;
            stringstream src_sign;
            string srcsign;
            string filename;
            CV_Assert(programCache != NULL);

            if(NULL != build_options)
            {
                src_sign << (int64)(*source) << clCxt->impl->oclcontext << "_" << build_options;
            }
            else
            {
                src_sign << (int64)(*source) << clCxt->impl->oclcontext;
            }
            srcsign = src_sign.str();

            program = NULL;
            program = programCache->progLookup(srcsign);

            if(!program)
            {
                //config build programs
                char all_build_options[1024];
                memset(all_build_options, 0, 1024);
                char zeromem[512] = {0};
                if(0 != memcmp(clCxt -> impl->extra_options, zeromem, 512))
                    strcat(all_build_options, clCxt -> impl->extra_options);
                strcat(all_build_options, " ");
                if(build_options != NULL)
                    strcat(all_build_options, build_options);
                if(all_build_options != NULL)
                {
                    filename = clCxt->impl->binpath  + kernelName + "_" + clCxt->impl->devName[clCxt->impl->devnum] + all_build_options + ".clb";
                }
                else
                {
                    filename = clCxt->impl->binpath  + kernelName + "_" + clCxt->impl->devName[clCxt->impl->devnum] + ".clb";
                }

                FILE *fp = fopen(filename.c_str(), "rb");
                if(fp == NULL || clCxt->impl->binpath.size() == 0)    //we should generate a binary file for the first time.
                {
                    if(fp != NULL)
                        fclose(fp);

                    program = clCreateProgramWithSource(
                                  clCxt->impl->oclcontext, 1, source, NULL, &status);
                    openCLVerifyCall(status);
                    status = clBuildProgram(program, 1, &(clCxt->impl->devices[clCxt->impl->devnum]), all_build_options, NULL, NULL);
                    if(status == CL_SUCCESS && clCxt->impl->binpath.size())
                        savetofile(clCxt, program, filename.c_str());
                }
                else
                {
                    fseek(fp, 0, SEEK_END);
                    size_t binarySize = ftell(fp);
                    fseek(fp, 0, SEEK_SET);
                    char *binary = new char[binarySize];
                    CV_Assert(1 == fread(binary, binarySize, 1, fp));
                    fclose(fp);
                    cl_int status = 0;
                    program = clCreateProgramWithBinary(clCxt->impl->oclcontext,
                                                        1,
                                                        &(clCxt->impl->devices[clCxt->impl->devnum]),
                                                        (const size_t *)&binarySize,
                                                        (const unsigned char **)&binary,
                                                        NULL,
                                                        &status);
                    openCLVerifyCall(status);
                    status = clBuildProgram(program, 1, &(clCxt->impl->devices[clCxt->impl->devnum]), all_build_options, NULL, NULL);
                    delete[] binary;
                }

                if(status != CL_SUCCESS)
                {
                    if(status == CL_BUILD_PROGRAM_FAILURE)
                    {
                        cl_int logStatus;
                        char *buildLog = NULL;
                        size_t buildLogSize = 0;
                        logStatus = clGetProgramBuildInfo(program,
                                                          clCxt->impl->devices[clCxt->impl->devnum], CL_PROGRAM_BUILD_LOG, buildLogSize,
                                                          buildLog, &buildLogSize);
                        if(logStatus != CL_SUCCESS)
                            cout << "Failed to build the program and get the build info." << endl;
                        buildLog = new char[buildLogSize];
                        CV_DbgAssert(!!buildLog);
                        memset(buildLog, 0, buildLogSize);
                        openCLSafeCall(clGetProgramBuildInfo(program, clCxt->impl->devices[clCxt->impl->devnum],
                                                             CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL));
                        cout << "\n\t\t\tBUILD LOG\n";
                        cout << buildLog << endl;
                        delete [] buildLog;
                    }
                    openCLVerifyCall(status);
                }
                //Cache the binary for future use if build_options is null
                if( (programCache->cacheSize += 1) < programCache->MAX_PROG_CACHE_SIZE)
                    programCache->addProgram(srcsign, program);
                else
                    cout << "Warning: code cache has been full.\n";
            }
            kernel = clCreateKernel(program, kernelName.c_str(), &status);
            openCLVerifyCall(status);
            return kernel;
        }

        void openCLVerifyKernel(const Context *clCxt, cl_kernel kernel, size_t *localThreads)
        {
            size_t kernelWorkGroupSize;
            openCLSafeCall(clGetKernelWorkGroupInfo(kernel, clCxt->impl->devices[clCxt->impl->devnum],
                                                    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0));
            CV_Assert( localThreads[0] <= clCxt->impl->maxWorkItemSizes[0] );
            CV_Assert( localThreads[1] <= clCxt->impl->maxWorkItemSizes[1] );
            CV_Assert( localThreads[2] <= clCxt->impl->maxWorkItemSizes[2] );
            CV_Assert( localThreads[0] * localThreads[1] * localThreads[2] <= kernelWorkGroupSize );
            CV_Assert( localThreads[0] * localThreads[1] * localThreads[2] <= clCxt->impl->maxWorkGroupSize );
        }

#ifdef PRINT_KERNEL_RUN_TIME
        static double total_execute_time = 0;
        static double total_kernel_time = 0;
#endif
        void openCLExecuteKernel_(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                  size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
                                  int depth, const char *build_options)
        {
            //construct kernel name
            //The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
            //for exmaple split_C2_D2, represent the split kernel with channels =2 and dataType Depth = 2(Data type is char)
            stringstream idxStr;
            if(channels != -1)
                idxStr << "_C" << channels;
            if(depth != -1)
                idxStr << "_D" << depth;
            kernelName += idxStr.str();

            cl_kernel kernel;
            kernel = openCLGetKernelFromSource(clCxt, source, kernelName, build_options);

            if ( localThreads != NULL)
            {
                globalThreads[0] = divUp(globalThreads[0], localThreads[0]) * localThreads[0];
                globalThreads[1] = divUp(globalThreads[1], localThreads[1]) * localThreads[1];
                globalThreads[2] = divUp(globalThreads[2], localThreads[2]) * localThreads[2];

                //size_t blockSize = localThreads[0] * localThreads[1] * localThreads[2];
                cv::ocl::openCLVerifyKernel(clCxt, kernel, localThreads);
            }
            for(size_t i = 0; i < args.size(); i ++)
                openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));

#ifndef PRINT_KERNEL_RUN_TIME
            openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
                                                  localThreads, 0, NULL, NULL));
#else
            cl_event event = NULL;
            openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
                                                  localThreads, 0, NULL, &event));

            cl_ulong start_time, end_time, queue_time;
            double execute_time = 0;
            double total_time   = 0;

            openCLSafeCall(clWaitForEvents(1, &event));
            openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                                   sizeof(cl_ulong), &start_time, 0));

            openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                                   sizeof(cl_ulong), &end_time, 0));

            openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                                   sizeof(cl_ulong), &queue_time, 0));

            execute_time = (double)(end_time - start_time) / (1000 * 1000);
            total_time = (double)(end_time - queue_time) / (1000 * 1000);

            //	cout << setiosflags(ios::left) << setw(15) << execute_time;
            //	cout << setiosflags(ios::left) << setw(15) << total_time - execute_time;
            //	cout << setiosflags(ios::left) << setw(15) << total_time << endl;

            total_execute_time += execute_time;
            total_kernel_time += total_time;
            clReleaseEvent(event);
#endif

            clFlush(clCxt->impl->clCmdQueue);
            openCLSafeCall(clReleaseKernel(kernel));
        }

        void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName,
                                 size_t globalThreads[3], size_t localThreads[3],
                                 vector< pair<size_t, const void *> > &args, int channels, int depth)
        {
            openCLExecuteKernel(clCxt, source, kernelName, globalThreads, localThreads, args,
                                channels, depth, NULL);
        }
        void openCLExecuteKernel(Context *clCxt , const char **source, string kernelName,
                                 size_t globalThreads[3], size_t localThreads[3],
                                 vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options)

        {
#ifndef PRINT_KERNEL_RUN_TIME
            openCLExecuteKernel_(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth,
                                 build_options);
#else
            string data_type[] = { "uchar", "char", "ushort", "short", "int", "float", "double"};
            cout << endl;
            cout << "Function Name: " << kernelName;
            if(depth >= 0)
                cout << " |data type: " << data_type[depth];
            cout << " |channels: " << channels;
            cout << " |Time Unit: " << "ms" << endl;

            total_execute_time = 0;
            total_kernel_time = 0;
            cout << "-------------------------------------" << endl;

            cout << setiosflags(ios::left) << setw(15) << "excute time";
            cout << setiosflags(ios::left) << setw(15) << "lauch time";
            cout << setiosflags(ios::left) << setw(15) << "kernel time" << endl;
            int i = 0;
            for(i = 0; i < RUN_TIMES; i++)
                openCLExecuteKernel_(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth,
                                     build_options);

            cout << "average kernel excute time: " << total_execute_time / RUN_TIMES << endl; // "ms" << endl;
            cout << "average kernel total time:  " << total_kernel_time / RUN_TIMES << endl; // "ms" << endl;
#endif
        }

       double openCLExecuteKernelInterop(Context *clCxt , const char **source, string kernelName,
                                 size_t globalThreads[3], size_t localThreads[3],
                                 vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options,
                                 bool finish, bool measureKernelTime, bool cleanUp)

        {
            //construct kernel name
            //The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
            //for exmaple split_C2_D2, represent the split kernel with channels =2 and dataType Depth = 2(Data type is char)
            stringstream idxStr;
            if(channels != -1)
                idxStr << "_C" << channels;
            if(depth != -1)
                idxStr << "_D" << depth;
            kernelName += idxStr.str();

            cl_kernel kernel;
            kernel = openCLGetKernelFromSource(clCxt, source, kernelName, build_options);

            double kernelTime = 0.0;

            if( globalThreads != NULL)
            {
                if ( localThreads != NULL)
                {
                    globalThreads[0] = divUp(globalThreads[0], localThreads[0]) * localThreads[0];
                    globalThreads[1] = divUp(globalThreads[1], localThreads[1]) * localThreads[1];
                    globalThreads[2] = divUp(globalThreads[2], localThreads[2]) * localThreads[2];

                    //size_t blockSize = localThreads[0] * localThreads[1] * localThreads[2];
                    cv::ocl::openCLVerifyKernel(clCxt, kernel, localThreads);
                }
                for(size_t i = 0; i < args.size(); i ++)
                    openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));

                if(measureKernelTime == false)
                {
                    openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
                                    localThreads, 0, NULL, NULL));
                }
                else
                {
                    cl_event event = NULL;
                    openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
                                    localThreads, 0, NULL, &event));

                    cl_ulong end_time, queue_time;

                    openCLSafeCall(clWaitForEvents(1, &event));

                    openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &end_time, 0));

                    openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                    sizeof(cl_ulong), &queue_time, 0));

                    kernelTime = (double)(end_time - queue_time) / (1000 * 1000);

                    clReleaseEvent(event);
                }
            }

            if(finish)
            {
                clFinish(clCxt->impl->clCmdQueue);
            }

            if(cleanUp)
            {
                openCLSafeCall(clReleaseKernel(kernel));
            }

            return kernelTime;
        }

        // Converts the contents of a file into a string
        static int convertToString(const char *filename, std::string& s)
        {
            size_t size;
            char*  str;

            std::fstream f(filename, (std::fstream::in | std::fstream::binary));
            if(f.is_open())
            {
                size_t fileSize;
                f.seekg(0, std::fstream::end);
                size = fileSize = (size_t)f.tellg();
                f.seekg(0, std::fstream::beg);

                str = new char[size+1];
                if(!str)
                {
                    f.close();
                    return -1;
                }

                f.read(str, fileSize);
                f.close();
                str[size] = '\0';

                s = str;
                delete[] str;
                return 0;
            }
            printf("Error: Failed to open file %s\n", filename);
            return -1;
        }

        double openCLExecuteKernelInterop(Context *clCxt , const char **fileName, const int numFiles, string kernelName,
                                 size_t globalThreads[3], size_t localThreads[3],
                                 vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options,
                                 bool finish, bool measureKernelTime, bool cleanUp)

        {
            std::vector<std::string> fsource;
            for (int i = 0 ; i < numFiles ; i++)
            {
                std::string str;
                if (convertToString(fileName[i], str) >= 0)
                    fsource.push_back(str);
            }
            const char **source = new const char *[numFiles];
            for (int i = 0 ; i < numFiles ; i++)
                source[i] = fsource[i].c_str();
            double kernelTime = openCLExecuteKernelInterop(clCxt ,source, kernelName, globalThreads, localThreads,
                                 args, channels, depth, build_options, finish, measureKernelTime, cleanUp);
            fsource.clear();
            delete []source;
            return kernelTime;
        }

        cl_mem load_constant(cl_context context, cl_command_queue command_queue, const void *value,
                             const size_t size)
        {
            int status;
            cl_mem con_struct;

            con_struct = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
            openCLSafeCall(status);

            openCLSafeCall(clEnqueueWriteBuffer(command_queue, con_struct, 1, 0, size,
                                                value, 0, 0, 0));

            return con_struct;

        }

        /////////////////////////////OpenCL initialization/////////////////
        auto_ptr<Context> Context::clCxt;
        int Context::val = 0;
        static Mutex cs;
        Context* Context::getContext()
        {
            if(*((volatile int*)&val) != 1)
            {
                AutoLock al(cs);
                if(*((volatile int*)&val) != 1)
                {
                    if( 0 == clCxt.get())
                        clCxt.reset(new Context);

                    std::vector<Info> oclinfo;
                    CV_Assert(getDevice(oclinfo, CVCL_DEVICE_TYPE_ALL) > 0);
                    oclinfo[0].impl->setDevice(0, 0, 0);
                    clCxt.get()->impl = oclinfo[0].impl->copy();

                    *((volatile int*)&val) = 1;
                }
            }
            return clCxt.get();
        }

        void Context::setContext(Info &oclinfo)
        {
            AutoLock guard(cs);
            if(*((volatile int*)&val) != 1)
            {
                if( 0 == clCxt.get())
                    clCxt.reset(new Context);

                clCxt.get()->impl = oclinfo.impl->copy();

                *((volatile int*)&val) = 1;
            }
            else
            {
                clCxt.get()->impl->release();
                clCxt.get()->impl = oclinfo.impl->copy();
            }
        }

        Context::Context()
        {
            impl = 0;
            programCache = ProgramCache::getProgramCache();
        }

        Context::~Context()
        {
            release();
        }

        void Context::release()
        {
            if (impl)
                impl->release();
            programCache->releaseProgram();
        }

        bool Context::supportsFeature(int ftype)
        {
            switch(ftype)
            {
            case CL_DOUBLE:
                return impl->double_support == 1;
            case CL_UNIFIED_MEM:
                return impl->unified_memory == 1;
            case CL_CPU:
                cl_device_type devicetype;
                clGetDeviceInfo(impl->devices[impl->devnum], 
                                CL_DEVICE_TYPE, sizeof(cl_device_type), 
                                &devicetype, NULL);
                return devicetype == CVCL_DEVICE_TYPE_CPU;
            default:
                return false;
            }
        }

        size_t Context::computeUnits()
        {
            return impl->maxComputeUnits;
        }

        void* Context::oclContext()
        {
            return impl->oclcontext;
        }

        void* Context::oclCommandQueue()
        {
            return impl->clCmdQueue;
        }

        Info::Info()
        {
            impl = new Impl;
        }

        void Info::release()
        {
            fft_teardown();
            impl->release();
            impl = new Impl;
            DeviceName.clear();
        }

        Info::~Info()
        {
            fft_teardown();
            impl->release();
        }

        Info &Info::operator = (const Info &m)
        {
            impl->release();
            impl = m.impl->copy();
            DeviceName = m.DeviceName;
            return *this;
        }

        Info::Info(const Info &m)
        {
            impl = m.impl->copy();
            DeviceName = m.DeviceName;
        }
    }//namespace ocl

}//namespace cv

#if defined BUILD_SHARED_LIBS && defined CVAPI_EXPORTS && defined WIN32 && !defined WINCE
#include <windows.h>
BOOL WINAPI DllMain( HINSTANCE, DWORD  fdwReason, LPVOID );

BOOL WINAPI DllMain( HINSTANCE, DWORD  fdwReason, LPVOID )
{
    if( fdwReason == DLL_PROCESS_DETACH )
    {
        // application hangs if call clReleaseCommandQueue here, so release context only
        // without context release application hangs as well
        cl_context ctx = (cl_context)getoclContext();
        if(ctx)
            openCLSafeCall(clReleaseContext(ctx));
    }
    return TRUE;
}
#endif
