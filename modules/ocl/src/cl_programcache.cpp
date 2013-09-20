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

#undef __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace cv { namespace ocl {
/*
 * The binary caching system to eliminate redundant program source compilation.
 * Strictly, this is not a cache because we do not implement evictions right now.
 * We shall add such features to trade-off memory consumption and performance when necessary.
 */

std::auto_ptr<ProgramCache> _programCache;
ProgramCache* ProgramCache::getProgramCache()
{
	if (NULL == _programCache.get())
		_programCache.reset(new ProgramCache());
	return _programCache.get();
}

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

static int enable_disk_cache =
#ifdef _DEBUG
        false;
#else
        true;
#endif
static int update_disk_cache = false;
static String binpath = "";

void setBinaryDiskCache(int mode, String path)
{
    if(mode == CACHE_NONE)
    {
        update_disk_cache = 0;
        enable_disk_cache = 0;
        return;
    }
    update_disk_cache |= (mode & CACHE_UPDATE) == CACHE_UPDATE;
    enable_disk_cache |=
#ifdef _DEBUG
        (mode & CACHE_DEBUG)   == CACHE_DEBUG;
#else
        (mode & CACHE_RELEASE) == CACHE_RELEASE;
#endif
    if(enable_disk_cache && !path.empty())
    {
        binpath = path;
    }
}

void setBinpath(const char *path)
{
    binpath = path;
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

cl_program ProgramCache::getProgram(const Context *ctx, const char **source, string kernelName,
                                    const char *build_options)
{
    cl_program program;
    cl_int status = 0;
    stringstream src_sign;
    string srcsign;
    string filename;

    if (NULL != build_options)
    {
        src_sign << (int64)(*source) << getClContext(ctx) << "_" << build_options;
    }
    else
    {
        src_sign << (int64)(*source) << getClContext(ctx);
    }
    srcsign = src_sign.str();

    program = NULL;
    program = ProgramCache::getProgramCache()->progLookup(srcsign);

    if (!program)
    {
        //config build programs
        std::string all_build_options;
        if (!ctx->getDeviceInfo().compilationExtraOptions.empty())
            all_build_options += ctx->getDeviceInfo().compilationExtraOptions;
        if (build_options != NULL)
        {
            all_build_options += " ";
            all_build_options += build_options;
        }
        filename = binpath + kernelName + "_" + ctx->getDeviceInfo().deviceName + all_build_options + ".clb";

        FILE *fp = enable_disk_cache ? fopen(filename.c_str(), "rb") : NULL;
        if(fp == NULL || update_disk_cache)
        {
            if(fp != NULL)
                fclose(fp);

            program = clCreateProgramWithSource(
                          getClContext(ctx), 1, source, NULL, &status);
            openCLVerifyCall(status);
            cl_device_id device = getClDeviceID(ctx);
            status = clBuildProgram(program, 1, &device, all_build_options.c_str(), NULL, NULL);
            if(status == CL_SUCCESS && enable_disk_cache)
                savetofile(ctx, program, filename.c_str());
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
            cl_device_id device = getClDeviceID(ctx);
            program = clCreateProgramWithBinary(getClContext(ctx),
                                                1,
                                                &device,
                                                (const size_t *)&binarySize,
                                                (const unsigned char **)&binary,
                                                NULL,
                                                &status);
            openCLVerifyCall(status);
            status = clBuildProgram(program, 1, &device, all_build_options.c_str(), NULL, NULL);
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
                        getClDeviceID(ctx), CL_PROGRAM_BUILD_LOG, buildLogSize,
                        buildLog, &buildLogSize);
                if(logStatus != CL_SUCCESS)
                    std::cout << "Failed to build the program and get the build info." << endl;
                buildLog = new char[buildLogSize];
                CV_DbgAssert(!!buildLog);
                memset(buildLog, 0, buildLogSize);
                openCLSafeCall(clGetProgramBuildInfo(program, getClDeviceID(ctx),
                                                     CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL));
                std::cout << "\n\t\t\tBUILD LOG\n";
                std::cout << buildLog << endl;
                delete [] buildLog;
            }
            openCLVerifyCall(status);
        }
        //Cache the binary for future use if build_options is null
        if( (this->cacheSize += 1) < MAX_PROG_CACHE_SIZE)
            this->addProgram(srcsign, program);
        else
            cout << "Warning: code cache has been full.\n";
    }
    return program;
}

//// Converts the contents of a file into a string
//static int convertToString(const char *filename, std::string& s)
//{
//    size_t size;
//    char*  str;
//
//    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
//    if(f.is_open())
//    {
//        size_t fileSize;
//        f.seekg(0, std::fstream::end);
//        size = fileSize = (size_t)f.tellg();
//        f.seekg(0, std::fstream::beg);
//
//        str = new char[size+1];
//        if(!str)
//        {
//            f.close();
//            return -1;
//        }
//
//        f.read(str, fileSize);
//        f.close();
//        str[size] = '\0';
//
//        s = str;
//        delete[] str;
//        return 0;
//    }
//    printf("Error: Failed to open file %s\n", filename);
//    return -1;
//}

} // namespace ocl
} // namespace cv
