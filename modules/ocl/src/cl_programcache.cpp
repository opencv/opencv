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

#include "precomp.hpp"
#include <iomanip>
#include <fstream>
#include "cl_programcache.hpp"

namespace cv { namespace ocl {

/*
 * The binary caching system to eliminate redundant program source compilation.
 * Strictly, this is not a cache because we do not implement evictions right now.
 * We shall add such features to trade-off memory consumption and performance when necessary.
 */

cv::Mutex ProgramCache::mutexFiles;
cv::Mutex ProgramCache::mutexCache;

ProgramCache* _programCache = NULL;
ProgramCache* ProgramCache::getProgramCache()
{
    if (NULL == _programCache)
    {
        cv::AutoLock lock(getInitializationMutex());
        if (NULL == _programCache)
            _programCache = new ProgramCache();
    }
    return _programCache;
}

ProgramCache::ProgramCache()
{
    codeCache.clear();
    cacheSize = 0;
}

ProgramCache::~ProgramCache()
{
    releaseProgram();
    if (this == _programCache)
    {
        cv::AutoLock lock(getInitializationMutex());
        if (this == _programCache)
            _programCache = NULL;
    }
}

cl_program ProgramCache::progLookup(const String& srcsign)
{
    std::map<String, cl_program>::iterator iter;
    iter = codeCache.find(srcsign);
    if(iter != codeCache.end())
        return iter->second;
    else
        return NULL;
}

void ProgramCache::addProgram(const String& srcsign, cl_program program)
{
    if (!progLookup(srcsign))
    {
        clRetainProgram(program);
        codeCache.insert(std::map<String, cl_program>::value_type(srcsign, program));
    }
}

void ProgramCache::releaseProgram()
{
    std::map<String, cl_program>::iterator iter;
    for(iter = codeCache.begin(); iter != codeCache.end(); iter++)
    {
        openCLSafeCall(clReleaseProgram(iter->second));
    }
    codeCache.clear();
    cacheSize = 0;
}

static bool enable_disk_cache = true;
static String binpath = "";

void setBinaryDiskCache(int mode, String path)
{
    enable_disk_cache = false;
    binpath = "";

    if(mode == CACHE_NONE)
    {
        return;
    }
    enable_disk_cache =
#if defined(_DEBUG) || defined(DEBUG)
        (mode & CACHE_DEBUG)   == CACHE_DEBUG;
#else
        (mode & CACHE_RELEASE) == CACHE_RELEASE;
#endif
    if(enable_disk_cache && !path.empty())
    {
        binpath = path;
    }
}

void setBinaryPath(const char *path)
{
    binpath = path;
}

static const int MAX_ENTRIES = 64;

struct ProgramFileCache
{
    struct CV_DECL_ALIGNED(1) ProgramFileHeader
    {
        int hashLength;
        //char hash[];
    };

    struct CV_DECL_ALIGNED(1) ProgramFileTable
    {
        int numberOfEntries;
        //int firstEntryOffset[];
    };

    struct CV_DECL_ALIGNED(1) ProgramFileConfigurationEntry
    {
        int nextEntry;
        int dataSize;
        int optionsLength;
        //char options[];
        // char data[];
    };

    String fileName_;
    const char* hash_;
    std::fstream f;

    ProgramFileCache(const String& fileName, const char* hash)
        : fileName_(fileName), hash_(hash)
    {
        if (hash_ != NULL)
        {
            f.open(fileName_.c_str(), std::ios::in|std::ios::out|std::ios::binary);
            if(f.is_open())
            {
                int hashLength = 0;
                f.read((char*)&hashLength, sizeof(int));
                std::vector<char> fhash(hashLength + 1);
                f.read(&fhash[0], hashLength);
                if (f.eof() || strncmp(hash_, &fhash[0], hashLength) != 0)
                {
                    f.close();
                    remove(fileName_.c_str());
                    return;
                }
            }
        }
    }

    int getHash(const String& options)
    {
        int hash = 0;
        for (size_t i = 0; i < options.length(); i++)
        {
            hash = (hash << 2) ^ (hash >> 17) ^ options[i];
        }
        return (hash + (hash >> 16)) & (MAX_ENTRIES - 1);
    }

    bool readConfigurationFromFile(const String& options, std::vector<char>& buf)
    {
        if (hash_ == NULL)
            return false;

        if (!f.is_open())
            return false;

        f.seekg(0, std::fstream::end);
        size_t fileSize = (size_t)f.tellg();
        if (fileSize == 0)
        {
            std::cerr << "Invalid file (empty): " << fileName_ << std::endl;
            f.close();
            remove(fileName_.c_str());
            return false;
        }
        f.seekg(0, std::fstream::beg);

        int hashLength = 0;
        f.read((char*)&hashLength, sizeof(int));
        CV_Assert(hashLength > 0);
        f.seekg(sizeof(hashLength) + hashLength, std::fstream::beg);

        int numberOfEntries = 0;
        f.read((char*)&numberOfEntries, sizeof(int));
        CV_Assert(numberOfEntries > 0);
        if (numberOfEntries != MAX_ENTRIES)
        {
            std::cerr << "Invalid file: " << fileName_ << std::endl;
            f.close();
            remove(fileName_.c_str());
            return false;
        }

        std::vector<int> firstEntryOffset(numberOfEntries);
        f.read((char*)&firstEntryOffset[0], sizeof(int)*numberOfEntries);

        int entryNum = getHash(options);

        int entryOffset = firstEntryOffset[entryNum];
        ProgramFileConfigurationEntry entry;
        while (entryOffset > 0)
        {
            f.seekg(entryOffset, std::fstream::beg);
            assert(sizeof(entry) == sizeof(int)*3);
            f.read((char*)&entry, sizeof(entry));
            std::vector<char> foptions(entry.optionsLength);
            if ((int)options.length() == entry.optionsLength)
            {
                if (entry.optionsLength > 0)
                    f.read(&foptions[0], entry.optionsLength);
                if (memcmp(&foptions[0], options.c_str(), entry.optionsLength) == 0)
                {
                    buf.resize(entry.dataSize);
                    f.read(&buf[0], entry.dataSize);
                    f.seekg(0, std::fstream::beg);
                    return true;
                }
            }
            if (entry.nextEntry <= 0)
                break;
            entryOffset = entry.nextEntry;
        }
        return false;
    }

    bool writeConfigurationToFile(const String& options, std::vector<char>& buf)
    {
        if (hash_ == NULL)
            return true; // don't save programs without hash

        if (!f.is_open())
        {
            f.open(fileName_.c_str(), std::ios::in|std::ios::out|std::ios::binary);
            if (!f.is_open())
            {
                f.open(fileName_.c_str(), std::ios::out|std::ios::binary);
                if (!f.is_open())
                    return false;
            }
        }

        f.seekg(0, std::fstream::end);
        size_t fileSize = (size_t)f.tellg();
        if (fileSize == 0)
        {
            f.seekp(0, std::fstream::beg);
            int hashLength = strlen(hash_);
            f.write((char*)&hashLength, sizeof(int));
            f.write(hash_, hashLength);

            int numberOfEntries = MAX_ENTRIES;
            f.write((char*)&numberOfEntries, sizeof(int));
            std::vector<int> firstEntryOffset(MAX_ENTRIES, 0);
            f.write((char*)&firstEntryOffset[0], sizeof(int)*numberOfEntries);
            f.close();
            f.open(fileName_.c_str(), std::ios::in|std::ios::out|std::ios::binary);
            CV_Assert(f.is_open());
            f.seekg(0, std::fstream::end);
            fileSize = (size_t)f.tellg();
        }
        f.seekg(0, std::fstream::beg);

        int hashLength = 0;
        f.read((char*)&hashLength, sizeof(int));
        CV_Assert(hashLength > 0);
        f.seekg(sizeof(hashLength) + hashLength, std::fstream::beg);

        int numberOfEntries = 0;
        f.read((char*)&numberOfEntries, sizeof(int));
        CV_Assert(numberOfEntries > 0);
        if (numberOfEntries != MAX_ENTRIES)
        {
            std::cerr << "Invalid file: " << fileName_ << std::endl;
            f.close();
            remove(fileName_.c_str());
            return false;
        }

        size_t tableEntriesOffset = (size_t)f.tellg();
        std::vector<int> firstEntryOffset(numberOfEntries);
        f.read((char*)&firstEntryOffset[0], sizeof(int)*numberOfEntries);

        int entryNum = getHash(options);

        int entryOffset = firstEntryOffset[entryNum];
        ProgramFileConfigurationEntry entry;
        while (entryOffset > 0)
        {
            f.seekg(entryOffset, std::fstream::beg);
            assert(sizeof(entry) == sizeof(int)*3);
            f.read((char*)&entry, sizeof(entry));
            std::vector<char> foptions(entry.optionsLength);
            if ((int)options.length() == entry.optionsLength)
            {
                if (entry.optionsLength > 0)
                    f.read(&foptions[0], entry.optionsLength);
                CV_Assert(memcmp(&foptions, options.c_str(), entry.optionsLength) != 0);
            }
            if (entry.nextEntry <= 0)
                break;
            entryOffset = entry.nextEntry;
        }
        if (entryOffset > 0)
        {
            f.seekp(entryOffset, std::fstream::beg);
            entry.nextEntry = fileSize;
            f.write((char*)&entry, sizeof(entry));
        }
        else
        {
            firstEntryOffset[entryNum] = fileSize;
            f.seekp(tableEntriesOffset, std::fstream::beg);
            f.write((char*)&firstEntryOffset[0], sizeof(int)*numberOfEntries);
        }
        f.seekp(fileSize, std::fstream::beg);
        entry.nextEntry = 0;
        entry.dataSize = buf.size();
        entry.optionsLength = options.length();
        f.write((char*)&entry, sizeof(entry));
        f.write(options.c_str(), entry.optionsLength);
        f.write(&buf[0], entry.dataSize);
        return true;
    }

    cl_program getOrBuildProgram(const Context* ctx, const cv::ocl::ProgramEntry* source, const String& options)
    {
        cl_int status = 0;
        cl_program program = NULL;
        std::vector<char> binary;
        if (!enable_disk_cache || !readConfigurationFromFile(options, binary))
        {
            program = clCreateProgramWithSource(getClContext(ctx), 1, (const char**)&source->programStr, NULL, &status);
            openCLVerifyCall(status);
            cl_device_id device = getClDeviceID(ctx);
            status = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);
            if(status == CL_SUCCESS)
            {
                if (enable_disk_cache)
                {
                    size_t binarySize;
                    openCLSafeCall(clGetProgramInfo(program,
                                            CL_PROGRAM_BINARY_SIZES,
                                            sizeof(size_t),
                                            &binarySize, NULL));

                    std::vector<char> binary(binarySize);

                    char* ptr = &binary[0];
                    openCLSafeCall(clGetProgramInfo(program,
                                            CL_PROGRAM_BINARIES,
                                            sizeof(char*),
                                            &ptr,
                                            NULL));

                    if (!writeConfigurationToFile(options, binary))
                    {
                        std::cerr << "Can't write data to file: " << fileName_ << std::endl;
                    }
                }
            }
        }
        else
        {
            cl_device_id device = getClDeviceID(ctx);
            size_t size = binary.size();
            const char* ptr = &binary[0];
            program = clCreateProgramWithBinary(getClContext(ctx),
                    1, &device,
                    (const size_t *)&size, (const unsigned char **)&ptr,
                    NULL, &status);
            openCLVerifyCall(status);
            status = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);
        }

        if(status != CL_SUCCESS)
        {
            if (status == CL_BUILD_PROGRAM_FAILURE || status == CL_INVALID_BUILD_OPTIONS)
            {
                size_t buildLogSize = 0;
                openCLSafeCall(clGetProgramBuildInfo(program, getClDeviceID(ctx),
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize));
                std::vector<char> buildLog; buildLog.resize(buildLogSize);
                memset(&buildLog[0], 0, buildLogSize);
                openCLSafeCall(clGetProgramBuildInfo(program, getClDeviceID(ctx),
                        CL_PROGRAM_BUILD_LOG, buildLogSize, &buildLog[0], NULL));
                std::cout << std::endl << "BUILD LOG: "
                        << (source->name ? source->name : "dynamic program") << ": "
                        << options << "\n";
                std::cout << &buildLog[0] << std::endl;
            }
            openCLVerifyCall(status);
        }
        return program;
    }
};

cl_program ProgramCache::getProgram(const Context *ctx, const cv::ocl::ProgramEntry* source,
                                    const char *build_options)
{
    std::stringstream src_sign;

    if (source->name)
    {
        src_sign << source->name;
        src_sign << getClContext(ctx);
        if (NULL != build_options)
        {
            src_sign << "_" << build_options;
        }

        {
            cv::AutoLock lockCache(mutexCache);
            cl_program program = ProgramCache::getProgramCache()->progLookup(src_sign.str());
            if (!!program)
            {
                clRetainProgram(program);
                return program;
            }
        }
    }

    cv::AutoLock lockCache(mutexFiles);

    // second check
    if (source->name)
    {
        cv::AutoLock lockCache(mutexCache);
        cl_program program = ProgramCache::getProgramCache()->progLookup(src_sign.str());
        if (!!program)
        {
            clRetainProgram(program);
            return program;
        }
    }

    String all_build_options;
    if (!ctx->getDeviceInfo().compilationExtraOptions.empty())
        all_build_options += ctx->getDeviceInfo().compilationExtraOptions;
    if (build_options != NULL)
    {
        all_build_options += " ";
        all_build_options += build_options;
    }
    const DeviceInfo& devInfo = ctx->getDeviceInfo();
    String filename = binpath + (source->name ? source->name : "NULL") + "_" + devInfo.platform->platformName + "_" + devInfo.deviceName + ".clb";

    ProgramFileCache programFileCache(filename, source->programHash);
    cl_program program = programFileCache.getOrBuildProgram(ctx, source, all_build_options);

    //Cache the binary for future use if build_options is null
    if (source->name)
    {
        cv::AutoLock lockCache(mutexCache);
        this->addProgram(src_sign.str(), program);
    }
    return program;
}

} // namespace ocl
} // namespace cv
