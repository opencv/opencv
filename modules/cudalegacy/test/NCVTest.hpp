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

#ifndef _ncvtest_hpp_
#define _ncvtest_hpp_

#if defined _MSC_VER
# pragma warning( disable : 4201 4408 4100)
#endif

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <fstream>

#include <cuda_runtime.h>

#include "opencv2/cudalegacy.hpp"


struct NCVTestReport
{
    std::map<std::string, Ncv32u> statsNums;
    std::map<std::string, std::string> statsText;
};


class INCVTest
{
public:
    virtual bool executeTest(NCVTestReport &report) = 0;
    virtual std::string getName() const = 0;
    virtual ~INCVTest(){}
};


class NCVTestProvider : public INCVTest
{
public:

    NCVTestProvider(std::string testName_)
        :
        testName(testName_)
    {
        int devId;
        ncvAssertPrintReturn(cudaSuccess == cudaGetDevice(&devId), "Error returned from cudaGetDevice", );
        ncvAssertPrintReturn(cudaSuccess == cudaGetDeviceProperties(&this->devProp, devId), "Error returned from cudaGetDeviceProperties", );
    }

    virtual bool init() = 0;
    virtual bool process() = 0;
    virtual bool deinit() = 0;
    virtual bool toString(std::ofstream &strOut) = 0;

    virtual std::string getName() const
    {
        return this->testName;
    }

    virtual ~NCVTestProvider()
    {
        deinitMemory();
    }

    virtual bool executeTest(NCVTestReport &report)
    {
        bool res;
        report.statsText["rcode"] = "FAILED";

        res = initMemory(report);
        if (!res)
        {
            dumpToFile(report);
            deinitMemory();
            return false;
        }

        res = init();
        if (!res)
        {
            dumpToFile(report);
            deinit();
            deinitMemory();
            return false;
        }

        res = process();
        if (!res)
        {
            dumpToFile(report);
            deinit();
            deinitMemory();
            return false;
        }

        res = deinit();
        if (!res)
        {
            dumpToFile(report);
            deinitMemory();
            return false;
        }

        deinitMemory();

        report.statsText["rcode"] = "Passed";
        return true;
    }

protected:

    cudaDeviceProp devProp;
    std::auto_ptr<INCVMemAllocator> allocatorGPU;
    std::auto_ptr<INCVMemAllocator> allocatorCPU;

private:

    std::string testName;

    bool initMemory(NCVTestReport &report)
    {
        this->allocatorGPU.reset(new NCVMemStackAllocator(static_cast<Ncv32u>(devProp.textureAlignment)));
        this->allocatorCPU.reset(new NCVMemStackAllocator(static_cast<Ncv32u>(devProp.textureAlignment)));

        if (!this->allocatorGPU.get()->isInitialized() ||
            !this->allocatorCPU.get()->isInitialized())
        {
            report.statsText["rcode"] = "Memory FAILED";
            return false;
        }

        if (!this->process())
        {
            report.statsText["rcode"] = "Memory FAILED";
            return false;
        }

        Ncv32u maxGPUsize = (Ncv32u)this->allocatorGPU.get()->maxSize();
        Ncv32u maxCPUsize = (Ncv32u)this->allocatorCPU.get()->maxSize();

        report.statsNums["MemGPU"] = maxGPUsize;
        report.statsNums["MemCPU"] = maxCPUsize;

        this->allocatorGPU.reset(new NCVMemStackAllocator(NCVMemoryTypeDevice, maxGPUsize, static_cast<Ncv32u>(devProp.textureAlignment)));

        this->allocatorCPU.reset(new NCVMemStackAllocator(NCVMemoryTypeHostPinned, maxCPUsize, static_cast<Ncv32u>(devProp.textureAlignment)));

        if (!this->allocatorGPU.get()->isInitialized() ||
            !this->allocatorCPU.get()->isInitialized())
        {
            report.statsText["rcode"] = "Memory FAILED";
            return false;
        }

        return true;
    }

    void deinitMemory()
    {
        this->allocatorGPU.reset();
        this->allocatorCPU.reset();
    }

    void dumpToFile(NCVTestReport &report)
    {
        bool bReasonMem = (0 == report.statsText["rcode"].compare("Memory FAILED"));
        std::string fname = "TestDump_";
        fname += (bReasonMem ? "m_" : "") + this->testName + ".log";
        std::ofstream stream(fname.c_str(), std::ios::trunc | std::ios::out);
        if (!stream.is_open()) return;

        stream << "NCV Test Failure Log: " << this->testName << std::endl;
        stream << "====================================================" << std::endl << std::endl;
        stream << "Test initialization report: " << std::endl;
        for (std::map<std::string,std::string>::iterator it=report.statsText.begin();
             it != report.statsText.end(); ++it)
        {
            stream << it->first << "=" << it->second << std::endl;
        }
        for (std::map<std::string,Ncv32u>::iterator it=report.statsNums.begin();
            it != report.statsNums.end(); ++it)
        {
            stream << it->first << "=" << it->second << std::endl;
        }
        stream << std::endl;

        stream << "Test initialization parameters: " << std::endl;
        bool bSerializeRes = false;
        try
        {
            bSerializeRes = this->toString(stream);
        }
        catch (...)
        {
        }

        if (!bSerializeRes)
        {
            stream << "Couldn't retrieve object dump" << std::endl;
        }

        stream.flush();
    }
};

#endif // _ncvtest_hpp_
