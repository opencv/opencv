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

#ifndef _ncvautotestlister_hpp_
#define _ncvautotestlister_hpp_

#include <vector>

#include "NCVTest.hpp"
#include <main_test_nvidia.h>
//enum OutputLevel
//{
//    OutputLevelNone,
//    OutputLevelCompact,
//    OutputLevelFull
//};

class NCVAutoTestLister
{
public:

    NCVAutoTestLister(std::string testSuiteName_, OutputLevel outputLevel_ = OutputLevelCompact, NcvBool bStopOnFirstFail_=false)
        :
    testSuiteName(testSuiteName_),
    outputLevel(outputLevel_),
    bStopOnFirstFail(bStopOnFirstFail_)
    {
    }

    void add(INCVTest *test)
    {
        this->tests.push_back(test);
    }

    bool invoke()
    {
        Ncv32u nPassed = 0;
        Ncv32u nFailed = 0;
        Ncv32u nFailedMem = 0;

        if (outputLevel == OutputLevelCompact)
        {
            printf("Test suite '%s' with %d tests\n",
                testSuiteName.c_str(),
                (int)(this->tests.size()));
        }

        for (Ncv32u i=0; i<this->tests.size(); i++)
        {
            INCVTest &curTest = *tests[i];

            NCVTestReport curReport;
            bool res = curTest.executeTest(curReport);

            if (outputLevel == OutputLevelFull)
            {
                printf("Test %3i %16s; Consumed mem GPU = %8d, CPU = %8d; %s\n",
                    i,
                    curTest.getName().c_str(),
                    curReport.statsNums["MemGPU"],
                    curReport.statsNums["MemCPU"],
                    curReport.statsText["rcode"].c_str());
            }

            if (res)
            {
                nPassed++;
                if (outputLevel == OutputLevelCompact)
                {
                    printf(".");
                }
            }
            else
            {
                if (!curReport.statsText["rcode"].compare("FAILED"))
                {
                    nFailed++;
                    if (outputLevel == OutputLevelCompact)
                    {
                        printf("x");
                    }
                    if (bStopOnFirstFail)
                    {
                        break;
                    }
                }
                else
                {
                    nFailedMem++;
                    if (outputLevel == OutputLevelCompact)
                    {
                        printf("m");
                    }
                }
            }
            fflush(stdout);
        }
        if (outputLevel == OutputLevelCompact)
        {
            printf("\n");
        }

        if (outputLevel != OutputLevelNone)
        {
            printf("Test suite '%s' complete: %d total, %d passed, %d memory errors, %d failed\n\n",
                testSuiteName.c_str(),
                (int)(this->tests.size()),
                nPassed,
                nFailedMem,
                nFailed);
        }

        bool passed = nFailed == 0 && nFailedMem == 0;
        return passed;
    }

    ~NCVAutoTestLister()
    {
        for (Ncv32u i=0; i<this->tests.size(); i++)
        {
            delete tests[i];
        }
    }

private:

    std::string testSuiteName;
    OutputLevel outputLevel;
    NcvBool bStopOnFirstFail;
    std::vector<INCVTest *> tests;
};

#endif // _ncvautotestlister_hpp_
