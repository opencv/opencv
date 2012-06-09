/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual 
 * property and proprietary rights in and to this software and 
 * related documentation and any modifications thereto.  
 * Any use, reproduction, disclosure, or distribution of this 
 * software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 */
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

    NCVAutoTestLister(std::string testSuiteName, OutputLevel outputLevel = OutputLevelCompact, NcvBool bStopOnFirstFail=false)
        :
    testSuiteName(testSuiteName),
    outputLevel(outputLevel),
    bStopOnFirstFail(bStopOnFirstFail)
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
