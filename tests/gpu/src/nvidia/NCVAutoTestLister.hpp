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

class NCVAutoTestLister
{
public:

    NCVAutoTestLister(std::string testSuiteName, NcvBool bStopOnFirstFail=false, NcvBool bCompactOutput=true)
        :
    testSuiteName(testSuiteName),
    bStopOnFirstFail(bStopOnFirstFail),
    bCompactOutput(bCompactOutput)
    {
    }

    void add(INCVTest *test)
    {
        this->tests.push_back(test);
    }

    void invoke()
    {
        Ncv32u nPassed = 0;
        Ncv32u nFailed = 0;
        Ncv32u nFailedMem = 0;

        if (bCompactOutput)
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

            if (!bCompactOutput)
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
                if (bCompactOutput)
                {
                    printf(".");
                }
            }
            else
            {
                if (!curReport.statsText["rcode"].compare("FAILED"))
                {
                    nFailed++;
                    if (bCompactOutput)
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
                    if (bCompactOutput)
                    {
                        printf("m");
                    }
                }
            }
            fflush(stdout);
        }
        if (bCompactOutput)
        {
            printf("\n");
        }

        printf("Test suite '%s' complete: %d total, %d passed, %d memory errors, %d failed\n\n", 
            testSuiteName.c_str(),
            (int)(this->tests.size()),
            nPassed,
            nFailedMem,
            nFailed);
    }

    ~NCVAutoTestLister()
    {
        for (Ncv32u i=0; i<this->tests.size(); i++)
        {
            delete tests[i];
        }
    }

private:

    NcvBool bStopOnFirstFail;
    NcvBool bCompactOutput;
    std::string testSuiteName;
    std::vector<INCVTest *> tests;
};

#endif // _ncvautotestlister_hpp_
