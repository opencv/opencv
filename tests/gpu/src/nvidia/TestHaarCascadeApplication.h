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
#ifndef _testhaarcascadeapplication_h_
#define _testhaarcascadeapplication_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestHaarCascadeApplication : public NCVTestProvider
{
public:

    TestHaarCascadeApplication(std::string testName, NCVTestSourceProvider<Ncv8u> &src,
                               std::string cascadeName, Ncv32u width, Ncv32u height);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestHaarCascadeApplication(const TestHaarCascadeApplication&);
	TestHaarCascadeApplication& operator=(const TestHaarCascadeApplication&);	


    NCVTestSourceProvider<Ncv8u> &src;
    std::string cascadeName;
    Ncv32u width;
    Ncv32u height;
};

#endif // _testhaarcascadeapplication_h_
