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
#ifndef _testintegralimagesquared_h_
#define _testintegralimagesquared_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestIntegralImageSquared : public NCVTestProvider
{
public:

    TestIntegralImageSquared(std::string testName, NCVTestSourceProvider<Ncv8u> &src,
                             Ncv32u width, Ncv32u height);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestIntegralImageSquared(const TestIntegralImageSquared&);
	TestIntegralImageSquared& operator=(const TestIntegralImageSquared&);	

    NCVTestSourceProvider<Ncv8u> &src;
    Ncv32u width;
    Ncv32u height;
};

#endif // _testintegralimagesquared_h_
