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
#ifndef _testrectstddev_h_
#define _testrectstddev_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestRectStdDev : public NCVTestProvider
{
public:

    TestRectStdDev(std::string testName, NCVTestSourceProvider<Ncv8u> &src,
                   Ncv32u width, Ncv32u height, NcvRect32u rect, Ncv32f scaleFactor,
                   NcvBool bTextureCache);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestRectStdDev(const TestRectStdDev&);
	TestRectStdDev& operator=(const TestRectStdDev&);	

    NCVTestSourceProvider<Ncv8u> &src;
    Ncv32u width;
    Ncv32u height;
    NcvRect32u rect;
    Ncv32f scaleFactor;

    NcvBool bTextureCache;
};

#endif // _testrectstddev_h_
