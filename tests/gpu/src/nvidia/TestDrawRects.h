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
#ifndef _testdrawrects_h_
#define _testdrawrects_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


template <class T>
class TestDrawRects : public NCVTestProvider
{
public:

    TestDrawRects(std::string testName, NCVTestSourceProvider<T> &src, NCVTestSourceProvider<Ncv32u> &src32u,
                  Ncv32u width, Ncv32u height, Ncv32u numRects, T color);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:

	TestDrawRects(const TestDrawRects&);
	TestDrawRects& operator=(const TestDrawRects&);	

    NCVTestSourceProvider<T> &src;
    NCVTestSourceProvider<Ncv32u> &src32u;
    Ncv32u width;
    Ncv32u height;
    Ncv32u numRects;
    T color;
};

#endif // _testdrawrects_h_
