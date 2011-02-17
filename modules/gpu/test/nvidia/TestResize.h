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
#ifndef _testresize_h_
#define _testresize_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"

template <class T>
class TestResize : public NCVTestProvider
{
public:

    TestResize(std::string testName, NCVTestSourceProvider<T> &src,
               Ncv32u width, Ncv32u height, Ncv32u scaleFactor, NcvBool bTextureCache);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestResize(const TestResize&);
	TestResize& operator=(const TestResize&);	

    NCVTestSourceProvider<T> &src;
    Ncv32u width;
    Ncv32u height;
    Ncv32u scaleFactor;

    NcvBool bTextureCache;
};

#endif // _testresize_h_
