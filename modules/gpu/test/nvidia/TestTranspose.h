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
#ifndef _testtranspose_h_
#define _testtranspose_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


template <class T>
class TestTranspose : public NCVTestProvider
{
public:

    TestTranspose(std::string testName, NCVTestSourceProvider<T> &src,
                  Ncv32u width, Ncv32u height);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:

	TestTranspose(const TestTranspose&);
	TestTranspose& operator=(const TestTranspose&);	

    NCVTestSourceProvider<T> &src;
    Ncv32u width;
    Ncv32u height;
};

#endif // _testtranspose_h_
