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
#ifndef _testhypothesesgrow_h_
#define _testhypothesesgrow_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestHypothesesGrow : public NCVTestProvider
{
public:

    TestHypothesesGrow(std::string testName, NCVTestSourceProvider<Ncv32u> &src,
                       Ncv32u rectWidth, Ncv32u rectHeight, Ncv32f rectScale, 
                       Ncv32u maxLenSrc, Ncv32u lenSrc, Ncv32u maxLenDst, Ncv32u lenDst);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestHypothesesGrow(const TestHypothesesGrow&);
	TestHypothesesGrow& operator=(const TestHypothesesGrow&);	


    NCVTestSourceProvider<Ncv32u> &src;
    Ncv32u rectWidth;
    Ncv32u rectHeight;
    Ncv32f rectScale;
    Ncv32u maxLenSrc;
    Ncv32u lenSrc;
    Ncv32u maxLenDst;
    Ncv32u lenDst;
};

#endif // _testhypothesesgrow_h_
