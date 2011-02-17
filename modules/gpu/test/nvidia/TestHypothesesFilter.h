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
#ifndef _testhypothesesfilter_h_
#define _testhypothesesfilter_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestHypothesesFilter : public NCVTestProvider
{
public:

    TestHypothesesFilter(std::string testName, NCVTestSourceProvider<Ncv32u> &src,
                         Ncv32u numDstRects, Ncv32u minNeighbors, Ncv32f eps);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:

	TestHypothesesFilter(const TestHypothesesFilter&);
	TestHypothesesFilter& operator=(const TestHypothesesFilter&);	

    NCVTestSourceProvider<Ncv32u> &src;
    Ncv32u numDstRects;
    Ncv32u minNeighbors;
    Ncv32f eps;

    Ncv32u canvasWidth;
    Ncv32u canvasHeight;
};

#endif // _testhypothesesfilter_h_
