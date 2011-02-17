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
#ifndef _testhypothesescompact_h_
#define _testhypothesescompact_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestCompact : public NCVTestProvider
{
public:

    TestCompact(std::string testName, NCVTestSourceProvider<Ncv32u> &src,
                          Ncv32u length, Ncv32u badElem, Ncv32u badElemPercentage);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:
	TestCompact(const TestCompact&);
	TestCompact& operator=(const TestCompact&);	


    NCVTestSourceProvider<Ncv32u> &src;
    Ncv32u length;
    Ncv32u badElem;
    Ncv32u badElemPercentage;
};

#endif // _testhypothesescompact_h_
