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
#ifndef _testhaarcascadeloader_h_
#define _testhaarcascadeloader_h_

#include "NCVTest.hpp"
#include "NCVTestSourceProvider.hpp"


class TestHaarCascadeLoader : public NCVTestProvider
{
public:

    TestHaarCascadeLoader(std::string testName, std::string cascadeName);

    virtual bool init();
    virtual bool process();
    virtual bool deinit();
    virtual bool toString(std::ofstream &strOut);

private:

    std::string cascadeName;
};

#endif // _testhaarcascadeloader_h_
