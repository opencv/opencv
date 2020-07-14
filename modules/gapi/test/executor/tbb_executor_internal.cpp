// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "executor/gapi_tbb_executor.cpp"

#if defined(HAVE_TBB)


#include "../test_precomp.hpp"

#include <iostream>
namespace opencv_test
{
TEST(TBBExecutor, Basic)
{
    using namespace cv::gimpl::parallel;
    bool executed = false;
    tbb::concurrent_priority_queue<tile_node* , tile_node_indirect_priority_comparator> q;
    tile_node n([&](){
        executed = true;
    });
    q.push(&n);
    execute(q);
    EXPECT_EQ(true,executed);
}

} // namespace opencv_test
#endif //HAVE_TBB
