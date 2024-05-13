// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Zhangjie Chen <zhangjiec01@gmail.com>
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(SpC, FunctionTest) {
    // load point cloud
    auto folder = cvtest::TS::ptr()->get_data_path();
    String test_file_path = folder + "pointcloudio/two_cubes.ply";

    std::vector<cv::Point3f> vertices;
    std::vector<std::vector<int32_t>> indices;
    cv::loadPointCloud(test_file_path, vertices, cv::noArray(), cv::noArray(), indices);
    cv::SpectralCluster cluster(0.1, 0.1);

    std::vector<int> results;
    cluster.cluster(results, vertices, indices, 2);
    std::vector<int> truth_value = {0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0};

    for (size_t i = 0; i < results.size(); ++i)
        CV_Assert(results[i] == truth_value[i]);

}
}
}

