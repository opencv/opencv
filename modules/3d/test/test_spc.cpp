//
// Created by jeffery on 24-3-14.
//

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(SpC, FunctionTest) {
    // load point cloud
    // String test_file_path = "/home/jeffery/Desktop/Sustech/Thesis/ptclouds/cube.ply";
    String test_file_path = "/home/jeffery/Desktop/Sustech/Thesis/ptclouds/two_cubes.ply";
    // String test_file_path = "/home/jeffery/Desktop/Sustech/Thesis/ptclouds/bunny/bun_zipper_res4.ply";
    // String test_file_path = "/home/jeffery/Desktop/Sustech/Thesis/ptclouds/SHREC19_lores/models/scan_000.obj";
    std::vector<cv::Point3f> vertices;
    std::vector<std::vector<int32_t>> indices;
//    cv::loadMesh(test_file_path, vertices, indices);
    cv::loadPointCloud(test_file_path, vertices, cv::noArray(), cv::noArray(), indices);

    cv::SpectralCluster cluster(0.03, 0.15);

    std::vector<int> results;

    cluster.cluster(results, vertices, indices, 2);

    std::cout << "Clustering Complete!" << std::endl;
}

}
}

