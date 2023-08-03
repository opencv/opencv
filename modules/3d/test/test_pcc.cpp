// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <fstream>
// TODO: when pcc.h put into 3d.hpp, this need to change
#include "../src/pcc.h"
#include "../src/pcc.cpp"

namespace opencv_test { namespace {

using namespace cv;

class PccEntropyCodingTest: public testing::Test
{
protected:
    void SetUp() override
    {
        //setbuf(stdout, nullptr);
        // set test char vector to encode and decode
        testCharVector = {'a', 'a', 'b', 'b', 'b', 'c', '1', '2', '3'};
        String FileBase = R"(C:\Users\WYH\Desktop\PointCloud\dress)";
        // TODO lew resolution(big number) causes error!!
        double resolution = 1;
        String label = "Test_Color_";
        String res_str = std::to_string(resolution);
        res_str.erase(res_str.find_last_not_of('0') + 1);
        res_str.erase(res_str.find('.'), 1);

        auto start = std::chrono::high_resolution_clock::now();
        //load .ply file
        String loadFileName = FileBase + ".ply";
        // color: rgb
        loadPointCloud(loadFileName, pointCloud, normal_placeholder);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by loadPointCloud: "
                  << duration.count()/1e6<< std::endl;

        PointCloudCompression pcc;

        start = std::chrono::high_resolution_clock::now();
        std::ofstream vectorToStream;
        vectorToStream.open(FileBase + label + res_str + ".bin", std::ios_base::binary);
        pcc.compress(pointCloud,resolution,vectorToStream);
        vectorToStream.close();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by Compress: "
                  << duration.count()/1e6<< std::endl;

        start = std::chrono::high_resolution_clock::now();
        std::ifstream streamToVector;
        streamToVector.open(FileBase + label + res_str + ".bin", std::ios_base::binary);
        pcc.decompress(streamToVector,restorePointCloudData);
        streamToVector.close();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by Decompress: "
                  << duration.count()/1e6<< std::endl;

        start = std::chrono::high_resolution_clock::now();
        String saveFileName= FileBase + label + res_str + ".ply";
        savePointCloud(saveFileName,restorePointCloudData, normal_placeholder, restore_color_attribute);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken by SavePointCloud: "
                  << duration.count()/1e6<< std::endl;
    }

public:
    std::vector<unsigned char> testCharVector;
    std::stringstream binaryStream;
    std::vector<unsigned char> restoreCharVector;
    EntropyCoder testEntropyCoder;

    //Origin point cloud data
    std::vector<Point3f> pointCloud;

    //Point cloud data from octree
    std::vector<Point3f> restorePointCloudData;
    std::vector<Point3f> normal_placeholder;
    vector<Point3f> restore_color_attribute;
};

TEST_F(PccEntropyCodingTest, EntropyEncodingTest){
    EXPECT_NO_THROW(testEntropyCoder.encodeCharVectorToStream(testCharVector, binaryStream));
    binaryStream.seekg(0, std::ios::beg);
    EXPECT_NO_THROW(testEntropyCoder.decodeStreamToCharVector(binaryStream, restoreCharVector));
    EXPECT_EQ(testCharVector, restoreCharVector);
}

//TEST_F(PccTest, PointCloudCompression){
//
//}

} // namespace
} // opencv_test