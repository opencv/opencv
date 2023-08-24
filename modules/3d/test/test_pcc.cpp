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

class PccTest: public testing::Test
{
protected:
    void SetUp() override
    {
        testCharVector = {'a', 'a', 'b', 'b', 'b', 'c', '1', '2', '3'};

        resolution = 0.001;
        inputFilename=cvtest::TS::ptr()->get_data_path()+"pointcloudio/sphere";

        //The point cloud is generated randomly by the function generateSphere()

        //generateSphere(pointCloud,vector<float>{0.f,0.f,0.f,10.f},0.5,10000,vector<float>{-10.f,10.f,-10.f,10.f,-10.f,10.f});
        //savePointCloud(inputFilename+".ply",pointCloud, normal_placeholder, restore_color_attribute);

        loadPointCloud(inputFilename + ".ply", pointCloud, normalPlaceholder, colorAttribute);
        String res_str = std::to_string(resolution);
        res_str.erase(res_str.find_last_not_of('0') + 1);
        res_str.erase(res_str.find('.'), 1);
        outputFilename = inputFilename + "_" + res_str + ".ply";
        if (colorAttribute.empty()) {
            outputFilename = inputFilename + "_" + res_str + ".ply";
        } else {
            outputFilename = inputFilename + "_" + res_str;
            res_str = std::to_string(qStep);
            res_str.erase(res_str.find_last_not_of('0') + 1);
            res_str.erase(res_str.find('.'), 1);
            outputFilename = outputFilename + "_q" + res_str + ".ply";
        }
    }

public:
    std::vector<unsigned char> testCharVector;
    std::stringstream binaryStream;
    std::vector<unsigned char> restoreCharVector;
    EntropyCoder testEntropyCoder;

    String inputFilename;
    String outputFilename;
    double resolution;
    double qStep;
    PointCloudCompression pcc;
    std::ofstream vectorToStream;
    std::ifstream streamToVector;

    //Origin point cloud data
    std::vector<Point3f> pointCloud;
    //Point cloud data from octree
    std::vector<Point3f> restorePointCloudData;
    std::vector<Point3f> normalPlaceholder;
    vector<Point3f> colorAttribute;
};

TEST_F(PccTest, EntropyEncodingTest){
    EXPECT_NO_THROW(testEntropyCoder.encodeCharVectorToStream(testCharVector, binaryStream));
    binaryStream.seekg(0, std::ios::beg);
    EXPECT_NO_THROW(testEntropyCoder.decodeStreamToCharVector(binaryStream, restoreCharVector));
    EXPECT_EQ(testCharVector, restoreCharVector);
};

TEST_F(PccTest, PointCloudCompressionTest){

    vectorToStream.open(inputFilename+".bin", std::ios_base::binary);
    pcc.compress(pointCloud, resolution, vectorToStream, colorAttribute, qStep);
    vectorToStream.close();

    streamToVector.open(inputFilename+".bin", std::ios_base::binary);
    pcc.decompress(streamToVector,restorePointCloudData);
    streamToVector.close();

    savePointCloud(outputFilename,restorePointCloudData, normalPlaceholder, colorAttribute);

    // BPP (Bits Per Point)
    // BPP is used to quantify the compression efficiency of point cloud compression algorithms.
    // It indicates the number of bits required to store a single point. BPP is calculated as the ratio of the file size
    // to the number of points in the point cloud file. A smaller BPP value corresponds to higher compression efficiency.
    std::ifstream pccFile(inputFilename+".bin", std::ios::binary | std::ios::ate);
    std::streampos pccFileSize = pccFile.tellg();
    pccFile.close();

    std::ifstream oriFile(inputFilename+".ply", std::ios::binary | std::ios::ate);
    std::streampos oriFileSize = oriFile.tellg();
    oriFile.close();

    float pcc_bpp=(float)pccFileSize/(float)pointCloud.size();
    float ori_bpp=(float)oriFileSize/(float)pointCloud.size();
    CV_LOG_INFO(nullptr,"The BPP of pccFile is "+std::to_string(pcc_bpp));
    CV_LOG_INFO(nullptr,"The BPP of oriFile is "+std::to_string(ori_bpp));

    // PSNR (Peak signal-to-noise ratio)
    // PSNR is commonly used to quantify reconstruction quality for images and video subject to lossy compression.
    // The same applies to point clouds.
    // There is a tool provided by MPEG to calculate PSNR in pcc, see details in https://github.com/minhkstn/mpeg-pcc-dmetric.



}



} // namespace
} // opencv_test