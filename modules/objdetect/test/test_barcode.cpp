// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"

namespace opencv_test{namespace{

typedef std::vector<std::string> stringvec;
typedef std::map<std::string, std::string> datasetType;

inline stringvec explode(const std::string &s, const char &c)
{
    std::string buff;
    stringvec v;

    for (auto n:s)
    {
        if (n != c) { buff += n; }
        else if (n == c && !buff.empty())
        {
            v.push_back(buff);
            buff = "";
        }
    }
    if (!buff.empty()) { v.push_back(buff); }

    return v;
}

inline datasetType buildDataSet(std::string result_file_path)
{
    std::ifstream result_file;
    datasetType dataset;
    result_file.open(result_file_path);
    std::string line;
    if (result_file.is_open())
    {
        while (std::getline(result_file, line))
        {
            stringvec result = explode(line, ',');
            std::string filename = result[0];
            if (dataset.find(filename) == dataset.end())
            {
                dataset[filename] = result[1];
            }
        }
    }

    result_file.close();
    return dataset;
}

inline datasetType initValidation(std::string path)
{
    const std::string valid_path = findDataFile(path);
    return buildDataSet(valid_path);
}

//==============================================================================

TEST(BARCODE_BarcodeDetector_single, regression)
{
    const std::string root = "barcode/single/";
    datasetType validation = initValidation(root + "result.csv");
    auto bardet = barcode::BarcodeDetector();
    datasetType::iterator iterator = validation.begin();
    while (iterator != validation.end())
    {
        std::string img_name = iterator->first;
        std::string result = iterator->second;
        std::string image_path = findDataFile(root + img_name);
        Mat img = imread(image_path);
        EXPECT_FALSE(img.empty()) << "Can't read image: " << image_path;
        std::vector<cv::Point2f> points;
        std::vector<std::string> infos;
        std::vector<std::string> formats;
        bardet.detectAndDecodeWithType(img, infos, formats, points);
        EXPECT_FALSE(points.empty()) << "Nothing detected: " << image_path;
        bool is_correct = false;
        for (const auto &ans : infos)
        {
            if (ans == result)
            {
                is_correct = true;
                break;
            }
        }
        EXPECT_TRUE(is_correct) << "No results for " << img_name;
        iterator++;
    }
}

TEST(BARCODE_BarcodeDetector_detect_multi, detect_regression)
{
    const std::string root = "barcode/multiple/";
    datasetType validation = initValidation(root + "result.csv");
    auto bardet = barcode::BarcodeDetector();
    datasetType::iterator iterator = validation.begin();
    while (iterator != validation.end())
    {
        std::string img = iterator->first;
        size_t expect_corners_size = std::stoi(iterator->second);
        std::string image_path = findDataFile(root + img);
        Mat src = imread(image_path);
        EXPECT_FALSE(src.empty()) << "Can't read image: " << image_path;

        std::vector<Point> corners;
        bardet.detectMulti(src, corners);
        EXPECT_EQ(corners.size(), expect_corners_size) << "Can't detect all barcodes: " << img;
        iterator++;
    }
}

TEST(BARCODE_BarcodeDetector_basic, not_found_barcode)
{
    auto bardet = barcode::BarcodeDetector();
    std::vector<Point> corners;
    vector<cv::String> decoded_info;
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    EXPECT_FALSE(bardet.detectMulti(zero_image, corners));
    corners = std::vector<Point>(4);
    EXPECT_ANY_THROW(bardet.decodeMulti(zero_image, corners, decoded_info));
}

}} // opencv_test::<anonymous>::
