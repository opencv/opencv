// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"


namespace opencv_test { namespace {

std::string qrcode_images_name[] = {
    // "20110817_030.jpg",
    "20110817_048.jpg",
    "img_20120226_161648.jpg",
    "img_2714.jpg",
    "img_2716.jpg",
    "img_3011.jpg",
    "img_3029.jpg",
    "img_3070.jpg",
    "qr_test_030.jpg"
};

// #define UPDATE_QRCODE_TEST_DATA
#ifdef  UPDATE_QRCODE_TEST_DATA

TEST(Objdetect_QRCode, generate_test_data)
{
    const std::string root = "qrcode/";
    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(qrcode_images_name) / sizeof(qrcode_images_name[0]);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_name[i];
        std::string image_path = findDataFile(root + qrcode_images_name[i]);
        std::vector<Point> corners;
        Mat src = imread(image_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
        EXPECT_TRUE(detectQRCode(src, corners));
        file_config << "x" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < corners.size(); j++) { file_config << corners[j].y; }
        file_config << "]" << "}";
    }
    file_config << "]";
    file_config.release();
}

#else

typedef testing::TestWithParam< std::string > Objdetect_QRCode;
TEST_P(Objdetect_QRCode, regression)
{
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/";
    const int pixels_error = 3;

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector<Point> corners;
    ASSERT_TRUE(detectQRCode(src, corners));

    const std::string dataset_config = findDataFile(root + "dataset_config.json", false);
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["test_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++)
        {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image)
            {
                for (int i = 0; i < 4; i++)
                {
                    int x = config["x"][i];
                    int y = config["y"][i];
                    EXPECT_NEAR(x, corners[i].x, pixels_error);
                    EXPECT_NEAR(y, corners[i].y, pixels_error);
                }
                return; // done
            }
        }
        std::cerr
            << "Not found results for '" << name_current_image
            << "' image in config file:" << dataset_config << std::endl
            << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
            << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));



TEST(Objdetect_QRCode_basic, not_found_qrcode)
{
    std::vector<Point> corners;
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    EXPECT_FALSE(detectQRCode(zero_image, corners));
}



#endif // UPDATE_QRCODE_TEST_DATA

}} // namespace
