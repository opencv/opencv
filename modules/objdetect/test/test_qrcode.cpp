// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"


namespace opencv_test
{

String qrcode_images_name[] = {
    "20110817_030.jpg",
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
    String root = cvtest::TS::ptr()->get_data_path() + "qrcode/";
    String dataset_config = cvtest::TS::ptr()->get_data_path() + "qrcode/dataset_config.json";
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(qrcode_images_name) / sizeof(String);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << qrcode_images_name[i];
        String image_path = root + qrcode_images_name[i];
        std::vector<Point> transform;
        Mat src = imread(image_path, IMREAD_GRAYSCALE);
        EXPECT_TRUE(detectQRCode(src, transform));
        file_config << "x" << "[:";
        for (size_t j = 0; j < transform.size(); j++) { file_config << transform[j].x; }
        file_config << "]";
        file_config << "y" << "[:";
        for (size_t j = 0; j < transform.size(); j++) { file_config << transform[j].y; }
        file_config << "]" << "}";
    }
    file_config << "]";
    file_config.release();
}

#else

typedef testing::TestWithParam< String > Objdetect_QRCode;
TEST_P(Objdetect_QRCode, regression)
{
    String root = cvtest::TS::ptr()->get_data_path() + "qrcode/";
    String dataset_config = cvtest::TS::ptr()->get_data_path() + "qrcode/dataset_config.json";
    FileStorage file_config(dataset_config, FileStorage::READ);
    const int pixels_error = 3;

    std::vector<Point> corners;
    String image_path = root + String(GetParam());
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    EXPECT_TRUE(detectQRCode(src, corners));

    if (file_config.isOpened())
    {
        FileNode images_list = file_config["test_images"];
        int index = 0, images_count = static_cast<int>(images_list.size());
        ASSERT_GT(images_count, 0);

        bool runTestsFlag = false;
        String name_current_image = String(GetParam());
        for (; index < images_count; index++)
        {
            String name_test_image = images_list[index]["image_name"];
            if (name_test_image == name_current_image)
            {
                for (int i = 0; i < 4; i++)
                {
                    int x = images_list[index]["x"][i];
                    int y = images_list[index]["y"][i];
                    EXPECT_NEAR(x, corners[i].x, pixels_error);
                    EXPECT_NEAR(y, corners[i].y, pixels_error);
                }
                runTestsFlag = true;
            }
        }
        if (!runTestsFlag)
        {
            std::cout << "Not found results for " << name_current_image;
            std::cout << " image in dataset_config.json file." << std::endl;
        }

        file_config.release();
    }
    else
    {
        std::cout << " Not found dataset_config.json file." << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(objdetect, Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));

TEST(Objdetect_QRCode, not_found_qrcode)
{
    std::vector<Point> corners;
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    EXPECT_FALSE(detectQRCode(zero_image, corners));
}

#endif

} // namespace
