// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
namespace opencv_test { namespace {

std::string encode_qrcode_images_name[] = {
        "version1_mode1.png", "version1_mode2.png", "version1_mode4.png",
        "version2_mode1.png", "version2_mode2.png", "version2_mode4.png",
        "version3_mode2.png", "version3_mode4.png",
        "version4_mode4.png"
};

const Size fixed_size = Size(600, 600);
const float border_width = 2.0;
int countDiffPixels(cv::Mat in1, cv::Mat in2);
int countDiffPixels(cv::Mat in1, cv::Mat in2) {
    cv::Mat diff;
    cv::compare(in1, in2, diff, cv::CMP_NE);
    return cv::countNonZero(diff);
}
int establishCapacity(int mode, int version, int capacity);
int establishCapacity(int mode, int version, int capacity)
{
    int result = 0;
    capacity *= 8;
    capacity -= 4;
    switch (mode)
    {
        case 1:
        {
            if (version >= 10)
                capacity -= 12;
            else
                capacity -= 10;
            int tmp = capacity / 10;
            result = tmp * 3;
            if (tmp * 10 + 7 <= capacity)
                result += 2;
            else if (tmp * 10 + 4 <= capacity)
                result += 1;
            break;
        }
        case 2:
        {
            if (version < 10)
                capacity -= 9;
            else
                capacity -= 13;
            int tmp = capacity / 11;
            result = tmp * 2;
            if (tmp * 11 + 6 <= capacity)
                result++;
            break;
        }
        case 4:
        {
            if (version > 9)
                capacity -= 16;
            else
                capacity -= 8;
            result = capacity / 8;
            break;
        }
    }
    return result;
}

// #define UPDATE_TEST_DATA
#ifdef UPDATE_TEST_DATA

TEST(Objdetect_QRCode_Encode, generate_test_data)
{
    const std::string root = "qrcode/encode";
    const std::string dataset_config = findDataFile(root +"/"+ "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(encode_qrcode_images_name) / sizeof(encode_qrcode_images_name[0]);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << encode_qrcode_images_name[i];
        std::string image_path = findDataFile(root +"/"+ encode_qrcode_images_name[i]);

        /**read from test set*/
        Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
        std::vector<Point2f> corners(4);
        corners[0] = Point2f(border_width, border_width);
        corners[1] = Point2f(src.cols - border_width, border_width);
        corners[2] = Point2f(src.cols - border_width, src.rows - border_width);
        corners[3] = Point2f(border_width, src.rows - border_width);
        Mat resized_src;
        resize(src, resized_src, fixed_size, 0, 0, INTER_AREA);
        double width_ratio =  resized_src.cols * 1.0 / src.cols;
        double height_ratio = resized_src.rows * 1.0 / src.rows;
        for(size_t j = 0; j < corners.size(); j++)
        {
            corners[j].x = corners[j].x * width_ratio;
            corners[j].y = corners[j].y * height_ratio;
        }

        std::string decoded_info;
        ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
#ifdef HAVE_QUIRC
        EXPECT_TRUE(decodeQRCode(resized_src, corners, decoded_info, straight_barcode))<< "ERROR : " << image_path;
#endif
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}
#else

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Encode;
TEST_P(Objdetect_QRCode_Encode, regression){
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/encode";

    std::string image_path = findDataFile(root + "/" + name_current_image);
    const std::string dataset_config = findDataFile(root + "/" + "dataset_config.json");
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
                std::string original_info = config["info"];
                QRCodeEncoder encoder;
                Mat result ;
                bool success = encoder.generate(original_info, result);
                ASSERT_TRUE(success) << "Can't generate qr image :" << name_test_image;

                Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
                ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

                bool eq = countDiffPixels(result, src) == 0;
                ASSERT_TRUE(eq) << "The generated QRcode is not same as test data:" << name_test_image;

                return; // done
            }
        }
        std::cerr
                << "Not found results for '" << name_current_image
                << "' image in config file:" << dataset_config << std::endl
                << "Re-run tests with enabled UPDATE_ENCODE_TEST_DATA macro to update test data."
                << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode, testing::ValuesIn(encode_qrcode_images_name));

TEST(Objdetect_QRCode_Encode_Decode, regression){
    const std::string root = "qrcode/decode_encode";
    const int min_version = 1;
    const int test_max_version = 5;
    const int max_ecc = 3;
    const std::string dataset_config = findDataFile(root + "/" + "symbol_sets.json");
    const std::string version_config = findDataFile(root + "/" + "capacity.json");

    FileStorage file_config(dataset_config, FileStorage::READ);
    FileStorage capacity_config(version_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened() && capacity_config.isOpened()) << "Can't read validation data: " << dataset_config;

    FileNode mode_list = file_config["symbols_sets"];
    FileNode capacity_list = capacity_config["version_ecc_capacity"];

    size_t mode_count = static_cast<size_t>(mode_list.size());
    ASSERT_GT(mode_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

    int modes[] = {1, 2, 4};
    for (size_t i = 0; i < 3; i++)
    {
        int mode = modes[i];
        FileNode config = mode_list[(int)i];

        std::string symbol_set = config["symbols_set"];

        for(int v = min_version; v <= test_max_version; v++)
        {
            FileNode capa_config = capacity_list[v - 1];
            for(int m = 0; m <= max_ecc; m++)
            {
                std::string cur_level = capa_config["verison_level"];
                const int cur_capacity = capa_config["ecc_level"][m];

                int true_capacity = establishCapacity(mode, v, cur_capacity);

                std::string input_info = symbol_set;
                std::random_shuffle(input_info.begin(),input_info.end());
                int count = 0;
                if((int)input_info.length() > true_capacity)
                {
                    input_info = input_info.substr(0, true_capacity);
                }
                else
                {
                    while ((int)input_info.length() != true_capacity)
                    {
                        input_info += input_info.substr(count, 1);
                        count++;
                    }
                }

                QRCodeEncoder my_encoder;
                vector<Mat> qrcodes;
                bool generate_success = my_encoder.generate(input_info, qrcodes, v, m, mode);
                ASSERT_TRUE(generate_success) << "Can't generate this QR image :("<<"mode : "<<mode<<
                                                " version : "<<v<<" ecc_level : "<<m<<")";
                std::string output_info = "";
                for(size_t n = 0; n < qrcodes.size(); n++)
                {
                    Mat src = qrcodes[n];

                    std::vector<Point2f> corners(4);
                    corners[0] = Point2f(border_width, border_width);
                    corners[1] = Point2f(src.cols * 1.0f - border_width, border_width);
                    corners[2] = Point2f(src.cols * 1.0f - border_width, src.rows * 1.0f - border_width);
                    corners[3] = Point2f(border_width, src.rows * 1.0f - border_width);

                    Mat resized_src;
                    resize(src, resized_src, fixed_size, 0, 0, INTER_AREA);
                    float width_ratio =  resized_src.cols * 1.0f / src.cols ;
                    float height_ratio = resized_src.rows * 1.0f / src.rows;
                    for(size_t p = 0; p < corners.size(); p ++)
                    {
                        corners[p].x = corners[p].x * width_ratio;
                        corners[p].y = corners[p].y * height_ratio;
                    }

                    std::string decoded_info ;
                    Mat straight_barcode;
#ifdef HAVE_QUIRC
                    bool success = decodeQRCode(resized_src, corners, decoded_info, straight_barcode);
                    ASSERT_TRUE(success) << "The generated QRcode cannot be decoded."<<" Mode : "<<mode<<
                                            " version : "<<v<<" ecc_level : "<<m;
                    output_info += decoded_info;
#endif
                }
                EXPECT_EQ(input_info, output_info) << "The generated QRcode is not same as test data."<<" Mode : "<<mode<<
                                                        " version : "<<v<<" ecc_level : "<<m;
            }
        }
    }

}

#endif // UPDATE_QRCODE_TEST_DATA

}} // namespace
