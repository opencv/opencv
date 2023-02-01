// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
namespace opencv_test { namespace {

#if !defined CV_CXX11
// Wrapper for generating seeded random number via std::rand.
template<unsigned Seed>
class SeededRandFunctor {
public:
    SeededRandFunctor() { std::srand(Seed); }
    int operator()(int i) { return std::rand() % (i + 1); }
};
#endif

std::string encode_qrcode_images_name[] = {
        "version1_mode1.png", "version1_mode2.png", "version1_mode4.png",
        "version2_mode1.png", "version2_mode2.png", "version2_mode4.png",
        "version3_mode2.png", "version3_mode4.png",
        "version4_mode4.png"
};

std::string encode_qrcode_eci_images_name[] = {
        "version1_mode7.png",
        "version2_mode7.png",
        "version3_mode7.png",
        "version4_mode7.png",
        "version5_mode7.png"
};

const Size fixed_size = Size(200, 200);
const float border_width = 2.0;

int establishCapacity(QRCodeEncoder::EncodeMode mode, int version, int capacity)
{
    int result = 0;
    capacity *= 8;
    capacity -= 4;
    switch (mode)
    {
        case QRCodeEncoder::MODE_NUMERIC:
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
        case QRCodeEncoder::MODE_ALPHANUMERIC:
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
        case QRCodeEncoder::MODE_BYTE:
        {
            if (version > 9)
                capacity -= 16;
            else
                capacity -= 8;
            result = capacity / 8;
            break;
        }
        default:
            break;
    }
    return result;
}

// #define UPDATE_TEST_DATA
#ifdef UPDATE_TEST_DATA

TEST(Objdetect_QRCode_Encode, generate_test_data)
{
    const std::string root = "qrcode/encode";
    const std::string dataset_config = findDataFile(root + "/" + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::WRITE);

    file_config << "test_images" << "[";
    size_t images_count = sizeof(encode_qrcode_images_name) / sizeof(encode_qrcode_images_name[0]);
    for (size_t i = 0; i < images_count; i++)
    {
        file_config << "{:" << "image_name" << encode_qrcode_images_name[i];
        std::string image_path = findDataFile(root + "/" + encode_qrcode_images_name[i]);

        Mat src = imread(image_path, IMREAD_GRAYSCALE);
        Mat straight_barcode;
        EXPECT_TRUE(!src.empty()) << "Can't read image: " << image_path;

        std::vector<Point2f> corners(4);
        corners[0] = Point2f(border_width, border_width);
        corners[1] = Point2f(qrcode.cols * 1.0f - border_width, border_width);
        corners[2] = Point2f(qrcode.cols * 1.0f - border_width, qrcode.rows * 1.0f - border_width);
        corners[3] = Point2f(border_width, qrcode.rows * 1.0f - border_width);

        Mat resized_src;
        resize(qrcode, resized_src, fixed_size, 0, 0, INTER_AREA);
        float width_ratio =  resized_src.cols * 1.0f / qrcode.cols;
        float height_ratio = resized_src.rows * 1.0f / qrcode.rows;
        for(size_t j = 0; j < corners.size(); j++)
        {
            corners[j].x = corners[j].x * width_ratio;
            corners[j].y = corners[j].y * height_ratio;
        }

        std::string decoded_info = "";
#ifdef HAVE_QUIRC
        EXPECT_TRUE(decodeQRCode(resized_src, corners, decoded_info, straight_barcode)) << "The QR code cannot be decoded: " << image_path;
#endif
        file_config << "info" << decoded_info;
        file_config << "}";
    }
    file_config << "]";
    file_config.release();
}
#else

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Encode;
TEST_P(Objdetect_QRCode_Encode, regression) {
    const int pixels_error = 3;
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
                Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create();
                Mat result;
                encoder->encode(original_info, result);
                EXPECT_FALSE(result.empty()) << "Can't generate QR code image";

                Mat src = imread(image_path, IMREAD_GRAYSCALE);
                Mat straight_barcode;
                EXPECT_TRUE(!src.empty()) << "Can't read image: " << image_path;

                double diff_norm = cvtest::norm(result - src, NORM_L1);
                EXPECT_NEAR(diff_norm, 0.0, pixels_error) << "The generated QRcode is not same as test data. The difference: " << diff_norm;

                return; // done
            }
        }
        FAIL()  << "Not found results in config file:" << dataset_config
                << "\nRe-run tests with enabled UPDATE_ENCODE_TEST_DATA macro to update test data.";
    }
}

typedef testing::TestWithParam< std::string > Objdetect_QRCode_Encode_ECI;
TEST_P(Objdetect_QRCode_Encode_ECI, regression) {
    const int pixels_error = 3;
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
        QRCodeEncoder::Params params;
        params.mode = QRCodeEncoder::MODE_ECI;

        for (size_t index = 0; index < images_count; index++)
        {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image)
            {
                std::string original_info = config["info"];
                Mat result;
                Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(params);
                encoder->encode(original_info, result);
                EXPECT_FALSE(result.empty()) << "Can't generate QR code image";

                Mat src = imread(image_path, IMREAD_GRAYSCALE);
                Mat straight_barcode;
                EXPECT_TRUE(!src.empty()) << "Can't read image: " << image_path;

                double diff_norm = cvtest::norm(result - src, NORM_L1);
                EXPECT_NEAR(diff_norm, 0.0, pixels_error) << "The generated QRcode is not same as test data. The difference: " << diff_norm;

                return; // done
            }
        }
        FAIL()  << "Not found results in config file:" << dataset_config
                << "\nRe-run tests with enabled UPDATE_ENCODE_TEST_DATA macro to update test data.";
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode, testing::ValuesIn(encode_qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode_ECI, testing::ValuesIn(encode_qrcode_eci_images_name));

TEST(Objdetect_QRCode_Encode_Decode, regression)
{
    const std::string root = "qrcode/decode_encode";
    const int min_version = 1;
    const int test_max_version = 5;
    const int max_ec_level = 3;
    const std::string dataset_config = findDataFile(root + "/" + "symbol_sets.json");
    const std::string version_config = findDataFile(root + "/" + "capacity.json");

    FileStorage file_config(dataset_config, FileStorage::READ);
    FileStorage capacity_config(version_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    ASSERT_TRUE(capacity_config.isOpened()) << "Can't read validation data: " << version_config;

    FileNode mode_list = file_config["symbols_sets"];
    FileNode capacity_list = capacity_config["version_ecc_capacity"];

    size_t mode_count = static_cast<size_t>(mode_list.size());
    ASSERT_GT(mode_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

    const int testing_modes = 3;
    QRCodeEncoder::EncodeMode modes[testing_modes] = {
        QRCodeEncoder::MODE_NUMERIC,
        QRCodeEncoder::MODE_ALPHANUMERIC,
        QRCodeEncoder::MODE_BYTE
    };

    for (int i = 0; i < testing_modes; i++)
    {
        QRCodeEncoder::EncodeMode mode = modes[i];
        FileNode config = mode_list[i];

        std::string symbol_set = config["symbols_set"];

        for(int version = min_version; version <= test_max_version; version++)
        {
            FileNode capa_config = capacity_list[version - 1];
            for(int level = 0; level <= max_ec_level; level++)
            {
                const int cur_capacity = capa_config["ecc_level"][level];

                int true_capacity = establishCapacity(mode, version, cur_capacity);

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
                        input_info += input_info.substr(count%(int)input_info.length(), 1);
                        count++;
                    }
                }

                QRCodeEncoder::Params params;
                params.version = version;
                params.correction_level = static_cast<QRCodeEncoder::CorrectionLevel>(level);
                params.mode = mode;
                Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(params);
                Mat qrcode;
                encoder->encode(input_info, qrcode);
                EXPECT_TRUE(!qrcode.empty()) << "Can't generate this QR image (" << "mode: " << (int)mode <<
                                                " version: "<< version <<" error correction level: "<< (int)level <<")";

                std::vector<Point2f> corners(4);
                corners[0] = Point2f(border_width, border_width);
                corners[1] = Point2f(qrcode.cols * 1.0f - border_width, border_width);
                corners[2] = Point2f(qrcode.cols * 1.0f - border_width, qrcode.rows * 1.0f - border_width);
                corners[3] = Point2f(border_width, qrcode.rows * 1.0f - border_width);

                Mat resized_src;
                resize(qrcode, resized_src, fixed_size, 0, 0, INTER_AREA);
                float width_ratio =  resized_src.cols * 1.0f / qrcode.cols;
                float height_ratio = resized_src.rows * 1.0f / qrcode.rows;
                for(size_t k = 0; k < corners.size(); k++)
                {
                    corners[k].x = corners[k].x * width_ratio;
                    corners[k].y = corners[k].y * height_ratio;
                }

#ifdef HAVE_QUIRC
                Mat straight_barcode;
                std::string output_info = QRCodeDetector().decode(resized_src, corners, straight_barcode);
                EXPECT_FALSE(output_info.empty())
                    << "The generated QRcode cannot be decoded." << " Mode: " << (int)mode
                    << " version: " << version << " error correction level: " << (int)level;
                EXPECT_EQ(input_info, output_info) << "The generated QRcode is not same as test data." << " Mode: " << (int)mode <<
                                                        " version: " << version << " error correction level: " << (int)level;
#endif
            }
        }
    }

}

TEST(Objdetect_QRCode_Encode_Kanji, regression)
{
    QRCodeEncoder::Params params;
    params.mode = QRCodeEncoder::MODE_KANJI;

    Mat qrcode;

    const int testing_versions = 3;
    std::string input_infos[testing_versions] = {"\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd\x90\xa2\x8a\x45", // "Hello World" in Japanese
                                                 "\x82\xa8\x95\xa0\x82\xaa\x8b\xf3\x82\xa2\x82\xc4\x82\xa2\x82\xdc\x82\xb7", // "I am hungry" in Japanese
                                                 "\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd\x81\x41\x8e\x84\x82\xcd\x8f\xad\x82\xb5\x93\xfa\x96\x7b\x8c\xea\x82\xf0\x98\x62\x82\xb5\x82\xdc\x82\xb7" // "Hello, I speak a little Japanese" in Japanese
                                                };

    for (int i = 0; i < testing_versions; i++)
    {
        std::string input_info = input_infos[i];
        Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(params);
        encoder->encode(input_info, qrcode);

        std::vector<Point2f> corners(4);
        corners[0] = Point2f(border_width, border_width);
        corners[1] = Point2f(qrcode.cols * 1.0f - border_width, border_width);
        corners[2] = Point2f(qrcode.cols * 1.0f - border_width, qrcode.rows * 1.0f - border_width);
        corners[3] = Point2f(border_width, qrcode.rows * 1.0f - border_width);

        Mat resized_src;
        resize(qrcode, resized_src, fixed_size, 0, 0, INTER_AREA);
        float width_ratio =  resized_src.cols * 1.0f / qrcode.cols;
        float height_ratio = resized_src.rows * 1.0f / qrcode.rows;
        for(size_t j = 0; j < corners.size(); j++)
        {
            corners[j].x = corners[j].x * width_ratio;
            corners[j].y = corners[j].y * height_ratio;
        }

#ifdef HAVE_QUIRC
        Mat straight_barcode;
        std::string decoded_info = QRCodeDetector().decode(resized_src, corners, straight_barcode);
        EXPECT_FALSE(decoded_info.empty()) << "The generated QRcode cannot be decoded.";
        EXPECT_EQ(input_info, decoded_info);
#endif
    }
}

TEST(Objdetect_QRCode_Encode_Decode_Structured_Append, DISABLED_regression)
{
    // disabled since QR decoder probably doesn't support structured append mode qr codes
    const std::string root = "qrcode/decode_encode";
    const std::string dataset_config = findDataFile(root + "/" + "symbol_sets.json");
    const std::string version_config = findDataFile(root + "/" + "capacity.json");

    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;

    FileNode mode_list = file_config["symbols_sets"];

    size_t mode_count = static_cast<size_t>(mode_list.size());
    ASSERT_GT(mode_count, 0u) << "Can't find validation data entries in 'test_images': " << dataset_config;

    int modes[] = {1, 2, 4};
    const int min_stuctures_num = 2;
    const int max_stuctures_num = 5;
    for (int i = 0; i < 3; i++)
    {
        int mode = modes[i];
        FileNode config = mode_list[i];

        std::string symbol_set = config["symbols_set"];

        std::string input_info = symbol_set;
#if defined CV_CXX11
        // std::random_shuffle is deprecated since C++11 and removed in C++17.
        // Use manually constructed RNG with a fixed seed and std::shuffle instead.
        std::mt19937 rand_gen {1};
        std::shuffle(input_info.begin(), input_info.end(), rand_gen);
#else
        SeededRandFunctor<1> rand_gen;
        std::random_shuffle(input_info.begin(), input_info.end(), rand_gen);
#endif
        for (int j = min_stuctures_num; j < max_stuctures_num; j++)
        {
            QRCodeEncoder::Params params;
            params.structure_number = j;
            Ptr<QRCodeEncoder> encoder = QRCodeEncoder::create(params);
            vector<Mat> qrcodes;
            encoder->encodeStructuredAppend(input_info, qrcodes);
            EXPECT_TRUE(!qrcodes.empty()) << "Can't generate this QR images";

            std::string output_info = "";
            for (size_t k = 0; k < qrcodes.size(); k++)
            {
                Mat qrcode = qrcodes[k];

                std::vector<Point2f> corners(4);
                corners[0] = Point2f(border_width, border_width);
                corners[1] = Point2f(qrcode.cols * 1.0f - border_width, border_width);
                corners[2] = Point2f(qrcode.cols * 1.0f - border_width, qrcode.rows * 1.0f - border_width);
                corners[3] = Point2f(border_width, qrcode.rows * 1.0f - border_width);

                Mat resized_src;
                resize(qrcode, resized_src, fixed_size, 0, 0, INTER_AREA);
                float width_ratio =  resized_src.cols * 1.0f / qrcode.cols;
                float height_ratio = resized_src.rows * 1.0f / qrcode.rows;
                for(size_t m = 0; m < corners.size(); m++)
                {
                    corners[m].x = corners[m].x * width_ratio;
                    corners[m].y = corners[m].y * height_ratio;
                }

#ifdef HAVE_QUIRC
                Mat straight_barcode;
                std::string decoded_info = QRCodeDetector().decode(resized_src, corners, straight_barcode);
                EXPECT_FALSE(decoded_info.empty())
                    << "The generated QRcode cannot be decoded." << " Mode: " << modes[i]
                    << " structures number: " << k << "/" << j;
                output_info += decoded_info;
#endif
            }
#ifdef HAVE_QUIRC
            EXPECT_EQ(input_info, output_info) << "The generated QRcode is not same as test data." << " Mode: " << mode <<
                                                  " structures number: " << j;
#else
            std::cout << "Mode=" << mode << ": Unable to verify generated QR codes - QUIRC is disabled" << std::endl;
#endif
        }
    }
}

#endif // UPDATE_QRCODE_TEST_DATA

CV_ENUM(EncodeModes, QRCodeEncoder::EncodeMode::MODE_NUMERIC,
                     QRCodeEncoder::EncodeMode::MODE_ALPHANUMERIC,
                     QRCodeEncoder::EncodeMode::MODE_BYTE)

typedef ::testing::TestWithParam<EncodeModes> Objdetect_QRCode_Encode_Decode_Structured_Append_Parameterized;
TEST_P(Objdetect_QRCode_Encode_Decode_Structured_Append_Parameterized, regression_22205)
{
    const std::string input_data = "the quick brown fox jumps over the lazy dog";

    std::vector<cv::Mat> result_qrcodes;

    cv::QRCodeEncoder::Params params;
    int encode_mode = GetParam();
    params.mode = static_cast<cv::QRCodeEncoder::EncodeMode>(encode_mode);

    for(size_t struct_num = 2; struct_num < 5; ++struct_num)
    {
        params.structure_number = static_cast<int>(struct_num);
        cv::Ptr<cv::QRCodeEncoder> encoder = cv::QRCodeEncoder::create(params);
        encoder->encodeStructuredAppend(input_data, result_qrcodes);
        EXPECT_EQ(result_qrcodes.size(), struct_num) << "The number of QR Codes requested is not equal"<<
                                                    "to the one returned";
    }
}
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Encode_Decode_Structured_Append_Parameterized, EncodeModes::all());

TEST(Objdetect_QRCode_Encode_Decode, regression_issue22029)
{
    const cv::String msg = "OpenCV";
    const int min_version = 1;
    const int max_version = 40;

    for ( int v = min_version ; v <= max_version ; v++ )
    {
        SCOPED_TRACE(cv::format("version=%d",v));

        Mat qrimg;
        QRCodeEncoder::Params params;
        params.version = v;
        Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
        qrcode_enc->encode(msg, qrimg);

        const int white_margin = 2;
        const int finder_width = 7;

        const int timing_pos = white_margin + 6;
        int i;

        // Horizontal Check
        // (1) White margin(Left)
        for(i = 0; i < white_margin ; i++ )
        {
            ASSERT_EQ((uint8_t)255, qrimg.at<uint8_t>(i, timing_pos)) << "i=" << i;
        }
        // (2) Finder pattern(Left)
        for(     ; i < white_margin + finder_width ; i++ )
        {
            ASSERT_EQ((uint8_t)0, qrimg.at<uint8_t>(i, timing_pos)) << "i=" << i;
        }
        // (3) Timing pattern
        for(     ; i < qrimg.rows - finder_width - white_margin; i++ )
        {
            ASSERT_EQ((uint8_t)(i % 2 == 0)?0:255, qrimg.at<uint8_t>(i, timing_pos)) << "i=" << i;
        }
        // (4) Finder pattern(Right)
        for(     ; i < qrimg.rows - white_margin; i++ )
        {
            ASSERT_EQ((uint8_t)0, qrimg.at<uint8_t>(i, timing_pos)) << "i=" << i;
        }
        // (5) White margin(Right)
        for(     ; i < qrimg.rows ; i++ )
        {
            ASSERT_EQ((uint8_t)255, qrimg.at<uint8_t>(i, timing_pos)) << "i=" << i;
        }

        // Vertical Check
        // (1) White margin(Top)
        for(i = 0; i < white_margin ; i++ )
        {
            ASSERT_EQ((uint8_t)255, qrimg.at<uint8_t>(timing_pos, i)) << "i=" << i;
        }
        // (2) Finder pattern(Top)
        for(     ; i < white_margin + finder_width ; i++ )
        {
            ASSERT_EQ((uint8_t)0, qrimg.at<uint8_t>(timing_pos, i)) << "i=" << i;
        }
        // (3) Timing pattern
        for(     ; i < qrimg.rows - finder_width - white_margin; i++ )
        {
            ASSERT_EQ((uint8_t)(i % 2 == 0)?0:255, qrimg.at<uint8_t>(timing_pos, i)) << "i=" << i;
        }
        // (4) Finder pattern(Bottom)
        for(     ; i < qrimg.rows - white_margin; i++ )
        {
            ASSERT_EQ((uint8_t)0, qrimg.at<uint8_t>(timing_pos, i)) << "i=" << i;
        }
        // (5) White margin(Bottom)
        for(     ; i < qrimg.rows ; i++ )
        {
            ASSERT_EQ((uint8_t)255, qrimg.at<uint8_t>(timing_pos, i)) << "i=" << i;
        }
    }
}

}} // namespace
