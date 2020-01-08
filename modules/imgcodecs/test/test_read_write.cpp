// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

/* Flag_Resize_Data = <file_name, resize_flag_and_dims> */
typedef tuple< string, tuple < int, int, int > > Flag_Resize_Data;

typedef testing::TestWithParam< Flag_Resize_Data > Imgcodecs_Resize;

/* resize_flag_and_dims = <imread_flag, expected_cols, expected_rows>*/
const tuple < int, int, int > resize_flag_and_dims[] =
{
    make_tuple(IMREAD_UNCHANGED, 640, 480),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_2, 320, 240),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_4, 160, 120),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_8, 80, 60),
    make_tuple(IMREAD_REDUCED_COLOR_2, 320, 240),
    make_tuple(IMREAD_REDUCED_COLOR_4, 160, 120),
    make_tuple(IMREAD_REDUCED_COLOR_8, 80, 60)
};

const string images[] =
{
#ifdef HAVE_JPEG
    "../cv/imgproc/stuff.jpg",
#endif
    "../cv/shared/1_itseez-0002524.png"
};

TEST_P(Imgcodecs_Resize, imread_reduce_flags)
{
    const string file_name = TS::ptr()->get_data_path() + get<0>(GetParam());
    const tuple< int, int, int > resize_flag_and_dim = get<1>(GetParam());
    const int imread_flag = get<0>(resize_flag_and_dim);
    const int cols = get<1>(resize_flag_and_dim);
    const int rows = get<2>(resize_flag_and_dim);
    {
        Mat img = imread(file_name, imread_flag);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cols, img.cols);
        EXPECT_EQ(rows, img.rows);
    }
}

//==================================================================================================

TEST_P(Imgcodecs_Resize, imdecode_reduce_flags)
{
    const string file_name = TS::ptr()->get_data_path() + get<0>(GetParam());

    std::vector<char> content;

    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(file_name.c_str(), mode);
    ASSERT_TRUE(ifs.is_open());

    content.clear();

    ifs.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(ifs.tellg());
    content.resize(sz);
    ifs.seekg(0, std::ios::beg);

    ifs.read((char*)content.data(), sz);
    ASSERT_FALSE(ifs.fail());

    const tuple< int, int, int > resize_flag_and_dim = get<1>(GetParam());
    const int imread_flag = get<0>(resize_flag_and_dim);
    const int cols = get<1>(resize_flag_and_dim);
    const int rows = get<2>(resize_flag_and_dim);
    {
        Mat img = imdecode(Mat(content), imread_flag);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cols, img.cols);
        EXPECT_EQ(rows, img.rows);
    }
}

//==================================================================================================

INSTANTIATE_TEST_CASE_P(Imread_Imdecode_Resize, Imgcodecs_Resize,
        testing::Combine(
            testing::ValuesIn(images),
            testing::ValuesIn(resize_flag_and_dims)
            )
        );

//==================================================================================================

TEST(Imgcodecs_Image, read_write_bmp)
{
    const size_t IMAGE_COUNT = 10;
    const double thresDbell = 32;

    for (size_t i = 0; i < IMAGE_COUNT; ++i)
    {
        stringstream s; s << i;
        const string digit = s.str();
        const string src_name = TS::ptr()->get_data_path() + "../python/images/QCIF_0" + digit + ".bmp";
        const string dst_name = cv::tempfile((digit + ".bmp").c_str());
        Mat image = imread(src_name);
        ASSERT_FALSE(image.empty());

        resize(image, image, Size(968, 757), 0.0, 0.0, INTER_CUBIC);
        imwrite(dst_name, image);
        Mat loaded = imread(dst_name);
        ASSERT_FALSE(loaded.empty());

        double psnr = cvtest::PSNR(loaded, image);
        EXPECT_GT(psnr, thresDbell);

        vector<uchar> from_file;

        FILE *f = fopen(dst_name.c_str(), "rb");
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        from_file.resize((size_t)len);
        fseek(f, 0, SEEK_SET);
        from_file.resize(fread(&from_file[0], 1, from_file.size(), f));
        fclose(f);

        vector<uchar> buf;
        imencode(".bmp", image, buf);
        ASSERT_EQ(buf, from_file);

        Mat buf_loaded = imdecode(Mat(buf), 1);
        ASSERT_FALSE(buf_loaded.empty());

        psnr = cvtest::PSNR(buf_loaded, image);
        EXPECT_GT(psnr, thresDbell);

        EXPECT_EQ(0, remove(dst_name.c_str()));
    }
}

//==================================================================================================

typedef string Ext;
typedef testing::TestWithParam<Ext> Imgcodecs_Image;

TEST_P(Imgcodecs_Image, read_write)
{
    const string ext = this->GetParam();
    const string full_name = cv::tempfile(ext.c_str());
    const string _name = TS::ptr()->get_data_path() + "../cv/shared/baboon.png";
    const double thresDbell = 32;

    Mat image = imread(_name);
    image.convertTo(image, CV_8UC3);
    ASSERT_FALSE(image.empty());

    imwrite(full_name, image);
    Mat loaded = imread(full_name);
    ASSERT_FALSE(loaded.empty());

    double psnr = cvtest::PSNR(loaded, image);
    EXPECT_GT(psnr, thresDbell);

    vector<uchar> from_file;
    FILE *f = fopen(full_name.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    from_file.resize((size_t)len);
    fseek(f, 0, SEEK_SET);
    from_file.resize(fread(&from_file[0], 1, from_file.size(), f));
    fclose(f);
    vector<uchar> buf;
    imencode("." + ext, image, buf);
    ASSERT_EQ(buf, from_file);

    Mat buf_loaded = imdecode(Mat(buf), 1);
    ASSERT_FALSE(buf_loaded.empty());

    psnr = cvtest::PSNR(buf_loaded, image);
    EXPECT_GT(psnr, thresDbell);

    EXPECT_EQ(0, remove(full_name.c_str()));
}

const string exts[] = {
#ifdef HAVE_PNG
    "png",
#endif
#ifdef HAVE_TIFF
    "tiff",
#endif
#ifdef HAVE_JPEG
    "jpg",
#endif
#if defined(HAVE_JASPER) && defined(OPENCV_IMGCODECS_ENABLE_JASPER_TESTS)
    "jp2",
#endif
#if 0 /*defined HAVE_OPENEXR && !defined __APPLE__*/
    "exr",
#endif
    "bmp",
#ifdef HAVE_IMGCODEC_PXM
    "ppm",
#endif
#ifdef HAVE_IMGCODEC_SUNRASTER
    "ras",
#endif
};

INSTANTIATE_TEST_CASE_P(imgcodecs, Imgcodecs_Image, testing::ValuesIn(exts));

TEST(Imgcodecs_Image, regression_9376)
{
    String path = findDataFile("readwrite/regression_9376.bmp");
    Mat m = imread(path);
    ASSERT_FALSE(m.empty());
    EXPECT_EQ(32, m.cols);
    EXPECT_EQ(32, m.rows);
}

//==================================================================================================

TEST(Imgcodecs_Image, write_umat)
{
    const string src_name = TS::ptr()->get_data_path() + "../python/images/baboon.bmp";
    const string dst_name = cv::tempfile(".bmp");

    Mat image1 = imread(src_name);
    ASSERT_FALSE(image1.empty());

    UMat image1_umat = image1.getUMat(ACCESS_RW);

    imwrite(dst_name, image1_umat);

    Mat image2 = imread(dst_name);
    ASSERT_FALSE(image2.empty());

    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), image1, image2);
    EXPECT_EQ(0, remove(dst_name.c_str()));
}

}} // namespace
