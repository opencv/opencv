// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

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
