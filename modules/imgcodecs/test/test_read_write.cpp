// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test { namespace {

/* < <file_name, image_size>, <imread mode, scale> > */
typedef tuple< tuple<string, Size>, tuple<ImreadModes, int> > Imgcodecs_Resize_t;

typedef testing::TestWithParam< Imgcodecs_Resize_t > Imgcodecs_Resize;

/* resize_flag_and_dims = <imread_flag, scale>*/
const tuple <ImreadModes, int> resize_flag_and_dims[] =
{
    make_tuple(IMREAD_UNCHANGED, 1),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_2, 2),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_4, 4),
    make_tuple(IMREAD_REDUCED_GRAYSCALE_8, 8),
    make_tuple(IMREAD_REDUCED_COLOR_2, 2),
    make_tuple(IMREAD_REDUCED_COLOR_4, 4),
    make_tuple(IMREAD_REDUCED_COLOR_8, 8)
};

const tuple<string, Size> images[] =
{
#ifdef HAVE_JPEG
    make_tuple<string, Size>("../cv/imgproc/stuff.jpg", Size(640, 480)),
#endif
#ifdef HAVE_PNG
    make_tuple<string, Size>("../cv/shared/pic1.png", Size(400, 300)),
#endif
};

TEST_P(Imgcodecs_Resize, imread_reduce_flags)
{
    const string file_name = findDataFile(get<0>(get<0>(GetParam())));
    const Size imageSize = get<1>(get<0>(GetParam()));

    const int imread_flag = get<0>(get<1>(GetParam()));
    const int scale = get<1>(get<1>(GetParam()));

    const int cols = imageSize.width / scale;
    const int rows = imageSize.height / scale;
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
    const string file_name = findDataFile(get<0>(get<0>(GetParam())));
    const Size imageSize = get<1>(get<0>(GetParam()));

    const int imread_flag = get<0>(get<1>(GetParam()));
    const int scale = get<1>(get<1>(GetParam()));

    const int cols = imageSize.width / scale;
    const int rows = imageSize.height / scale;

    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(file_name.c_str(), mode);
    ASSERT_TRUE(ifs.is_open());

    ifs.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    std::vector<char> content(sz);
    ifs.read((char*)content.data(), sz);
    ASSERT_FALSE(ifs.fail());

    {
        Mat img = imdecode(Mat(content), imread_flag);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(cols, img.cols);
        EXPECT_EQ(rows, img.rows);
    }
}

//==================================================================================================

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Resize,
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
#if (defined(HAVE_JASPER) && defined(OPENCV_IMGCODECS_ENABLE_JASPER_TESTS)) \
    || defined(HAVE_OPENJPEG)
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

static
void test_image_io(const Mat& image, const std::string& fname, const std::string& ext, int imreadFlag, double psnrThreshold)
{
    vector<uchar> buf;
    ASSERT_NO_THROW(imencode("." + ext, image, buf));

    ASSERT_NO_THROW(imwrite(fname, image));

    FILE *f = fopen(fname.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    cout << "File size: " << len << " bytes" << endl;
    EXPECT_GT(len, 1024) << "File is small. Test or implementation is broken";
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    EXPECT_EQ(buf, file_buf) << "imwrite() / imencode() calls must provide the same output (bit-exact)";

    Mat buf_loaded = imdecode(Mat(buf), imreadFlag);
    EXPECT_FALSE(buf_loaded.empty());

    Mat loaded = imread(fname, imreadFlag);
    EXPECT_FALSE(loaded.empty());

    EXPECT_EQ(0, cv::norm(loaded, buf_loaded, NORM_INF)) << "imread() and imdecode() calls must provide the same result (bit-exact)";

    double psnr = cvtest::PSNR(loaded, image);
    EXPECT_GT(psnr, psnrThreshold);

    // not necessary due bitexact check above
    //double buf_psnr = cvtest::PSNR(buf_loaded, image);
    //EXPECT_GT(buf_psnr, psnrThreshold);

#if 0  // debug
    if (psnr <= psnrThreshold /*|| buf_psnr <= thresDbell*/)
    {
        cout << "File: " << fname << endl;
        imshow("origin", image);
        imshow("imread", loaded);
        imshow("imdecode", buf_loaded);
        waitKey();
    }
#endif
}

TEST_P(Imgcodecs_Image, read_write_BGR)
{
    const string ext = this->GetParam();
    const string fname = cv::tempfile(ext.c_str());

    double psnrThreshold = 100;
    if (ext == "jpg")
        psnrThreshold = 32;
#ifdef HAVE_JASPER
    if (ext == "jp2")
        psnrThreshold = 95;
#endif

    Mat image = generateTestImageBGR();
    EXPECT_NO_THROW(test_image_io(image, fname, ext, IMREAD_COLOR, psnrThreshold));

    EXPECT_EQ(0, remove(fname.c_str()));
}

TEST_P(Imgcodecs_Image, read_write_GRAYSCALE)
{
    const string ext = this->GetParam();

    if (false
        || ext == "ppm"  // grayscale is not implemented
        || ext == "ras"  // broken (black result)
    )
        throw SkipTestException("GRAYSCALE mode is not supported");

    const string fname = cv::tempfile(ext.c_str());

    double psnrThreshold = 100;
    if (ext == "jpg")
        psnrThreshold = 40;
#ifdef HAVE_JASPER
    if (ext == "jp2")
        psnrThreshold = 70;
#endif

    Mat image = generateTestImageGrayscale();
    EXPECT_NO_THROW(test_image_io(image, fname, ext, IMREAD_GRAYSCALE, psnrThreshold));

    EXPECT_EQ(0, remove(fname.c_str()));
}

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
