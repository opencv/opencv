/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";
const string DESCRIPTOR_DIR = FEATURES2D_DIR + "/descriptor_extractors";

/****************************************************************************************\
*                     Regression tests for descriptor extractors.                        *
\****************************************************************************************/
static void writeMatInBin( const Mat& mat, const string& filename )
{
    FILE* f = fopen( filename.c_str(), "wb");
    if( f )
    {
        CV_Assert(4 == sizeof(int));
        int type = mat.type();
        fwrite( (void*)&mat.rows, sizeof(int), 1, f );
        fwrite( (void*)&mat.cols, sizeof(int), 1, f );
        fwrite( (void*)&type, sizeof(int), 1, f );
        int dataSize = (int)(mat.step * mat.rows);
        fwrite( (void*)&dataSize, sizeof(int), 1, f );
        fwrite( (void*)mat.ptr(), 1, dataSize, f );
        fclose(f);
    }
}

static Mat readMatFromBin( const string& filename )
{
    FILE* f = fopen( filename.c_str(), "rb" );
    if( f )
    {
        CV_Assert(4 == sizeof(int));
        int rows, cols, type, dataSize;
        size_t elements_read1 = fread( (void*)&rows, sizeof(int), 1, f );
        size_t elements_read2 = fread( (void*)&cols, sizeof(int), 1, f );
        size_t elements_read3 = fread( (void*)&type, sizeof(int), 1, f );
        size_t elements_read4 = fread( (void*)&dataSize, sizeof(int), 1, f );
        CV_Assert(elements_read1 == 1 && elements_read2 == 1 && elements_read3 == 1 && elements_read4 == 1);

        int step = dataSize / rows / CV_ELEM_SIZE(type);
        CV_Assert(step >= cols);

        Mat returnMat = Mat(rows, step, type).colRange(0, cols);

        size_t elements_read = fread( returnMat.ptr(), 1, dataSize, f );
        CV_Assert(elements_read == (size_t)(dataSize));

        fclose(f);

        return returnMat;
    }
    return Mat();
}

template<class Distance>
class CV_DescriptorExtractorTest : public cvtest::BaseTest
{
public:
    typedef typename Distance::ValueType ValueType;
    typedef typename Distance::ResultType DistanceType;

    CV_DescriptorExtractorTest( const string _name, DistanceType _maxDist, const Ptr<DescriptorExtractor>& _dextractor,
                                Distance d = Distance(), Ptr<FeatureDetector> _detector = Ptr<FeatureDetector>()):
        name(_name), maxDist(_maxDist), dextractor(_dextractor), distance(d) , detector(_detector) {}

    ~CV_DescriptorExtractorTest()
    {
    }
protected:
    virtual void createDescriptorExtractor() {}

    void compareDescriptors( const Mat& validDescriptors, const Mat& calcDescriptors )
    {
        if( validDescriptors.size != calcDescriptors.size || validDescriptors.type() != calcDescriptors.type() )
        {
            ts->printf(cvtest::TS::LOG, "Valid and computed descriptors matrices must have the same size and type.\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        CV_Assert( DataType<ValueType>::type == validDescriptors.type() );

        int dimension = validDescriptors.cols;
        DistanceType curMaxDist = 0;
        size_t exact_count = 0, failed_count = 0;
        for( int y = 0; y < validDescriptors.rows; y++ )
        {
            DistanceType dist = distance( validDescriptors.ptr<ValueType>(y), calcDescriptors.ptr<ValueType>(y), dimension );
            if (dist == 0)
                exact_count++;
            if( dist > curMaxDist )
            {
                if (dist > maxDist)
                    failed_count++;
                curMaxDist = dist;
            }
#if 0
            if (dist > 0)
            {
                std::cout << "i=" << y << " fail_count=" << failed_count << " dist=" << dist << std::endl;
                std::cout << "valid: " << validDescriptors.row(y) << std::endl;
                std::cout << " calc: " << calcDescriptors.row(y) << std::endl;
            }
#endif
        }

        float exact_percents = (100 * (float)exact_count / validDescriptors.rows);
        float failed_percents = (100 * (float)failed_count / validDescriptors.rows);
        std::stringstream ss;
        ss << "Exact count (dist == 0): " << exact_count << " (" << (int)exact_percents << "%)" << std::endl
                << "Failed count (dist > " << maxDist << "): " << failed_count << " (" << (int)failed_percents << "%)" << std::endl
                << "Max distance between valid and computed descriptors (" << validDescriptors.size() << "): " << curMaxDist;
        EXPECT_LE(failed_percents, 20.0f);
        std::cout << ss.str() << std::endl;
    }

    void emptyDataTest()
    {
        assert( dextractor );

        // One image.
        Mat image;
        vector<KeyPoint> keypoints;
        Mat descriptors;

        try
        {
            dextractor->compute( image, keypoints, descriptors );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on empty image and empty keypoints must not generate exception (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }

        RNG rng;
        image = cvtest::randomMat(rng, Size(50, 50), CV_8UC3, 0, 255, false);
        try
        {
            dextractor->compute( image, keypoints, descriptors );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on nonempty image and empty keypoints must not generate exception (1).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }

        // Several images.
        vector<Mat> images;
        vector<vector<KeyPoint> > keypointsCollection;
        vector<Mat> descriptorsCollection;
        try
        {
            dextractor->compute( images, keypointsCollection, descriptorsCollection );
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "compute() on empty images and empty keypoints collection must not generate exception (2).\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        }
    }

    void regressionTest()
    {
        assert( dextractor );

        // Read the test image.
        string imgFilename =  string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;
        Mat img = imread( imgFilename );
        if( img.empty() )
        {
            ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }
        const std::string keypoints_filename = string(ts->get_data_path()) +
                (detector.empty()
                        ? (FEATURES2D_DIR + "/" + std::string("keypoints.xml.gz"))
                        : (DESCRIPTOR_DIR + "/" + name + "_keypoints.xml.gz"));
        FileStorage fs(keypoints_filename, FileStorage::READ);

        vector<KeyPoint> keypoints;
        EXPECT_TRUE(fs.isOpened()) << "Keypoint testdata is missing. Re-computing and re-writing keypoints testdata...";
        if (!fs.isOpened())
        {
            fs.open(keypoints_filename, FileStorage::WRITE);
            ASSERT_TRUE(fs.isOpened()) << "File for writing keypoints can not be opened.";
            if (detector.empty())
            {
                Ptr<ORB> fd = ORB::create();
                fd->detect(img, keypoints);
            }
            else
            {
                detector->detect(img, keypoints);
            }
            write(fs, "keypoints", keypoints);
            fs.release();
        }
        else
        {
            read(fs.getFirstTopLevelNode(), keypoints);
            fs.release();
        }

        if(!detector.empty())
        {
            vector<KeyPoint> calcKeypoints;
            detector->detect(img, calcKeypoints);
            // TODO validate received keypoints
            int diff = abs((int)calcKeypoints.size() - (int)keypoints.size());
            if (diff > 0)
            {
                std::cout << "Keypoints difference: " << diff << std::endl;
                EXPECT_LE(diff, (int)(keypoints.size() * 0.03f));
            }
        }
        ASSERT_FALSE(keypoints.empty());
        {
            Mat calcDescriptors;
            double t = (double)getTickCount();
            dextractor->compute(img, keypoints, calcDescriptors);
            t = getTickCount() - t;
            ts->printf(cvtest::TS::LOG, "\nAverage time of computing one descriptor = %g ms.\n", t/((double)getTickFrequency()*1000.)/calcDescriptors.rows);

            if (calcDescriptors.rows != (int)keypoints.size())
            {
                ts->printf( cvtest::TS::LOG, "Count of computed descriptors and keypoints count must be equal.\n" );
                ts->printf( cvtest::TS::LOG, "Count of keypoints is            %d.\n", (int)keypoints.size() );
                ts->printf( cvtest::TS::LOG, "Count of computed descriptors is %d.\n", calcDescriptors.rows );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            if (calcDescriptors.cols != dextractor->descriptorSize() || calcDescriptors.type() != dextractor->descriptorType())
            {
                ts->printf( cvtest::TS::LOG, "Incorrect descriptor size or descriptor type.\n" );
                ts->printf( cvtest::TS::LOG, "Expected size is   %d.\n", dextractor->descriptorSize() );
                ts->printf( cvtest::TS::LOG, "Calculated size is %d.\n", calcDescriptors.cols );
                ts->printf( cvtest::TS::LOG, "Expected type is   %d.\n", dextractor->descriptorType() );
                ts->printf( cvtest::TS::LOG, "Calculated type is %d.\n", calcDescriptors.type() );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }

            // TODO read and write descriptor extractor parameters and check them
            Mat validDescriptors = readDescriptors();
            EXPECT_FALSE(validDescriptors.empty()) << "Descriptors testdata is missing. Re-writing descriptors testdata...";
            if (!validDescriptors.empty())
            {
                compareDescriptors(validDescriptors, calcDescriptors);
            }
            else
            {
                ASSERT_TRUE(writeDescriptors(calcDescriptors)) << "Descriptors can not be written.";
            }
        }
    }

    void run(int)
    {
        createDescriptorExtractor();
        if( !dextractor )
        {
            ts->printf(cvtest::TS::LOG, "Descriptor extractor is empty.\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        emptyDataTest();
        regressionTest();

        ts->set_failed_test_info( cvtest::TS::OK );
    }

    virtual Mat readDescriptors()
    {
        Mat res = readMatFromBin( string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return res;
    }

    virtual bool writeDescriptors( Mat& descs )
    {
        writeMatInBin( descs,  string(ts->get_data_path()) + DESCRIPTOR_DIR + "/" + string(name) );
        return true;
    }

    string name;
    const DistanceType maxDist;
    Ptr<DescriptorExtractor> dextractor;
    Distance distance;
    Ptr<FeatureDetector> detector;

private:
    CV_DescriptorExtractorTest& operator=(const CV_DescriptorExtractorTest&) { return *this; }
};

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST( Features2d_DescriptorExtractor_SIFT, regression )
{
    CV_DescriptorExtractorTest<L1<float> > test( "descriptor-sift", 1.0f,
                                                SIFT::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_BRISK, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-brisk",
                                             (CV_DescriptorExtractorTest<Hamming>::DistanceType)2.f,
                                            BRISK::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_ORB, regression )
{
    // TODO adjust the parameters below
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-orb",
#if CV_NEON
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)25.f,
#else
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)12.f,
#endif
                                             ORB::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_KAZE, regression )
{
    CV_DescriptorExtractorTest< L2<float> > test( "descriptor-kaze",  0.03f,
                                                 KAZE::create(),
                                                 L2<float>(), KAZE::create() );
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_AKAZE, regression )
{
    CV_DescriptorExtractorTest<Hamming> test( "descriptor-akaze",
                                              (CV_DescriptorExtractorTest<Hamming>::DistanceType)(486*0.05f),
                                              AKAZE::create(),
                                              Hamming(), AKAZE::create());
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor_AKAZE_DESCRIPTOR_KAZE, regression )
{
    CV_DescriptorExtractorTest< L2<float> > test( "descriptor-akaze-with-kaze-desc", 0.03f,
                                              AKAZE::create(AKAZE::DESCRIPTOR_KAZE),
                                              L2<float>(), AKAZE::create(AKAZE::DESCRIPTOR_KAZE));
    test.safe_run();
}

TEST( Features2d_DescriptorExtractor, batch_ORB )
{
    string path = string(cvtest::TS::ptr()->get_data_path() + "detectors_descriptors_evaluation/images_datasets/graf");
    vector<Mat> imgs, descriptors;
    vector<vector<KeyPoint> > keypoints;
    int i, n = 6;
    Ptr<ORB> orb = ORB::create();

    for( i = 0; i < n; i++ )
    {
        string imgname = format("%s/img%d.png", path.c_str(), i+1);
        Mat img = imread(imgname, 0);
        imgs.push_back(img);
    }

    orb->detect(imgs, keypoints);
    orb->compute(imgs, keypoints, descriptors);

    ASSERT_EQ((int)keypoints.size(), n);
    ASSERT_EQ((int)descriptors.size(), n);

    for( i = 0; i < n; i++ )
    {
        EXPECT_GT((int)keypoints[i].size(), 100);
        EXPECT_GT(descriptors[i].rows, 100);
    }
}

TEST( Features2d_DescriptorExtractor, batch_SIFT )
{
    string path = string(cvtest::TS::ptr()->get_data_path() + "detectors_descriptors_evaluation/images_datasets/graf");
    vector<Mat> imgs, descriptors;
    vector<vector<KeyPoint> > keypoints;
    int i, n = 6;
    Ptr<SIFT> sift = SIFT::create();

    for( i = 0; i < n; i++ )
    {
        string imgname = format("%s/img%d.png", path.c_str(), i+1);
        Mat img = imread(imgname, 0);
        imgs.push_back(img);
    }

    sift->detect(imgs, keypoints);
    sift->compute(imgs, keypoints, descriptors);

    ASSERT_EQ((int)keypoints.size(), n);
    ASSERT_EQ((int)descriptors.size(), n);

    for( i = 0; i < n; i++ )
    {
        EXPECT_GT((int)keypoints[i].size(), 100);
        EXPECT_GT(descriptors[i].rows, 100);
    }
}


class DescriptorImage : public TestWithParam<std::string>
{
protected:
    virtual void SetUp() {
        pattern = GetParam();
    }

    std::string pattern;
};

TEST_P(DescriptorImage, no_crash)
{
    vector<String> fnames;
    glob(cvtest::TS::ptr()->get_data_path() + pattern, fnames, false);
    sort(fnames.begin(), fnames.end());

    Ptr<AKAZE> akaze_mldb = AKAZE::create(AKAZE::DESCRIPTOR_MLDB);
    Ptr<AKAZE> akaze_mldb_upright = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT);
    Ptr<AKAZE> akaze_mldb_256 = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 256);
    Ptr<AKAZE> akaze_mldb_upright_256 = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 256);
    Ptr<AKAZE> akaze_kaze = AKAZE::create(AKAZE::DESCRIPTOR_KAZE);
    Ptr<AKAZE> akaze_kaze_upright = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
    Ptr<ORB> orb = ORB::create();
    Ptr<KAZE> kaze = KAZE::create();
    Ptr<BRISK> brisk = BRISK::create();
    size_t n = fnames.size();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->setMaxFeatures(5000);

    for(size_t i = 0; i < n; i++ )
    {
        printf("%d. image: %s:\n", (int)i, fnames[i].c_str());
        if( strstr(fnames[i].c_str(), "MP.png") != 0 )
        {
            printf("\tskip\n");
            continue;
        }
        bool checkCount = strstr(fnames[i].c_str(), "templ.png") == 0;

        Mat img = imread(fnames[i], -1);

        printf("\t%dx%d\n", img.cols, img.rows);

#define TEST_DETECTOR(name, descriptor) \
        keypoints.clear(); descriptors.release(); \
        printf("\t" name "\n"); fflush(stdout); \
        descriptor->detectAndCompute(img, noArray(), keypoints, descriptors); \
        printf("\t\t\t(%d keypoints, descriptor size = %d)\n", (int)keypoints.size(), descriptors.cols); fflush(stdout); \
        if (checkCount) \
        { \
            EXPECT_GT((int)keypoints.size(), 0); \
        } \
        ASSERT_EQ(descriptors.rows, (int)keypoints.size());

        TEST_DETECTOR("AKAZE:MLDB", akaze_mldb);
        TEST_DETECTOR("AKAZE:MLDB_UPRIGHT", akaze_mldb_upright);
        TEST_DETECTOR("AKAZE:MLDB_256", akaze_mldb_256);
        TEST_DETECTOR("AKAZE:MLDB_UPRIGHT_256", akaze_mldb_upright_256);
        TEST_DETECTOR("AKAZE:KAZE", akaze_kaze);
        TEST_DETECTOR("AKAZE:KAZE_UPRIGHT", akaze_kaze_upright);
        TEST_DETECTOR("KAZE", kaze);
        TEST_DETECTOR("ORB", orb);
        TEST_DETECTOR("BRISK", brisk);
    }
}

INSTANTIATE_TEST_CASE_P(Features2d, DescriptorImage,
        testing::Values(
            "shared/lena.png",
            "shared/box*.png",
            "shared/fruits*.png",
            "shared/airplane.png",
            "shared/graffiti.png",
            "shared/1_itseez-0001*.png",
            "shared/pic*.png",
            "shared/templ.png"
        )
);

}} // namespace
