// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace opencv_test { namespace {

/****************************************************************************************\
*                     Regression tests for descriptor extractors.                        *
\****************************************************************************************/
static void double_image(Mat& src, Mat& dst) {

    dst.create(Size(src.cols*2, src.rows*2), src.type());

    Mat H = Mat::zeros(2, 3, CV_32F);
    H.at<float>(0, 0) = 0.5f;
    H.at<float>(1, 1) = 0.5f;
    cv::warpAffine(src, dst, H, dst.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REFLECT);

}

static Mat prepare_img(bool rows_indexed) {
    int rows = 5;
    int columns = 5;
    Mat img(rows, columns, CV_32F);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (rows_indexed) {
                img.at<float>(i, j) = (float)i;
            } else {
                img.at<float>(i, j) = (float)j;
            }
        }
    }
    return img;
}

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

        image = prepare_img(false);
        Mat dbl;
        try
        {
            double_image(image, dbl);

            Mat downsized_back(dbl.rows/2, dbl.cols/2, CV_32F);
            resize(dbl, downsized_back, Size(dbl.cols/2, dbl.rows/2), 0, 0, INTER_NEAREST);

            cv::Mat diff = (image != downsized_back);
            ASSERT_EQ(0, cv::norm(image, downsized_back, NORM_INF));
        }
        catch(...)
        {
            ts->printf( cvtest::TS::LOG, "double_image() must not generate exception (1).\n");
            ts->printf( cvtest::TS::LOG, "double_image() when downsized back by NEAREST must generate the same original image (1).\n");
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

}} // namespace
