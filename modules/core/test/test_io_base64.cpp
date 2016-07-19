#include "test_precomp.hpp"

using namespace cv;
using namespace std;

struct data_t
{
    typedef uchar  u;
    typedef char   b;
    typedef ushort w;
    typedef short  s;
    typedef int    i;
    typedef float  f;
    typedef double d;

    u u1   ;u u2   ;                i i1                           ;
    i i2                           ;i i3                           ;
    d d1                                                           ;
    d d2                                                           ;
    i i4                           ;

    static inline const char * signature() { return "2u3i2di"; }
};


TEST(Core_InputOutput_Base64, basic)
{
    char const * filenames[] = {
        "core_io_base64_basic_test.yml",
        "core_io_base64_basic_test.xml",
        0
    };

    for (char const ** ptr = filenames; *ptr; ptr++)
    {
        char const * name = *ptr;

        std::vector<data_t> rawdata;

        cv::Mat _em_out, _em_in;
        cv::Mat _2d_out, _2d_in;
        cv::Mat _nd_out, _nd_in;
        cv::Mat _rd_out(64, 64, CV_64FC1), _rd_in;

        {   /* init */

            /* a normal mat */
            _2d_out = cv::Mat(100, 100, CV_8UC3, cvScalar(1U, 2U, 127U));
            for (int i = 0; i < _2d_out.rows; ++i)
                for (int j = 0; j < _2d_out.cols; ++j)
                    _2d_out.at<cv::Vec3b>(i, j)[1] = (i + j) % 256;

            /* a 4d mat */
            const int Size[] = {4, 4, 4, 4};
            cv::Mat _4d(4, Size, CV_64FC4, cvScalar(0.888, 0.111, 0.666, 0.444));
            const cv::Range ranges[] = {
                cv::Range(0, 2),
                cv::Range(0, 2),
                cv::Range(1, 2),
                cv::Range(0, 2) };
            _nd_out = _4d(ranges);

            /* a random mat */
            cv::randu(_rd_out, cv::Scalar(0.0), cv::Scalar(1.0));

            /* raw data */
            for (int i = 0; i < 1000; i++) {
                data_t tmp;
                tmp.u1 = 1;
                tmp.u2 = 2;
                tmp.i1 = 1;
                tmp.i2 = 2;
                tmp.i3 = 3;
                tmp.d1 = 0.1;
                tmp.d2 = 0.2;
                tmp.i4 = i;
                rawdata.push_back(tmp);
            }
        }

        {   /* write */
            cv::FileStorage fs(name, cv::FileStorage::WRITE_BASE64);
            fs << "normal_2d_mat" << _2d_out;
            fs << "normal_nd_mat" << _nd_out;
            fs << "empty_2d_mat"  << _em_out;
            fs << "random_mat"    << _rd_out;

            cvStartWriteStruct( *fs, "rawdata", CV_NODE_SEQ | CV_NODE_FLOW, "binary" );
            for (int i = 0; i < 10; i++)
                cvWriteRawDataBase64(*fs, rawdata.data() + i * 100, 100, data_t::signature());
            cvEndWriteStruct( *fs );

            fs.release();
        }

        {   /* read */
            cv::FileStorage fs(name, cv::FileStorage::READ);

            /* mat */
            fs["empty_2d_mat"]  >> _em_in;
            fs["normal_2d_mat"] >> _2d_in;
            fs["normal_nd_mat"] >> _nd_in;
            fs["random_mat"]    >> _rd_in;

            /* raw data */
            std::vector<data_t>(1000).swap(rawdata);
            cvReadRawData(*fs, fs["rawdata"].node, rawdata.data(), data_t::signature());

            fs.release();
        }

        for (int i = 0; i < 1000; i++) {
            // TODO: Solve this bug in `cvReadRawData`
            //EXPECT_EQ(rawdata[i].u1, 1);
            //EXPECT_EQ(rawdata[i].u2, 2);
            //EXPECT_EQ(rawdata[i].i1, 1);
            //EXPECT_EQ(rawdata[i].i2, 2);
            //EXPECT_EQ(rawdata[i].i3, 3);
            //EXPECT_EQ(rawdata[i].d1, 0.1);
            //EXPECT_EQ(rawdata[i].d2, 0.2);
            //EXPECT_EQ(rawdata[i].i4, i);
        }

        EXPECT_EQ(_em_in.rows   , _em_out.rows);
        EXPECT_EQ(_em_in.cols   , _em_out.cols);
        EXPECT_EQ(_em_in.dims   , _em_out.dims);
        EXPECT_EQ(_em_in.depth(), _em_out.depth());
        EXPECT_TRUE(_em_in.empty());

        EXPECT_EQ(_2d_in.rows   , _2d_out.rows);
        EXPECT_EQ(_2d_in.cols   , _2d_out.cols);
        EXPECT_EQ(_2d_in.dims   , _2d_out.dims);
        EXPECT_EQ(_2d_in.depth(), _2d_out.depth());
        for(int i = 0; i < _2d_out.rows; ++i)
            for (int j = 0; j < _2d_out.cols; ++j)
                EXPECT_EQ(_2d_in.at<cv::Vec3b>(i, j), _2d_out.at<cv::Vec3b>(i, j));

        EXPECT_EQ(_nd_in.rows   , _nd_out.rows);
        EXPECT_EQ(_nd_in.cols   , _nd_out.cols);
        EXPECT_EQ(_nd_in.dims   , _nd_out.dims);
        EXPECT_EQ(_nd_in.depth(), _nd_out.depth());
        EXPECT_EQ(cv::countNonZero(cv::mean(_nd_in != _nd_out)), 0);

        EXPECT_EQ(_rd_in.rows   , _rd_out.rows);
        EXPECT_EQ(_rd_in.cols   , _rd_out.cols);
        EXPECT_EQ(_rd_in.dims   , _rd_out.dims);
        EXPECT_EQ(_rd_in.depth(), _rd_out.depth());
        EXPECT_EQ(cv::countNonZero(cv::mean(_rd_in != _rd_out)), 0);

        remove(name);
    }
}

TEST(Core_InputOutput_Base64, valid)
{
    char const * filenames[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.yml?base64",
        "core_io_base64_other_test.xml?base64",
        0
    };
    char const * real_name[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        0
    };

    std::vector<int> rawdata(10, static_cast<int>(0x00010203));
    cv::String str_out = "test_string";

    for (char const ** ptr = filenames; *ptr; ptr++)
    {
        char const * name = *ptr;

        EXPECT_NO_THROW(
        {
            cv::FileStorage fs(name, cv::FileStorage::WRITE_BASE64);

            cvStartWriteStruct(*fs, "manydata", CV_NODE_SEQ);
            cvStartWriteStruct(*fs, 0, CV_NODE_SEQ | CV_NODE_FLOW);
            for (int i = 0; i < 10; i++)
                cvWriteRawData(*fs, rawdata.data(), rawdata.size(), "i");
            cvEndWriteStruct(*fs);
            cvWriteString(*fs, 0, str_out.c_str(), 1);
            cvEndWriteStruct(*fs);

            fs.release();
        });

        {
            cv::FileStorage fs(name, cv::FileStorage::READ);
            std::vector<int> data_in(rawdata.size());
            fs["manydata"][0].readRaw("i", (uchar *)data_in.data(), data_in.size());
            EXPECT_TRUE(fs["manydata"][0].isSeq());
            EXPECT_TRUE(std::equal(rawdata.begin(), rawdata.end(), data_in.begin()));
            cv::String str_in;
            fs["manydata"][1] >> str_in;
            EXPECT_TRUE(fs["manydata"][1].isString());
            EXPECT_EQ(str_in, str_out);
            fs.release();
        }

        EXPECT_NO_THROW(
        {
            cv::FileStorage fs(name, cv::FileStorage::WRITE);

            cvStartWriteStruct(*fs, "manydata", CV_NODE_SEQ);
            cvWriteString(*fs, 0, str_out.c_str(), 1);
            cvStartWriteStruct(*fs, 0, CV_NODE_SEQ | CV_NODE_FLOW, "binary");
            for (int i = 0; i < 10; i++)
                cvWriteRawData(*fs, rawdata.data(), rawdata.size(), "i");
            cvEndWriteStruct(*fs);
            cvEndWriteStruct(*fs);

            fs.release();
        });

        {
            cv::FileStorage fs(name, cv::FileStorage::READ);
            cv::String str_in;
            fs["manydata"][0] >> str_in;
            EXPECT_TRUE(fs["manydata"][0].isString());
            EXPECT_EQ(str_in, str_out);
            std::vector<int> data_in(rawdata.size());
            fs["manydata"][1].readRaw("i", (uchar *)data_in.data(), data_in.size());
            EXPECT_TRUE(fs["manydata"][1].isSeq());
            EXPECT_TRUE(std::equal(rawdata.begin(), rawdata.end(), data_in.begin()));
            fs.release();
        }

        remove(real_name[ptr - filenames]);
    }
}

TEST(Core_InputOutput_Base64, invalid)
{
    char const * filenames[] = {
        "core_io_base64_other_test.yml",
        "core_io_base64_other_test.xml",
        0
    };

    for (char const ** ptr = filenames; *ptr; ptr++)
    {
        char const * name = *ptr;

        EXPECT_ANY_THROW({
            cv::FileStorage fs(name, cv::FileStorage::WRITE);
            cvStartWriteStruct(*fs, "rawdata", CV_NODE_SEQ, "binary");
            cvStartWriteStruct(*fs, 0, CV_NODE_SEQ | CV_NODE_FLOW);
        });

        EXPECT_ANY_THROW({
            cv::FileStorage fs(name, cv::FileStorage::WRITE);
            cvStartWriteStruct(*fs, "rawdata", CV_NODE_SEQ);
            cvStartWriteStruct(*fs, 0, CV_NODE_SEQ | CV_NODE_FLOW);
            cvWriteRawDataBase64(*fs, name, 1, "u");
        });

        remove(name);
    }
}

TEST(Core_InputOutput_Base64, TODO_compatibility)
{
    // TODO:
}
