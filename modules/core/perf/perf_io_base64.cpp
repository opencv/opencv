#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<cv::Size, MatType, String> Size_MatType_Str_t;
typedef TestBaseWithParam<Size_MatType_Str_t> Size_Mat_StrType;

#define MAT_SIZES      ::perf::sz1080p/*, ::perf::sz4320p*/
#define MAT_TYPES      CV_8UC1, CV_32FC1
#define FILE_EXTENSION String(".xml"), String(".yml"), String(".json")


PERF_TEST_P(Size_Mat_StrType, fs_text,
            testing::Combine(testing::Values(MAT_SIZES),
                             testing::Values(MAT_TYPES),
                             testing::Values(FILE_EXTENSION))
             )
{
    Size   size = get<0>(GetParam());
    int    type = get<1>(GetParam());
    String ext  = get<2>(GetParam());

    Mat src(size.height, size.width, type);
    Mat dst = src.clone();

    declare.in(src, WARMUP_RNG).out(dst);

    cv::String file_name = cv::tempfile(ext.c_str());
    cv::String key       = "test_mat";

    TEST_CYCLE_MULTIRUN(2)
    {
        {
            FileStorage fs(file_name, cv::FileStorage::WRITE);
            fs << key << src;
            fs.release();
        }
        {
            FileStorage fs(file_name, cv::FileStorage::READ);
            fs[key] >> dst;
            fs.release();
        }
    }

    remove(file_name.c_str());
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_Mat_StrType, fs_base64,
            testing::Combine(testing::Values(MAT_SIZES),
                             testing::Values(MAT_TYPES),
                             testing::Values(FILE_EXTENSION))
             )
{
    Size   size = get<0>(GetParam());
    int    type = get<1>(GetParam());
    String ext  = get<2>(GetParam());

    Mat src(size.height, size.width, type);
    Mat dst = src.clone();

    cv::String file_name = cv::tempfile(ext.c_str());
    cv::String key       = "test_mat";

    declare.in(src, WARMUP_RNG).out(dst);
    TEST_CYCLE_MULTIRUN(2)
    {
        {
            FileStorage fs(file_name, cv::FileStorage::WRITE_BASE64);
            fs << key << src;
            fs.release();
        }
        {
            FileStorage fs(file_name, cv::FileStorage::READ);
            fs[key] >> dst;
            fs.release();
        }
    }

    remove(file_name.c_str());
    SANITY_CHECK_NOTHING();
}
