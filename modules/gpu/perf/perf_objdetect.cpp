#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

///////////////////////////////////////////////////////////////
// HOG

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, ObjDetect_HOG, Values<string>("gpu/hog/road.png"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Rect> found_locations;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_img(img);

        cv::gpu::HOGDescriptor d_hog;
        d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        d_hog.detectMultiScale(d_img, found_locations);

        TEST_CYCLE()
        {
            d_hog.detectMultiScale(d_img, found_locations);
        }
    }
    else
    {
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        hog.detectMultiScale(img, found_locations);

        TEST_CYCLE()
        {
            hog.detectMultiScale(img, found_locations);
        }
    }

    SANITY_CHECK(found_locations);
}

//===========test for CalTech data =============//
DEF_PARAM_TEST_1(HOG, string);

PERF_TEST_P(HOG, CalTech, Values<string>("gpu/caltech/image_00000009_0.png", "gpu/caltech/image_00000032_0.png",
    "gpu/caltech/image_00000165_0.png", "gpu/caltech/image_00000261_0.png", "gpu/caltech/image_00000469_0.png",
    "gpu/caltech/image_00000527_0.png", "gpu/caltech/image_00000574_0.png"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Rect> found_locations;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_img(img);

        cv::gpu::HOGDescriptor d_hog;
        d_hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        d_hog.detectMultiScale(d_img, found_locations);

        TEST_CYCLE()
        {
            d_hog.detectMultiScale(d_img, found_locations);
        }
    }
    else
    {
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());

        hog.detectMultiScale(img, found_locations);

        TEST_CYCLE()
        {
            hog.detectMultiScale(img, found_locations);
        }
    }

    SANITY_CHECK(found_locations);
}

//================================================= ICF SoftCascade =================================================//

typedef pair<string, string> pair_string;
DEF_PARAM_TEST_1(SoftCascade, pair_string);


// struct SoftCascadeTest : public perf::TestBaseWithParam<roi_fixture_t>
// {
//     typedef cv::gpu::SoftCascade::Detection detection_t;
//     static cv::Rect getFromTable(int idx)
//     {
//         static const cv::Rect rois[] =
//         {
//             cv::Rect( 65,  20,  35, 80),
//             cv::Rect( 95,  35,  45, 40),
//             cv::Rect( 45,  35,  45, 40),
//             cv::Rect( 25,  27,  50, 45),
//             cv::Rect(100,  50,  45, 40),

//             cv::Rect( 60,  30,  45, 40),
//             cv::Rect( 40,  55,  50, 40),
//             cv::Rect( 48,  37,  72, 80),
//             cv::Rect( 48,  32,  85, 58),
//             cv::Rect( 48,   0,  32, 27)
//         };

//         return rois[idx];
//     }

//     static std::string itoa(long i)
//     {
//         static char s[65];
//         sprintf(s, "%ld", i);
//         return std::string(s);
//     }

//     static std::string getImageName(int level)
//     {
//         time_t rawtime;
//         struct tm * timeinfo;
//         char buffer [80];

//         time ( &rawtime );
//         timeinfo = localtime ( &rawtime );

//         strftime (buffer,80,"%Y-%m-%d--%H-%M-%S",timeinfo);
//         return "gpu_rec_level_" + itoa(level)+ "_" + std::string(buffer) + ".png";
//     }

//     static void print(std::ostream &out, const detection_t& d)
//     {
//         out << "\x1b[32m[ detection]\x1b[0m ("
//             << std::setw(4)  << d.x
//             << " "
//             << std::setw(4)  << d.y
//             << ") ("
//             << std::setw(4)  << d.w
//             << " "
//             << std::setw(4)  << d.h
//             << ") "
//             << std::setw(12) << d.confidence
//             <<  std::endl;
//     }

//     static void printTotal(std::ostream &out, int detbytes)
//     {
//         out << "\x1b[32m[          ]\x1b[0m Total detections " << (detbytes / sizeof(detection_t)) << std::endl;
//     }

//     static void writeResult(const cv::Mat& result, const int level)
//     {
//         std::string path = cv::tempfile(getImageName(level).c_str());
//         cv::imwrite(path, result);
//         std::cout << "\x1b[32m" << "[          ]" << std::endl << "[ stored in]"<< "\x1b[0m" << path << std::endl;
//     }
// };

typedef std::tr1::tuple<std::string, std::string> fixture_t;
typedef perf::TestBaseWithParam<fixture_t> SoftCascadeTest;

PERF_TEST_P(SoftCascadeTest, detect,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png"))))
{
    if (runOnGpu)
    {
        cv::Mat cpu = readImage (GetParam().second);
        ASSERT_FALSE(cpu.empty());
        cv::gpu::GpuMat colored(cpu);

        cv::gpu::SoftCascade cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GetParam().first)));

        cv::gpu::GpuMat objectBoxes(1, 16384, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1);

        rois.setTo(0);
        cv::gpu::GpuMat sub(rois, cv::Rect(rois.cols / 4, rois.rows / 4,rois.cols / 2, rois.rows / 2));
        sub.setTo(cv::Scalar::all(1));
        cascade.detectMultiScale(colored, rois, objectBoxes);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(colored, rois, objectBoxes);
        }
    } else
    {
        cv::Mat colored = readImage(GetParam().second);
        ASSERT_FALSE(colored.empty());

        cv::SoftCascade cascade;
        ASSERT_TRUE(cascade.load(getDataPath(GetParam().first)));

        std::vector<cv::Rect> rois, objectBoxes;
        cascade.detectMultiScale(colored, rois, objectBoxes);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(colored, rois, objectBoxes);
        }
    }
}

static cv::Rect getFromTable(int idx)
{
    static const cv::Rect rois[] =
    {
        cv::Rect( 65,  20,  35, 80),
        cv::Rect( 95,  35,  45, 40),
        cv::Rect( 45,  35,  45, 40),
        cv::Rect( 25,  27,  50, 45),
        cv::Rect(100,  50,  45, 40),

        cv::Rect( 60,  30,  45, 40),
        cv::Rect( 40,  55,  50, 40),
        cv::Rect( 48,  37,  72, 80),
        cv::Rect( 48,  32,  85, 58),
        cv::Rect( 48,   0,  32, 27)
    };

    return rois[idx];
}

typedef std::tr1::tuple<std::string, std::string, int> roi_fixture_t;
typedef perf::TestBaseWithParam<roi_fixture_t> SoftCascadeTestRoi;

PERF_TEST_P(SoftCascadeTestRoi, detectInRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 5)))
{
    if (runOnGpu)
    {
        cv::Mat cpu = readImage (GET_PARAM(1));
        ASSERT_FALSE(cpu.empty());
        cv::gpu::GpuMat colored(cpu);

        cv::gpu::SoftCascade cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GET_PARAM(0))));

        cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1);
        rois.setTo(0);

        int nroi = GET_PARAM(2);
        cv::RNG rng;
        for (int i = 0; i < nroi; ++i)
        {
            cv::Rect r = getFromTable(rng(10));
            cv::gpu::GpuMat sub(rois, r);
            sub.setTo(1);
        }

        cv::gpu::GpuMat curr = objectBoxes;
        cascade.detectMultiScale(colored, rois, curr);

        TEST_CYCLE()
        {
            curr = objectBoxes;
            cascade.detectMultiScale(colored, rois, curr);
        }
    }
    else
    {
        FAIL();
    }
}

PERF_TEST_P(SoftCascadeTestRoi, detectEachRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 10)))
{
    if (runOnGpu)
    {
        cv::Mat cpu = readImage (GET_PARAM(1));
        ASSERT_FALSE(cpu.empty());
        cv::gpu::GpuMat colored(cpu);

        cv::gpu::SoftCascade cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GET_PARAM(0))));

        cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1);
        rois.setTo(0);

        int idx = GET_PARAM(2);
        cv::Rect r = getFromTable(idx);
        cv::gpu::GpuMat sub(rois, r);
        sub.setTo(1);

        cv::gpu::GpuMat curr = objectBoxes;
        cascade.detectMultiScale(colored, rois, curr);

        TEST_CYCLE()
        {
            curr = objectBoxes;
            cascade.detectMultiScale(colored, rois, curr);
        }
    }
    else
    {
        FAIL();
    }
}


///////////////////////////////////////////////////////////////
// HaarClassifier

typedef pair<string, string> pair_string;
DEF_PARAM_TEST_1(ImageAndCascade, pair_string);

PERF_TEST_P(ImageAndCascade, ObjDetect_HaarClassifier,
    Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/perf/haarcascade_frontalface_alt.xml")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_objects_buffer;

        d_cascade.detectMultiScale(d_img, d_objects_buffer);

        TEST_CYCLE()
        {
            d_cascade.detectMultiScale(d_img, d_objects_buffer);
        }

        GPU_SANITY_CHECK(d_objects_buffer);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/perf/haarcascade_frontalface_alt.xml")));

        std::vector<cv::Rect> rects;

        cascade.detectMultiScale(img, rects);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(img, rects);
        }

        CPU_SANITY_CHECK(rects);
    }
}

///////////////////////////////////////////////////////////////
// LBP cascade

PERF_TEST_P(ImageAndCascade, ObjDetect_LBPClassifier,
    Values<pair_string>(make_pair("gpu/haarcascade/group_1_640x480_VGA.pgm", "gpu/lbpcascade/lbpcascade_frontalface.xml")))
{
    cv::Mat img = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::CascadeClassifier_GPU d_cascade;
        ASSERT_TRUE(d_cascade.load(perf::TestBase::getDataPath(GetParam().second)));

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_gpu_rects;

        d_cascade.detectMultiScale(d_img, d_gpu_rects);

        TEST_CYCLE()
        {
            d_cascade.detectMultiScale(d_img, d_gpu_rects);
        }

        GPU_SANITY_CHECK(d_gpu_rects);
    }
    else
    {
        cv::CascadeClassifier cascade;
        ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath("gpu/lbpcascade/lbpcascade_frontalface.xml")));

        std::vector<cv::Rect> rects;

        cascade.detectMultiScale(img, rects);

        TEST_CYCLE()
        {
            cascade.detectMultiScale(img, rects);
        }

        CPU_SANITY_CHECK(rects);
    }
}

} // namespace
