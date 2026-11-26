// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/geometry.hpp>
#include <opencv2/core/quaternion.hpp>

using namespace cv;

namespace opencv_test { namespace {

const int W = 640;
const int H = 480;
//int window_size = 5;
float focal_length = 525;
float cx = W / 2.f + 0.5f;
float cy = H / 2.f + 0.5f;

static Mat K() { static Mat res = Mat_<double>({3, 3}, {focal_length, 0, cx, 0, focal_length, cy, 0, 0, 1}); return res; }
static Mat Kinv() { static Mat res = K().inv(); return res; }

void points3dToDepth16U(const Mat_<Vec4f>& points3d, Mat& depthMap);

void points3dToDepth16U(const Mat_<Vec4f>& points3d, Mat& depthMap)
{
    std::vector<Point3f> points3dvec;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            points3dvec.push_back(Point3f(points3d(i, j)[0], points3d(i, j)[1], points3d(i, j)[2]));

    std::vector<Point2f> img_points;
    depthMap = Mat::zeros(H, W, CV_32F);
    Vec3f R(0.0, 0.0, 0.0);
    Vec3f T(0.0, 0.0, 0.0);
    cv::projectPoints(points3dvec, R, T, K(), Mat(), img_points);

    float maxv = 0.f;
    int index = 0;
    for (int i = 0; i < H; i++)
    {

        for (int j = 0; j < W; j++)
        {
            float value = (points3d(i, j))[2]; // value is the z
            depthMap.at<float>(cvRound(img_points[index].y), cvRound(img_points[index].x)) = value;
            maxv = std::max(maxv, value);
            index++;
        }
    }

    double scale = ((1 << 16) - 1) / maxv;
    depthMap.convertTo(depthMap, CV_16U, scale);
}


struct Plane
{
public:
    Vec4d nd;

    Plane() : nd(1, 0, 0, 0) { }

    static Plane generate(RNG& rng)
    {
        // Gaussian 3D distribution is separable and spherically symmetrical
        // Being normalized, its points represent uniformly distributed points on a sphere (i.e. normal directions)
        double sigma = 1.0;
        Vec3d ngauss;
        ngauss[0] = rng.gaussian(sigma);
        ngauss[1] = rng.gaussian(sigma);
        ngauss[2] = rng.gaussian(sigma);
        ngauss = ngauss * (1.0 / cv::norm(ngauss));

        double d = rng.uniform(-2.0, 2.0);
        Plane p;
        p.nd = Vec4d(ngauss[0], ngauss[1], ngauss[2], d);
        return p;
    }

    Vec3d pixelIntersection(double u, double v, const Matx33d& K_inv)
    {
        Vec3d uv1(u, v, 1);
        // pixel reprojected to camera space
        Matx31d pspace = K_inv * uv1;

        double d = this->nd[3];
        double dotp = pspace.ddot({this->nd[0], this->nd[1], this->nd[2]});
        double d_over_dotp = d / dotp;
        if (std::fabs(dotp) <= 1e-9)
        {
            d_over_dotp = 1.0;
            CV_LOG_INFO(NULL, "warning, dotp nearly 0! " << dotp);
        }

        Matx31d pmeet = pspace * (- d_over_dotp);
        return {pmeet(0, 0), pmeet(1, 0), pmeet(2, 0)};
    }
};

void gen_points_3d(std::vector<Plane>& planes_out, Mat_<unsigned char> &plane_mask, Mat& points3d, Mat& normals,
                   int n_planes, float scale, RNG& rng)
{
    const double minGoodZ = 0.0001;
    const double maxGoodZ = 1000.0;

    std::vector<Plane> planes;
    for (int i = 0; i < n_planes; i++)
    {
        bool found = false;
        for (int j = 0; j < 100; j++)
        {
            Plane px = Plane::generate(rng);

            // Check that area corners have good z values
            // So that they won't break rendering
            double x0 = double(i) * double(W) / double(n_planes);
            double x1 = double(i+1) * double(W) / double(n_planes);
            std::vector<Point2d> corners = {{x0, 0}, {x0, H - 1}, {x1, 0}, {x1, H - 1}};
            double minz = std::numeric_limits<double>::max();
            double maxz = 0.0;
            for (auto p : corners)
            {
                Vec3d v = px.pixelIntersection(p.x, p.y, Kinv());
                minz = std::min(minz, v[2]);
                maxz = std::max(maxz, v[2]);
            }
            if (minz > minGoodZ && maxz < maxGoodZ)
            {
                planes.push_back(px);
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found)  << "Failed to generate proper random plane" << std::endl;
    }
    Mat_ < Vec4f > outp(H, W);
    Mat_ < Vec4f > outn(H, W);
    plane_mask.create(H, W);

    // n  ( r - r_0) = 0
    // n * r_0 = d
    //
    // r_0 = (0,0,0)
    // r[0]
    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            unsigned int plane_index = (unsigned int)((u / float(W)) * planes.size());
            Plane plane = planes[plane_index];
            Vec3f pt = Vec3f(plane.pixelIntersection((double)u, (double)v, Kinv()) * scale);
            outp(v, u) = {pt[0], pt[1], pt[2], 0};
            outn(v, u) = {(float)plane.nd[0], (float)plane.nd[1], (float)plane.nd[2], 0};
            plane_mask(v, u) = (uchar)plane_index;
        }
    }
    planes_out = planes;
    points3d = outp;
    normals = outn;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_ENUM(NormalComputers, RgbdNormals::RGBD_NORMALS_METHOD_FALS,
                         RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,
                         RgbdNormals::RGBD_NORMALS_METHOD_SRI,
                         RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT);
typedef std::tuple<MatDepth, NormalComputers, bool, double, double, double, double, double> NormalsTestData;
typedef std::tuple<NormalsTestData, int> NormalsTestParams;

const double threshold3d1d = 1e-12;
// Right angle is the maximum angle possible between two normals
const double hpi = CV_PI / 2.0;
const int nTestCasesNormals = 5;

class NormalsRandomPlanes : public ::testing::TestWithParam<NormalsTestParams>
{
protected:
    void SetUp() override
    {
        p = GetParam();
        depth  = std::get<0>(std::get<0>(p));
        alg = static_cast<RgbdNormals::RgbdNormalsMethod>(int(std::get<1>(std::get<0>(p))));
        scale = std::get<2>(std::get<0>(p));
        idx = std::get<1>(p);
    }

    struct NormalsCompareResult
    {
        double meanErr;
        double maxErr;
    };

    static NormalsCompareResult checkNormals(Mat_<Vec4f> normals, Mat_<Vec4f> ground_normals)
    {
        double meanErr = 0, maxErr = 0;
        for (int y = 0; y < normals.rows; ++y)
        {
            for (int x = 0; x < normals.cols; ++x)
            {
                Vec4f vec1 = normals(y, x), vec2 = ground_normals(y, x);
                vec1 = vec1 / cv::norm(vec1);
                vec2 = vec2 / cv::norm(vec2);

                double dot = vec1.ddot(vec2);
                // Just for rounding errors
                double err = std::abs(dot) < 1.0 ? std::min(std::acos(dot), std::acos(-dot)) : 0.0;
                meanErr += err;
                maxErr = std::max(maxErr, err);
            }
        }
        meanErr /= normals.rows * normals.cols;
        return { meanErr, maxErr };
    }

    void runCase(bool scaleUp, int nPlanes, bool makeDepth,
                 double meanThreshold, double maxThreshold, double threshold3d)
    {
        // created here rather than in SetUp() so that skipped instances do not pay for cache()
        float diffThreshold = scale ? 100000.f : 50.f;
        normalsComputer = RgbdNormals::create(H, W, depth, K(), 5, diffThreshold, alg);
        normalsComputer->cache();

        RNG& rng = cv::theRNG();
        rng.state += idx + nTestCasesNormals*int(scale) + alg*16 + depth*64;

        std::vector<Plane> plane_params;
        Mat_<unsigned char> plane_mask;
        Mat points3d, ground_normals;

        gen_points_3d(plane_params, plane_mask, points3d, ground_normals, nPlanes, scaleUp ? 5000.f : 1.f, rng);

        Mat in;
        if (makeDepth)
        {
            points3dToDepth16U(points3d, in);
        }
        else
        {
            in = points3d;
        }

        TickMeter tm;
        tm.start();
        Mat in_normals, normals3d;
        //TODO: check other methods when 16U input is implemented for them
        if (normalsComputer->getMethod() == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD && in.channels() == 3)
        {
            std::vector<Mat> channels;
            split(in, channels);
            normalsComputer->apply(channels[2], in_normals);

            normalsComputer->apply(in, normals3d);
        }
        else
            normalsComputer->apply(in, in_normals);
        tm.stop();

        CV_LOG_INFO(NULL, "Speed: " << tm.getTimeMilli() << " ms");

        Mat_<Vec4f> normals;
        in_normals.convertTo(normals, CV_32FC4);

        NormalsCompareResult res = checkNormals(normals, ground_normals);
        double err3d = 0.0;
        if (!normals3d.empty())
        {
            Mat_<Vec4f> cvtNormals3d;
            normals3d.convertTo(cvtNormals3d, CV_32FC4);
            err3d = checkNormals(cvtNormals3d, ground_normals).maxErr;
        }

        EXPECT_LE(res.meanErr, meanThreshold);
        EXPECT_LE(res.maxErr, maxThreshold);
        EXPECT_LE(err3d, threshold3d);
    }

    NormalsTestParams p;
    int depth;
    RgbdNormals::RgbdNormalsMethod alg;
    bool scale;
    int idx;

    Ptr<RgbdNormals> normalsComputer;
};

//TODO Test NaNs in data

TEST_P(NormalsRandomPlanes, check1plane)
{
    double meanErr = std::get<3>(std::get<0>(p));
    double maxErr  = std::get<4>(std::get<0>(p));

    // 1 plane, continuous scene, very low error..
    runCase(scale, 1, false, meanErr, maxErr, threshold3d1d);
}

TEST_P(NormalsRandomPlanes, check3planes)
{
    double meanErr = std::get<5>(std::get<0>(p));
    double maxErr  = hpi;

    // 3 discontinuities, more error expected
    runCase(scale, 3, false, meanErr, maxErr, threshold3d1d);
}

TEST_P(NormalsRandomPlanes, check1plane16u)
{
    // TODO: check other algos as soon as they support 16U depth inputs
    if (alg == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD && scale)
    {
        double meanErr = std::get<6>(std::get<0>(p));
        double maxErr  = hpi;

        runCase(false, 1, true, meanErr, maxErr, threshold3d1d);
    }
    else
    {
        throw SkipTestException("Not implemented for anything except LINEMOD with scale");
    }
}

TEST_P(NormalsRandomPlanes, check3planes16u)
{
    // TODO: check other algos as soon as they support 16U depth inputs
    if (alg == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD && scale)
    {
        double meanErr = std::get<7>(std::get<0>(p));
        double maxErr  = hpi;

        runCase(false, 3, true, meanErr, maxErr, threshold3d1d);
    }
    else
    {
        throw SkipTestException("Not implemented for anything except LINEMOD with scale");
    }
}

INSTANTIATE_TEST_CASE_P(RGBD_Normals, NormalsRandomPlanes,
::testing::Combine(::testing::Values(
    // 3 normal computer params + 5 thresholds:
    //depth, alg, scale, 1plane mean, 1plane max, 3planes mean, 1plane16u mean, 3planes16 mean
    // maxErr is the maximum over all the pixels, it is very sensitive to the floating-point details of a platform
    // (FMA contraction, summation order in filters): 0.071 on x86-64, 0.094 on RISC-V (RVV)
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_FALS,  true, 0.00362, 0.1, 0.02175, 0, 0},
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_FALS, false, 0.00374, 0.10309, 0.02, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_FALS,  true, 0.00023, 0.00037, 0.01805, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_FALS, false, 0.00023, 0.00037, 0.01805, 0, 0},

    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,  true, 0.00186, 0.08974, 0.04528, 0.21220, 0.17314},
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD, false, 0.00157, 0.01225, 0.04528, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,  true, 0.00160, 0.06526, 0.04371, 0.28837, 0.28918},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD, false, 0.00154, 0.06877, 0.04323, 0, 0},

    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_SRI,  true, 0.01987, hpi, 0.036, 0, 0},
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_SRI, false, 0.01962, hpi, 0.037, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_SRI,  true, 0.01958, hpi, 0.037, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_SRI, false, 0.01995, hpi, 0.036, 0, 0},

    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT,  true, 0.000230, 0.00038, 0.00450, 0, 0},
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT, false, 0.000230, 0.00038, 0.00478, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT,  true, 0.000221, 0.00038, 0.00469, 0, 0},
    NormalsTestData {CV_64F, RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT, false, 0.000238, 0.00038, 0.00477, 0, 0}
), ::testing::Range(0, nTestCasesNormals)));

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::tuple<NormalComputers, std::pair<double, double>> NormalComputerThresholds;
struct RenderedNormals: public ::testing::TestWithParam<std::tuple<MatDepth, NormalComputerThresholds, bool>>
{
    static Mat readYaml(std::string fname)
    {
        Mat img;
        FileStorage fs(fname, FileStorage::Mode::READ);
        if (fs.isOpened() && fs.getFirstTopLevelNode().name() == "testImg")
        {
            fs["testImg"] >> img;
        }
        return img;
    }

    static Mat nanMask(Mat img)
    {
        int depth = img.depth();
        Mat mask(img.size(), CV_8U);
        for (int y = 0; y < img.rows; y++)
        {
            uchar* maskRow = mask.ptr<uchar>(y);
            if (depth == CV_32F)
            {
                Vec3f *imgrow = img.ptr<Vec3f>(y);
                for (int x = 0; x < img.cols; x++)
                {
                    maskRow[x] = (imgrow[x] == imgrow[x])*255;
                }
            }
            else if (depth == CV_64F)
            {
                Vec3d *imgrow = img.ptr<Vec3d>(y);
                for (int x = 0; x < img.cols; x++)
                {
                    maskRow[x] = (imgrow[x] == imgrow[x])*255;
                }
            }
        }
        return mask;
    }

    template<typename VT>
    static Mat flipAxesT(Mat pts, int flip)
    {
        Mat flipped(pts.size(), pts.type());
        for (int y = 0; y < pts.rows; y++)
        {
            VT *inrow = pts.ptr<VT>(y);
            VT *outrow = flipped.ptr<VT>(y);
            for (int x = 0; x < pts.cols; x++)
            {
                VT n = inrow[x];
                n[0] = (flip & FLIP_X) ? -n[0] : n[0];
                n[1] = (flip & FLIP_Y) ? -n[1] : n[1];
                n[2] = (flip & FLIP_Z) ? -n[2] : n[2];
                outrow[x] = n;
            }
        }
        return flipped;
    }

    static const int FLIP_X = 1;
    static const int FLIP_Y = 2;
    static const int FLIP_Z = 4;
    static Mat flipAxes(Mat pts, int flip)
    {
        int depth = pts.depth();
        if (depth == CV_32F)
        {
            return flipAxesT<Vec3f>(pts, flip);
        }
        else if (depth == CV_64F)
        {
            return flipAxesT<Vec3d>(pts, flip);
        }
        else
        {
            return Mat();
        }
    }

    template<typename VT>
    static Mat_<typename VT::value_type> normalsErrorT(Mat_<VT> srcNormals, Mat_<VT> dstNormals)
    {
        typedef typename VT::value_type Val;
        Mat out(srcNormals.size(), cv::traits::Depth<Val>::value, Scalar(0));
        for (int y = 0; y < srcNormals.rows; y++)
        {

            VT *srcrow = srcNormals[y];
            VT *dstrow = dstNormals[y];
            Val *outrow = out.ptr<Val>(y);
            for (int x = 0; x < srcNormals.cols; x++)
            {
                VT sn = srcrow[x];
                VT dn = dstrow[x];

                Val dot = sn.dot(dn);
                Val v(0.0);
                // Just for rounding errors
                if (std::abs(dot) < 1)
                    v = std::min(std::acos(dot), std::acos(-dot));

                outrow[x] = v;
            }
        }
        return out;
    }

    static Mat normalsError(Mat srcNormals, Mat dstNormals)
    {
        int depth = srcNormals.depth();
        int channels = srcNormals.channels();

        if (depth == CV_32F)
        {
            if (channels == 3)
            {
                return normalsErrorT<Vec3f>(srcNormals, dstNormals);
            }
            else if (channels == 4)
            {
                return normalsErrorT<Vec4f>(srcNormals, dstNormals);
            }
        }
        else if (depth == CV_64F)
        {
            if (channels == 3)
            {
                return normalsErrorT<Vec3d>(srcNormals, dstNormals);
            }
            else if (channels == 4)
            {
                return normalsErrorT<Vec4d>(srcNormals, dstNormals);
            }
        }
        else
        {
            CV_Error(Error::StsInternal, "This type is unsupported");
        }
        return Mat();
    }
};


TEST_P(RenderedNormals, check)
{
    auto p = GetParam();
    int depth  = std::get<0>(p);
    auto alg = static_cast<RgbdNormals::RgbdNormalsMethod>(int(std::get<0>(std::get<1>(p))));
    bool scale = std::get<2>(p);

    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    // The depth rendered from scene OPENCV_TEST_DATA_PATH + "/cv/rgbd/normals_check/normals_scene.blend"
    std::string srcDepthFilename = dataPath + "/cv/rgbd/normals_check/depth.yaml.gz";
    std::string srcNormalsFilename = dataPath + "/cv/rgbd/normals_check/normals%d.yaml.gz";
    Mat srcDepth = readYaml(srcDepthFilename);

    ASSERT_FALSE(srcDepth.empty()) << "Failed to load depth data";

    Size depthSize = srcDepth.size();

    Mat srcNormals;
    std::array<Mat, 3> srcNormalsCh;
    for (int i = 0; i < 3; i++)
    {
        Mat m = readYaml(cv::format(srcNormalsFilename.c_str(), i));

        ASSERT_FALSE(m.empty()) << "Failed to load normals data";

        if (depth == CV_64F)
        {
            Mat c;
            m.convertTo(c, CV_64F);
            m = c;
        }

        srcNormalsCh[i] = m;
    }
    cv::merge(srcNormalsCh, srcNormals);

    // Convert saved normals from [0; 1] range to [-1; 1]
    srcNormals = srcNormals * 2.0 - 1.0;

    // Data obtained from Blender scene
    Matx33f intr(666.6667f, 0.f, 320.f,
                 0.f, 666.6667f, 240.f,
                 0.f, 0.f, 1.f);
    // Inverted camera rotation
    Matx33d rotm = cv::Quatd(0.7805, 0.4835, 0.2087, 0.3369).conjugate().toRotMat3x3();
    cv::transform(srcNormals, srcNormals, rotm);

    Mat srcMask = srcDepth > 0;

    float diffThreshold = 50.f;
    if (scale)
    {
        srcDepth = srcDepth * 5000.0;
        diffThreshold = 100000.f;
    }

    Mat srcCloud;
    // The function with mask produces 1x(w*h) vector, this is not what we need
    // depthTo3d(srcDepth, intr, srcCloud, srcMask);
    depthTo3d(srcDepth, intr, srcCloud);
    Scalar qnan = Scalar::all(std::numeric_limits<double>::quiet_NaN());
    srcCloud.setTo(qnan, ~srcMask);
    srcDepth.setTo(qnan, ~srcMask);

    // For further result comparison
    srcNormals.setTo(qnan, ~srcMask);

    Ptr<RgbdNormals> normalsComputer = RgbdNormals::create(depthSize.height, depthSize.width, depth, intr, 5, diffThreshold, alg);
    normalsComputer->cache();

    Mat dstNormals, dstNormalsOrig, dstNormalsDepth;
    normalsComputer->apply(srcCloud, dstNormals);
    //TODO: add for other methods too when it's implemented
    if (alg == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD)
    {
        normalsComputer->apply(srcDepth, dstNormalsDepth);
        dstNormalsOrig = dstNormals.clone();
    }

    // Remove 4th channel from dstNormals
    Mat newDstNormals;
    std::vector<Mat> dstNormalsCh;
    split(dstNormals, dstNormalsCh);
    dstNormalsCh.resize(3);
    merge(dstNormalsCh, newDstNormals);
    dstNormals = newDstNormals;

    Mat dstMask = nanMask(dstNormals);
    // div by 8 because uchar is 8-bit
    double maskl2 = cv::norm(dstMask, srcMask, NORM_HAMMING) / 8;

    // Flipping Y and Z to correspond to srcNormals
    Mat flipped = flipAxes(dstNormals, FLIP_Y | FLIP_Z);
    dstNormals = flipped;

    Mat absdot = normalsError(srcNormals, dstNormals);

    Mat cmpMask = srcMask & dstMask;

    EXPECT_GT(countNonZero(cmpMask), 0);

    double nrml2 = cv::norm(absdot, NORM_L2, cmpMask);

    if (!dstNormalsDepth.empty())
    {
        Mat abs3d = normalsError(dstNormalsOrig, dstNormalsDepth);
        double errInf = cv::norm(abs3d, NORM_INF, cmpMask);
        double errL2 = cv::norm(abs3d, NORM_L2, cmpMask);
        EXPECT_LE(errInf, 0.00085);
        EXPECT_LE(errL2, 0.07718);
    }

    auto th = std::get<1>(std::get<1>(p));
    EXPECT_LE(nrml2,  th.first);
    EXPECT_LE(maskl2, th.second);
}

INSTANTIATE_TEST_CASE_P(RGBD_Normals, RenderedNormals, ::testing::Combine(::testing::Values(CV_32F, CV_64F),
                                                                          ::testing::Values(
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_FALS,          {  81.8218,     0}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,       { 107.2710, 29168}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_SRI,           {  73.2027, 17693}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT, {  57.9832,  2531}}),
                                                                          ::testing::Values(true, false)));

const Scalar kSentinelVal(11.0, 22.0, 33.0, 44.0);

template<typename T>
void expectNormalsExactT(const Mat& expected, const Mat& actual, const char* label)
{
    if (expected.size() != actual.size())
    {
        ADD_FAILURE() << label << " size mismatch: expected " << expected.size()
                      << ", actual " << actual.size();
        return;
    }
    if (expected.type() != actual.type())
    {
        ADD_FAILURE() << label << " type mismatch: expected " << expected.type()
                      << ", actual " << actual.type();
        return;
    }
    const int ch = expected.channels();
    for (int y = 0; y < expected.rows; ++y)
    {
        const T* ep = expected.ptr<T>(y);
        const T* ap = actual.ptr<T>(y);
        for (int i = 0; i < expected.cols * ch; ++i)
        {
            if (cvIsNaN(ep[i]) || cvIsNaN(ap[i]))
            {
                if (!(cvIsNaN(ep[i]) && cvIsNaN(ap[i])))
                {
                    ADD_FAILURE() << label << " NaN mismatch at row=" << y << " idx=" << i
                                  << " expected=" << ep[i] << " actual=" << ap[i];
                    return;
                }
            }
            else if (ep[i] != ap[i])
            {
                ADD_FAILURE() << label << " value mismatch at row=" << y << " idx=" << i
                              << " expected=" << ep[i] << " actual=" << ap[i];
                return;
            }
        }
    }
}

void expectNormalsExact(const Mat& expected, const Mat& actual, const char* label)
{
    if (expected.depth() == CV_32F)
        expectNormalsExactT<float>(expected, actual, label);
    else if (expected.depth() == CV_64F)
        expectNormalsExactT<double>(expected, actual, label);
    else
        ADD_FAILURE() << label << " unsupported depth: " << expected.depth();
}

template<typename T>
void expectBorderSentinelT(const Mat& canvas, const Rect& roi, const Vec<T, 4>& sentinel, const char* label)
{
    if (canvas.dims != 2 || canvas.channels() != 4 ||
        roi.x < 0 || roi.y < 0 || roi.width < 0 || roi.height < 0 ||
        roi.x + roi.width > canvas.cols || roi.y + roi.height > canvas.rows)
    {
        ADD_FAILURE() << label << " invalid canvas/roi shape: dims=" << canvas.dims
                      << " ch=" << canvas.channels() << " size=" << canvas.size() << " roi=" << roi;
        return;
    }
    for (int y = 0; y < canvas.rows; ++y)
    {
        const Vec<T, 4>* row = canvas.ptr<Vec<T, 4>>(y);
        for (int x = 0; x < canvas.cols; ++x)
        {
            if (x >= roi.x && x < roi.x + roi.width && y >= roi.y && y < roi.y + roi.height)
                continue;
            if (row[x] != sentinel)
            {
                ADD_FAILURE() << label << " wrote outside ROI at (" << x << ", " << y
                              << "): expected=" << sentinel << " actual=" << row[x];
                return;
            }
        }
    }
}

void expectBorderSentinel(const Mat& canvas, const Rect& roi, const char* label)
{
    if (canvas.depth() == CV_32F)
    {
        expectBorderSentinelT<float>(canvas, roi,
            Vec4f(float(kSentinelVal[0]), float(kSentinelVal[1]), float(kSentinelVal[2]), float(kSentinelVal[3])),
            label);
    }
    else if (canvas.depth() == CV_64F)
    {
        expectBorderSentinelT<double>(canvas, roi,
            Vec4d(kSentinelVal[0], kSentinelVal[1], kSentinelVal[2], kSentinelVal[3]),
            label);
    }
    else
    {
        ADD_FAILURE() << label << " unsupported depth: " << canvas.depth();
    }
}

template<typename T>
int countFiniteUnitNormalsT(const Mat& m)
{
    int count = 0;
    for (int y = 0; y < m.rows; ++y)
    {
        const Vec<T, 4>* row = m.ptr<Vec<T, 4>>(y);
        for (int x = 0; x < m.cols; ++x)
        {
            const Vec<T, 4>& v = row[x];
            if (cvIsNaN(v[0]) || cvIsNaN(v[1]) || cvIsNaN(v[2]) ||
                cvIsInf(v[0]) || cvIsInf(v[1]) || cvIsInf(v[2]))
                continue;
            const double x0 = double(v[0]), y0 = double(v[1]), z0 = double(v[2]);
            const double sq = x0 * x0 + y0 * y0 + z0 * z0;
            if (std::abs(sq - 1.0) < 1e-4 && v[3] == T(0))
                ++count;
        }
    }
    return count;
}

int countFiniteUnitNormals(const Mat& m)
{
    if (m.dims != 2 || m.channels() != 4)
        return -1;
    if (m.depth() == CV_32F)
        return countFiniteUnitNormalsT<float>(m);
    if (m.depth() == CV_64F)
        return countFiniteUnitNormalsT<double>(m);
    return -1;
}

template<typename T>
void expectOutsideRectMatchesT(const Mat& before, const Mat& after, const Rect& skip, const char* label)
{
    if (before.size() != after.size() || before.type() != after.type())
    {
        ADD_FAILURE() << label << " canvas size/type changed";
        return;
    }
    const int ch = before.channels();
    for (int y = 0; y < before.rows; ++y)
    {
        const T* bp = before.ptr<T>(y);
        const T* ap = after.ptr<T>(y);
        for (int x = 0; x < before.cols; ++x)
        {
            if (x >= skip.x && x < skip.x + skip.width && y >= skip.y && y < skip.y + skip.height)
                continue;
            for (int c = 0; c < ch; ++c)
            {
                const int idx = x * ch + c;
                if (cvIsNaN(bp[idx]) || cvIsNaN(ap[idx]))
                {
                    if (!(cvIsNaN(bp[idx]) && cvIsNaN(ap[idx])))
                    {
                        ADD_FAILURE() << label << " corrupted NaN outside ROI at (" << x << ", " << y << ")";
                        return;
                    }
                }
                else if (bp[idx] != ap[idx])
                {
                    ADD_FAILURE() << label << " modified pixel outside ROI at (" << x << ", " << y << ")";
                    return;
                }
            }
        }
    }
}

void expectOutsideRectMatches(const Mat& before, const Mat& after, const Rect& skip, const char* label)
{
    if (before.depth() == CV_32F)
        expectOutsideRectMatchesT<float>(before, after, skip, label);
    else if (before.depth() == CV_64F)
        expectOutsideRectMatchesT<double>(before, after, skip, label);
    else
        ADD_FAILURE() << label << " unsupported depth: " << before.depth();
}

typedef ::testing::TestWithParam<std::tuple<MatDepth, NormalComputers>> NormalsInputSize;

TEST_P(NormalsInputSize, rejects_mismatched_input_size)
{
    const int depth = std::get<0>(GetParam());
    const RgbdNormals::RgbdNormalsMethod method = static_cast<RgbdNormals::RgbdNormalsMethod>(int(std::get<1>(GetParam())));
    const int cfgRows = 48, cfgCols = 64;
    const int type4 = CV_MAKETYPE(depth, 4);
    Mat intr(Matx33d(262.5, 0.0, 31.5, 0.0, 262.5, 23.5, 0.0, 0.0, 1.0));

    auto makePoints4 = [&](int r, int c) -> Mat {
        if (r == 0 || c == 0)
            return Mat(r, c, type4);
        Mat depth32(r, c, CV_32FC1);
        theRNG().fill(depth32, RNG::UNIFORM, 1.5, 3.0);
        Mat pts;
        depthTo3d(depth32, intr, pts);
        if (depth == CV_64F)
            pts.convertTo(pts, CV_64F);
        return pts;
    };

    Ptr<RgbdNormals> comp = RgbdNormals::create(cfgRows, cfgCols, depth, intr, 5, 50.f, method);

    const Mat ptsOk = makePoints4(cfgRows, cfgCols);
    Mat okOut;
    ASSERT_NO_THROW(comp->apply(ptsOk, okOut));
    EXPECT_EQ(Size(cfgCols, cfgRows), okOut.size());
    EXPECT_EQ(type4, okOut.type());
    EXPECT_GT(countFiniteUnitNormals(okOut), 0);

    // setRows() and setCols() must reject non-positive values without corrupting prior state or cache
    for (int badDim : { 0, -1, -17 })
    {
        SCOPED_TRACE(cv::format("badDim=%d", badDim));
        EXPECT_THROW(comp->setRows(badDim), cv::Exception);
        EXPECT_EQ(cfgRows, comp->getRows());
        EXPECT_THROW(comp->setCols(badDim), cv::Exception);
        EXPECT_EQ(cfgCols, comp->getCols());
    }
    if (method == RgbdNormals::RGBD_NORMALS_METHOD_SRI)
    {
        // SRI::cache() requires at least 2x2 grids (cols/2 - 1 index & rows-1/cols-1 divisors)
        EXPECT_THROW(RgbdNormals::create(1, cfgCols, depth, intr, 5, 50.f, method), cv::Exception);
        EXPECT_THROW(RgbdNormals::create(cfgRows, 1, depth, intr, 5, 50.f, method), cv::Exception);
        EXPECT_THROW(comp->setRows(1), cv::Exception);
        EXPECT_EQ(cfgRows, comp->getRows());
        EXPECT_THROW(comp->setCols(1), cv::Exception);
        EXPECT_EQ(cfgCols, comp->getCols());
    }
    Mat afterBadSetter;
    ASSERT_NO_THROW(comp->apply(ptsOk, afterBadSetter));
    expectNormalsExact(okOut, afterBadSetter, "after_bad_setter");

    // Empty 0-D Mat, 3-D Mat, wrong channels/depth, and overlapping-row broadcast step[0] == 0
    {
        Mat empty0d;
        Mat sentinel(1, 1, type4, kSentinelVal);
        const uchar* sPtr = sentinel.data;
        EXPECT_THROW(comp->apply(empty0d, sentinel), cv::Exception);
        EXPECT_EQ(sPtr, sentinel.data);

        const int sz3d[3] = { 2, cfgRows, cfgCols };
        Mat pts3d(3, sz3d, type4, Scalar::all(1.0));
        EXPECT_THROW(comp->apply(pts3d, sentinel), cv::Exception);
        EXPECT_EQ(sPtr, sentinel.data);

        Mat wrongCh(cfgRows, cfgCols, CV_MAKETYPE(depth, 3), Scalar::all(1.0));
        EXPECT_THROW(comp->apply(wrongCh, sentinel), cv::Exception);
        EXPECT_EQ(sPtr, sentinel.data);

        Mat wrongDepth(cfgRows, cfgCols, CV_8UC4, Scalar::all(1));
        EXPECT_THROW(comp->apply(wrongDepth, sentinel), cv::Exception);
        EXPECT_EQ(sPtr, sentinel.data);

        Mat rowBuf(1, cfgCols, type4, Scalar::all(1.0));
        const int sz2d[2] = { cfgRows, cfgCols };
        const size_t zeroSteps[2] = { 0, rowBuf.elemSize() };
        Mat broadcastIn(2, sz2d, type4, rowBuf.data, zeroSteps);
        EXPECT_THROW(comp->apply(broadcastIn, sentinel), cv::Exception);
        EXPECT_EQ(sPtr, sentinel.data);
        Mat broadcastOut(2, sz2d, type4, rowBuf.data, zeroSteps);
        EXPECT_THROW(comp->apply(ptsOk, broadcastOut), cv::Exception);
        expectNormalsExact(Mat(1, 1, type4, kSentinelVal), sentinel, "sentinel_misc_bad");
    }

    const Size badSizes[7] = {
        Size(32, 24),          // both smaller
        Size(96, 72),          // both larger
        Size(cfgCols, 24),     // height mismatch only
        Size(32, cfgRows),     // width mismatch only
        Size(cfgCols, 0),      // zero height
        Size(0, cfgRows),      // zero width
        Size(0, 0)             // both zero (2-D typed empty)
    };
    for (const Size& sz : badSizes)
    {
        SCOPED_TRACE(cv::format("badSize=%dx%d", sz.width, sz.height));
        const Mat badPts = makePoints4(sz.height, sz.width);

        Mat sentinel(1, 1, type4, kSentinelVal);
        const uchar* sPtr = sentinel.data;
        EXPECT_THROW(comp->apply(badPts, sentinel), cv::Exception);
        EXPECT_EQ(Size(1, 1), sentinel.size());
        EXPECT_EQ(sPtr, sentinel.data);
        expectNormalsExact(Mat(1, 1, type4, kSentinelVal), sentinel, "sentinel_scalar");

        Mat canvas(cfgRows + 8, cfgCols + 12, type4, kSentinelVal);
        Mat roiOut = canvas(Rect(6, 4, cfgCols, cfgRows));
        const uchar* rPtr = roiOut.data;
        EXPECT_THROW(comp->apply(badPts, roiOut), cv::Exception);
        EXPECT_EQ(Size(cfgCols, cfgRows), roiOut.size());
        EXPECT_EQ(rPtr, roiOut.data);
        expectNormalsExact(Mat(canvas.size(), canvas.type(), kSentinelVal), canvas, "sentinel_roi");
    }

    if (method == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD)
    {
        for (int dDepth : { CV_16U, CV_32F, CV_64F })
        {
            for (const Size& sz : badSizes)
            {
                SCOPED_TRACE(cv::format("LINEMOD dDepth=%d badSize=%dx%d", dDepth, sz.width, sz.height));
                const Mat badDepth(sz.height, sz.width, CV_MAKETYPE(dDepth, 1), Scalar::all(2000));
                Mat out(1, 1, type4, kSentinelVal);
                const uchar* oPtr = out.data;
                EXPECT_THROW(comp->apply(badDepth, out), cv::Exception);
                EXPECT_EQ(oPtr, out.data);
                expectNormalsExact(Mat(1, 1, type4, kSentinelVal), out, "linemod_sentinel");
            }
        }
    }

    // setRows() and setCols() change the accepted input size
    comp->setRows(24);
    comp->setCols(32);
    Mat outResized;
    EXPECT_THROW(comp->apply(ptsOk, outResized), cv::Exception);
    EXPECT_TRUE(outResized.empty());
    const Mat ptsResized = makePoints4(24, 32);
    ASSERT_NO_THROW(comp->apply(ptsResized, outResized));
    EXPECT_EQ(Size(32, 24), outResized.size());
    EXPECT_EQ(type4, outResized.type());

    // 1-D-shaped grids (1xN and Nx1) must run safely for FALS, LINEMOD (including rows>=12, cols==1), and CrossProduct;
    // SRI accepts a minimum 2x2 grid.
    if (method == RgbdNormals::RGBD_NORMALS_METHOD_SRI)
    {
        const Mat pts2x2 = makePoints4(2, 2);
        comp->setRows(2);
        comp->setCols(2);
        Mat out2x2;
        ASSERT_NO_THROW(comp->apply(pts2x2, out2x2));
        EXPECT_EQ(Size(2, 2), out2x2.size());
        EXPECT_EQ(type4, out2x2.type());
    }
    else
    {
        const Mat ptsRow = makePoints4(1, 20);
        comp->setRows(1);
        comp->setCols(20);
        Mat outRow;
        ASSERT_NO_THROW(comp->apply(ptsRow, outRow));
        EXPECT_EQ(Size(20, 1), outRow.size());

        const Mat ptsCol = makePoints4(20, 1);
        comp->setRows(20);
        comp->setCols(1);
        Mat outCol;
        ASSERT_NO_THROW(comp->apply(ptsCol, outCol));
        EXPECT_EQ(Size(1, 20), outCol.size());

        // 1-D OutputArray wrappers (std::vector<Vec4f/d>, _OutputArray(raw, n)) for 20x1 and 1x20
        if (depth == CV_32F)
        {
            const Vec4f sent(static_cast<float>(kSentinelVal[0]), static_cast<float>(kSentinelVal[1]),
                             static_cast<float>(kSentinelVal[2]), static_cast<float>(kSentinelVal[3]));
            std::vector<Vec4f> vec(20, sent);
            comp->apply(ptsCol, vec);
            expectNormalsExact(outCol, Mat(20, 1, type4, vec.data()), "vector_out_32f");
            ptsCol.copyTo(Mat(20, 1, type4, vec.data()));
            comp->apply(Mat(20, 1, type4, vec.data()), vec);
            expectNormalsExact(outCol, Mat(20, 1, type4, vec.data()), "vector_inplace_32f");

            Vec4f rawBuf[20];
            std::fill_n(rawBuf, 20, sent);
            comp->apply(ptsCol, _OutputArray(rawBuf, 20));
            expectNormalsExact(outCol, Mat(20, 1, type4, rawBuf), "raw_ptr_out_32f");

            comp->setRows(1);
            comp->setCols(20);
            std::fill(vec.begin(), vec.end(), sent);
            comp->apply(ptsRow, vec);
            expectNormalsExact(outRow, Mat(1, 20, type4, vec.data()), "vector_row_32f");
        }
        else
        {
            const Vec4d sent(kSentinelVal[0], kSentinelVal[1], kSentinelVal[2], kSentinelVal[3]);
            std::vector<Vec4d> vec(20, sent);
            comp->apply(ptsCol, vec);
            expectNormalsExact(outCol, Mat(20, 1, type4, vec.data()), "vector_out_64f");
            ptsCol.copyTo(Mat(20, 1, type4, vec.data()));
            comp->apply(Mat(20, 1, type4, vec.data()), vec);
            expectNormalsExact(outCol, Mat(20, 1, type4, vec.data()), "vector_inplace_64f");

            Vec4d rawBuf[20];
            std::fill_n(rawBuf, 20, sent);
            comp->apply(ptsCol, _OutputArray(rawBuf, 20));
            expectNormalsExact(outCol, Mat(20, 1, type4, rawBuf), "raw_ptr_out_64f");

            comp->setRows(1);
            comp->setCols(20);
            std::fill(vec.begin(), vec.end(), sent);
            comp->apply(ptsRow, vec);
            expectNormalsExact(outRow, Mat(1, 20, type4, vec.data()), "vector_row_64f");
        }
    }
}

TEST_P(NormalsInputSize, roi_input_and_output)
{
    const int depth = std::get<0>(GetParam());
    const RgbdNormals::RgbdNormalsMethod method = static_cast<RgbdNormals::RgbdNormalsMethod>(int(std::get<1>(GetParam())));
    const int cfgRows = 32, cfgCols = 40;
    const int type4 = CV_MAKETYPE(depth, 4);
    Mat intr(Matx33d(262.5, 0.0, 19.5, 0.0, 262.5, 15.5, 0.0, 0.0, 1.0));

    // Build non-uniform synthetic depth so any stride bug shifts values measurably
    Mat depthBase(cfgRows, cfgCols, CV_32FC1);
    theRNG().fill(depthBase, RNG::UNIFORM, 1.0, 4.0);
    Mat ptsRef;
    depthTo3d(depthBase, intr, ptsRef);
    if (depth == CV_64F)
        ptsRef.convertTo(ptsRef, CV_64F);
    ASSERT_TRUE(ptsRef.isContinuous());

    Ptr<RgbdNormals> comp = RgbdNormals::create(cfgRows, cfgCols, depth, intr, 5, 50.f, method);

    Mat refOut;
    comp->apply(ptsRef, refOut);
    ASSERT_EQ(Size(cfgCols, cfgRows), refOut.size());
    ASSERT_EQ(type4, refOut.type());
    ASSERT_GT(countFiniteUnitNormals(refOut), 0);

    // 1. Preallocated non-continuous output ROI (both lvalue header and direct submatrix rvalue)
    const Rect outRect(11, 7, cfgCols, cfgRows);
    Mat outCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    Mat outRoi = outCanvas(outRect);
    const uchar* outRoiPtr = outRoi.data;
    ASSERT_FALSE(outRoi.isContinuous());
    comp->apply(ptsRef, outRoi);
    EXPECT_EQ(outRoiPtr, outRoi.data);
    expectBorderSentinel(outCanvas, outRect, "lvalue_roi_out");
    expectNormalsExact(refOut, outCanvas(outRect), "lvalue_roi_out");

    Mat outCanvasRv(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    comp->apply(ptsRef, outCanvasRv(outRect));
    expectBorderSentinel(outCanvasRv, outRect, "rvalue_roi_out");
    expectNormalsExact(refOut, outCanvasRv(outRect), "rvalue_roi_out");

    // 2. Non-continuous 4-channel input ROI surrounded by NaNs -> non-continuous output ROI
    const Rect inRect(9, 5, cfgCols, cfgRows);
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    Mat inCanvas(cfgRows + 14, cfgCols + 21, type4, Scalar::all(qnan));
    ptsRef.copyTo(inCanvas(inRect));
    Mat inRoi = inCanvas(inRect);
    ASSERT_FALSE(inRoi.isContinuous());

    Mat bothCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    Mat bothOutRoi = bothCanvas(outRect);
    comp->apply(inRoi, bothOutRoi);
    expectBorderSentinel(bothCanvas, outRect, "both_roi_4ch");
    expectNormalsExact(refOut, bothCanvas(outRect), "both_roi_4ch");

    // 2b. Cross-depth 4-channel ROI input (CV_64FC4 into CV_32F computer and CV_32FC4 into CV_64F computer)
    const int otherDepth = (depth == CV_32F) ? CV_64F : CV_32F;
    const int otherType4 = CV_MAKETYPE(otherDepth, 4);
    Mat ptsOtherRef;
    ptsRef.convertTo(ptsOtherRef, otherDepth);
    Mat refOtherOut;
    comp->apply(ptsOtherRef, refOtherOut);
    ASSERT_EQ(Size(cfgCols, cfgRows), refOtherOut.size());
    ASSERT_EQ(type4, refOtherOut.type());
    ASSERT_GT(countFiniteUnitNormals(refOtherOut), 0);

    Mat otherInCanvas(cfgRows + 14, cfgCols + 21, otherType4, Scalar::all(qnan));
    ptsOtherRef.copyTo(otherInCanvas(inRect));
    Mat otherInRoi = otherInCanvas(inRect);
    ASSERT_FALSE(otherInRoi.isContinuous());
    Mat otherOutCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    Mat otherOutRoi = otherOutCanvas(outRect);
    comp->apply(otherInRoi, otherOutRoi);
    expectBorderSentinel(otherOutCanvas, outRect, "cross_depth_roi_4ch");
    expectNormalsExact(refOtherOut, otherOutCanvas(outRect), "cross_depth_roi_4ch");

    // 3. Row-padded external buffer (not a submatrix, non-continuous step)
    const int paddedCols = cfgCols + 13;
    Mat paddedBacking(cfgRows, paddedCols, type4, kSentinelVal);
    Mat paddedOut(cfgRows, cfgCols, type4, paddedBacking.data, paddedBacking.step[0]);
    ASSERT_FALSE(paddedOut.isContinuous());
    comp->apply(ptsRef, paddedOut);
    EXPECT_EQ(paddedBacking.data, paddedOut.data);
    expectBorderSentinel(paddedBacking, Rect(0, 0, cfgCols, cfgRows), "padded_step_out");
    expectNormalsExact(refOut, paddedBacking(Rect(0, 0, cfgCols, cfgRows)), "padded_step_out");

    // 4. Non-continuous UMat input and output ROI
    UMat uInCanvas(cfgRows + 14, cfgCols + 21, type4, Scalar::all(qnan));
    ptsRef.copyTo(uInCanvas(inRect));
    UMat uInRoi = uInCanvas(inRect);
    ASSERT_FALSE(uInRoi.isContinuous());
    UMat uCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    UMat uRoi = uCanvas(outRect);
    ASSERT_FALSE(uRoi.isContinuous());
    comp->apply(uInRoi, uRoi);
    Mat uGot;
    uCanvas.copyTo(uGot);
    expectBorderSentinel(uGot, outRect, "umat_roi_out");
    expectNormalsExact(refOut, uGot(outRect), "umat_roi_out");

    // 5. Same-canvas overlapping ROIs (shifted ROI, forward/reverse continuous rowRange, and in-place)
    Mat sharedCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
    ptsRef.copyTo(sharedCanvas(Rect(8, 5, cfgCols, cfgRows)));
    const Mat sharedBefore = sharedCanvas.clone();
    const Rect shiftedOut(9, 6, cfgCols, cfgRows);
    Mat sharedSrc = sharedCanvas(Rect(8, 5, cfgCols, cfgRows));
    Mat sharedDst = sharedCanvas(shiftedOut);
    comp->apply(sharedSrc, sharedDst);
    expectOutsideRectMatches(sharedBefore, sharedCanvas, shiftedOut, "overlapping_roi_border");
    expectNormalsExact(refOut, sharedCanvas(shiftedOut), "overlapping_roi");

    Mat vertCanvas(cfgRows + 4, cfgCols, type4, kSentinelVal);
    ptsRef.copyTo(vertCanvas.rowRange(0, cfgRows));
    const Mat vertBefore = vertCanvas.clone();
    Mat vertSrc = vertCanvas.rowRange(0, cfgRows);
    Mat vertDst = vertCanvas.rowRange(1, cfgRows + 1);
    ASSERT_TRUE(vertSrc.isContinuous() && vertDst.isContinuous());
    comp->apply(vertSrc, vertDst);
    expectOutsideRectMatches(vertBefore, vertCanvas, Rect(0, 1, cfgCols, cfgRows), "overlapping_rowrange_fwd_border");
    expectNormalsExact(refOut, vertCanvas.rowRange(1, cfgRows + 1), "overlapping_rowrange_fwd");

    Mat vertRevCanvas(cfgRows + 4, cfgCols, type4, kSentinelVal);
    ptsRef.copyTo(vertRevCanvas.rowRange(1, cfgRows + 1));
    const Mat vertRevBefore = vertRevCanvas.clone();
    Mat vertRevSrc = vertRevCanvas.rowRange(1, cfgRows + 1);
    Mat vertRevDst = vertRevCanvas.rowRange(0, cfgRows);
    ASSERT_TRUE(vertRevSrc.isContinuous() && vertRevDst.isContinuous());
    comp->apply(vertRevSrc, vertRevDst);
    expectOutsideRectMatches(vertRevBefore, vertRevCanvas, Rect(0, 0, cfgCols, cfgRows), "overlapping_rowrange_rev_border");
    expectNormalsExact(refOut, vertRevCanvas.rowRange(0, cfgRows), "overlapping_rowrange_rev");

    Mat inplace = ptsRef.clone();
    const uchar* inplacePtr = inplace.data;
    comp->apply(inplace, inplace);
    EXPECT_EQ(inplacePtr, inplace.data);
    expectNormalsExact(refOut, inplace, "inplace");

    // 6. LINEMOD non-continuous & overlapping continuous 1-channel depth across CV_16U, CV_32F, CV_64F
    if (method == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD)
    {
        const int expectedInterior = (cfgRows - 2 * 5 - 1) * (cfgCols - 2 * 5 - 1);
        for (int dDepth : { CV_16U, CV_32F, CV_64F })
        {
            SCOPED_TRACE(cv::format("LINEMOD dDepth=%d", dDepth));
            Mat dCont;
            if (dDepth == CV_16U)
                depthBase.convertTo(dCont, CV_16U, 1000.0);
            else
                depthBase.convertTo(dCont, dDepth);
            ASSERT_TRUE(dCont.isContinuous());

            Mat refDepthOut;
            comp->apply(dCont, refDepthOut);
            ASSERT_EQ(Size(cfgCols, cfgRows), refDepthOut.size());
            ASSERT_EQ(type4, refDepthOut.type());
            ASSERT_EQ(expectedInterior, countFiniteUnitNormals(refDepthOut));

            // Keep border padding below differenceThreshold (50 / 50000) so stray reads perturb gradients
            Mat dCanvas(cfgRows + 18, cfgCols + 29, dDepth);
            theRNG().fill(dCanvas, RNG::UNIFORM, dDepth == CV_16U ? 5000.0 : 10.0,
                          dDepth == CV_16U ? 45000.0 : 40.0);
            const Rect dRect(13, 9, cfgCols, cfgRows);
            dCont.copyTo(dCanvas(dRect));
            Mat dRoi = dCanvas(dRect);
            ASSERT_FALSE(dRoi.isContinuous());

            Mat dOutCanvas(cfgRows + 16, cfgCols + 25, type4, kSentinelVal);
            Mat dOutRoi = dOutCanvas(outRect);
            comp->apply(dRoi, dOutRoi);
            expectBorderSentinel(dOutCanvas, outRect, "linemod_depth_roi");
            expectNormalsExact(refDepthOut, dOutCanvas(outRect), "linemod_depth_roi");

            // Continuous 1-channel depth overlapping the output 4-channel normals buffer
            Mat overlapBuf(cfgRows, cfgCols, type4, kSentinelVal);
            Mat dAlias(cfgRows, cfgCols, dDepth, overlapBuf.data);
            dCont.copyTo(dAlias);
            ASSERT_TRUE(dAlias.isContinuous() && overlapBuf.isContinuous());
            comp->apply(dAlias, overlapBuf);
            expectNormalsExact(refDepthOut, overlapBuf, "linemod_depth_overlap");
        }
    }
}

INSTANTIATE_TEST_CASE_P(RGBD_Normals, NormalsInputSize, ::testing::Combine(::testing::Values(CV_32F, CV_64F),
                                                                           NormalComputers::all()));

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RgbdPlaneGenerate : public ::testing::TestWithParam<std::tuple<int, bool, int>>
{
protected:
    void SetUp() override
    {
        auto p = GetParam();
        idx = std::get<0>(p);
        checkNormals = std::get<1>(p);
        nPlanes = std::get<2>(p);
    }

    int idx;
    bool checkNormals;
    int nPlanes;
};

TEST_P(RgbdPlaneGenerate, compute)
{
    RNG &rng = cvtest::TS::ptr()->get_rng();
    rng.state += idx;

    std::vector<Plane> planes;
    Mat points3d, ground_normals;
    Mat_<unsigned char> gt_plane_mask;
    gen_points_3d(planes, gt_plane_mask, points3d, ground_normals, nPlanes, 1.f, rng);

    Mat plane_mask;
    std::vector<Vec4f> plane_coefficients;

    Mat normals;
    if (checkNormals)
    {
        // First, get the normals
        int depth = CV_32F;
        Ptr<RgbdNormals> normalsComputer = RgbdNormals::create(H, W, depth, K(), 5, 50.f, RgbdNormals::RGBD_NORMALS_METHOD_FALS);
        normalsComputer->apply(points3d, normals);
    }

    findPlanes(points3d, normals, plane_mask, plane_coefficients);

    // Compare each found plane to each ground truth plane
    int n_planes = (int)plane_coefficients.size();
    int n_gt_planes = (int)planes.size();
    Mat_<int> matching(n_gt_planes, n_planes);
    for (int j = 0; j < n_gt_planes; ++j)
    {
        Mat gt_mask = gt_plane_mask == j;
        int n_gt = countNonZero(gt_mask);
        int n_max = 0, i_max = 0;
        for (int i = 0; i < n_planes; ++i)
        {
            Mat dst;
            bitwise_and(gt_mask, plane_mask == i, dst);
            matching(j, i) = countNonZero(dst);
            if (matching(j, i) > n_max)
            {
                n_max = matching(j, i);
                i_max = i;
            }
        }
        // Get the best match
        ASSERT_LE(float(n_max - n_gt) / n_gt, 0.001);
        // Compare the normals
        Vec3d normal(plane_coefficients[i_max][0], plane_coefficients[i_max][1], plane_coefficients[i_max][2]);
        Vec4d nd = planes[j].nd;
        ASSERT_GE(std::abs(Vec3d(nd[0], nd[1], nd[2]).dot(normal)), 0.95);
    }
}

// 1 plane, continuous scene, very low error
// 3 planes, 3 discontinuities, more error expected
INSTANTIATE_TEST_CASE_P(RGBD_Plane, RgbdPlaneGenerate, ::testing::Combine(::testing::Range(0, 10),
                                                                            ::testing::Values(false, true),
                                                                            ::testing::Values(1, 3)));

TEST(RGBD_Plane, regression2309ValgrindCheck)
{
    Mat points(640, 480, CV_32FC3, Scalar::all(0));
    // Note, 640%9 is 1 and 480%9 is 3
    int blockSize = 9;

    Mat mask;
    std::vector<cv::Vec4f> planes;
    // Will corrupt memory; valgrind gets triggered
    findPlanes(points, noArray(), mask, planes, blockSize);
}

TEST(RGBD_Plane, regression_3channel_matches_4channel)
{
    const int rows = 240, cols = 320;

    Mat points3(rows, cols, CV_32FC3);
    Mat points4(rows, cols, CV_32FC4);
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            const float z = 2.f + 0.001f * u + 0.002f * v;
            const Vec3f p((float)u * 0.01f, (float)v * 0.01f, z);
            points3.at<Vec3f>(v, u) = p;
            points4.at<Vec4f>(v, u) = Vec4f(p[0], p[1], p[2], 0.f);
        }
    }

    Mat mask3, mask4;
    std::vector<Vec4f> planes3, planes4;
    findPlanes(points3, noArray(), mask3, planes3);
    findPlanes(points4, noArray(), mask4, planes4);

    EXPECT_EQ(points3.size(), mask3.size());
    EXPECT_EQ(CV_8U, mask3.type());

    ASSERT_EQ(mask4.size(), mask3.size());
    EXPECT_EQ(0, cv::countNonZero(mask3 != mask4));
    ASSERT_EQ(planes4.size(), planes3.size());
    for (size_t i = 0; i < planes3.size(); i++)
        EXPECT_LE(cv::norm(planes3[i], planes4[i], NORM_INF), 1e-6) << "plane " << i;
}

}} // namespace
