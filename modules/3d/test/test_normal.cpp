// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/3d.hpp>
#include <opencv2/core/quaternion.hpp>

namespace opencv_test { namespace {

const int W = 640;
const int H = 480;
//int window_size = 5;
float focal_length = 525;
float cx = W / 2.f + 0.5f;
float cy = H / 2.f + 0.5f;

static Mat K() { static Mat res = (Mat_<double>(3, 3) << focal_length, 0, cx, 0, focal_length, cy, 0, 0, 1); return res; }
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

        float diffThreshold = scale ? 100000.f : 50.f;
        normalsComputer = RgbdNormals::create(H, W, depth, K(), 5, diffThreshold, alg);
        normalsComputer->cache();
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
    NormalsTestData {CV_32F, RgbdNormals::RGBD_NORMALS_METHOD_FALS,  true, 0.00362, 0.08881, 0.02175, 0, 0},
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
    };

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
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_FALS,          {  81.8213,     0}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD,       { 107.2710, 29168}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_SRI,           {  73.2027, 17693}},
    NormalComputerThresholds { RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT, {  57.9832,  2531}}),
                                                                          ::testing::Values(true, false)));

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

}} // namespace
