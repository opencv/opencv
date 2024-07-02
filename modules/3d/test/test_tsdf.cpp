// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

using namespace cv;

/** Reprojects screen point to camera space given z coord. */
struct Reprojector
{
    Reprojector() {}
    inline Reprojector(Matx33f intr)
    {
        fxinv = 1.f / intr(0, 0), fyinv = 1.f / intr(1, 1);
        cx = intr(0, 2), cy = intr(1, 2);
    }
    template<typename T>
    inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
    {
        T x = p.z * (p.x - cx) * fxinv;
        T y = p.z * (p.y - cy) * fyinv;
        return cv::Point3_<T>(x, y, p.z);
    }

    float fxinv, fyinv, cx, cy;
};

template<class Scene>
struct RenderInvoker : ParallelLoopBody
{
    RenderInvoker(Mat_<float>& _frame, Affine3f _pose,
        Reprojector _reproj, float _depthFactor, bool _onlySemisphere)
        : ParallelLoopBody(),
        frame(_frame),
        pose(_pose),
        reproj(_reproj),
        depthFactor(_depthFactor),
        onlySemisphere(_onlySemisphere)
    { }

    virtual void operator ()(const cv::Range& r) const
    {
        for (int y = r.start; y < r.end; y++)
        {
            float* frameRow = frame[y];
            for (int x = 0; x < frame.cols; x++)
            {
                float pix = 0;

                Point3f orig = pose.translation();
                // direction through pixel
                Point3f screenVec = reproj(Point3f((float)x, (float)y, 1.f));
                float xyt = 1.f / (screenVec.x * screenVec.x +
                    screenVec.y * screenVec.y + 1.f);
                Point3f dir = cv::normalize(Vec3f(pose.rotation() * screenVec));
                // screen space axis
                dir.y = -dir.y;

                const float maxDepth = 20.f;
                const float maxSteps = 256;
                float t = 0.f;
                for (int step = 0; step < maxSteps && t < maxDepth; step++)
                {
                    Point3f p = orig + dir * t;
                    float d = Scene::map(p, onlySemisphere);
                    if (d < 0.000001f)
                    {
                        float depth = std::sqrt(t * t * xyt);
                        pix = depth * depthFactor;
                        break;
                    }
                    t += d;
                }

                frameRow[x] = pix;
            }
        }
    }

    Mat_<float>& frame;
    Affine3f pose;
    Reprojector reproj;
    float depthFactor;
    bool onlySemisphere;
};

template<class Scene>
struct RenderColorInvoker : ParallelLoopBody
{
    RenderColorInvoker(Mat_<Vec3f>& _frame, Affine3f _pose,
        Reprojector _reproj,
        float _depthFactor, bool _onlySemisphere) : ParallelLoopBody(),
        frame(_frame),
        pose(_pose),
        reproj(_reproj),
        depthFactor(_depthFactor),
        onlySemisphere(_onlySemisphere)
    { }

    virtual void operator ()(const cv::Range& r) const
    {
        for (int y = r.start; y < r.end; y++)
        {
            Vec3f* frameRow = frame[y];
            for (int x = 0; x < frame.cols; x++)
            {
                Vec3f pix = 0;

                Point3f orig = pose.translation();
                // direction through pixel
                Point3f screenVec = reproj(Point3f((float)x, (float)y, 1.f));
                Point3f dir = cv::normalize(Vec3f(pose.rotation() * screenVec));
                // screen space axis
                dir.y = -dir.y;

                const float maxDepth = 20.f;
                const float maxSteps = 256;
                float t = 0.f;
                for (int step = 0; step < maxSteps && t < maxDepth; step++)
                {
                    Point3f p = orig + dir * t;
                    float d = Scene::map(p, onlySemisphere);
                    if (d < 0.000001f)
                    {
                        float m = 0.25f;
                        float p0 = float(abs(fmod(p.x, m)) > m / 2.f);
                        float p1 = float(abs(fmod(p.y, m)) > m / 2.f);
                        float p2 = float(abs(fmod(p.z, m)) > m / 2.f);

                        pix[0] = p0 + p1;
                        pix[1] = p1 + p2;
                        pix[2] = p0 + p2;

                        pix *= 128.f;
                        break;
                    }
                    t += d;
                }

                frameRow[x] = pix;
            }
        }
    }

    Mat_<Vec3f>& frame;
    Affine3f pose;
    Reprojector reproj;
    float depthFactor;
    bool onlySemisphere;
};


struct Scene
{
    virtual ~Scene() {}
    static Ptr<Scene> create(Size sz, Matx33f _intr, float _depthFactor, bool onlySemisphere);
    virtual Mat_<float> depth(Affine3f pose) = 0;
    virtual Mat_<Vec3f> rgb(Affine3f pose) = 0;
    virtual std::vector<Affine3f> getPoses() = 0;
};

struct SemisphereScene : Scene
{
    const int framesPerCycle = 72;
    const float nCycles = 0.25f;
    const Affine3f startPose = Affine3f(Vec3f(0.f, 0.f, 0.f), Vec3f(1.5f, 0.3f, -2.1f));

    Size frameSize;
    Matx33f intr;
    float depthFactor;
    bool onlySemisphere;

    SemisphereScene(Size sz, Matx33f _intr, float _depthFactor, bool _onlySemisphere) :
        frameSize(sz), intr(_intr), depthFactor(_depthFactor), onlySemisphere(_onlySemisphere)
    { }

    static float map(Point3f p, bool onlySemisphere)
    {
        float plane = p.y + 0.5f;
        Point3f spherePose = p - Point3f(-0.0f, 0.3f, 1.1f);
        float sphereRadius = 0.5f;
        float sphere = (float)cv::norm(spherePose) - sphereRadius;
        float sphereMinusBox = sphere;

        float subSphereRadius = 0.05f;
        Point3f subSpherePose = p - Point3f(0.3f, -0.1f, -0.3f);
        float subSphere = (float)cv::norm(subSpherePose) - subSphereRadius;

        float res;
        if (!onlySemisphere)
            res = min({ sphereMinusBox, subSphere, plane });
        else
            res = sphereMinusBox;

        return res;
    }

    Mat_<float> depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<SemisphereScene>(frame, pose, reproj, depthFactor, onlySemisphere));

        return frame;
    }

    Mat_<Vec3f> rgb(Affine3f pose) override
    {
        Mat_<Vec3f> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderColorInvoker<SemisphereScene>(frame, pose, reproj, depthFactor, onlySemisphere));

        return frame;
    }

    std::vector<Affine3f> getPoses() override
    {
        std::vector<Affine3f> poses;
        for (int i = 0; i < framesPerCycle * nCycles; i++)
        {
            float angle = (float)(CV_2PI * i / framesPerCycle);
            Affine3f pose;
            pose = pose.rotate(startPose.rotation());
            pose = pose.rotate(Vec3f(0.f, -0.5f, 0.f) * angle);
            pose = pose.translate(Vec3f(startPose.translation()[0] * sin(angle),
                                        startPose.translation()[1],
                                        startPose.translation()[2] * cos(angle)));
            poses.push_back(pose);
        }

        return poses;
    }

};

Ptr<Scene> Scene::create(Size sz, Matx33f _intr, float _depthFactor, bool _onlySemisphere)
{
    return makePtr<SemisphereScene>(sz, _intr, _depthFactor, _onlySemisphere);
}

// this is a temporary solution
// ----------------------------

typedef cv::Vec4f ptype;
typedef cv::Mat_< ptype > Points;
typedef cv::Mat_< ptype > Colors;
typedef Points Normals;
typedef Size2i Size;

template<int p>
inline float specPow(float x)
{
    if (p % 2 == 0)
    {
        float v = specPow<p / 2>(x);
        return v * v;
    }
    else
    {
        float v = specPow<(p - 1) / 2>(x);
        return v * v * x;
    }
}

template<>
inline float specPow<0>(float /*x*/)
{
    return 1.f;
}

template<>
inline float specPow<1>(float x)
{
    return x;
}

inline cv::Vec3f fromPtype(const ptype& x)
{
    return cv::Vec3f(x[0], x[1], x[2]);
}

void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, Affine3f lightPose)
{
    Size sz = _points.size();
    image.create(sz, CV_8UC4);

    Points  points = _points.getMat();
    Normals normals = _normals.getMat();

    Mat goods;
    finiteMask(points, goods);

    Mat_<Vec4b> img = image.getMat();

    Range range(0, sz.height);
    const int nstripes = -1;
    parallel_for_(range, [&](const Range&)
        {
            for (int y = range.start; y < range.end; y++)
            {
                Vec4b* imgRow = img[y];
                const ptype* ptsRow = points[y];
                const ptype* nrmRow = normals[y];
                const uchar* goodRow = goods.ptr<uchar>(y);

                for (int x = 0; x < sz.width; x++)
                {
                    Point3f p = fromPtype(ptsRow[x]);
                    Point3f n = fromPtype(nrmRow[x]);

                    Vec4b color;

                    if (!goodRow[x])
                    {
                        color = Vec4b(0, 32, 0, 0);
                    }
                    else
                    {
                        const float Ka = 0.3f;  //ambient coeff
                        const float Kd = 0.5f;  //diffuse coeff
                        const float Ks = 0.2f;  //specular coeff
                        const int   sp = 20;  //specular power

                        const float Ax = 1.f;   //ambient color,  can be RGB
                        const float Dx = 1.f;   //diffuse color,  can be RGB
                        const float Sx = 1.f;   //specular color, can be RGB
                        const float Lx = 1.f;   //light color

                        Point3f l = cv::normalize(lightPose.translation() - Vec3f(p));
                        Point3f v = cv::normalize(-Vec3f(p));
                        Point3f r = cv::normalize(Vec3f(2.f * n * n.dot(l) - l));

                        uchar ix = (uchar)((Ax * Ka * Dx + Lx * Kd * Dx * max(0.f, n.dot(l)) +
                            Lx * Ks * Sx * specPow<sp>(max(0.f, r.dot(v)))) * 255.f);
                        color = Vec4b(ix, ix, ix, 0);
                    }

                    imgRow[x] = color;
                }
            }
        }, nstripes);
}
void renderPointsNormalsColors(InputArray _points, InputArray, InputArray _colors, OutputArray image, Affine3f)
{
    Size sz = _points.size();
    image.create(sz, CV_8UC4);

    Points  points  = _points.getMat();
    Colors  colors  = _colors.getMat();

    Mat goods, goodc, goodp;
    finiteMask(points, goodp);
    finiteMask(colors, goodc);
    goods = goodp & goodc;

    Mat_<Vec4b> img = image.getMat();

    Range range(0, sz.height);
    const int nstripes = -1;
    parallel_for_(range, [&](const Range&)
        {
            for (int y = range.start; y < range.end; y++)
            {
                Vec4b* imgRow = img[y];
                const ptype* clrRow = colors[y];
                const uchar* goodRow = goods.ptr<uchar>(y);

                for (int x = 0; x < sz.width; x++)
                {
                    Point3f c = fromPtype(clrRow[x]);

                    Vec4b color;

                    if (!goodRow[x])
                    {
                        color = Vec4b(0, 32, 0, 0);
                    }
                    else
                    {
                        color = Vec4b((uchar)c.x, (uchar)c.y, (uchar)c.z, (uchar)0);
                    }

                    imgRow[x] = color;
                }
            }
        }, nstripes);
}
// ----------------------------

void displayImage(Mat depth, Mat points, Mat normals, float depthFactor, Vec3f lightPose)
{
    Mat image;
    patchNaNs(points);
    imshow("depth", depth * (1.f / depthFactor / 4.f));
    renderPointsNormals(points, normals, image, lightPose);
    imshow("render", image);
    waitKey(2000);
    destroyAllWindows();
}

void displayColorImage(Mat depth, Mat rgb, Mat points, Mat normals, Mat colors, float depthFactor, Vec3f lightPose)
{
    Mat image;
    patchNaNs(points);
    imshow("depth", depth * (1.f / depthFactor / 4.f));
    imshow("rgb", rgb * (1.f / 255.f));
    renderPointsNormalsColors(points, normals, colors, image, lightPose);
    imshow("render", image);
    waitKey(2000);
    destroyAllWindows();
}

void normalsCheck(Mat normals)
{
    Vec4f vector;
    int counter = 0;
    for (auto pvector = normals.begin<Vec4f>(); pvector < normals.end<Vec4f>(); pvector++)
    {
        vector = *pvector;
        if (!(cvIsNaN(vector[0]) || cvIsNaN(vector[1]) || cvIsNaN(vector[2])))
        {
            counter++;
            float l2 = vector[0] * vector[0] +
                       vector[1] * vector[1] +
                       vector[2] * vector[2];
            ASSERT_LT(abs(1.f - l2), 0.0001f) << "There is normal with length != 1";
        }
    }
    ASSERT_GT(counter, 0) << "There are no normals";
}

int counterOfValid(Mat points)
{
    Vec4f* v;
    int i, j;
    int count = 0;
    for (i = 0; i < points.rows; ++i)
    {
        v = (points.ptr<Vec4f>(i));
        for (j = 0; j < points.cols; ++j)
        {
            if ((v[j])[0] != 0 ||
                (v[j])[1] != 0 ||
                (v[j])[2] != 0)
            {
                count++;
            }
        }
    }
    return count;
}


enum class VolumeTestFunction
{
    RAYCAST = 0,
    FETCH_NORMALS = 1,
    FETCH_POINTS_NORMALS = 2
};

enum class VolumeTestSrcType
{
    MAT = 0,
    ODOMETRY_FRAME = 1
};

enum class FrameSizeType
{
    DEFAULT = 0,
    CUSTOM = 1
};


void debugVolumeDraw(const Volume &volume, Affine3f pose, Mat depth, float depthFactor, std::string objFname)
{
    Vec3f lightPose = Vec3f::all(0.f);
    Mat points, normals;
    volume.raycast(pose.matrix, points, normals);

    Mat ptsList, ptsList3, nrmList, nrmList3;
    volume.fetchPointsNormals(ptsList, nrmList);
    // transform 4 channels to 3 channels
    cvtColor(ptsList, ptsList3, COLOR_BGRA2BGR);
    cvtColor(ptsList, nrmList3, COLOR_BGRA2BGR);
    savePointCloud(objFname, ptsList3, nrmList3);

    displayImage(depth, points, normals, depthFactor, lightPose);
}


// For fixed volumes which are TSDF and ColorTSDF
void staticBoundingBoxTest(VolumeType volumeType)
{
    VolumeSettings vs(volumeType);
    Volume volume(volumeType, vs);

    Vec3i res;
    vs.getVolumeResolution(res);
    float voxelSize = vs.getVoxelSize();
    Matx44f pose;
    vs.getVolumePose(pose);
    Vec3f end = voxelSize * Vec3f(res);
    Vec6f truebb(0, 0, 0, end[0], end[1], end[2]);
    Vec6f bb;
    volume.getBoundingBox(bb, Volume::BoundingBoxPrecision::VOLUME_UNIT);
    Vec6f diff = bb - truebb;
    double normdiff = std::sqrt(diff.ddot(diff));
    ASSERT_LE(normdiff, std::numeric_limits<double>::epsilon());
}


// For HashTSDF only
void boundingBoxGrowthTest(bool enableGrowth)
{
    VolumeSettings vs(VolumeType::HashTSDF);
    Volume volume(VolumeType::HashTSDF, vs);

    Size frameSize(vs.getRaycastWidth(), vs.getRaycastHeight());
    Matx33f intrIntegrate, intrRaycast;
    vs.getCameraIntegrateIntrinsics(intrIntegrate);
    vs.getCameraRaycastIntrinsics(intrRaycast);
    bool onlySemisphere = false;
    float depthFactor = vs.getDepthFactor();
    Ptr<Scene> scene = Scene::create(frameSize, intrIntegrate, depthFactor, onlySemisphere);
    std::vector<Affine3f> poses = scene->getPoses();

    Mat depth = scene->depth(poses[0]);
    UMat udepth;
    depth.copyTo(udepth);

    // depth is integrated with multiple weight
    // TODO: add weight parameter to integrate() call (both scalar and array of 8u/32f)
    const int nIntegrations = 1;
    for (int i = 0; i < nIntegrations; i++)
        volume.integrate(udepth, poses[0].matrix);

    Vec6f bb;
    volume.getBoundingBox(bb, Volume::BoundingBoxPrecision::VOLUME_UNIT);
    Vec6f truebb(-0.9375f, 1.3125f, -0.8906f, 3.9375f, 2.6133f, 1.4004f);
    Vec6f diff = bb - truebb;
    double bbnorm = std::sqrt(diff.ddot(diff));

    Vec3f vuRes;
    vs.getVolumeResolution(vuRes);
    double vuSize = vs.getVoxelSize() * vuRes[0];
    // it's OK to have such big difference since this is volume unit size-grained BB calculation
    // Theoretical max difference can be sqrt(6) =(approx)= 2.4494
    EXPECT_LE(bbnorm, vuSize * 2.38);

    if (cvtest::debugLevel > 0)
    {
        debugVolumeDraw(volume, poses[0], depth, depthFactor, "pts.obj");
    }

    // Integrate another depth growth changed

    Mat depth2 = scene->depth(poses[0].translate(Vec3f(0, -0.25f, 0)));
    UMat udepth2;
    depth2.copyTo(udepth2);

    volume.setEnableGrowth(enableGrowth);

    for (int i = 0; i < nIntegrations; i++)
        volume.integrate(udepth2, poses[0].matrix);

    Vec6f bb2;
    volume.getBoundingBox(bb2, Volume::BoundingBoxPrecision::VOLUME_UNIT);

    Vec6f truebb2 = truebb + Vec6f(0, -(1.3125f - 1.0723f), -(-0.8906f - (-1.4238f)), 0, 0, 0);
    Vec6f diff2 = enableGrowth ? bb2 - truebb2 : bb2 - bb;
    double bbnorm2 = std::sqrt(diff2.ddot(diff2));
    EXPECT_LE(bbnorm2, enableGrowth ? (vuSize * 2.3) : std::numeric_limits<double>::epsilon());

    if (cvtest::debugLevel > 0)
    {
        debugVolumeDraw(volume, poses[0], depth, depthFactor, enableGrowth ? "pts_growth.obj" : "pts_no_growth.obj");
    }

    // Reset check

    volume.reset();
    Vec6f bb3;
    volume.getBoundingBox(bb3, Volume::BoundingBoxPrecision::VOLUME_UNIT);
    double bbnorm3 = std::sqrt(bb3.ddot(bb3));
    EXPECT_LE(bbnorm3, std::numeric_limits<double>::epsilon());
}


template <typename VT>
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

void regressionVolPoseRot()
{
    // Make 2 volumes which differ only in their pose (especially rotation)
    VolumeSettings vs(VolumeType::HashTSDF);
    Volume volume0(VolumeType::HashTSDF, vs);

    VolumeSettings vsRot(vs);
    Matx44f pose;
    vsRot.getVolumePose(pose);
    pose = Affine3f(Vec3f(1, 1, 1), Vec3f()).matrix;
    vsRot.setVolumePose(pose);
    Volume volumeRot(VolumeType::HashTSDF, vsRot);

    Size frameSize(vs.getRaycastWidth(), vs.getRaycastHeight());
    Matx33f intrIntegrate, intrRaycast;
    vs.getCameraIntegrateIntrinsics(intrIntegrate);
    vs.getCameraRaycastIntrinsics(intrRaycast);
    bool onlySemisphere = false;
    float depthFactor = vs.getDepthFactor();
    Vec3f lightPose = Vec3f::all(0.f);
    Ptr<Scene> scene = Scene::create(frameSize, intrIntegrate, depthFactor, onlySemisphere);
    std::vector<Affine3f> poses = scene->getPoses();

    Mat depth = scene->depth(poses[0]);
    UMat udepth;
    depth.copyTo(udepth);

    volume0.integrate(udepth, poses[0].matrix);
    volumeRot.integrate(udepth, poses[0].matrix);

    UMat upts, unrm, uptsRot, unrmRot;

    volume0.raycast(poses[0].matrix, upts, unrm);
    volumeRot.raycast(poses[0].matrix, uptsRot, unrmRot);

    Mat mpts = upts.getMat(ACCESS_READ), mnrm = unrm.getMat(ACCESS_READ);
    Mat mptsRot = uptsRot.getMat(ACCESS_READ), mnrmRot = unrmRot.getMat(ACCESS_READ);

    if (cvtest::debugLevel > 0)
    {
        displayImage(depth, mpts, mnrm, depthFactor, lightPose);
        displayImage(depth, mptsRot, mnrmRot, depthFactor, lightPose);
    }

    std::vector<Mat> ptsCh(3), ptsRotCh(3);
    split(mpts, ptsCh);
    split(uptsRot, ptsRotCh);
    Mat maskPts0 = ptsCh[2] > 0;
    Mat maskPtsRot = ptsRotCh[2] > 0;
    Mat maskNrm0, maskNrmRot;
    finiteMask(mnrm, maskNrm0);
    finiteMask(mnrmRot, maskNrmRot);
    Mat maskPtsDiff, maskNrmDiff;
    cv::bitwise_xor(maskPts0, maskPtsRot, maskPtsDiff);
    cv::bitwise_xor(maskNrm0, maskNrmRot, maskNrmDiff);
    double ptsDiffNorm = cv::sum(maskPtsDiff)[0]/255.0;
    double nrmDiffNorm = cv::sum(maskNrmDiff)[0]/255.0;

    EXPECT_LE(ptsDiffNorm, 786);
    EXPECT_LE(nrmDiffNorm, 786);

    double normPts = cv::norm(mpts, mptsRot, NORM_INF, (maskPts0 & maskPtsRot));
    Mat absdot = normalsError(mnrm, mnrmRot);
    double normNrm = cv::norm(absdot, NORM_L2, (maskNrm0 & maskNrmRot));

    EXPECT_LE(normPts, 2.0);
    EXPECT_LE(normNrm, 73.08);
}

///////// Parametrized tests

enum PlatformType
{
    CPU = 0, GPU = 1
};
CV_ENUM(PlatformTypeEnum, PlatformType::CPU, PlatformType::GPU);

// used to store current OpenCL status (on/off) and revert it after test is done
// works even after exceptions thrown in test body
struct OpenCLStatusRevert
{
#ifdef HAVE_OPENCL
    OpenCLStatusRevert()
    {
        originalOpenCLStatus = cv::ocl::useOpenCL();
    }
    ~OpenCLStatusRevert()
    {
        cv::ocl::setUseOpenCL(originalOpenCLStatus);
    }
    void off()
    {
        cv::ocl::setUseOpenCL(false);
    }
    bool originalOpenCLStatus;
#else
    void off() { }
#endif
};


// CV_ENUM does not support enum class types, so let's implement the class explicitly
namespace
{
    struct VolumeTypeEnum
    {
        static const std::array<VolumeType, 3> vals;
        static const std::array<std::string, 3> svals;

        VolumeTypeEnum(VolumeType v = VolumeType::TSDF) : val(v) {}
        operator VolumeType() const { return val; }
        void PrintTo(std::ostream *os) const
        {
            int v = int(val);
            if (v >= 0 && v < 3)
            {
                *os << svals[v];
            }
            else
            {
                *os << "UNKNOWN";
            }
        }
        static ::testing::internal::ParamGenerator<VolumeTypeEnum> all()
        {
            return ::testing::Values(VolumeTypeEnum(vals[0]), VolumeTypeEnum(vals[1]), VolumeTypeEnum(vals[2]));
        }

    private:
        VolumeType val;
    };
    const std::array<VolumeType, 3> VolumeTypeEnum::vals{VolumeType::TSDF, VolumeType::HashTSDF, VolumeType::ColorTSDF};
    const std::array<std::string, 3> VolumeTypeEnum::svals{std::string("TSDF"), std::string("HashTSDF"), std::string("ColorTSDF")};

    static inline void PrintTo(const VolumeTypeEnum &t, std::ostream *os) { t.PrintTo(os); }


    struct VolumeTestSrcTypeEnum
    {
        static const std::array<VolumeTestSrcType, 2> vals;
        static const std::array<std::string, 2> svals;

        VolumeTestSrcTypeEnum(VolumeTestSrcType v = VolumeTestSrcType::MAT) : val(v) {}
        operator VolumeTestSrcType() const { return val; }
        void PrintTo(std::ostream *os) const
        {
            int v = int(val);
            if (v >= 0 && v < 3)
            {
                *os << svals[v];
            }
            else
            {
                *os << "UNKNOWN";
            }
        }
        static ::testing::internal::ParamGenerator<VolumeTestSrcTypeEnum> all()
        {
            return ::testing::Values(VolumeTestSrcTypeEnum(vals[0]), VolumeTestSrcTypeEnum(vals[1]));
        }

    private:
        VolumeTestSrcType val;
    };
    const std::array<VolumeTestSrcType, 2> VolumeTestSrcTypeEnum::vals{VolumeTestSrcType::MAT, VolumeTestSrcType::ODOMETRY_FRAME};
    const std::array<std::string, 2> VolumeTestSrcTypeEnum::svals{std::string("UMat"), std::string("OdometryFrame")};

    static inline void PrintTo(const VolumeTestSrcTypeEnum &t, std::ostream *os) { t.PrintTo(os); }


    struct FrameSizeTypeEnum
    {
        static const std::array<FrameSizeType, 2> vals;
        static const std::array<std::string, 2> svals;

        FrameSizeTypeEnum(FrameSizeType v = FrameSizeType::DEFAULT) : val(v) {}
        operator FrameSizeType() const { return val; }
        void PrintTo(std::ostream *os) const
        {
            int v = int(val);
            if (v >= 0 && v < 3)
            {
                *os << svals[v];
            }
            else
            {
                *os << "UNKNOWN";
            }
        }
        static ::testing::internal::ParamGenerator<FrameSizeTypeEnum> all()
        {
            return ::testing::Values(FrameSizeTypeEnum(vals[0]), FrameSizeTypeEnum(vals[1]));
        }

    private:
        FrameSizeType val;
    };
    const std::array<FrameSizeType, 2> FrameSizeTypeEnum::vals{FrameSizeType::DEFAULT, FrameSizeType::CUSTOM};
    const std::array<std::string, 2> FrameSizeTypeEnum::svals{std::string("DefaultSize"), std::string("CustomSize")};

    static inline void PrintTo(const FrameSizeTypeEnum &t, std::ostream *os) { t.PrintTo(os); }
}


typedef std::tuple<PlatformTypeEnum, VolumeTypeEnum> PlatformVolumeType;
struct VolumeTestFixture : public ::testing::TestWithParam<std::tuple<PlatformVolumeType, VolumeTestSrcTypeEnum, FrameSizeTypeEnum>>
{
protected:
    void SetUp() override
    {
        auto p = GetParam();
        gpu = (std::get<0>(std::get<0>(p)) == PlatformType::GPU);
        volumeType = std::get<1>(std::get<0>(p));

        testSrcType = std::get<1>(p);
        frameSizeSpecified = std::get<2>(p);

        if (!gpu)
            oclStatus.off();

        vs = makePtr<VolumeSettings>(volumeType);
        volume = makePtr<Volume>(volumeType, *vs);

        frameSize = Size(vs->getRaycastWidth(), vs->getRaycastHeight());
        vs->getCameraIntegrateIntrinsics(intrIntegrate);
        vs->getCameraRaycastIntrinsics(intrRaycast);
        bool onlySemisphere = true; //TODO: check both
        depthFactor = vs->getDepthFactor();
        lightPose = Vec3f::all(0.f);
        scene = Scene::create(frameSize, intrIntegrate, depthFactor, onlySemisphere);
        poses = scene->getPoses();

        depth = scene->depth(poses[0]);
        rgb = scene->rgb(poses[0]);
        UMat udepth, urgb;
        depth.copyTo(udepth);
        rgb.copyTo(urgb);

        OdometryFrame odf(udepth, urgb);

        if (testSrcType == VolumeTestSrcType::MAT)
        {
            if (volumeType == VolumeType::ColorTSDF)
                volume->integrate(udepth, urgb, poses[0].matrix);
            else
                volume->integrate(udepth, poses[0].matrix);
        }
        else if (testSrcType == VolumeTestSrcType::ODOMETRY_FRAME)
        {
            volume->integrate(odf, poses[0].matrix);
        }
    }

    void saveObj(std::string funcName, Mat points, Mat normals);
    void raycast_test();
    void fetch_points_normals_test();
    void fetch_normals_test();
    void valid_points_test();

    bool gpu;
    VolumeType volumeType;
    VolumeTestSrcType testSrcType;
    FrameSizeType frameSizeSpecified;

    OpenCLStatusRevert oclStatus;

    Ptr<Volume> volume;
    Ptr<VolumeSettings> vs;
    Size frameSize;
    Matx33f intrIntegrate, intrRaycast;
    Ptr<Scene> scene;
    std::vector<Affine3f> poses;
    float depthFactor;
    Vec3f lightPose;

    Mat depth, rgb;
};


void VolumeTestFixture::saveObj(std::string funcName, Mat points, Mat normals)
{
    Mat pts3, nrm3;
    cvtColor(points, pts3, COLOR_RGBA2RGB);
    cvtColor(normals, nrm3, COLOR_RGBA2RGB);
    string platformString = gpu ? "GPU" : "CPU";
    string volumeTypeString = volumeType == VolumeType::TSDF ? "TSDF" :
                              volumeType == VolumeType::HashTSDF ? "HashTSDF" :
                              volumeType == VolumeType::ColorTSDF  ? "ColorTSDF" : "";
    string testSrcTypeString = testSrcType == VolumeTestSrcType::MAT ? "MAT" :
                               testSrcType == VolumeTestSrcType::ODOMETRY_FRAME ? "OFRAME" : "";
    string frameSizeSpecifiedString = frameSizeSpecified == FrameSizeType::DEFAULT ? "DefaultSize" :
                                      frameSizeSpecified == FrameSizeType::CUSTOM ? "CustomSize" : "";
    savePointCloud(cv::format("pts_%s_%s_%s_%s_%s.obj", funcName.c_str(), platformString.c_str(), volumeTypeString.c_str(),
                              testSrcTypeString.c_str(), frameSizeSpecifiedString.c_str()),
                   pts3.reshape(3, 1), nrm3.reshape(3, 1));
}

void VolumeTestFixture::raycast_test()
{
    UMat upoints, unormals, ucolors;
    if (frameSizeSpecified == FrameSizeType::CUSTOM)
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[0].matrix, frameSize.height, frameSize.width, intrRaycast, upoints, unormals, ucolors);
        else
            volume->raycast(poses[0].matrix, frameSize.height, frameSize.width, intrRaycast, upoints, unormals);
    }
    else if (frameSizeSpecified == FrameSizeType::DEFAULT)
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[0].matrix, upoints, unormals, ucolors);
        else
            volume->raycast(poses[0].matrix, upoints, unormals);
    }

    Mat points, normals, colors;
    points  = upoints.getMat(ACCESS_READ);
    normals = unormals.getMat(ACCESS_READ);
    colors  = ucolors.getMat(ACCESS_READ);

    if (cvtest::debugLevel > 0)
    {
        if (volumeType == VolumeType::ColorTSDF)
            displayColorImage(depth, rgb, points, normals, colors, depthFactor, lightPose);
        else
            displayImage(depth, points, normals, depthFactor, lightPose);

        saveObj("raycast", points, normals);
    }

    normalsCheck(normals);
}


void VolumeTestFixture::fetch_normals_test()
{
    UMat upoints, unormals;
    volume->fetchPointsNormals(upoints, noArray());

    volume->fetchNormals(upoints, unormals);

    Mat points, normals;
    points  = upoints.getMat(ACCESS_READ);
    normals = unormals.getMat(ACCESS_READ);

    if (cvtest::debugLevel > 0)
    {
        saveObj("fetch_normals", points, normals);
    }

    normalsCheck(normals);
}


void VolumeTestFixture::fetch_points_normals_test()
{
    UMat upoints, unormals;
    volume->fetchPointsNormals(upoints, unormals);

    Mat points, normals;
    points  = upoints.getMat(ACCESS_READ);
    normals = unormals.getMat(ACCESS_READ);

    if (cvtest::debugLevel > 0)
    {
        saveObj("fetch_points_normals", points, normals);
    }

    normalsCheck(normals);
}


void VolumeTestFixture::valid_points_test()
{
    UMat upoints, unormals, ucolors;
    if (frameSizeSpecified == FrameSizeType::CUSTOM)
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[0].matrix, frameSize.height, frameSize.width, intrRaycast, upoints, unormals, ucolors);
        else
            volume->raycast(poses[0].matrix, frameSize.height, frameSize.width, intrRaycast, upoints, unormals);
    }
    else if (frameSizeSpecified == FrameSizeType::DEFAULT)
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[0].matrix, upoints, unormals, ucolors);
        else
            volume->raycast(poses[0].matrix, upoints, unormals);
    }

    Mat points, normals, colors;
    points = upoints.getMat(ACCESS_READ);
    normals = unormals.getMat(ACCESS_READ);
    colors = ucolors.getMat(ACCESS_READ);

    patchNaNs(points);
    int enface = counterOfValid(points);

    if (cvtest::debugLevel > 0)
    {
        if (volumeType == VolumeType::ColorTSDF)
            displayColorImage(depth, rgb, points, normals, colors, depthFactor, lightPose);
        else
            displayImage(depth, points, normals, depthFactor, lightPose);
    }

    UMat upoints2, unormals2, ucolors2;
    Mat points2, normals2, colors2;

    if (frameSizeSpecified == FrameSizeType::CUSTOM)
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[17].matrix, frameSize.height, frameSize.width, intrRaycast, upoints2, unormals2, ucolors2);
        else
            volume->raycast(poses[17].matrix, frameSize.height, frameSize.width, intrRaycast, upoints2, unormals2);
    }
    else
    {
        if (volumeType == VolumeType::ColorTSDF)
            volume->raycast(poses[17].matrix, upoints2, unormals2, ucolors2);
        else
            volume->raycast(poses[17].matrix, upoints2, unormals2);
    }

    points2 = upoints2.getMat(ACCESS_READ);
    normals2 = unormals2.getMat(ACCESS_READ);
    colors2 = ucolors2.getMat(ACCESS_READ);

    patchNaNs(points2);
    int profile = counterOfValid(points2);

    if (cvtest::debugLevel > 0)
    {
        if (volumeType == VolumeType::ColorTSDF)
            displayColorImage(depth, rgb, points2, normals2, colors2, depthFactor, lightPose);
        else
            displayImage(depth, points2, normals2, depthFactor, lightPose);
    }

    ASSERT_GT(profile, 0) << "There are no points in profile";
    ASSERT_GT(enface, 0) << "There are no points in enface";

    // TODO: why profile == 2*enface ?
    float percentValidity = float(enface) / float(profile) * 100;

    ASSERT_NEAR(percentValidity, 50, 6);
}

TEST_P(VolumeTestFixture, valid_points)
{
    valid_points_test();
}

TEST_P(VolumeTestFixture, raycast_normals)
{
    raycast_test();
}

//TODO: this test should run just 1 time, not 4
TEST_P(VolumeTestFixture, fetch_points_normals)
{
    fetch_points_normals_test();
}
//TODO: this test should run just 1 time, not 4
TEST_P(VolumeTestFixture, fetch_normals)
{
    fetch_normals_test();
}

//TODO: fix it when ColorTSDF gets GPU version
INSTANTIATE_TEST_CASE_P(Volume, VolumeTestFixture, /*::testing::Combine(PlatformTypeEnum::all(), VolumeTypeEnum::all())*/
                        ::testing::Combine(
                        ::testing::Values(PlatformVolumeType {PlatformType::CPU, VolumeType::TSDF},
                                          PlatformVolumeType {PlatformType::CPU, VolumeType::HashTSDF},
                                          PlatformVolumeType {PlatformType::CPU, VolumeType::ColorTSDF},
                                          PlatformVolumeType {PlatformType::GPU, VolumeType::TSDF},
                                          PlatformVolumeType {PlatformType::GPU, VolumeType::HashTSDF}),
                        VolumeTestSrcTypeEnum::all(), FrameSizeTypeEnum::all()));


class StaticVolumeBoundingBox : public ::testing::TestWithParam<PlatformVolumeType>
{ };

TEST_P(StaticVolumeBoundingBox, staticBoundingBox)
{
    auto p = GetParam();
    bool gpu = (std::get<0>(p) == PlatformType::GPU);
    VolumeType volumeType = std::get<1>(p);

    OpenCLStatusRevert oclStatus;
    if (!gpu)
        oclStatus.off();

    staticBoundingBoxTest(volumeType);
}

//TODO: edit this list when ColorTSDF gets GPU support
INSTANTIATE_TEST_CASE_P(Volume, StaticVolumeBoundingBox, ::testing::Values(
                        PlatformVolumeType {PlatformType::CPU, VolumeType::TSDF},
                        PlatformVolumeType {PlatformType::CPU, VolumeType::ColorTSDF},
                        PlatformVolumeType {PlatformType::GPU, VolumeType::TSDF}));


class ReproduceVolPoseRotTest : public ::testing::TestWithParam<PlatformTypeEnum>
{ };

TEST_P(ReproduceVolPoseRotTest, reproduce_volPoseRot)
{
    bool gpu = (GetParam() == PlatformType::GPU);

    OpenCLStatusRevert oclStatus;

    if (!gpu)
        oclStatus.off();

    regressionVolPoseRot();
}

INSTANTIATE_TEST_CASE_P(Volume, ReproduceVolPoseRotTest, PlatformTypeEnum::all());


enum Growth
{
    OFF = 0, ON = 1
};
CV_ENUM(GrowthEnum, Growth::OFF, Growth::ON);

class BoundingBoxEnableGrowthTest : public ::testing::TestWithParam<std::tuple<PlatformTypeEnum, GrowthEnum>>
{ };

TEST_P(BoundingBoxEnableGrowthTest, boundingBoxEnableGrowth)
{
    auto p = GetParam();
    bool gpu = (std::get<0>(p) == PlatformType::GPU);
    bool enableGrowth = (std::get<1>(p) == Growth::ON);

    OpenCLStatusRevert oclStatus;

    if (!gpu)
        oclStatus.off();

    boundingBoxGrowthTest(enableGrowth);
}

INSTANTIATE_TEST_CASE_P(Volume, BoundingBoxEnableGrowthTest, ::testing::Combine(PlatformTypeEnum::all(), GrowthEnum::all()));

}
}  // namespace
