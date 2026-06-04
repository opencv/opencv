// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

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
                Point3f dir = normalize(Vec3f(pose.rotation() * screenVec));
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
                Point3f dir = normalize(Vec3f(pose.rotation() * screenVec));
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

inline Point3f normalize(const Vec3f& v)
{
    double nv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return v * (nv ? 1. / nv : 0.);
}

void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, Affine3f lightPose)
{
    Size sz = _points.size();
    image.create(sz, CV_8UC4);

    Points  points = _points.getMat();
    Normals normals = _normals.getMat();

    Mat_<Vec4b> img = image.getMat();

    Mat goods;
    finiteMask(points, goods);

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

                    if ( !goodRow[x] )
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

                        Point3f l = normalize(lightPose.translation() - Vec3f(p));
                        Point3f v = normalize(-Vec3f(p));
                        Point3f r = normalize(Vec3f(2.f * n * n.dot(l) - l));

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

    Points  points = _points.getMat();
    Colors  colors = _colors.getMat();

    Mat goods, goodp, goodc;
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

                    if ( !goodRow[x] )
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
    waitKey(100);
}

void displayColorImage(Mat depth, Mat rgb, Mat points, Mat normals, Mat colors, float depthFactor, Vec3f lightPose)
{
    Mat image;
    patchNaNs(points);
    imshow("depth", depth * (1.f / depthFactor / 4.f));
    imshow("rgb", rgb * (1.f / 255.f));
    renderPointsNormalsColors(points, normals, colors, image, lightPose);
    imshow("render", image);
    waitKey(100);
}

static const bool display = false;

enum PlatformType
{
    CPU = 0, GPU = 1
};
CV_ENUM(PlatformTypeEnum, PlatformType::CPU, PlatformType::GPU);

enum Sequence
{
    ALL = 0, FIRST = 1
};
CV_ENUM(SequenceEnum, Sequence::ALL, Sequence::FIRST);

enum class VolumeTestSrcType
{
    MAT = 0,
    ODOMETRY_FRAME = 1
};

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
}

typedef std::tuple<PlatformTypeEnum, VolumeTypeEnum> PlatformVolumeType;
class VolumePerfFixture : public perf::TestBaseWithParam<std::tuple<PlatformVolumeType, VolumeTestSrcTypeEnum, SequenceEnum>>
{
protected:
    void SetUp() override
    {
        TestBase::SetUp();

        auto p = GetParam();
        gpu = (std::get<0>(std::get<0>(p)) == PlatformType::GPU);
        volumeType = std::get<1>(std::get<0>(p));

        testSrcType = std::get<1>(p);

        repeat1st = (std::get<2>(p) == Sequence::FIRST);

        if (!gpu)
            oclStatus.off();

        VolumeSettings vs(volumeType);
        volume = makePtr<Volume>(volumeType, vs);

        frameSize = Size(vs.getRaycastWidth(), vs.getRaycastHeight());
        Matx33f intrIntegrate;
        vs.getCameraIntegrateIntrinsics(intrIntegrate);
        vs.getCameraRaycastIntrinsics(intrRaycast);
        bool onlySemisphere = false;
        depthFactor = vs.getDepthFactor();
        scene = Scene::create(frameSize, intrIntegrate, depthFactor, onlySemisphere);
        poses = scene->getPoses();
    }

    bool gpu;
    VolumeType volumeType;
    VolumeTestSrcType testSrcType;
    bool repeat1st;

    OpenCLStatusRevert oclStatus;

    Ptr<Volume> volume;
    Size frameSize;
    Matx33f intrRaycast;
    Ptr<Scene> scene;
    std::vector<Affine3f> poses;
    float depthFactor;
};


PERF_TEST_P_(VolumePerfFixture, integrate)
{
    for (size_t i = 0; i < (repeat1st ? 1 : poses.size()); i++)
    {
        Matx44f pose = poses[i].matrix;
        Mat depth = scene->depth(pose);
        Mat rgb = scene->rgb(pose);
        UMat urgb, udepth;
        depth.copyTo(udepth);
        rgb.copyTo(urgb);
        OdometryFrame odf(udepth, urgb);

        bool done = false;
        while (repeat1st ? next() : !done)
        {
            startTimer();
            if (testSrcType == VolumeTestSrcType::MAT)
                if (volumeType == VolumeType::ColorTSDF)
                    volume->integrate(udepth, urgb, pose);
                else
                    volume->integrate(udepth, pose);
            else if (testSrcType == VolumeTestSrcType::ODOMETRY_FRAME)
                volume->integrate(odf, pose);
            stopTimer();

            // perf check makes sense only for identical states
            if (repeat1st)
                volume->reset();

            done = true;
        }
    }
    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(VolumePerfFixture, raycast)
{
    for (size_t i = 0; i < (repeat1st ? 1 : poses.size()); i++)
    {
        Matx44f pose = poses[i].matrix;
        Mat depth = scene->depth(pose);
        Mat rgb = scene->rgb(pose);
        UMat urgb, udepth;
        depth.copyTo(udepth);
        rgb.copyTo(urgb);

        OdometryFrame odf(udepth, urgb);

        if (testSrcType == VolumeTestSrcType::MAT)
            if (volumeType == VolumeType::ColorTSDF)
                volume->integrate(udepth, urgb, pose);
            else
                volume->integrate(udepth, pose);
        else if (testSrcType == VolumeTestSrcType::ODOMETRY_FRAME)
            volume->integrate(odf, pose);

        UMat upoints, unormals, ucolors;

        bool done = false;
        while (repeat1st ? next() : !done)
        {
            startTimer();
            if (volumeType == VolumeType::ColorTSDF)
                volume->raycast(pose, frameSize.height, frameSize.width, intrRaycast, upoints, unormals, ucolors);
            else
                volume->raycast(pose, frameSize.height, frameSize.width, intrRaycast, upoints, unormals);
            stopTimer();

            done = true;
        }

        if (display)
        {
            Mat points, normals, colors;
            points = upoints.getMat(ACCESS_READ);
            normals = unormals.getMat(ACCESS_READ);
            colors = ucolors.getMat(ACCESS_READ);

            Vec3f lightPose = Vec3f::all(0.f);
            if (volumeType == VolumeType::ColorTSDF)
                displayColorImage(depth, rgb, points, normals, colors, depthFactor, lightPose);
            else
                displayImage(depth, points, normals, depthFactor, lightPose);
        }
    }
    SANITY_CHECK_NOTHING();
}

//TODO: fix it when ColorTSDF gets GPU version
INSTANTIATE_TEST_CASE_P(Volume, VolumePerfFixture, /*::testing::Combine(PlatformTypeEnum::all(), VolumeTypeEnum::all())*/
                        ::testing::Combine(
                        ::testing::Values(PlatformVolumeType {PlatformType::CPU, VolumeType::TSDF},
                                          PlatformVolumeType {PlatformType::CPU, VolumeType::HashTSDF},
                                          PlatformVolumeType {PlatformType::CPU, VolumeType::ColorTSDF},
                                          PlatformVolumeType {PlatformType::GPU, VolumeType::TSDF},
                                          PlatformVolumeType {PlatformType::GPU, VolumeType::HashTSDF}),
                        VolumeTestSrcTypeEnum::all(), SequenceEnum::all()));

}} // namespace
