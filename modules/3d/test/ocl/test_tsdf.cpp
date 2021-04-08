// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

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

struct Scene
{
    virtual ~Scene() {}
    static Ptr<Scene> create(Size sz, Matx33f _intr, float _depthFactor, bool onlySemisphere);
    virtual Mat depth(Affine3f pose) = 0;
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

    Mat depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<SemisphereScene>(frame, pose, reproj, depthFactor, onlySemisphere));

        return std::move(frame);
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

    Range range(0, sz.height);
    const int nstripes = -1;
    parallel_for_(range, [&](const Range&)
        {
            for (int y = range.start; y < range.end; y++)
            {
                Vec4b* imgRow = img[y];
                const ptype* ptsRow = points[y];
                const ptype* nrmRow = normals[y];

                for (int x = 0; x < sz.width; x++)
                {
                    Point3f p = fromPtype(ptsRow[x]);
                    Point3f n = fromPtype(nrmRow[x]);

                    Vec4b color;

                    if (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z))
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
// ----------------------------

static const bool display = false;
static const bool parallelCheck = false;

class Settings
{
public:
    Ptr<kinfu::Params> params;
    Ptr<kinfu::Volume> volume;
    Ptr<Scene> scene;
    std::vector<Affine3f> poses;

    Settings(bool useHashTSDF, bool onlySemisphere)
    {
        if (useHashTSDF)
            params = kinfu::Params::hashTSDFParams(true);
        else
            params = kinfu::Params::coarseParams();

        volume = kinfu::makeVolume(params->volumeType, params->voxelSize, params->volumePose.matrix,
            params->raycast_step_factor, params->tsdf_trunc_dist, params->tsdf_max_weight,
            params->truncateThreshold, params->volumeDims);

        scene = Scene::create(params->frameSize, params->intr, params->depthFactor, onlySemisphere);
        poses = scene->getPoses();
    }
};

void displayImage(Mat depth, Mat points, Mat normals, float depthFactor, Vec3f lightPose)
{
    Mat image;
    patchNaNs(points);
    imshow("depth", depth * (1.f / depthFactor / 4.f));
    renderPointsNormals(points, normals, image, lightPose);
    imshow("render", image);
    waitKey(2000);
}

void normalsCheck(Mat normals)
{
    Vec4f vector;
    for (auto pvector = normals.begin<Vec4f>(); pvector < normals.end<Vec4f>(); pvector++)
    {
        vector = *pvector;
        if (!cvIsNaN(vector[0]))
        {
            float length = vector[0] * vector[0] +
                vector[1] * vector[1] +
                vector[2] * vector[2];
            ASSERT_LT(abs(1 - length), 0.0001f) << "There is normal with length != 1";
        }
    }
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

void normal_test(bool isHashTSDF, bool isRaycast, bool isFetchPointsNormals, bool isFetchNormals)
{
    auto normalCheck = [](Vec4f& vector, const int*)
    {
        if (!cvIsNaN(vector[0]))
        {
            float length = vector[0] * vector[0] +
                vector[1] * vector[1] +
                vector[2] * vector[2];
            ASSERT_LT(abs(1 - length), 0.0001f) << "There is normal with length != 1";
        }
    };

    Settings settings(isHashTSDF, false);

    Mat depth = settings.scene->depth(settings.poses[0]);
    UMat _points, _normals, _tmpnormals;
    UMat _newPoints, _newNormals;
    Mat  points, normals;
    AccessFlag af = ACCESS_READ;

    settings.volume->integrate(depth, settings.params->depthFactor, settings.poses[0].matrix, settings.params->intr);

    if (isRaycast)
    {
        settings.volume->raycast(settings.poses[0].matrix, settings.params->intr, settings.params->frameSize, _points, _normals);
    }
    if (isFetchPointsNormals)
    {
        settings.volume->fetchPointsNormals(_points, _normals);
    }
    if (isFetchNormals)
    {
        settings.volume->fetchPointsNormals(_points, _tmpnormals);
        settings.volume->fetchNormals(_points, _normals);
    }

    normals = _normals.getMat(af);
    points = _points.getMat(af);

    if (parallelCheck)
        normals.forEach<Vec4f>(normalCheck);
    else
        normalsCheck(normals);

    if (isRaycast && display)
        displayImage(depth, points, normals, settings.params->depthFactor, settings.params->lightPose);

    if (isRaycast)
    {
        settings.volume->raycast(settings.poses[17].matrix, settings.params->intr, settings.params->frameSize, _newPoints, _newNormals);
        normals = _newNormals.getMat(af);
        points = _newPoints.getMat(af);
        normalsCheck(normals);

        if (parallelCheck)
            normals.forEach<Vec4f>(normalCheck);
        else
            normalsCheck(normals);

        if (display)
            displayImage(depth, points, normals, settings.params->depthFactor, settings.params->lightPose);
    }

    points.release(); normals.release();
}

void valid_points_test(bool isHashTSDF)
{
    Settings settings(isHashTSDF, true);

    Mat depth = settings.scene->depth(settings.poses[0]);
    UMat _points, _normals, _newPoints, _newNormals;
    AccessFlag af = ACCESS_READ;
    Mat  points, normals;
    int anfas, profile;

    settings.volume->integrate(depth, settings.params->depthFactor, settings.poses[0].matrix, settings.params->intr);
    settings.volume->raycast(settings.poses[0].matrix, settings.params->intr, settings.params->frameSize, _points, _normals);
    normals = _normals.getMat(af);
    points = _points.getMat(af);
    patchNaNs(points);
    anfas = counterOfValid(points);

    if (display)
        displayImage(depth, points, normals, settings.params->depthFactor, settings.params->lightPose);

    settings.volume->raycast(settings.poses[17].matrix, settings.params->intr, settings.params->frameSize, _newPoints, _newNormals);
    normals = _newNormals.getMat(af);
    points = _newPoints.getMat(af);
    patchNaNs(points);
    profile = counterOfValid(points);

    if (display)
        displayImage(depth, points, normals, settings.params->depthFactor, settings.params->lightPose);

    // TODO: why profile == 2*anfas ?
    float percentValidity = float(anfas) / float(profile);

    ASSERT_NE(profile, 0) << "There is no points in profile";
    ASSERT_NE(anfas, 0) << "There is no points in anfas";
    ASSERT_LT(abs(0.5 - percentValidity), 0.3) << "percentValidity out of [0.3; 0.7] (percentValidity=" << percentValidity << ")";
}

TEST(TSDF_GPU, raycast_normals) { normal_test(false, true, false, false); }
TEST(TSDF_GPU, fetch_points_normals) { normal_test(false, false, true, false); }
TEST(TSDF_GPU, fetch_normals) { normal_test(false, false, false, true); }
TEST(TSDF_GPU, valid_points) { valid_points_test(false); }

TEST(HashTSDF_GPU, raycast_normals) { normal_test(true, true, false, false); }
TEST(HashTSDF_GPU, fetch_points_normals) { normal_test(true, false, true, false); }
TEST(HashTSDF_GPU, fetch_normals) { normal_test(true, false, false, true); }
TEST(HashTSDF_GPU, valid_points) { valid_points_test(true); }

}
}  // namespace

#endif
