// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

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
        Reprojector _reproj,
        float _depthFactor) : ParallelLoopBody(),
        frame(_frame),
        pose(_pose),
        reproj(_reproj),
        depthFactor(_depthFactor)
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
                    float d = Scene::map(p);
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
};

struct Scene
{
    virtual ~Scene() {}
    static Ptr<Scene> create(Size sz, Matx33f _intr, float _depthFactor);
    virtual Mat depth(Affine3f pose) = 0;
    virtual std::vector<Affine3f> getPoses() = 0;
};

struct SemisphereScene : Scene
{
    const int framesPerCycle = 72;
    const float nCycles = 0.25f;
    const Affine3f startPose = Affine3f(Vec3f(0.f, 0.f, 0.f), Vec3f(1.5f, 0.3f, -2.3f));

    Size frameSize;
    Matx33f intr;
    float depthFactor;

    SemisphereScene(Size sz, Matx33f _intr, float _depthFactor) :
        frameSize(sz), intr(_intr), depthFactor(_depthFactor)
    { }

    static float map(Point3f p)
    {
        float plane = p.y + 0.5f;

        Point3f boxPose = p - Point3f(-0.0f, 0.3f, 0.5f);
        float boxSize = 0.5f;
        float roundness = 0.08f;
        Point3f boxTmp;
        boxTmp.x = max(abs(boxPose.x) - boxSize, 0.0f);
        boxTmp.y = max(abs(boxPose.y) - boxSize, 0.0f);
        boxTmp.z = max(abs(boxPose.z) - boxSize, 0.0f);
        float roundBox = (float)cv::norm(boxTmp) - roundness;

        Point3f spherePose = p - Point3f(-0.0f, 0.3f, 0.0f);
        float sphereRadius = 0.5f;
        float sphere = (float)cv::norm(spherePose) - sphereRadius;
        float sphereMinusBox = max(sphere, -roundBox);

        float subSphereRadius = 0.05f;
        Point3f subSpherePose = p - Point3f(0.3f, -0.1f, -0.3f);
        float subSphere = (float)cv::norm(subSpherePose) - subSphereRadius;

        float res = min({sphereMinusBox, subSphere, plane});
        return res;
    }

    Mat depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<SemisphereScene>(frame, pose, reproj, depthFactor));

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
            pose = pose.rotate(Vec3f(0.f, -1.f, 0.f) * angle);
            pose = pose.translate(Vec3f(startPose.translation()[0] * sin(angle),
                startPose.translation()[1],
                startPose.translation()[2] * cos(angle)));
            poses.push_back(pose);
        }

        return poses;
    }

};

Ptr<Scene> Scene::create(Size sz, Matx33f _intr, float _depthFactor)
{
    return makePtr<SemisphereScene>(sz, _intr, _depthFactor);
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

                    if (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z) )
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
            ASSERT_LT(abs(1 - length), 0.0001f);
        }
    }
}

void normal_test(bool isHashTSDF, bool isRaycast, bool isFetchPointsNormals, bool isFetchNormals)
{
    Ptr<kinfu::Params> _params;
    if (isHashTSDF)
        _params = kinfu::Params::hashTSDFParams(true);
    else
        _params = kinfu::Params::coarseParams();

    Ptr<Scene> scene = Scene::create(_params->frameSize, _params->intr, _params->depthFactor);
    std::vector<Affine3f> poses = scene->getPoses();

    Mat depth = scene->depth(poses[0]);
    UMat _points, _normals, _tmpnormals;
    UMat _newPoints, _newNormals;
    Mat  points,  normals;
    Mat image;
    AccessFlag af = ACCESS_READ;

    auto normalCheck = [](Vec4f& vector, const int*)
    {
        if (!cvIsNaN(vector[0]))
        {
            float length = vector[0] * vector[0] +
                vector[1] * vector[1] +
                vector[2] * vector[2];
            ASSERT_LT(abs(1 - length), 0.0001f);
        }
    };

    Ptr<kinfu::Volume> volume = kinfu::makeVolume(_params->volumeType, _params->voxelSize, _params->volumePose.matrix,
                                _params->raycast_step_factor, _params->tsdf_trunc_dist, _params->tsdf_max_weight,
                                _params->truncateThreshold, _params->volumeDims);
    volume->integrate(depth, _params->depthFactor, poses[0].matrix, _params->intr);

    if (isRaycast)
    {
        volume->raycast(poses[0].matrix, _params->intr, _params->frameSize, _points, _normals);
    }
    if (isFetchPointsNormals)
    {
        volume->fetchPointsNormals(_points, _normals);
    }
    if (isFetchNormals)
    {
        volume->fetchPointsNormals(_points, _tmpnormals);
        volume->fetchNormals(_points, _normals);
    }

    normals = _normals.getMat(af);

    if (parallelCheck)
    {
        normals.forEach<Vec4f>(normalCheck);
    }
    else
    {
        normalsCheck(normals);
    }

    if (isRaycast && display)
    {
        imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
        points = _points.getMat(af);
        renderPointsNormals(points, normals, image, _params->lightPose);
        imshow("render", image);
        waitKey(20000);
    }

    if (isRaycast)
    {
        volume->raycast(poses[17].matrix, _params->intr, _params->frameSize, _newPoints, _newNormals);

        normals = _newNormals.getMat(af);
        normalsCheck(normals);

        if (parallelCheck)
        {
            normals.forEach<Vec4f>(normalCheck);
        }
        else
        {
            normalsCheck(normals);
        }


        if (display)
        {
            imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
            points = _newPoints.getMat(af);
            renderPointsNormals(points, normals, image, _params->lightPose);
            imshow("render", image);
            waitKey(20000);
        }

    }

    points.release(); normals.release();
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

void valid_points_test(bool isHashTSDF)
{
    Ptr<kinfu::Params> _params;
    if (isHashTSDF)
        _params = kinfu::Params::hashTSDFParams(true);
    else
        _params = kinfu::Params::coarseParams();

    Ptr<Scene> scene = Scene::create(_params->frameSize, _params->intr, _params->depthFactor);
    std::vector<Affine3f> poses = scene->getPoses();

    Mat depth = scene->depth(poses[0]);
    UMat _points, _normals;
    UMat _newPoints, _newNormals;
    Mat  points, normals;
    Mat image;
    int anfas, profile;
    AccessFlag af = ACCESS_READ;

    Ptr<kinfu::Volume> volume = kinfu::makeVolume(_params->volumeType, _params->voxelSize, _params->volumePose.matrix,
        _params->raycast_step_factor, _params->tsdf_trunc_dist, _params->tsdf_max_weight,
        _params->truncateThreshold, _params->volumeDims);
    volume->integrate(depth, _params->depthFactor, poses[0].matrix, _params->intr);

    volume->raycast(poses[0].matrix, _params->intr, _params->frameSize, _points, _normals);
    normals = _normals.getMat(af);
    points = _points.getMat(af);
    patchNaNs(points);
    anfas = counterOfValid(points);

    if (display)
    {
        imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
        renderPointsNormals(points, normals, image, _params->lightPose);
        imshow("render", image);
        waitKey(20000);
    }

    volume->raycast(poses[17].matrix, _params->intr, _params->frameSize, _newPoints, _newNormals);

    normals = _newNormals.getMat(af);
    points = _newPoints.getMat(af);
    patchNaNs(points);
    profile = counterOfValid(points);

    if (display)
    {
        imshow("depth", depth * (1.f / _params->depthFactor / 4.f));
        renderPointsNormals(points, normals, image, _params->lightPose);
        imshow("render", image);
        waitKey(20000);
    }

    float percentValidity = float(profile) / float(anfas);
    ASSERT_LT(0.5 - percentValidity, 0.3);
}

TEST(TSDF, raycast_normals)
{
    normal_test(false, true, false, false);
}

TEST(HashTSDF, raycast_normals)
{
    normal_test(true, true, false, false);
}

TEST(TSDF, fetch_points_normals)
{
    normal_test(false, false, true, false);
}

TEST(HashTSDF, fetch_points_normals)
{
    normal_test(true, false, true, false);
}

TEST(TSDF, fetch_normals)
{
    normal_test(false, false, false, true);
}

TEST(HashTSDF, fetch_normals)
{
    normal_test(true, false, false, true);
}

TEST(TSDF, valid_points)
{
    valid_points_test(false);
}

TEST(HashTSDF, valid_points)
{
    valid_points_test(true);
}

}}  // namespace
