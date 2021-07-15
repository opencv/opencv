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
typedef int maskType;
typedef cv::Mat_< maskType > Mask;

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

void renderPointsNormals(InputArray _points, InputArray _normals, InputArray _mask, OutputArray image, Affine3f lightPose)
{
    Size sz = _points.size();
    image.create(sz, CV_8UC4);

    Points  points = _points.getMat();
    Normals normals = _normals.getMat();
    Mask  mask = _mask.getMat();

    Mat_<Vec4b> img = image.getMat();

    Range range(0, sz.height);
    const int nstripes = -1;
    auto render = [&](const Range&)
    {
        for (int y = range.start; y < range.end; y++)
        {
            Vec4b* imgRow = img[y];
            const ptype* ptsRow = points[y];
            const ptype* nrmRow = normals[y];
            const maskType* maskRow = mask[y];

            for (int x = 0; x < sz.width; x++)
            {
                Point3f p = fromPtype(ptsRow[x]);
                Point3f n = fromPtype(nrmRow[x]);

                Vec4b color;

                //if (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z))
                if (maskRow[x] == 0)
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
    };
    parallel_for_(range, render, nstripes);
    //render(range);
}
// ----------------------------

class Settings
{
public:
    float depthFactor;
    Matx33f intr;
    Size frameSize;
    Vec3f lightPose;

    Ptr<Volume> volume;
    Ptr<Scene> scene;
    std::vector<Affine3f> poses;

    Settings(bool useHashTSDF)
    {
        frameSize = Size(640, 480);

        float fx, fy, cx, cy;
        fx = fy = 525.f;
        cx = frameSize.width / 2 - 0.5f;
        cy = frameSize.height / 2 - 0.5f;
        intr = Matx33f(fx,  0, cx,
                        0, fy, cy,
                        0,  0,  1);

        // 5000 for the 16-bit PNG files
        // 1 for the 32-bit float images in the ROS bag files
        depthFactor = 5000;

        Vec3i volumeDims = Vec3i::all(512); //number of voxels

        float volSize = 3.f;
        float voxelSize = volSize / 512.f; //meters

        // default pose of volume cube
        Affine3f volumePose = Affine3f().translate(Vec3f(-volSize / 2.f, -volSize / 2.f, 0.5f));
        float tsdf_trunc_dist = 7 * voxelSize; // about 0.04f in meters
        int tsdf_max_weight = 64;   //frames

        float raycast_step_factor = 0.25f;  //in voxel sizes
        // gradient delta factor is fixed at 1.0f and is not used
        //p.gradient_delta_factor = 0.5f; //in voxel sizes

        //p.lightPose = p.volume_pose.translation()/4; //meters
        lightPose = Vec3f::all(0.f); //meters

        // depth truncation is not used by default but can be useful in some scenes
        float truncateThreshold = 0.f; //meters

        VolumeParams::VolumeKind volumeKind = VolumeParams::VolumeKind::TSDF;

        if (useHashTSDF)
        {
            volumeKind = VolumeParams::VolumeKind::HASHTSDF;
            truncateThreshold = Odometry::DEFAULT_MAX_DEPTH();
        }
        else
        {
            volSize = 3.f;
            volumeDims = Vec3i::all(128); //number of voxels
            voxelSize = volSize / 128.f;
            tsdf_trunc_dist = 2 * voxelSize; // 0.04f in meters

            raycast_step_factor = 0.75f;  //in voxel sizes
        }

        volume = makeVolume(volumeKind, voxelSize, volumePose.matrix,
                            raycast_step_factor, tsdf_trunc_dist, tsdf_max_weight,
                            truncateThreshold, volumeDims[0], volumeDims[1], volumeDims[2]);

        scene = Scene::create(frameSize, intr, depthFactor, true);
        poses = scene->getPoses();
    }
};

void displayImage(Mat depth, UMat _points, UMat _normals, UMat _pointsMask, float depthFactor, Vec3f lightPose)
{
    Mat  points, normals, pointsMask, image;
    AccessFlag af = ACCESS_READ;
    normals = _normals.getMat(af);
    points = _points.getMat(af);
    pointsMask = _pointsMask.getMat(af);
    patchNaNs(points);

    imshow("depth", depth * (1.f / depthFactor / 4.f));
    renderPointsNormals(points, normals, pointsMask, image, lightPose);
    imshow("render", image);
    waitKey(2000);
}

static const bool display = false;

PERF_TEST(Perf_TSDF, integrate)
{
    Settings settings(false);
    for (size_t i = 0; i < settings.poses.size(); i++)
    {
        Matx44f pose = settings.poses[i].matrix;
        Mat depth = settings.scene->depth(pose);
        Mat mask(depth.size(), CV_32S, Scalar(255));
        for (int y = 0; y < depth.rows; y++)
            for (int x = 0; x < depth.cols; x++)
                if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > 10 || depth.at<float>(y, x) <= FLT_EPSILON)
                    mask.at<int>(y, x) = 0;
        startTimer();
        settings.volume->integrate(depth, mask, settings.depthFactor, pose, settings.intr);
        stopTimer();
        depth.release();
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Perf_TSDF, raycast)
{
    Settings settings(false);
    for (size_t i = 0; i < settings.poses.size(); i++)
    {
        UMat _points, _normals, _pointsMask;
        Matx44f pose = settings.poses[i].matrix;
        Mat depth = settings.scene->depth(pose);
        Mat mask(depth.size(), CV_32S, Scalar(255));
        for (int y = 0; y < depth.rows; y++)
            for (int x = 0; x < depth.cols; x++)
                if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > 10 || depth.at<float>(y, x) <= FLT_EPSILON)
                    mask.at<int>(y, x) = 0;

        settings.volume->integrate(depth, mask, settings.depthFactor, pose, settings.intr);
        startTimer();
        settings.volume->raycast(pose, settings.intr, settings.frameSize, _points, _normals, _pointsMask);
        stopTimer();

        if (display)
            displayImage(depth, _points, _normals, _pointsMask, settings.depthFactor, settings.lightPose);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Perf_HashTSDF, integrate)
{
    Settings settings(true);

    for (size_t i = 0; i < settings.poses.size(); i++)
    {
        Matx44f pose = settings.poses[i].matrix;
        Mat depth = settings.scene->depth(pose);
        Mat mask(depth.size(), CV_32S, Scalar(255));
        for (int y = 0; y < depth.rows; y++)
            for (int x = 0; x < depth.cols; x++)
                if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > 10 || depth.at<float>(y, x) <= FLT_EPSILON)
                    mask.at<int>(y, x) = 0;

        startTimer();
        settings.volume->integrate(depth, mask, settings.depthFactor, pose, settings.intr);
        stopTimer();
        depth.release();
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Perf_HashTSDF, raycast)
{
    Settings settings(true);
    for (size_t i = 0; i < settings.poses.size(); i++)
    {
        UMat _points, _normals, _pointsMask;
        Matx44f pose = settings.poses[i].matrix;
        Mat depth = settings.scene->depth(pose);
        Mat mask(depth.size(), CV_32S, Scalar(255));
        for (int y = 0; y < depth.rows; y++)
            for (int x = 0; x < depth.cols; x++)
                if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > 10 || depth.at<float>(y, x) <= FLT_EPSILON)
                    mask.at<int>(y, x) = 0;

        settings.volume->integrate(depth, mask, settings.depthFactor, pose, settings.intr);
        startTimer();
        settings.volume->raycast(pose, settings.intr, settings.frameSize, _points, _normals, _pointsMask);
        stopTimer();

        if (display)
            displayImage(depth, _points, _normals, _pointsMask, settings.depthFactor, settings.lightPose);
    }
    SANITY_CHECK_NOTHING();
}

}} // namespace
