// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "test_precomp.hpp"

// Inspired by Inigo Quilez' raymarching guide:
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

namespace opencv_test { namespace {

using namespace cv;

/** Reprojects screen point to camera space given z coord. */
struct Reprojector
{
    Reprojector() {}
    inline Reprojector(Matx33f intr)
    {
        fxinv = 1.f/intr(0, 0), fyinv = 1.f/intr(1, 1);
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
        for(int y = r.start; y < r.end; y++)
        {
            float* frameRow = frame[y];
            for(int x = 0; x < frame.cols; x++)
            {
                float pix = 0;

                Point3f orig = pose.translation();
                // direction through pixel
                Point3f screenVec = reproj(Point3f((float)x, (float)y, 1.f));
                float xyt = 1.f/(screenVec.x*screenVec.x +
                                 screenVec.y*screenVec.y + 1.f);
                Point3f dir = normalize(Vec3f(pose.rotation() * screenVec));
                // screen space axis
                dir.y = - dir.y;

                const float maxDepth = 20.f;
                const float maxSteps = 256;
                float t = 0.f;
                for(int step = 0; step < maxSteps && t < maxDepth; step++)
                {
                    Point3f p = orig + dir*t;
                    float d = Scene::map(p);
                    if(d < 0.000001f)
                    {
                        float depth = std::sqrt(t*t*xyt);
                        pix = depth*depthFactor;
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
    static Ptr<Scene> create(int nScene, Size sz, Matx33f _intr, float _depthFactor);
    virtual Mat depth(Affine3f pose) = 0;
    virtual std::vector<Affine3f> getPoses() = 0;
};

struct CubeSpheresScene : Scene
{
    const int framesPerCycle = 32;
    const float nCycles = 0.25f;
    const Affine3f startPose = Affine3f(Vec3f(-0.5f, 0.f, 0.f), Vec3f(2.1f, 1.4f, -2.1f));

    CubeSpheresScene(Size sz, Matx33f _intr, float _depthFactor) :
        frameSize(sz), intr(_intr), depthFactor(_depthFactor)
    { }

    static float map(Point3f p)
    {
        float plane = p.y + 0.5f;

        Point3f boxPose = p - Point3f(-0.0f, 0.3f, 0.0f);
        float boxSize = 0.5f;
        float roundness = 0.08f;
        Point3f boxTmp;
        boxTmp.x = max(abs(boxPose.x) - boxSize, 0.0f);
        boxTmp.y = max(abs(boxPose.y) - boxSize, 0.0f);
        boxTmp.z = max(abs(boxPose.z) - boxSize, 0.0f);
        float roundBox = (float)cv::norm(boxTmp) - roundness;

        float sphereRadius = 0.7f;
        float sphere = (float)cv::norm(boxPose) - sphereRadius;

        float boxMinusSphere = max(roundBox, -sphere);

        float sphere2 = (float)cv::norm(p - Point3f(0.3f, 1.f, 0.f)) - 0.1f;
        float sphere3 = (float)cv::norm(p - Point3f(0.0f, 1.f, 0.f)) - 0.2f;
        float res = min(min(plane, boxMinusSphere), min(sphere2, sphere3));

        return res;
    }

    Mat depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<CubeSpheresScene>(frame, pose, reproj, depthFactor));

        return std::move(frame);
    }

    std::vector<Affine3f> getPoses() override
    {
        std::vector<Affine3f> poses;
        for(int i = 0; i < (int)(framesPerCycle*nCycles); i++)
        {
            float angle = (float)(CV_2PI*i/framesPerCycle);
            Affine3f pose;
            pose = pose.rotate(startPose.rotation());
            pose = pose.rotate(Vec3f(0.f, -1.f, 0.f)*angle);
            pose = pose.translate(Vec3f(startPose.translation()[0]*sin(angle),
                                        startPose.translation()[1],
                                        startPose.translation()[2]*cos(angle)));
            poses.push_back(pose);
        }

        return poses;
    }

    Size frameSize;
    Matx33f intr;
    float depthFactor;
};


struct RotatingScene : Scene
{
    const int framesPerCycle = 32;
    const float nCycles = 0.5f;
    const Affine3f startPose = Affine3f(Vec3f(-1.f, 0.f, 0.f), Vec3f(1.5f, 2.f, -1.5f));

    RotatingScene(Size sz, Matx33f _intr, float _depthFactor) :
        frameSize(sz), intr(_intr), depthFactor(_depthFactor)
    {
        cv::RNG rng(0);
        rng.fill(randTexture, cv::RNG::UNIFORM, 0.f, 1.f);
    }

    static float noise(Point2f pt)
    {
        pt.x = abs(pt.x - (int)pt.x);
        pt.y = abs(pt.y - (int)pt.y);
        pt *= 256.f;

        int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

        const float* row0 = randTexture[(yi+0)%256];
        const float* row1 = randTexture[(yi+1)%256];

        float v00 = row0[(xi+0)%256];
        float v01 = row0[(xi+1)%256];
        float v10 = row1[(xi+0)%256];
        float v11 = row1[(xi+1)%256];

        float tx = pt.x - xi, ty = pt.y - yi;
        float v0 = v00 + tx*(v01 - v00);
        float v1 = v10 + tx*(v11 - v10);
        return v0 + ty*(v1 - v0);
    }

    static float map(Point3f p)
    {
        const Point3f torPlace(0.f, 0.f, 0.f);
        Point3f torPos(p - torPlace);
        const Point2f torusParams(1.f, 0.2f);
        Point2f torq(std::sqrt(torPos.x*torPos.x + torPos.z*torPos.z) - torusParams.x, torPos.y);
        float torus = (float)cv::norm(torq) - torusParams.y;

        const Point3f cylShift(0.25f, 0.25f, 0.25f);

        Point3f cylPos = Point3f(abs(std::fmod(p.x-0.1f, cylShift.x)),
                                 p.y,
                                 abs(std::fmod(p.z-0.2f, cylShift.z)))  - cylShift*0.5f;

        const Point2f cylParams(0.1f,
                                0.1f+0.1f*sin(p.x*p.y*5.f /* +std::log(1.f+abs(p.x*0.1f)) */));
        Point2f cyld = Point2f(abs(std::sqrt(cylPos.x*cylPos.x + cylPos.z*cylPos.z)), abs(cylPos.y)) - cylParams;
        float pins = min(max(cyld.x, cyld.y), 0.0f) + (float)cv::norm(Point2f(max(cyld.x, 0.f), max(cyld.y, 0.f)));

        float terrain = p.y + 0.25f*noise(Point2f(p.x, p.z)*0.01f);

        float res = min(terrain, max(-pins, torus));

        return res;
    }

    Mat depth(Affine3f pose) override
    {
        Mat_<float> frame(frameSize);
        Reprojector reproj(intr);

        Range range(0, frame.rows);
        parallel_for_(range, RenderInvoker<RotatingScene>(frame, pose, reproj, depthFactor));

        return std::move(frame);
    }

    std::vector<Affine3f> getPoses() override
    {
        std::vector<Affine3f> poses;
        for(int i = 0; i < framesPerCycle*nCycles; i++)
        {
            float angle = (float)(CV_2PI*i/framesPerCycle);
            Affine3f pose;
            pose = pose.rotate(startPose.rotation());
            pose = pose.rotate(Vec3f(0.f, -1.f, 0.f)*angle);
            pose = pose.translate(Vec3f(startPose.translation()[0]*sin(angle),
                                        startPose.translation()[1],
                                        startPose.translation()[2]*cos(angle)));
            poses.push_back(pose);
        }

        return poses;
    }

    Size frameSize;
    Matx33f intr;
    float depthFactor;
    static cv::Mat_<float> randTexture;
};

Mat_<float> RotatingScene::randTexture(256, 256);

Ptr<Scene> Scene::create(int nScene, Size sz, Matx33f _intr, float _depthFactor)
{
    if(nScene == 0)
        return makePtr<RotatingScene>(sz, _intr, _depthFactor);
    else
        return makePtr<CubeSpheresScene>(sz, _intr, _depthFactor);
}

static const bool display = false;

void flyTest(bool hiDense, bool inequal, bool hashTsdf = false)
{
    Ptr<kinfu::Params> params;
    if(hiDense)
        params = kinfu::Params::defaultParams();
    else
        params = kinfu::Params::coarseParams();

    if(hashTsdf)
        params = kinfu::Params::hashTSDFParams(!hiDense);

    if(inequal)
    {
        params->volumeDims[0] += 32;
        params->volumeDims[1] -= 32;
    }

    Ptr<Scene> scene = Scene::create(hiDense, params->frameSize, params->intr, params->depthFactor);

    Ptr<kinfu::KinFu> kf = kinfu::KinFu::create(params);

    std::vector<Affine3f> poses = scene->getPoses();
    Affine3f startPoseGT = poses[0], startPoseKF;
    Affine3f pose, kfPose;
    for(size_t i = 0; i < poses.size(); i++)
    {
        pose = poses[i];

        Mat depth = scene->depth(pose);

        ASSERT_TRUE(kf->update(depth));

        kfPose = kf->getPose();
        if(i == 0)
            startPoseKF = kfPose;

        pose = (  startPoseGT.inv() * pose  )*startPoseKF;

        if(display)
        {
            imshow("depth", depth*(1.f/params->depthFactor/4.f));
            Mat rendered;
            kf->render(rendered);
            imshow("render", rendered);
            waitKey(10);
        }
    }

    double rvecThreshold = hiDense ? 0.01 : 0.02;
    ASSERT_LT(cv::norm(kfPose.rvec() - pose.rvec()), rvecThreshold);
    double poseThreshold = hiDense ? 0.03 : 0.1;
    ASSERT_LT(cv::norm(kfPose.translation() - pose.translation()), poseThreshold);
}


#ifdef OPENCV_ENABLE_NONFREE
TEST( KinectFusion, lowDense )
#else
TEST(KinectFusion, DISABLED_lowDense)
#endif
{
    flyTest(false, false);
}

#ifdef OPENCV_ENABLE_NONFREE
TEST( KinectFusion, highDense )
#else
TEST(KinectFusion, DISABLED_highDense)
#endif
{
    flyTest(true, false);
}

#ifdef OPENCV_ENABLE_NONFREE
TEST( KinectFusion, inequal )
#else
TEST(KinectFusion, DISABLED_inequal)
#endif
{
    flyTest(false, true);
}

#ifdef HAVE_OPENCL
#ifdef OPENCV_ENABLE_NONFREE
TEST( KinectFusion, OCL )
#else
TEST(KinectFusion, DISABLED_OCL)
#endif
{
    cv::ocl::setUseOpenCL(false);
    flyTest(false, false);
    cv::ocl::setUseOpenCL(true);
    flyTest(false, false);
}
#endif

TEST( KinectFusion, DISABLED_hashTsdf )
{
    flyTest(false, false, true);
    //! hashTSDF does not support non-equal volumeDims
    flyTest(true, false, true);
}
}} // namespace
