// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "utils.hpp"
#include "opencv2/3d/odometry.hpp"
#include "odometry_functions.hpp"

namespace cv
{

class Odometry::Impl
{
private:

public:
    Impl() {};
    virtual ~Impl() {};
    virtual OdometryFrame createOdometryFrame() const = 0;
    virtual void prepareFrame(OdometryFrame& frame) = 0;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) = 0;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const = 0;
    virtual bool compute(InputArray srcFrame, InputArray dstFrame, OutputArray Rt) const = 0;
    virtual bool compute(InputArray srcDepthFrame, InputArray srcRGBFrame, 
                         InputArray dstDepthFrame, InputArray dstRGBFrame, OutputArray Rt) const = 0;
};


class OdometryICP : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryICP(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryICP();

    virtual OdometryFrame createOdometryFrame() const override;
    virtual void prepareFrame(OdometryFrame& frame) override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcFrame, InputArray dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepthFrame, InputArray srcRGBFrame, 
                         InputArray dstDepthFrame, InputArray dstRGBFrame, OutputArray Rt) const override;
};

OdometryICP::OdometryICP(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryICP::~OdometryICP()
{
}

OdometryFrame OdometryICP::createOdometryFrame() const
{
#ifdef HAVE_OPENCL
    return OdometryFrame(OdometryFrameStoreType::UMAT);
#else
    return OdometryFrame(OdometryFrameStoreType::MAT);
#endif
}

void OdometryICP::prepareFrame(OdometryFrame& frame)
{
    prepareICPFrame(frame, frame, this->settings, this->algtype);
}

void OdometryICP::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame)
{
    prepareICPFrame(srcFrame, dstFrame, this->settings, this->algtype);
}

bool OdometryICP::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
        this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
        iterCounts, this->settings.getMaxTranslation(),
        this->settings.getMaxRotation(), settings.getSobelScale(),
        OdometryType::DEPTH, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

bool OdometryICP::compute(InputArray _srcFrame, InputArray _dstFrame, OutputArray Rt) const
{
    OdometryFrame srcFrame = this->createOdometryFrame();
    OdometryFrame dstFrame = this->createOdometryFrame();
    srcFrame.setDepth(_srcFrame);
    dstFrame.setDepth(_dstFrame);

    prepareICPFrame(srcFrame, dstFrame, this->settings, this->algtype);

    bool isCorrect = compute(srcFrame, dstFrame, Rt);
    return isCorrect;
}

bool OdometryICP::compute(InputArray, InputArray, InputArray, InputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume does not work with depth and rgb data simultaneously");
    return false;
}

class OdometryRGB : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryRGB(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryRGB();

    virtual OdometryFrame createOdometryFrame() const override;
    virtual void prepareFrame(OdometryFrame& frame) override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcFrame, InputArray dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepthFrame, InputArray srcRGBFrame, 
                         InputArray dstDepthFrame, InputArray dstRGBFrame, OutputArray Rt) const override;
};

OdometryRGB::OdometryRGB(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryRGB::~OdometryRGB()
{
}

OdometryFrame OdometryRGB::createOdometryFrame() const
{
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

void OdometryRGB::prepareFrame(OdometryFrame& frame)
{
    prepareRGBFrame(frame, frame, this->settings);
}

void OdometryRGB::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame)
{
    prepareRGBFrame(srcFrame, dstFrame, this->settings);
}

bool OdometryRGB::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    CV_CheckTypeEQ(miterCounts.type(), CV_32S, "");
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
        this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
        iterCounts, this->settings.getMaxTranslation(),
        this->settings.getMaxRotation(), settings.getSobelScale(),
        OdometryType::RGB, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}
bool OdometryRGB::compute(InputArray _srcFrame, InputArray _dstFrame, OutputArray Rt) const
{
    OdometryFrame srcFrame = this->createOdometryFrame();
    OdometryFrame dstFrame = this->createOdometryFrame();
    srcFrame.setImage(_srcFrame);
    dstFrame.setImage(_dstFrame);

    prepareRGBFrame(srcFrame, dstFrame, this->settings);

    bool isCorrect = compute(srcFrame, dstFrame, Rt);
    return isCorrect;
}

bool OdometryRGB::compute(InputArray, InputArray, InputArray, InputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume does not work with depth and rgb data simultaneously");
    return false;
}

class OdometryRGBD : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryRGBD(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryRGBD();

    virtual OdometryFrame createOdometryFrame() const override;
    virtual void prepareFrame(OdometryFrame& frame) override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcFrame, InputArray dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepthFrame, InputArray srcRGBFrame, 
                         InputArray dstDepthFrame, InputArray dstRGBFrame, OutputArray Rt) const override;
};

OdometryRGBD::OdometryRGBD(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryRGBD::~OdometryRGBD()
{
}

OdometryFrame OdometryRGBD::createOdometryFrame() const
{
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

void OdometryRGBD::prepareFrame(OdometryFrame& frame)
{
    prepareRGBDFrame(frame, frame, this->settings, this->algtype);
}

void OdometryRGBD::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame)
{
    prepareRGBDFrame(srcFrame, dstFrame, this->settings, this->algtype);
}

bool OdometryRGBD::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
        this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
        iterCounts, this->settings.getMaxTranslation(),
        this->settings.getMaxRotation(), settings.getSobelScale(),
        OdometryType::RGB_DEPTH, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

bool OdometryRGBD::compute(InputArray, InputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume needs depth and rgb data simultaneously");
    return false;
}

bool OdometryRGBD::compute(InputArray _srcDepthFrame, InputArray _srcRGBFrame, 
                          InputArray _dstDepthFrame, InputArray _dstRGBFrame, OutputArray Rt) const
{
    OdometryFrame srcFrame = this->createOdometryFrame();
    OdometryFrame dstFrame = this->createOdometryFrame();
    srcFrame.setDepth(_srcDepthFrame);
    srcFrame.setImage(_srcRGBFrame);
    dstFrame.setDepth(_dstDepthFrame);
    dstFrame.setImage(_dstRGBFrame);

    prepareRGBDFrame(srcFrame, dstFrame, this->settings, this->algtype);

    bool isCorrect = compute(srcFrame, dstFrame, Rt);
    return isCorrect;
}

Odometry::Odometry()
{
    OdometrySettings settings;
    this->impl = makePtr<OdometryICP>(settings, OdometryAlgoType::COMMON);
}

Odometry::Odometry(OdometryType otype)
{
    OdometrySettings settings;
    switch (otype)
    {
    case OdometryType::DEPTH:
        this->impl = makePtr<OdometryICP>(settings, OdometryAlgoType::FAST);
        break;
    case OdometryType::RGB:
        this->impl = makePtr<OdometryRGB>(settings, OdometryAlgoType::COMMON);
        break;
    case OdometryType::RGB_DEPTH:
        this->impl = makePtr<OdometryRGBD>(settings, OdometryAlgoType::COMMON);
        break;
    default:
        CV_Error(Error::StsInternal,
            "Incorrect OdometryType, you are able to use only { DEPTH = 0, RGB = 1, RGB_DEPTH = 2 }");
        break;
    }
}

Odometry::Odometry(OdometryType otype, OdometrySettings settings, OdometryAlgoType algtype)
{
    switch (otype)
    {
    case OdometryType::DEPTH:
        this->impl = makePtr<OdometryICP>(settings, algtype);
        break;
    case OdometryType::RGB:
        this->impl = makePtr<OdometryRGB>(settings, algtype);
        break;
    case OdometryType::RGB_DEPTH:
        this->impl = makePtr<OdometryRGBD>(settings, algtype);
        break;
    default:
        CV_Error(Error::StsInternal,
            "Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }");
        break;
    }

}

Odometry::~Odometry()
{
}

OdometryFrame Odometry::createOdometryFrame() const
{
    return this->impl->createOdometryFrame();
}

OdometryFrame Odometry::createOdometryFrame(OdometryFrameStoreType matType) const
{
    return OdometryFrame(matType);
}

void Odometry::prepareFrame(OdometryFrame& frame)
{
    this->impl->prepareFrame(frame);
}

void Odometry::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame)
{
    this->impl->prepareFrames(srcFrame, dstFrame);
}

bool Odometry::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt)
{
    //this->prepareFrames(srcFrame, dstFrame);
    return this->impl->compute(srcFrame, dstFrame, Rt);
}

bool Odometry::compute(InputArray srcFrame, InputArray dstFrame, OutputArray Rt) const
{
    return this->impl->compute(srcFrame, dstFrame, Rt);
}
bool Odometry::compute(InputArray srcDepthFrame, InputArray srcRGBFrame,
                       InputArray dstDepthFrame, InputArray dstRGBFrame, OutputArray Rt) const
{
    return this->impl->compute(srcDepthFrame, srcRGBFrame, dstDepthFrame, dstRGBFrame, Rt);
}


template<class ImageElemType>
static void
warpFrameImpl(InputArray _image, InputArray depth, InputArray _mask,
    const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
    OutputArray _warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    CV_Assert(_image.size() == depth.size());

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    Mat cloud3;
    cloud3.create(cloud.size(), CV_32FC3);
    for (int y = 0; y < cloud3.rows; y++)
    {
        for (int x = 0; x < cloud3.cols; x++)
        {
            Vec4f p = cloud.at<Vec4f>(y, x);
            cloud3.at<Vec3f>(y, x) = Vec3f(p[0], p[1], p[2]);
        }
    }

    std::vector<Point2f> points2d;
    Mat transformedCloud;
    perspectiveTransform(cloud3, transformedCloud, Rt);
    projectPoints(transformedCloud.reshape(3, 1), Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1), cameraMatrix,
        distCoeff, points2d);

    Mat image = _image.getMat();
    Size sz = _image.size();
    Mat mask = _mask.getMat();
    _warpedImage.create(sz, image.type());
    Mat warpedImage = _warpedImage.getMat();

    Mat zBuffer(sz, CV_32FC1, std::numeric_limits<float>::max());
    const Rect rect = Rect(Point(), sz);

    for (int y = 0; y < sz.height; y++)
    {
        //const Point3f* cloud_row = cloud.ptr<Point3f>(y);
        const Point3f* transformedCloud_row = transformedCloud.ptr<Point3f>(y);
        const Point2f* points2d_row = &points2d[y * sz.width];
        const ImageElemType* image_row = image.ptr<ImageElemType>(y);
        const uchar* mask_row = mask.empty() ? 0 : mask.ptr<uchar>(y);
        for (int x = 0; x < sz.width; x++)
        {
            const float transformed_z = transformedCloud_row[x].z;
            const Point2i p2d = points2d_row[x];
            if ((!mask_row || mask_row[x]) && transformed_z > 0 && rect.contains(p2d) && /*!cvIsNaN(cloud_row[x].z) && */zBuffer.at<float>(p2d) > transformed_z)
            {
                warpedImage.at<ImageElemType>(p2d) = image_row[x];
                zBuffer.at<float>(p2d) = transformed_z;
            }
        }
    }

    if (warpedMask.needed())
        Mat(zBuffer != std::numeric_limits<float>::max()).copyTo(warpedMask);

    if (warpedDepth.needed())
    {
        zBuffer.setTo(std::numeric_limits<float>::quiet_NaN(), zBuffer == std::numeric_limits<float>::max());
        zBuffer.copyTo(warpedDepth);
    }
}

void warpFrame(InputArray image, InputArray depth, InputArray mask,
    InputArray Rt, InputArray cameraMatrix, InputArray distCoeff,
    OutputArray warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    if (image.type() == CV_8UC1)
        warpFrameImpl<uchar>(image, depth, mask, Rt.getMat(), cameraMatrix.getMat(), distCoeff.getMat(), warpedImage, warpedDepth, warpedMask);
    else if (image.type() == CV_8UC3)
        warpFrameImpl<Point3_<uchar> >(image, depth, mask, Rt.getMat(), cameraMatrix.getMat(), distCoeff.getMat(), warpedImage, warpedDepth, warpedMask);
    else
        CV_Error(Error::StsBadArg, "Image has to be type of CV_8UC1 or CV_8UC3");
}

}
