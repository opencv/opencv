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
    ~Impl() {};

    virtual OdometryFrame createOdometryFrame() = 0;
    virtual void prepareFrame(OdometryFrame frame) = 0;
    virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame) = 0;
    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const = 0;
};


class OdometryICP : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryICP(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryICP();

    virtual OdometryFrame createOdometryFrame();
    virtual void prepareFrame(OdometryFrame frame);
    virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame) override;
    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};

OdometryICP::OdometryICP(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryICP::~OdometryICP()
{
}

OdometryFrame OdometryICP::createOdometryFrame()
{
#ifdef HAVE_OPENCL
    return OdometryFrame(OdometryFrameStoreType::UMAT);
#endif
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

void OdometryICP::prepareFrame(OdometryFrame frame)
{
    prepareICPFrame(frame, frame, this->settings, this->algtype);
}

void OdometryICP::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
    prepareICPFrame(srcFrame, dstFrame, this->settings, this->algtype);
}

bool OdometryICP::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
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


class OdometryRGB : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryRGB(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryRGB();

    virtual OdometryFrame createOdometryFrame();
    virtual void prepareFrame(OdometryFrame frame);
    virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};

OdometryRGB::OdometryRGB(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryRGB::~OdometryRGB()
{
}

OdometryFrame OdometryRGB::createOdometryFrame()
{
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

void OdometryRGB::prepareFrame(OdometryFrame frame)
{
    prepareRGBFrame(frame, frame, this->settings);
}

void OdometryRGB::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
    prepareRGBFrame(srcFrame, dstFrame, this->settings);
}

bool OdometryRGB::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
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
        OdometryType::RGB, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}


class OdometryRGBD : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryRGBD(OdometrySettings settings, OdometryAlgoType algtype);
    ~OdometryRGBD();

    virtual OdometryFrame createOdometryFrame();
    virtual void prepareFrame(OdometryFrame frame);
    virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};

OdometryRGBD::OdometryRGBD(OdometrySettings _settings, OdometryAlgoType _algtype)
{
    this->settings = _settings;
    this->algtype = _algtype;
}

OdometryRGBD::~OdometryRGBD()
{
}

OdometryFrame OdometryRGBD::createOdometryFrame()
{
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

void OdometryRGBD::prepareFrame(OdometryFrame frame)
{
    prepareRGBDFrame(frame, frame, this->settings, this->algtype);
}

void OdometryRGBD::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
    prepareRGBDFrame(srcFrame, dstFrame, this->settings, this->algtype);
}

bool OdometryRGBD::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
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


Odometry::Odometry()
{
    OdometrySettings settings;
    this->impl = makePtr<OdometryICP>(settings, OdometryAlgoType::COMMON);
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

OdometryFrame Odometry::createOdometryFrame()
{
    return this->impl->createOdometryFrame();
}

OdometryFrame Odometry::createOdometryFrame(OdometryFrameStoreType matType)
{
    return OdometryFrame(matType);
}

void Odometry::prepareFrame(OdometryFrame frame)
{
    this->impl->prepareFrame(frame);
}

void Odometry::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
    this->impl->prepareFrames(srcFrame, dstFrame);
}

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt)
{
    this->prepareFrames(srcFrame, dstFrame);
    return this->impl->compute(srcFrame, dstFrame, Rt);
}


}
