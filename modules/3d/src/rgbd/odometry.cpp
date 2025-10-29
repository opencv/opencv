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
public:
    Impl() {};
    virtual ~Impl() {};
    virtual void prepareFrame(OdometryFrame& frame) const = 0;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const = 0;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const = 0;
    virtual bool compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const = 0;
    virtual bool compute(InputArray srcDepth, InputArray srcRGB,
                         InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const = 0;
    virtual Ptr<RgbdNormals> getNormalsComputer() const = 0;
};


class OdometryICP : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;
    mutable Ptr<RgbdNormals> normalsComputer;

public:
    OdometryICP(OdometrySettings _settings, OdometryAlgoType _algtype) :
        settings(_settings), algtype(_algtype), normalsComputer()
    { }
    ~OdometryICP() { }

    virtual void prepareFrame(OdometryFrame& frame) const override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray srcRGB,
                         InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const override;
    virtual Ptr<RgbdNormals> getNormalsComputer() const override;
};

Ptr<RgbdNormals> OdometryICP::getNormalsComputer() const
{
    return this->normalsComputer;
}

void OdometryICP::prepareFrame(OdometryFrame& frame) const
{
    prepareICPFrame(frame, frame, this->normalsComputer, this->settings, this->algtype);
}

void OdometryICP::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const
{
    prepareICPFrame(srcFrame, dstFrame, this->normalsComputer, this->settings, this->algtype);
}

bool OdometryICP::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
                                         this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
                                         iterCounts, this->settings.getMaxTranslation(),
                                         this->settings.getMaxRotation(), settings.getSobelScale(),
                                         OdometryType::DEPTH, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

bool OdometryICP::compute(InputArray _srcDepth, InputArray _dstDepth, OutputArray Rt) const
{
    OdometryFrame srcFrame(_srcDepth);
    OdometryFrame dstFrame(_dstDepth);

    prepareICPFrame(srcFrame, dstFrame, this->normalsComputer, this->settings, this->algtype);

    bool isCorrect = compute(srcFrame, dstFrame, Rt);
    return isCorrect;
}

bool OdometryICP::compute(InputArray srcDepth, InputArray srcRGB,
                          InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const
{
    CV_UNUSED(srcDepth);
    CV_UNUSED(srcRGB);
    CV_UNUSED(dstDepth);
    CV_UNUSED(dstRGB);
    CV_UNUSED(Rt);
    CV_Error(cv::Error::StsBadFunc, "This odometry does not work with rgb data");
}

class OdometryRGB : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;

public:
    OdometryRGB(OdometrySettings _settings, OdometryAlgoType _algtype) : settings(_settings), algtype(_algtype) { }
    ~OdometryRGB() { }

    virtual void prepareFrame(OdometryFrame& frame) const override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray srcRGB,
                         InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const override;
    virtual Ptr<RgbdNormals> getNormalsComputer() const override { return Ptr<RgbdNormals>(); }
};


void OdometryRGB::prepareFrame(OdometryFrame& frame) const
{
    prepareRGBFrame(frame, frame, this->settings);
}

void OdometryRGB::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const
{
    prepareRGBFrame(srcFrame, dstFrame, this->settings);
}

bool OdometryRGB::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
                                         this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
                                         iterCounts, this->settings.getMaxTranslation(),
                                         this->settings.getMaxRotation(), settings.getSobelScale(),
                                         OdometryType::RGB, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

bool OdometryRGB::compute(InputArray _srcDepth, InputArray _dstDepth, OutputArray Rt) const
{
    CV_UNUSED(_srcDepth);
    CV_UNUSED(_dstDepth);
    CV_UNUSED(Rt);
    CV_Error(cv::Error::StsBadFunc, "This odometry algorithm requires depth and rgb data simultaneously");
}

bool OdometryRGB::compute(InputArray srcDepth, InputArray srcRGB, InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const
{
    OdometryFrame srcFrame(srcDepth, srcRGB);
    OdometryFrame dstFrame(dstDepth, dstRGB);

    prepareRGBFrame(srcFrame, dstFrame, this->settings);

    return compute(srcFrame, dstFrame, Rt);
}

class OdometryRGBD : public Odometry::Impl
{
private:
    OdometrySettings settings;
    OdometryAlgoType algtype;
    mutable Ptr<RgbdNormals> normalsComputer;

public:
    OdometryRGBD(OdometrySettings _settings, OdometryAlgoType _algtype) : settings(_settings), algtype(_algtype), normalsComputer() { }
    ~OdometryRGBD() { }

    virtual void prepareFrame(OdometryFrame& frame) const override;
    virtual void prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const override;
    virtual bool compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const override;
    virtual bool compute(InputArray srcDepth, InputArray srcRGB,
                         InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const override;
    virtual Ptr<RgbdNormals> getNormalsComputer() const override;
};

Ptr<RgbdNormals> OdometryRGBD::getNormalsComputer() const
{
    return normalsComputer;
}

void OdometryRGBD::prepareFrame(OdometryFrame& frame) const
{
    prepareRGBDFrame(frame, frame, this->normalsComputer, this->settings, this->algtype);
}

void OdometryRGBD::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const
{
    prepareRGBDFrame(srcFrame, dstFrame, this->normalsComputer, this->settings, this->algtype);
}

bool OdometryRGBD::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);
    bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
                                         this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
                                         iterCounts, this->settings.getMaxTranslation(),
                                         this->settings.getMaxRotation(), settings.getSobelScale(),
                                         OdometryType::RGB_DEPTH, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

bool OdometryRGBD::compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const
{
    CV_UNUSED(srcDepth);
    CV_UNUSED(dstDepth);
    CV_UNUSED(Rt);
    CV_Error(cv::Error::StsBadFunc, "This odometry algorithm needs depth and rgb data simultaneously");
}

bool OdometryRGBD::compute(InputArray _srcDepth, InputArray _srcRGB,
                           InputArray _dstDepth, InputArray _dstRGB, OutputArray Rt) const
{
    OdometryFrame srcFrame(_srcDepth, _srcRGB);
    OdometryFrame dstFrame(_dstDepth, _dstRGB);

    prepareRGBDFrame(srcFrame, dstFrame, this->normalsComputer, this->settings, this->algtype);
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

Odometry::Odometry(OdometryType otype, const OdometrySettings& settings, OdometryAlgoType algtype)
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

void Odometry::prepareFrame(OdometryFrame& frame) const
{
    this->impl->prepareFrame(frame);
}

void Odometry::prepareFrames(OdometryFrame& srcFrame, OdometryFrame& dstFrame) const
{
    this->impl->prepareFrames(srcFrame, dstFrame);
}

bool Odometry::compute(const OdometryFrame& srcFrame, const OdometryFrame& dstFrame, OutputArray Rt) const
{
    return this->impl->compute(srcFrame, dstFrame, Rt);
}

bool Odometry::compute(InputArray srcDepth, InputArray dstDepth, OutputArray Rt) const
{
    return this->impl->compute(srcDepth, dstDepth, Rt);
}

bool Odometry::compute(InputArray srcDepth, InputArray srcRGB,
                       InputArray dstDepth, InputArray dstRGB, OutputArray Rt) const
{
    return this->impl->compute(srcDepth, srcRGB, dstDepth, dstRGB, Rt);
}

Ptr<RgbdNormals> Odometry::getNormalsComputer() const
{
    return this->impl->getNormalsComputer();
}


void warpFrame(InputArray depth, InputArray image, InputArray mask,
               InputArray Rt, InputArray cameraMatrix,
               OutputArray warpedDepth, OutputArray warpedImage, OutputArray warpedMask)
{
    CV_Assert(cameraMatrix.size() == Size(3, 3));
    CV_Assert(cameraMatrix.depth() == CV_32F || cameraMatrix.depth() == CV_64F);
    Matx33d K, Kinv;
    cameraMatrix.getMat().convertTo(K, CV_64F);
    std::vector<bool> camPlaces { /* fx */ true, false, /* cx */ true, false, /* fy */ true, /* cy */ true,  false, false, /* 1 */ true};
    for (int i = 0; i < 9; i++)
    {
        CV_Assert(camPlaces[i] == (K.val[i] > DBL_EPSILON));
    }
    Kinv = K.inv();

    CV_Assert((Rt.cols() == 4) && (Rt.rows() == 3 || Rt.rows() == 4));
    CV_Assert(Rt.depth() == CV_32F || Rt.depth() == CV_64F);
    Mat rtmat;
    Rt.getMat().convertTo(rtmat, CV_64F);
    Affine3d rt(rtmat);

    CV_Assert(!depth.empty());
    CV_Assert(depth.channels() == 1);
    double maxDepth = 0;
    int depthDepth = depth.depth();
    switch (depthDepth)
    {
    case CV_16U:
        maxDepth = std::numeric_limits<unsigned short>::max();
        break;
    case CV_32F:
        maxDepth = std::numeric_limits<float>::max();
        break;
    case CV_64F:
        maxDepth = std::numeric_limits<double>::max();
        break;
    default:
        CV_Error(Error::StsBadArg, "Unsupported depth data type");
    }
    Mat_<double> depthDbl;
    depth.getMat().convertTo(depthDbl, CV_64F);
    Size sz = depth.size();

    Mat_<uchar> maskMat;
    if (!mask.empty())
    {
        CV_Assert(mask.type() == CV_8UC1 || mask.type() == CV_8SC1 || mask.type() == CV_BoolC1);
        CV_Assert(mask.size() == sz);
        maskMat = mask.getMat();
    }

    int imageType = -1;
    Mat imageMat;
    if (!image.empty())
    {
        imageType = image.type();
        CV_Assert(imageType == CV_8UC1 || imageType == CV_8UC3 || imageType == CV_8UC4);
        CV_Assert(image.size() == sz);
        CV_Assert(warpedImage.needed());
        imageMat = image.getMat();
    }

    CV_Assert(warpedDepth.needed() || warpedImage.needed() || warpedMask.needed());

    // Getting new coords for depth point

    // see the explanation in the loop below
    Matx33d krki = K * rt.rotation() * Kinv;
    Matx32d krki_cols01 = krki.get_minor<3, 2>(0, 0);
    Vec3d krki_col2(krki.col(2).val);

    Vec3d ktmat = K * rt.translation();
    Mat_<Vec3d> reprojBack(depth.size());
    for (int y = 0; y < sz.height; y++)
    {
        const uchar* maskRow = maskMat.empty() ? nullptr : maskMat[y];
        const double* depthRow = depthDbl[y];
        Vec3d* reprojRow = reprojBack[y];
        for (int x = 0; x < sz.width; x++)
        {
            double z = depthRow[x];
            bool badz = cvIsNaN(z) || cvIsInf(z) || z <= 0 || z >= maxDepth || (maskRow && !maskRow[x]);
            Vec3d v;
            if (!badz)
            {
                // Reproject pixel (x, y) using known z, rotate+translate and project back
                // getting new pixel in projective coordinates:
                // v = K * Rt * K^-1 * ([x, y, 1] * z) = [new_x*new_z, new_y*new_z, new_z]
                // v = K * (R * K^-1 * ([x, y, 1] * z) + t) =
                // v = krki * [x, y, 1] * z + ktmat =
                // v = (krki_cols01 * [x, y] + krki_col2) * z + K * t
                v = (krki_cols01 * Vec2d(x, y) + krki_col2) * z + ktmat;
            }
            else
            {
                v = Vec3d();
            }
            reprojRow[x] = v;
        }
    }

    // Draw new depth in z-buffer manner

    Mat warpedImageMat;
    if (warpedImage.needed())
    {
        warpedImage.create(sz, imageType);
        warpedImage.setZero();
        warpedImageMat = warpedImage.getMat();
    }

    const double infinity = std::numeric_limits<double>::max();

    Mat zBuffer(sz, CV_32FC1, infinity);

    const Rect rect = Rect(Point(), sz);

    for (int y = 0; y < sz.height; y++)
    {
        uchar* imageRow1ch = nullptr;
        Vec3b* imageRow3ch = nullptr;
        Vec4b* imageRow4ch = nullptr;
        switch (imageType)
        {
        case -1:
            break;
        case CV_8UC1:
            imageRow1ch = imageMat.ptr<uchar>(y);
            break;
        case CV_8UC3:
            imageRow3ch = imageMat.ptr<Vec3b>(y);
            break;
        case CV_8UC4:
            imageRow4ch = imageMat.ptr<Vec4b>(y);
            break;
        default:
            break;
        }

        const Vec3d* reprojRow = reprojBack[y];
        for (int x = 0; x < sz.width; x++)
        {
            Vec3d v = reprojRow[x];
            double z = v[2];

            if (z > 0)
            {
                Point uv(cvFloor(v[0] / z), cvFloor(v[1] / z));
                if (rect.contains(uv))
                {
                    float oldz = zBuffer.at<float>(uv);

                    if (z < oldz)
                    {
                        zBuffer.at<float>(uv) = (float)z;

                        switch (imageType)
                        {
                        case -1:
                            break;
                        case CV_8UC1:
                            warpedImageMat.at<uchar>(uv) = imageRow1ch[x];
                            break;
                        case CV_8UC3:
                            warpedImageMat.at<Vec3b>(uv) = imageRow3ch[x];
                            break;
                        case CV_8UC4:
                            warpedImageMat.at<Vec4b>(uv) = imageRow4ch[x];
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
        }
    }

    if (warpedDepth.needed() || warpedMask.needed())
    {
        Mat goodMask = (zBuffer < infinity);

        if (warpedDepth.needed())
        {
            warpedDepth.create(sz, depthDepth);

            double badVal;
            switch (depthDepth)
            {
            case CV_16U:
                badVal = 0;
                break;
            case CV_32F:
                badVal = std::numeric_limits<float>::quiet_NaN();
                break;
            case CV_64F:
                badVal = std::numeric_limits<double>::quiet_NaN();
                break;
            default:
                break;
            }

            zBuffer.convertTo(warpedDepth, depthDepth);
            warpedDepth.setTo(badVal, ~goodMask);
        }

        if (warpedMask.needed())
        {
            warpedMask.create(sz, CV_8UC1);
            goodMask.copyTo(warpedMask);
        }
    }
}

}
