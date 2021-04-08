// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_RGBS_IO_UTILS_HPP
#define OPENCV_RGBS_IO_UTILS_HPP

#include <fstream>
#include <iostream>
#include <opencv2/3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/kinfu.hpp>
#include <opencv2/rgbd/large_kinfu.hpp>
#include <opencv2/rgbd/colored_kinfu.hpp>

namespace cv
{
namespace io_utils
{

static std::vector<std::string> readDepth(const std::string& fileList)
{
    std::vector<std::string> v;

    std::fstream file(fileList);
    if (!file.is_open())
        throw std::runtime_error("Failed to read depth list");

    std::string dir;
    size_t slashIdx = fileList.rfind('/');
    slashIdx        = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
    dir             = fileList.substr(0, slashIdx);

    while (!file.eof())
    {
        std::string s, imgPath;
        std::getline(file, s);
        if (s.empty() || s[0] == '#')
            continue;
        std::stringstream ss;
        ss << s;
        double thumb;
        ss >> thumb >> imgPath;
        v.push_back(dir + '/' + imgPath);
    }

    return v;
}

struct DepthWriter
{
    DepthWriter(std::string fileList) : file(fileList, std::ios::out), count(0), dir()
    {
        size_t slashIdx = fileList.rfind('/');
        slashIdx        = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
        dir             = fileList.substr(0, slashIdx);

        if (!file.is_open())
            throw std::runtime_error("Failed to write depth list");

        file << "# depth maps saved from device" << std::endl;
        file << "# useless_number filename" << std::endl;
    }

    void append(InputArray _depth)
    {
        Mat depth                  = _depth.getMat();
        std::string depthFname     = cv::format("%04d.png", count);
        std::string fullDepthFname = dir + '/' + depthFname;
        if (!imwrite(fullDepthFname, depth))
            throw std::runtime_error("Failed to write depth to file " + fullDepthFname);
        file << count++ << " " << depthFname << std::endl;
    }

    std::fstream file;
    int count;
    std::string dir;
};

namespace Kinect2Params
{
static const Size depth_frameSize = Size(512, 424);
// approximate values, no guarantee to be correct
static const float depth_focal = 366.1f;
static const float depth_cx    = 258.2f;
static const float depth_cy    = 204.f;
static const float depth_k1    = 0.12f;
static const float depth_k2    = -0.34f;
static const float depth_k3    = 0.12f;

static const Size rgb_frameSize = Size(640, 480);
static const float rgb_focal = 525.0f;
static const float rgb_cx    = 319.5f;
static const float rgb_cy    = 239.5f;
static const float rgb_k1    = 0.0f;
static const float rgb_k2    = 0.0f;
static const float rgb_k3    = 0.0f;

};  // namespace Kinect2Params

namespace AstraParams
{
static const Size depth_frameSize = Size(640, 480);
// approximate values, no guarantee to be correct
static const float depth_fx = 535.4f;
static const float depth_fy = 539.2f;
static const float depth_cx = 320.1f;
static const float depth_cy = 247.6f;
static const float depth_k1 = 0.0f;
static const float depth_k2 = 0.0f;
static const float depth_k3 = 0.0f;

static const Size rgb_frameSize = Size(640, 480);
static const float rgb_focal = 525.0f;
static const float rgb_cx    = 319.5f;
static const float rgb_cy    = 239.5f;
static const float rgb_k1    = 0.0f;
static const float rgb_k2    = 0.0f;
static const float rgb_k3    = 0.0f;

};  // namespace Kinect2Params

struct DepthSource
{
   public:
    enum Type
    {
        DEPTH_LIST,
        DEPTH_KINECT2_LIST,
        DEPTH_KINECT2,
        DEPTH_REALSENSE,
        DEPTH_ASTRA
    };

    DepthSource(int cam) : DepthSource("", cam) {}

    DepthSource(String fileListName) : DepthSource(fileListName, -1) {}

    DepthSource(String fileListName, int cam)
        : depthFileList(fileListName.empty() ? std::vector<std::string>()
                                             : readDepth(fileListName)),
          frameIdx(0),
          undistortMap1(),
          undistortMap2()
    {
        if (cam >= 0)
        {
            vc = VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam);
            if (vc.isOpened())
            {
                if(cam == 20)
                    sourceType = Type::DEPTH_ASTRA;
                else
                    sourceType = Type::DEPTH_KINECT2;
            }
            else
            {
                vc = VideoCapture(VideoCaptureAPIs::CAP_REALSENSE + cam);
                if (vc.isOpened())
                {
                    sourceType = Type::DEPTH_REALSENSE;
                }
            }
        }
        else
        {
            vc         = VideoCapture();
            sourceType = Type::DEPTH_KINECT2_LIST;
        }
    }

    UMat getDepth()
    {
        UMat out;
        if (!vc.isOpened())
        {
            if (frameIdx < depthFileList.size())
            {
                Mat f = cv::imread(depthFileList[frameIdx++], IMREAD_ANYDEPTH);
                f.copyTo(out);
            }
            else
            {
                return UMat();
            }
        }
        else
        {
            vc.grab();
            switch (sourceType)
            {
                case Type::DEPTH_KINECT2: vc.retrieve(out, CAP_OPENNI_DEPTH_MAP); break;
                case Type::DEPTH_REALSENSE: vc.retrieve(out, CAP_INTELPERC_DEPTH_MAP); break;
                default:
                    // unknown depth source
                    vc.retrieve(out);
            }

            // workaround for Kinect 2
            if (sourceType == Type::DEPTH_KINECT2)
            {
                out = out(Rect(Point(), Kinect2Params::depth_frameSize));

                UMat outCopy;
                // linear remap adds gradient between valid and invalid pixels
                // which causes garbage, use nearest instead
                remap(out, outCopy, undistortMap1, undistortMap2, cv::INTER_NEAREST);

                cv::flip(outCopy, out, 1);
            }
        }
        if (out.empty())
            throw std::runtime_error("Matrix is empty");
        return out;
    }

    bool empty() { return depthFileList.empty() && !(vc.isOpened()); }

    void updateIntrinsics(Matx33f& _intrinsics, Size& _frameSize, float& _depthFactor)
    {
        if (vc.isOpened())
        {
            // this should be set in according to user's depth sensor
            int w = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
            int h = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float fx, fy, cx, cy;
            float depthFactor = 1000.f;
            Size frameSize;
            if (sourceType == Type::DEPTH_KINECT2)
            {
                fx = fy = Kinect2Params::depth_focal;
                cx      = Kinect2Params::depth_cx;
                cy      = Kinect2Params::depth_cy;

                frameSize = Kinect2Params::depth_frameSize;
            }
            else if (sourceType == Type::DEPTH_ASTRA)
            {
                fx      = AstraParams::depth_fx;
                fy      = AstraParams::depth_fy;
                cx      = AstraParams::depth_cx;
                cy      = AstraParams::depth_cy;

                frameSize = AstraParams::depth_frameSize;
            }
            else
            {
                if (sourceType == Type::DEPTH_REALSENSE)
                {
                    fx          = (float)vc.get(CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ);
                    fy          = (float)vc.get(CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT);
                    depthFactor = 1.f / (float)vc.get(CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE);
                }
                else
                {
                    fx = fy =
                        (float)vc.get(CAP_OPENNI_DEPTH_GENERATOR | CAP_PROP_OPENNI_FOCAL_LENGTH);
                }

                cx = w / 2 - 0.5f;
                cy = h / 2 - 0.5f;

                frameSize = Size(w, h);
            }

            Matx33f camMatrix = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);
            _intrinsics       = camMatrix;
            _frameSize        = frameSize;
            _depthFactor      = depthFactor;
        }
    }

    void updateVolumeParams(const Vec3i& _resolution, float& _voxelSize, float& _tsdfTruncDist,
                            Affine3f& _volumePose, float& _depthTruncateThreshold)
    {
        float volumeSize        = 3.0f;
        _depthTruncateThreshold = 0.0f;
        // RealSense has shorter depth range, some params should be tuned
        if (sourceType == Type::DEPTH_REALSENSE)
        {
            volumeSize              = 1.f;
            _voxelSize              = volumeSize / _resolution[0];
            _tsdfTruncDist          = 0.01f;
            _depthTruncateThreshold = 2.5f;
        }
        _volumePose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.05f));
    }

    void updateICPParams(float& _icpDistThresh, float& _bilateralSigmaDepth)
    {
        _icpDistThresh       = 0.1f;
        _bilateralSigmaDepth = 0.04f;
        // RealSense has shorter depth range, some params should be tuned
        if (sourceType == Type::DEPTH_REALSENSE)
        {
            _icpDistThresh       = 0.01f;
            _bilateralSigmaDepth = 0.01f;
        }
    }

    void updateParams(large_kinfu::Params& params)
    {
        if (vc.isOpened())
        {
            updateIntrinsics(params.intr, params.frameSize, params.depthFactor);
            updateVolumeParams(params.volumeParams.resolution, params.volumeParams.voxelSize,
                               params.volumeParams.tsdfTruncDist, params.volumeParams.pose,
                               params.truncateThreshold);
            updateICPParams(params.icpDistThresh, params.bilateral_sigma_depth);

            if (sourceType == Type::DEPTH_KINECT2)
            {
                Matx<float, 1, 5> distCoeffs;
                distCoeffs(0) = Kinect2Params::depth_k1;
                distCoeffs(1) = Kinect2Params::depth_k2;
                distCoeffs(4) = Kinect2Params::depth_k3;

                initUndistortRectifyMap(params.intr, distCoeffs, cv::noArray(), params.intr,
                                        params.frameSize, CV_16SC2, undistortMap1, undistortMap2);
            }
        }
    }

    void updateParams(kinfu::Params& params)
    {
        if (vc.isOpened())
        {
            updateIntrinsics(params.intr, params.frameSize, params.depthFactor);
            updateVolumeParams(params.volumeDims, params.voxelSize,
                               params.tsdf_trunc_dist, params.volumePose, params.truncateThreshold);
            updateICPParams(params.icpDistThresh, params.bilateral_sigma_depth);

            if (sourceType == Type::DEPTH_KINECT2)
            {
                Matx<float, 1, 5> distCoeffs;
                distCoeffs(0) = Kinect2Params::depth_k1;
                distCoeffs(1) = Kinect2Params::depth_k2;
                distCoeffs(4) = Kinect2Params::depth_k3;

                initUndistortRectifyMap(params.intr, distCoeffs, cv::noArray(), params.intr,
                                        params.frameSize, CV_16SC2, undistortMap1, undistortMap2);
            }
        }
    }

    void updateParams(colored_kinfu::Params& params)
    {
        if (vc.isOpened())
        {
            updateIntrinsics(params.intr, params.frameSize, params.depthFactor);
            updateVolumeParams(params.volumeDims, params.voxelSize,
                               params.tsdf_trunc_dist, params.volumePose, params.truncateThreshold);
            updateICPParams(params.icpDistThresh, params.bilateral_sigma_depth);

            if (sourceType == Type::DEPTH_KINECT2)
            {
                Matx<float, 1, 5> distCoeffs;
                distCoeffs(0) = Kinect2Params::depth_k1;
                distCoeffs(1) = Kinect2Params::depth_k2;
                distCoeffs(4) = Kinect2Params::depth_k3;

                initUndistortRectifyMap(params.intr, distCoeffs, cv::noArray(), params.intr,
                                        params.frameSize, CV_16SC2, undistortMap1, undistortMap2);
            }
        }
    }

    std::vector<std::string> depthFileList;
    size_t frameIdx;
    VideoCapture vc;
    UMat undistortMap1, undistortMap2;
    Type sourceType;
};


static std::vector<std::string> readRGB(const std::string& fileList)
{
    std::vector<std::string> v;

    std::fstream file(fileList);
    if (!file.is_open())
        throw std::runtime_error("Failed to read rgb list");

    std::string dir;
    size_t slashIdx = fileList.rfind('/');
    slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
    dir = fileList.substr(0, slashIdx);

    while (!file.eof())
    {
        std::string s, imgPath;
        std::getline(file, s);
        if (s.empty() || s[0] == '#')
            continue;
        std::stringstream ss;
        ss << s;
        double thumb;
        ss >> thumb >> imgPath;
        v.push_back(dir + '/' + imgPath);
    }

    return v;
}

struct RGBWriter
{
    RGBWriter(std::string fileList) : file(fileList, std::ios::out), count(0), dir()
    {
        size_t slashIdx = fileList.rfind('/');
        slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
        dir = fileList.substr(0, slashIdx);

        if (!file.is_open())
            throw std::runtime_error("Failed to write rgb list");

        file << "# rgb maps saved from device" << std::endl;
        file << "# useless_number filename" << std::endl;
    }

    void append(InputArray _rgb)
    {
        Mat rgb = _rgb.getMat();
        std::string rgbFname = cv::format("%04d.png", count);
        std::string fullRGBFname = dir + '/' + rgbFname;
        if (!imwrite(fullRGBFname, rgb))
            throw std::runtime_error("Failed to write rgb to file " + fullRGBFname);
        file << count++ << " " << rgbFname << std::endl;
    }

    std::fstream file;
    int count;
    std::string dir;
};

struct RGBSource
{
   public:
    enum Type
    {
        RGB_LIST,
        RGB_KINECT2_LIST,
        RGB_KINECT2,
        RGB_REALSENSE,
        RGB_ASTRA
    };

    RGBSource(int cam) : RGBSource("", cam) {}

    RGBSource(String fileListName) : RGBSource(fileListName, -1) {}

    RGBSource(String fileListName, int cam)
        : rgbFileList(fileListName.empty() ? std::vector<std::string>()
                                             : readRGB(fileListName)),
          frameIdx(0),
          undistortMap1(),
          undistortMap2()
    {
        if (cam >= 0)
        {
            vc = VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam);
            if (vc.isOpened())
            {
                if(cam == 20)
                    sourceType = Type::RGB_ASTRA;
                else
                    sourceType = Type::RGB_KINECT2;
            }
            else
            {
                vc = VideoCapture(VideoCaptureAPIs::CAP_REALSENSE + cam);
                if (vc.isOpened())
                {
                    sourceType = Type::RGB_REALSENSE;
                }
            }
        }
        else
        {
            vc         = VideoCapture();
            sourceType = Type::RGB_KINECT2_LIST;
        }
    }

    UMat getRGB()
    {
        UMat out;
        if (!vc.isOpened())
        {
            if (frameIdx < rgbFileList.size())
            {
                Mat f = cv::imread(rgbFileList[frameIdx++], IMREAD_COLOR);
                f.copyTo(out);
            }
            else
            {
                return UMat();
            }
        }
        else
        {
            vc.grab();
            switch (sourceType)
            {
                case Type::RGB_KINECT2: vc.retrieve(out, CAP_OPENNI_BGR_IMAGE); break;
                case Type::RGB_REALSENSE: vc.retrieve(out, CAP_INTELPERC_IMAGE); break;
                default:
                    // unknown rgb source
                    vc.retrieve(out);
            }

            // workaround for Kinect 2
            if (sourceType == Type::RGB_KINECT2)
            {
                out = out(Rect(Point(), Kinect2Params::rgb_frameSize));

                UMat outCopy;
                // linear remap adds gradient between valid and invalid pixels
                // which causes garbage, use nearest instead
                remap(out, outCopy, undistortMap1, undistortMap2, cv::INTER_NEAREST);

                cv::flip(outCopy, out, 1);
            }
        }
        if (out.empty())
            throw std::runtime_error("Matrix is empty");
        return out;
    }

    bool empty() { return rgbFileList.empty() && !(vc.isOpened()); }

    void updateIntrinsics(Matx33f& _rgb_intrinsics, Size& _rgb_frameSize)
    {
        if (vc.isOpened())
        {
            // this should be set in according to user's rgb sensor
            int w = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
            int h = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float rgb_fx, rgb_fy, rgb_cx, rgb_cy;
            Size rgb_frameSize;
            if (sourceType == Type::RGB_KINECT2)
            {
                rgb_fx = rgb_fy = Kinect2Params::rgb_focal;
                rgb_cx      = Kinect2Params::rgb_cx;
                rgb_cy      = Kinect2Params::rgb_cy;

                rgb_frameSize = Kinect2Params::rgb_frameSize;
            }
            else if (sourceType == Type::RGB_ASTRA)
            {
                rgb_fx = rgb_fy = AstraParams::rgb_focal;
                rgb_cx = AstraParams::rgb_cx;
                rgb_cy = AstraParams::rgb_cy;

                rgb_frameSize = AstraParams::rgb_frameSize;
            }
            else
            {
                // TODO: replace to rgb types
                rgb_fx = rgb_fy = Kinect2Params::rgb_focal;
                rgb_cx = Kinect2Params::rgb_cx;
                rgb_cy = Kinect2Params::rgb_cy;
                rgb_frameSize = Size(w, h);
            }

            Matx33f rgb_camMatrix = Matx33f(rgb_fx, 0, rgb_cx, 0, rgb_fy, rgb_cy, 0, 0, 1);
            _rgb_intrinsics = rgb_camMatrix;
            _rgb_frameSize  = rgb_frameSize;
        }
    }

    void updateVolumeParams(const Vec3i&, float&, float&, Affine3f&)
    {
        // TODO: do this settings for rgb image
    }

    void updateICPParams(float&)
    {
        // TODO: do this settings for rgb image icp
    }

    void updateParams(colored_kinfu::Params& params)
    {
        if (vc.isOpened())
        {
            updateIntrinsics(params.rgb_intr, params.rgb_frameSize);
            updateVolumeParams(params.volumeDims, params.voxelSize,
                               params.tsdf_trunc_dist, params.volumePose);
            updateICPParams(params.icpDistThresh);

            if (sourceType == Type::RGB_KINECT2)
            {
                Matx<float, 1, 5> distCoeffs;
                distCoeffs(0) = Kinect2Params::rgb_k1;
                distCoeffs(1) = Kinect2Params::rgb_k2;
                distCoeffs(4) = Kinect2Params::rgb_k3;

                initUndistortRectifyMap(params.intr, distCoeffs, cv::noArray(), params.intr,
                                        params.frameSize, CV_16SC2, undistortMap1, undistortMap2);
            }
        }
    }

    std::vector<std::string> rgbFileList;
    size_t frameIdx;
    VideoCapture vc;
    UMat undistortMap1, undistortMap2;
    Type sourceType;
};
}  // namespace io_utils

}  // namespace cv
#endif /* ifndef OPENCV_RGBS_IO_UTILS_HPP */
