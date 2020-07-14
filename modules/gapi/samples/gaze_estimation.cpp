#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser

const std::string about =
    "This is an OpenCV-based version of Gaze Estimation example";
const std::string keys =
    "{ h help |                                    | Print this help message }"
    "{ input  |                                    | Path to the input video file }"
    "{ facem  | face-detection-retail-0005.xml     | Path to OpenVINO face detection model (.xml) }"
    "{ faced  | CPU                                | Target device for the face detection (e.g. CPU, GPU, VPU, ...) }"
    "{ landm  | facial-landmarks-35-adas-0002.xml  | Path to OpenVINO landmarks detector model (.xml) }"
    "{ landd  | CPU                                | Target device for the landmarks detector (e.g. CPU, GPU, VPU, ...) }"
    "{ headm  | head-pose-estimation-adas-0001.xml | Path to OpenVINO head pose estimation model (.xml) }"
    "{ headd  | CPU                                | Target device for the head pose estimation inference (e.g. CPU, GPU, VPU, ...) }"
    "{ gazem  | gaze-estimation-adas-0002.xml      | Path to OpenVINO gaze vector estimaiton model (.xml) }"
    "{ gazed  | CPU                                | Target device for the gaze vector estimation inference (e.g. CPU, GPU, VPU, ...) }"
    ;

namespace {
std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext   = model_path.substr(sz - EXT_LEN);
    auto lower = [](unsigned char c) {
        return static_cast<unsigned char>(std::tolower(c));
    };
    std::transform(ext.begin(), ext.end(), ext.begin(), lower);
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // anonymous namespace

namespace custom {
namespace {
using GMat3  = std::tuple<cv::GMat,cv::GMat,cv::GMat>;
using GMats  = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;
using GSize  = cv::GOpaque<cv::Size>;
G_API_NET(Faces,     <cv::GMat(cv::GMat)>, "face-detector"   );
G_API_NET(Landmarks, <cv::GMat(cv::GMat)>, "facial-landmarks");
G_API_NET(HeadPose,  <   GMat3(cv::GMat)>, "head-pose");
G_API_NET(Gaze,      <cv::GMat(cv::GMat,cv::GMat,cv::GMat)>, "gaze-vector");

G_API_OP(Size, <GSize(cv::GMat)>, "custom.gapi.size") {
    static cv::GOpaqueDesc outMeta(const cv::GMatDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(ParseSSD,
         <GRects(cv::GMat, GSize, bool)>,
         "custom.gaze_estimation.parseSSD") {
    static cv::GArrayDesc outMeta( const cv::GMatDesc &
                                 , const cv::GOpaqueDesc &
                                 , bool) {
        return cv::empty_array_desc();
    }
};

// Left/Right eye per every face
G_API_OP(ParseEyes,
         <std::tuple<GRects, GRects>(GMats, GRects, GSize)>,
         "custom.gaze_estimation.parseEyes") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc>
        outMeta(  const cv::GArrayDesc &
                , const cv::GArrayDesc &
                , const cv::GOpaqueDesc &) {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

// Combine three scalars into a 1x3 vector (per every face)
G_API_OP(ProcessPoses,
         <GMats(GMats, GMats, GMats)>,
         "custom.gaze_estimation.processPoses") {
    static cv::GArrayDesc outMeta(  const cv::GArrayDesc &
                                  , const cv::GArrayDesc &
                                  , const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

void adjustBoundingBox(cv::Rect& boundingBox) {
    auto w = boundingBox.width;
    auto h = boundingBox.height;

    boundingBox.x -= static_cast<int>(0.067 * w);
    boundingBox.y -= static_cast<int>(0.028 * h);

    boundingBox.width += static_cast<int>(0.15 * w);
    boundingBox.height += static_cast<int>(0.13 * h);

    if (boundingBox.width < boundingBox.height) {
        auto dx = (boundingBox.height - boundingBox.width);
        boundingBox.x -= dx / 2;
        boundingBox.width += dx;
    } else {
        auto dy = (boundingBox.width - boundingBox.height);
        boundingBox.y -= dy / 2;
        boundingBox.height += dy;
    }
}

void gazeVectorToGazeAngles(const cv::Point3f& gazeVector,
                                  cv::Point2f& gazeAngles) {
    auto r = cv::norm(gazeVector);

    double v0 = static_cast<double>(gazeVector.x);
    double v1 = static_cast<double>(gazeVector.y);
    double v2 = static_cast<double>(gazeVector.z);

    gazeAngles.x = static_cast<float>(180.0 / M_PI * (M_PI_2 + std::atan2(v2, v0)));
    gazeAngles.y = static_cast<float>(180.0 / M_PI * (M_PI_2 - std::acos(v1 / r)));
}

GAPI_OCV_KERNEL(OCVSize, Size) {
    static void run(const cv::Mat &in, cv::Size &out) {
        out = in.size();
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Size &upscale,
                    const bool filter_out_of_bounds,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Rect surface({0,0}, upscale);
        out_objects.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0];
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            (void) label;
            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < 0.5f) {
                continue; // skip objects with low confidence
            }
            cv::Rect rc;  // map relative coordinates to the original image scale
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            adjustBoundingBox(rc);                // TODO: new option?

            const auto clipped_rc = rc & surface; // TODO: new option?
            if (filter_out_of_bounds) {
                if (clipped_rc.area() != rc.area()) {
                    continue;
                }
            }
            out_objects.emplace_back(clipped_rc);
        }
    }
};

cv::Rect eyeBox(const cv::Rect &face_rc,
                float p1_x, float p1_y, float p2_x, float p2_y,
                float scale = 1.8f) {
    const auto &up = face_rc.size();
    const cv::Point p1 = {
        static_cast<int>(p1_x*up.width),
        static_cast<int>(p1_y*up.height)
    };
    const cv::Point p2 = {
        static_cast<int>(p2_x*up.width),
        static_cast<int>(p2_y*up.height)
    };
    cv::Rect result;

    const auto size     = static_cast<float>(cv::norm(p1 - p2));
    const auto midpoint = (p1 + p2) / 2;

    result.width = static_cast<int>(scale * size);
    result.height = result.width;
    result.x = face_rc.x + midpoint.x - (result.width / 2);
    result.y = face_rc.y + midpoint.y - (result.height / 2);
    // Shift result to the original frame's absolute coordinates
    return result;
}

GAPI_OCV_KERNEL(OCVParseEyes, ParseEyes) {
    static void run(const std::vector<cv::Mat> &in_landmarks_per_face,
                    const std::vector<cv::Rect> &in_face_rcs,
                    const cv::Size &frame_size,
                    std::vector<cv::Rect> &out_left_eyes,
                    std::vector<cv::Rect> &out_right_eyes) {
        const size_t numFaces = in_landmarks_per_face.size();
        const cv::Rect surface(cv::Point(0,0), frame_size);
        GAPI_Assert(numFaces == in_face_rcs.size());
        out_left_eyes.clear();
        out_right_eyes.clear();
        out_left_eyes.reserve(numFaces);
        out_right_eyes.reserve(numFaces);

        for (std::size_t i = 0u; i < numFaces; i++) {
            const auto &lm = in_landmarks_per_face[i];
            const auto &rc = in_face_rcs[i];
            // Left eye is defined by points 0/1 (x2),
            // Right eye is defined by points 2/3 (x2)
            const float *data = lm.ptr<float>();
            out_left_eyes .push_back(surface & eyeBox(rc, data[0], data[1], data[2], data[3]));
            out_right_eyes.push_back(surface & eyeBox(rc, data[4], data[5], data[6], data[7]));
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessPoses, ProcessPoses) {
    static void run(const std::vector<cv::Mat> &in_ys,
                    const std::vector<cv::Mat> &in_ps,
                    const std::vector<cv::Mat> &in_rs,
                    std::vector<cv::Mat> &out_poses) {
        const std::size_t sz = in_ys.size();
        GAPI_Assert(sz == in_ps.size() && sz == in_rs.size());
        out_poses.clear();
        for (std::size_t idx = 0u; idx < sz; idx++) {
            cv::Mat pose(1, 3, CV_32FC1);
            float *ptr = pose.ptr<float>();
            ptr[0] = in_ys[idx].ptr<float>()[0];
            ptr[1] = in_ps[idx].ptr<float>()[0];
            ptr[2] = in_rs[idx].ptr<float>()[0];
            out_poses.push_back(std::move(pose));
        }
    }
};
} // anonymous namespace
} // namespace custom

namespace vis {
namespace {
cv::Point2f midp(const cv::Rect &rc) {
    return (rc.tl() + rc.br()) / 2;
};
void bbox(cv::Mat &m, const cv::Rect &rc) {
    cv::rectangle(m, rc, cv::Scalar{0,255,0}, 2, cv::LINE_8, 0);
};
void pose(cv::Mat &m, const cv::Mat &p, const cv::Rect &face_rc) {
    const auto *posePtr = p.ptr<float>();
    const auto yaw   = static_cast<double>(posePtr[0]);
    const auto pitch = static_cast<double>(posePtr[1]);
    const auto roll  = static_cast<double>(posePtr[2]);

    const auto sinY = std::sin(yaw   * M_PI / 180.0);
    const auto sinP = std::sin(pitch * M_PI / 180.0);
    const auto sinR = std::sin(roll  * M_PI / 180.0);

    const auto cosY = std::cos(yaw   * M_PI / 180.0);
    const auto cosP = std::cos(pitch * M_PI / 180.0);
    const auto cosR = std::cos(roll  * M_PI / 180.0);

    const auto axisLength = 0.4 * face_rc.width;
    const auto xCenter = face_rc.x + face_rc.width  / 2;
    const auto yCenter = face_rc.y + face_rc.height / 2;

    const auto center = cv::Point{xCenter, yCenter};
    const auto axisln = cv::Point2d{axisLength, axisLength};
    const auto ctr    = cv::Matx<double,2,2>(cosR*cosY, sinY*sinP*sinR, 0.f,  cosP*sinR);
    const auto ctt    = cv::Matx<double,2,2>(cosR*sinY*sinP, cosY*sinR, 0.f, -cosP*cosR);
    const auto ctf    = cv::Matx<double,2,2>(sinY*cosP, 0.f, 0.f, sinP);

    // center to right
    cv::line(m, center, center + static_cast<cv::Point>(ctr*axisln), cv::Scalar(0, 0, 255), 2);
    // center to top
    cv::line(m, center, center + static_cast<cv::Point>(ctt*axisln), cv::Scalar(0, 255, 0), 2);
    // center to forward
    cv::line(m, center, center + static_cast<cv::Point>(ctf*axisln), cv::Scalar(255, 0, 255), 2);
}
void vvec(cv::Mat &m, const cv::Mat &v, const cv::Rect &face_rc,
          const cv::Rect &left_rc, const cv::Rect &right_rc) {
    const auto scale =  0.002 * face_rc.width;

    cv::Point3f gazeVector;
    const auto *gazePtr = v.ptr<float>();
    gazeVector.x = gazePtr[0];
    gazeVector.y = gazePtr[1];
    gazeVector.z = gazePtr[2];
    gazeVector = gazeVector / cv::norm(gazeVector);

    const double arrowLength = 0.4 * face_rc.width;
    const auto left_mid = midp(left_rc);
    const auto right_mid = midp(right_rc);

    cv::Point2f gazeArrow;
    gazeArrow.x =  gazeVector.x;
    gazeArrow.y = -gazeVector.y;
    gazeArrow  *= arrowLength;

    cv::arrowedLine(m, left_mid,  left_mid  + gazeArrow, cv::Scalar(255, 0, 0), 2);
    cv::arrowedLine(m, right_mid, right_mid + gazeArrow, cv::Scalar(255, 0, 0), 2);

    cv::Point2f gazeAngles;
    custom::gazeVectorToGazeAngles(gazeVector, gazeAngles);

    cv::putText(m,
                cv::format("gaze angles: (h=%0.0f, v=%0.0f)",
                           static_cast<double>(std::round(gazeAngles.x)),
                           static_cast<double>(std::round(gazeAngles.y))),
                cv::Point(static_cast<int>(face_rc.tl().x),
                          static_cast<int>(face_rc.br().y + 12. * face_rc.width / 100.)),
                cv::FONT_HERSHEY_PLAIN, scale * 2, cv::Scalar::all(255), 1);
};
} // anonymous namespace
} // namespace vis

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    cv::GMat in;
    cv::GMat faces = cv::gapi::infer<custom::Faces>(in);
    cv::GOpaque<cv::Size> sz = custom::Size::on(in); // FIXME
    cv::GArray<cv::Rect> faces_rc = custom::ParseSSD::on(faces, sz, true);
    cv::GArray<cv::GMat> angles_y, angles_p, angles_r;
    std::tie(angles_y, angles_p, angles_r) = cv::gapi::infer<custom::HeadPose>(faces_rc, in);
    cv::GArray<cv::GMat> heads_pos = custom::ProcessPoses::on(angles_y, angles_p, angles_r);
    cv::GArray<cv::GMat> landmarks = cv::gapi::infer<custom::Landmarks>(faces_rc, in);
    cv::GArray<cv::Rect> left_eyes, right_eyes;
    std::tie(left_eyes, right_eyes) = custom::ParseEyes::on(landmarks, faces_rc, sz);
    cv::GArray<cv::GMat> gaze_vectors = cv::gapi::infer2<custom::Gaze>( in
                                                                      , left_eyes
                                                                      , right_eyes
                                                                      , heads_pos);
    cv::GComputation graph(cv::GIn(in),
                           cv::GOut( cv::gapi::copy(in)
                                   , faces_rc
                                   , left_eyes
                                   , right_eyes
                                   , heads_pos
                                   , gaze_vectors));

    const auto input_file_name = cmd.get<std::string>("input");
    const auto face_model_path = cmd.get<std::string>("facem");
    const auto head_model_path = cmd.get<std::string>("headm");
    const auto lmrk_model_path = cmd.get<std::string>("landm");
    const auto gaze_model_path = cmd.get<std::string>("gazem");

    auto face_net = cv::gapi::ie::Params<custom::Faces> {
        face_model_path,                // path to topology IR
        weights_path(face_model_path),  // path to weights
        cmd.get<std::string>("faced"),  /// device specifier
    };
    auto head_net = cv::gapi::ie::Params<custom::HeadPose> {
        head_model_path,                // path to topology IR
        weights_path(head_model_path),  // path to weights
        cmd.get<std::string>("headd"),  // device specifier
    }.cfgOutputLayers({"angle_y_fc", "angle_p_fc", "angle_r_fc"});
    auto landmarks_net = cv::gapi::ie::Params<custom::Landmarks> {
        lmrk_model_path,                // path to topology IR
        weights_path(lmrk_model_path),  // path to weights
        cmd.get<std::string>("landd"),  // device specifier
    };
    auto gaze_net = cv::gapi::ie::Params<custom::Gaze> {
        gaze_model_path,                // path to topology IR
        weights_path(gaze_model_path),  // path to weights
        cmd.get<std::string>("gazed"),  // device specifier
    }.cfgInputLayers({"left_eye_image", "right_eye_image", "head_pose_angles"});

    auto kernels = cv::gapi::kernels< custom::OCVSize
                                    , custom::OCVParseSSD
                                    , custom::OCVParseEyes
                                    , custom::OCVProcessPoses>();
    auto networks = cv::gapi::networks(face_net, head_net, landmarks_net, gaze_net);
    auto pipeline = graph.compileStreaming(cv::compile_args(networks, kernels));

    cv::TickMeter tm;
    cv::Mat image;
    std::vector<cv::Rect> out_faces, out_right_eyes, out_left_eyes;
    std::vector<cv::Mat> out_poses;
    std::vector<cv::Mat> out_gazes;
    std::size_t frames = 0u;
    std::cout << "Reading " << input_file_name << std::endl;

    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name));
    pipeline.start();
    tm.start();
    while (pipeline.pull(cv::gout( image
                                 , out_faces
                                 , out_left_eyes
                                 , out_right_eyes
                                 , out_poses
                                 , out_gazes))) {
        frames++;
        // Visualize results on the frame
        for (auto &&rc : out_faces) vis::bbox(image, rc);
        for (auto &&rc : out_left_eyes) vis::bbox(image, rc);
        for (auto &&rc : out_right_eyes) vis::bbox(image, rc);
        for (std::size_t i = 0u; i < out_faces.size(); i++) {
            vis::pose(image, out_poses[i], out_faces[i]);
            vis::vvec(image, out_gazes[i], out_faces[i], out_left_eyes[i], out_right_eyes[i]);
        }
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        cv::putText(image, fps_str, {0,32}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        cv::imshow("Out", image);
        cv::waitKey(1);
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
