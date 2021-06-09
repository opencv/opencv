#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/highgui.hpp>

const std::string about =
"This is an OpenCV-based version of OMZ MTCNN Face Detection example";
const std::string keys =
"{ h help           |                           | Print this help message }"
"{ input            |                           | Path to the input video file }"
"{ mtcnnpm          | mtcnn-p.xml               | Path to OpenVINO MTCNN P (Proposal) detection model (.xml)}"
"{ mtcnnpd          | CPU                       | Target device for the MTCNN P (e.g. CPU, GPU, VPU, ...) }"
"{ mtcnnrm          | mtcnn-r.xml               | Path to OpenVINO MTCNN R (Refinement) detection model (.xml)}"
"{ mtcnnrd          | CPU                       | Target device for the MTCNN R (e.g. CPU, GPU, VPU, ...) }"
"{ mtcnnom          | mtcnn-o.xml               | Path to OpenVINO MTCNN O (Output) detection model (.xml)}"
"{ mtcnnod          | CPU                       | Target device for the MTCNN O (e.g. CPU, GPU, VPU, ...) }"
"{ thrp             | 0.6                       | MTCNN P confidence threshold}"
"{ thrr             | 0.7                       | MTCNN R confidence threshold}"
"{ thro             | 0.7                       | MTCNN O confidence threshold}"
"{ half_scale       | false                     | MTCNN P use half scale pyramid}"
"{ queue_capacity   | 1                         | Streaming executor queue capacity. Calculated automaticaly if 0}"
;

namespace {
std::string weights_path(const std::string& model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    const auto ext = model_path.substr(sz - EXT_LEN);
    CV_Assert(cv::toLowerCase(ext) == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
//////////////////////////////////////////////////////////////////////
} // anonymous namespace

namespace custom {
namespace {

// Define custom structures and operations
#define NUM_REGRESSIONS 4
#define NUM_PTS 5

struct BBox {
    int x1;
    int y1;
    int x2;
    int y2;

    cv::Rect getRect() const { return cv::Rect(x1,
                                               y1,
                                               x2 - x1,
                                               y2 - y1); }

    BBox getSquare() const {
        BBox bbox;
        float bboxWidth = static_cast<float>(x2 - x1);
        float bboxHeight = static_cast<float>(y2 - y1);
        float side = std::max(bboxWidth, bboxHeight);
        bbox.x1 = static_cast<int>(static_cast<float>(x1) + (bboxWidth - side) * 0.5f);
        bbox.y1 = static_cast<int>(static_cast<float>(y1) + (bboxHeight - side) * 0.5f);
        bbox.x2 = static_cast<int>(static_cast<float>(bbox.x1) + side);
        bbox.y2 = static_cast<int>(static_cast<float>(bbox.y1) + side);
        return bbox;
    }
};

struct Face {
    BBox bbox;
    float score;
    std::array<float, NUM_REGRESSIONS> regression;
    std::array<float, 2 * NUM_PTS> ptsCoords;

    static void applyRegression(std::vector<Face>& faces, bool addOne = false) {
        for (auto& face : faces) {
            float bboxWidth =
                face.bbox.x2 - face.bbox.x1 + static_cast<float>(addOne);
            float bboxHeight =
                face.bbox.y2 - face.bbox.y1 + static_cast<float>(addOne);
            face.bbox.x1 = static_cast<int>(static_cast<float>(face.bbox.x1) + (face.regression[1] * bboxWidth));
            face.bbox.y1 = static_cast<int>(static_cast<float>(face.bbox.y1) + (face.regression[0] * bboxHeight));
            face.bbox.x2 = static_cast<int>(static_cast<float>(face.bbox.x2) + (face.regression[3] * bboxWidth));
            face.bbox.y2 = static_cast<int>(static_cast<float>(face.bbox.y2) + (face.regression[2] * bboxHeight));
        }
    }

    static void bboxes2Squares(std::vector<Face>& faces) {
        for (auto& face : faces) {
            face.bbox = face.bbox.getSquare();
        }
    }

    static std::vector<Face> runNMS(std::vector<Face>& faces, const float threshold,
                                    const bool useMin = false) {
        std::vector<Face> facesNMS;
        if (faces.empty()) {
            return facesNMS;
        }

        std::sort(faces.begin(), faces.end(), [](const Face& f1, const Face& f2) {
            return f1.score > f2.score;
        });

        std::vector<int> indices(faces.size());
        std::iota(indices.begin(), indices.end(), 0);

        while (indices.size() > 0) {
            const int idx = indices[0];
            facesNMS.push_back(faces[idx]);
            std::vector<int> tmpIndices = indices;
            indices.clear();
            const float area1 = static_cast<float>(faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) *
                static_cast<float>(faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
            for (size_t i = 1; i < tmpIndices.size(); ++i) {
                int tmpIdx = tmpIndices[i];
                const float interX1 = static_cast<float>(std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1));
                const float interY1 = static_cast<float>(std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1));
                const float interX2 = static_cast<float>(std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2));
                const float interY2 = static_cast<float>(std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2));

                const float bboxWidth = std::max(0.0f, (interX2 - interX1 + 1));
                const float bboxHeight = std::max(0.0f, (interY2 - interY1 + 1));

                const float interArea = bboxWidth * bboxHeight;
                const float area2 = static_cast<float>(faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) *
                    static_cast<float>(faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
                float overlap = 0.0;
                if (useMin) {
                    overlap = interArea / std::min(area1, area2);
                } else {
                    overlap = interArea / (area1 + area2 - interArea);
                }
                if (overlap <= threshold) {
                    indices.push_back(tmpIdx);
                }
            }
        }
        return facesNMS;
    }
};

const float P_NET_WINDOW_SIZE = 12.0f;

std::vector<Face> buildFaces(const cv::Mat& scores,
                             const cv::Mat& regressions,
                             const float scaleFactor,
                             const float threshold) {

    auto w = scores.size[3];
    auto h = scores.size[2];
    auto size = w * h;

    const float* scores_data = scores.ptr<float>();
    scores_data += size;

    const float* reg_data = regressions.ptr<float>();

    auto out_side = std::max(h, w);
    auto in_side = 2 * out_side + 11;
    float stride = 0.0f;
    if (out_side != 1)
    {
        stride = static_cast<float>(in_side - P_NET_WINDOW_SIZE) / static_cast<float>(out_side - 1);
    }

    std::vector<Face> boxes;

    for (int i = 0; i < size; i++) {
        if (scores_data[i] >= (threshold)) {
            float y = static_cast<float>(i / w);
            float x = static_cast<float>(i - w * y);

            Face faceInfo;
            BBox& faceBox = faceInfo.bbox;

            faceBox.x1 = std::max(0, static_cast<int>((x * stride) / scaleFactor));
            faceBox.y1 = std::max(0, static_cast<int>((y * stride) / scaleFactor));
            faceBox.x2 = static_cast<int>((x * stride + P_NET_WINDOW_SIZE - 1.0f) / scaleFactor);
            faceBox.y2 = static_cast<int>((y * stride + P_NET_WINDOW_SIZE - 1.0f) / scaleFactor);
            faceInfo.regression[0] = reg_data[i];
            faceInfo.regression[1] = reg_data[i + size];
            faceInfo.regression[2] = reg_data[i + 2 * size];
            faceInfo.regression[3] = reg_data[i + 3 * size];
            faceInfo.score = scores_data[i];
            boxes.push_back(faceInfo);
        }
    }

    return boxes;
}

// Define networks for this sample
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
using GMats = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;

G_API_NET(MTCNNRefinement,
          <GMat2(cv::GMat)>,
          "sample.custom.mtcnn_refinement");

G_API_NET(MTCNNOutput,
          <GMat3(cv::GMat)>,
          "sample.custom.mtcnn_output");

using GFaces = cv::GArray<Face>;
G_API_OP(BuildFaces,
         <GFaces(cv::GMat, cv::GMat, float, float)>,
         "sample.custom.mtcnn.build_faces") {
         static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                       const cv::GMatDesc&,
                                       const float,
                                       const float) {
              return cv::empty_array_desc();
    }
};

G_API_OP(RunNMS,
         <GFaces(GFaces, float, bool)>,
         "sample.custom.mtcnn.run_nms") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const float, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(AccumulatePyramidOutputs,
         <GFaces(GFaces, GFaces)>,
         "sample.custom.mtcnn.accumulate_pyramid_outputs") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ApplyRegression,
         <GFaces(GFaces, bool)>,
         "sample.custom.mtcnn.apply_regression") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(BBoxesToSquares,
         <GFaces(GFaces)>,
         "sample.custom.mtcnn.bboxes_to_squares") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(R_O_NetPreProcGetROIs,
         <GRects(GFaces, GSize)>,
         "sample.custom.mtcnn.bboxes_r_o_net_preproc_get_rois") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GOpaqueDesc&) {
              return cv::empty_array_desc();
    }
};


G_API_OP(RNetPostProc,
         <GFaces(GFaces, GMats, GMats, float)>,
         "sample.custom.mtcnn.rnet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const float) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ONetPostProc,
         <GFaces(GFaces, GMats, GMats, GMats, float)>,
         "sample.custom.mtcnn.onet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const float) {
             return cv::empty_array_desc();
    }
};

G_API_OP(SwapFaces,
         <GFaces(GFaces)>,
         "sample.custom.mtcnn.swap_faces") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

//Custom kernels implementation
GAPI_OCV_KERNEL(OCVBuildFaces, BuildFaces) {
    static void run(const cv::Mat & in_scores,
                    const cv::Mat & in_regresssions,
                    const float scaleFactor,
                    const float threshold,
                    std::vector<Face> &out_faces) {
        out_faces = buildFaces(in_scores, in_regresssions, scaleFactor, threshold);
    }
};// GAPI_OCV_KERNEL(BuildFaces)

GAPI_OCV_KERNEL(OCVRunNMS, RunNMS) {
    static void run(const std::vector<Face> &in_faces,
                    const float threshold,
                    const bool useMin,
                    std::vector<Face> &out_faces) {
                    std::vector<Face> in_faces_copy = in_faces;
        out_faces = Face::runNMS(in_faces_copy, threshold, useMin);
    }
};// GAPI_OCV_KERNEL(RunNMS)

GAPI_OCV_KERNEL(OCVAccumulatePyramidOutputs, AccumulatePyramidOutputs) {
    static void run(const std::vector<Face> &total_faces,
                    const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
                    out_faces = total_faces;
        out_faces.insert(out_faces.end(), in_faces.begin(), in_faces.end());
    }
};// GAPI_OCV_KERNEL(AccumulatePyramidOutputs)

GAPI_OCV_KERNEL(OCVApplyRegression, ApplyRegression) {
    static void run(const std::vector<Face> &in_faces,
                    const bool addOne,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::applyRegression(in_faces_copy, addOne);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
    }
};// GAPI_OCV_KERNEL(ApplyRegression)

GAPI_OCV_KERNEL(OCVBBoxesToSquares, BBoxesToSquares) {
    static void run(const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        Face::bboxes2Squares(in_faces_copy);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
    }
};// GAPI_OCV_KERNEL(BBoxesToSquares)

GAPI_OCV_KERNEL(OCVR_O_NetPreProcGetROIs, R_O_NetPreProcGetROIs) {
    static void run(const std::vector<Face> &in_faces,
                    const cv::Size & in_image_size,
                    std::vector<cv::Rect> &outs) {
        outs.clear();
        for (const auto& face : in_faces) {
            cv::Rect tmp_rect = face.bbox.getRect();
            //Compare to transposed sizes width<->height
            tmp_rect &= cv::Rect(tmp_rect.x, tmp_rect.y, in_image_size.height - tmp_rect.x, in_image_size.width - tmp_rect.y) &
                        cv::Rect(0, 0, in_image_size.height, in_image_size.width);
            outs.push_back(tmp_rect);
        }
    }
};// GAPI_OCV_KERNEL(R_O_NetPreProcGetROIs)


GAPI_OCV_KERNEL(OCVRNetPostProc, RNetPostProc) {
    static void run(const std::vector<Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const float threshold,
                    std::vector<Face> &out_faces) {
        out_faces.clear();
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                Face info = in_faces[k];
                info.score = scores_data[1];
                std::copy_n(reg_data, NUM_REGRESSIONS, info.regression.begin());
                out_faces.push_back(info);
            }
        }
    }
};// GAPI_OCV_KERNEL(RNetPostProc)

GAPI_OCV_KERNEL(OCVONetPostProc, ONetPostProc) {
    static void run(const std::vector<Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const std::vector<cv::Mat> &in_landmarks,
                    const float threshold,
                    std::vector<Face> &out_faces) {
        out_faces.clear();
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            const float* landmark_data = in_landmarks[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                Face info = in_faces[k];
                info.score = scores_data[1];
                for (size_t i = 0; i < 4; ++i) {
                    info.regression[i] = reg_data[i];
                }
                float w = info.bbox.x2 - info.bbox.x1 + 1.0f;
                float h = info.bbox.y2 - info.bbox.y1 + 1.0f;

                for (size_t p = 0; p < NUM_PTS; ++p) {
                    info.ptsCoords[2 * p] =
                        info.bbox.x1 + static_cast<float>(landmark_data[NUM_PTS + p]) * w - 1;
                    info.ptsCoords[2 * p + 1] = info.bbox.y1 + static_cast<float>(landmark_data[p]) * h - 1;
                }

                out_faces.push_back(info);
            }
        }
    }
};// GAPI_OCV_KERNEL(ONetPostProc)

GAPI_OCV_KERNEL(OCVSwapFaces, SwapFaces) {
    static void run(const std::vector<Face> &in_faces,
                    std::vector<Face> &out_faces) {
        std::vector<Face> in_faces_copy = in_faces;
        out_faces.clear();
        if (!in_faces_copy.empty()) {
            for (size_t i = 0; i < in_faces_copy.size(); ++i) {
                std::swap(in_faces_copy[i].bbox.x1, in_faces_copy[i].bbox.y1);
                std::swap(in_faces_copy[i].bbox.x2, in_faces_copy[i].bbox.y2);
                for (size_t p = 0; p < NUM_PTS; ++p) {
                    std::swap(in_faces_copy[i].ptsCoords[2 * p], in_faces_copy[i].ptsCoords[2 * p + 1]);
                }
            }
            out_faces = in_faces_copy;
        }
    }
};// GAPI_OCV_KERNEL(SwapFaces)

} // anonymous namespace
} // namespace custom

namespace vis {
namespace {
void bbox(const cv::Mat& m, const cv::Rect& rc) {
    cv::rectangle(m, rc, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
};

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat& img,
    const std::vector<rectPoints> data) {
    cv::Mat outImg;
    img.copyTo(outImg);

    for (const auto& el : data) {
        vis::bbox(outImg, el.first);
        auto pts = el.second;
        for (size_t i = 0; i < pts.size(); ++i) {
            cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
        }
    }
    return outImg;
}
} // anonymous namespace
} // namespace vis


//Infer helper function
namespace {
static inline std::tuple<cv::GMat, cv::GMat> run_mtcnn_p(cv::GMat &in, const std::string &id) {
    cv::GInferInputs inputs;
    inputs["data"] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(id, inputs);
    auto regressions = outputs.at("conv4-2");
    auto scores = outputs.at("prob1");
    return std::make_tuple(regressions, scores);
}

static inline std::string get_pnet_level_name(const cv::Size &in_size) {
    return "MTCNNProposal_" + std::to_string(in_size.width) + "x" + std::to_string(in_size.height);
}

int calculate_scales(const cv::Size &input_size, std::vector<double> &out_scales, std::vector<cv::Size> &out_sizes ) {
    //calculate multi - scale and limit the maxinum side to 1000
    //pr_scale: limit the maxinum side to 1000, < 1.0
    double pr_scale = 1.0;
    double h = static_cast<double>(input_size.height);
    double w = static_cast<double>(input_size.width);
    if (std::min(w, h) > 1000)
    {
        pr_scale = 1000.0 / std::min(h, w);
        w = w * pr_scale;
        h = h * pr_scale;
    }
    else if (std::max(w, h) < 1000)
    {
        w = w * pr_scale;
        h = h * pr_scale;
    }
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.709;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12)
    {
        const double current_scale = pr_scale * std::pow(factor, factor_count);
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
    }
    return factor_count;
}

int calculate_half_scales(const cv::Size &input_size, std::vector<double>& out_scales, std::vector<cv::Size>& out_sizes) {
    double pr_scale = 0.5;
    const double h = static_cast<double>(input_size.height);
    const double w = static_cast<double>(input_size.width);
    //multi - scale
    out_scales.clear();
    out_sizes.clear();
    const double factor = 0.5;
    int factor_count = 0;
    double minl = std::min(h, w);
    while (minl >= 12.0*2.0)
    {
        const double current_scale = pr_scale;
        cv::Size current_size(static_cast<int>(static_cast<double>(input_size.width) * current_scale),
                              static_cast<int>(static_cast<double>(input_size.height) * current_scale));
        out_scales.push_back(current_scale);
        out_sizes.push_back(current_size);
        minl *= factor;
        factor_count += 1;
        pr_scale *= 0.5;
    }
    return factor_count;
}

const int MAX_PYRAMID_LEVELS = 13;
//////////////////////////////////////////////////////////////////////
} // anonymous namespace

int main(int argc, char* argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const auto input_file_name = cmd.get<std::string>("input");
    const auto model_path_p = cmd.get<std::string>("mtcnnpm");
    const auto target_dev_p = cmd.get<std::string>("mtcnnpd");
    const auto conf_thresh_p = cmd.get<float>("thrp");
    const auto model_path_r = cmd.get<std::string>("mtcnnrm");
    const auto target_dev_r = cmd.get<std::string>("mtcnnrd");
    const auto conf_thresh_r = cmd.get<float>("thrr");
    const auto model_path_o = cmd.get<std::string>("mtcnnom");
    const auto target_dev_o = cmd.get<std::string>("mtcnnod");
    const auto conf_thresh_o = cmd.get<float>("thro");
    const auto use_half_scale = cmd.get<bool>("half_scale");
    const auto streaming_queue_capacity = cmd.get<unsigned int>("queue_capacity");

    std::vector<cv::Size> level_size;
    std::vector<double> scales;
    //MTCNN input size
    cv::VideoCapture cap;
    cap.open(input_file_name);
    if (!cap.isOpened())
        CV_Assert(false);
    auto in_rsz = cv::Size{ static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) };
    //Calculate scales, number of pyramid levels and sizes for PNet pyramid
    auto pyramid_levels = use_half_scale ? calculate_half_scales(in_rsz, scales, level_size) :
                                           calculate_scales(in_rsz, scales, level_size);
    CV_Assert(pyramid_levels <= MAX_PYRAMID_LEVELS);

    //Proposal part of MTCNN graph
    //Preprocessing BGR2RGB + transpose (NCWH is expected instead of NCHW)
    cv::GMat in_original;
    cv::GMat in_originalRGB = cv::gapi::BGR2RGB(in_original);
    cv::GOpaque<cv::Size> in_sz = cv::gapi::streaming::size(in_original);
    cv::GMat in_resized[MAX_PYRAMID_LEVELS];
    cv::GMat in_transposed[MAX_PYRAMID_LEVELS];
    cv::GMat regressions[MAX_PYRAMID_LEVELS];
    cv::GMat scores[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> nms_p_faces[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> total_faces[MAX_PYRAMID_LEVELS];
    cv::GArray<custom::Face> faces_init(std::vector<custom::Face>{});

    //The very first PNet pyramid layer to init total_faces[0]
    in_resized[0] = cv::gapi::resize(in_originalRGB, level_size[0]);
    in_transposed[0] = cv::gapi::transpose(in_resized[0]);
    std::tie(regressions[0], scores[0]) = run_mtcnn_p(in_transposed[0], get_pnet_level_name(level_size[0]));
    cv::GArray<custom::Face> faces0 = custom::BuildFaces::on(scores[0], regressions[0], static_cast<float>(scales[0]), conf_thresh_p);
    cv::GArray<custom::Face> final_p_faces_for_bb2squares = custom::ApplyRegression::on(faces0, true);
    cv::GArray<custom::Face> final_faces_pnet0 = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares);
    nms_p_faces[0] = custom::RunNMS::on(final_faces_pnet0, 0.5f, false);
    total_faces[0] = custom::AccumulatePyramidOutputs::on(faces_init, nms_p_faces[0]);
    //The rest PNet pyramid layers to accumlate all layers result in total_faces[PYRAMID_LEVELS - 1]]
    for (int i = 1; i < pyramid_levels; ++i)
    {
        in_resized[i] = cv::gapi::resize(in_originalRGB, level_size[i]);
        in_transposed[i] = cv::gapi::transpose(in_resized[i]);
        std::tie(regressions[i], scores[i]) = run_mtcnn_p(in_transposed[i], get_pnet_level_name(level_size[i]));
        cv::GArray<custom::Face> faces = custom::BuildFaces::on(scores[i], regressions[i], static_cast<float>(scales[i]), conf_thresh_p);
        cv::GArray<custom::Face> final_p_faces_for_bb2squares_i = custom::ApplyRegression::on(faces, true);
        cv::GArray<custom::Face> final_faces_pnet_i = custom::BBoxesToSquares::on(final_p_faces_for_bb2squares_i);
        nms_p_faces[i] = custom::RunNMS::on(final_faces_pnet_i, 0.5f, false);
        total_faces[i] = custom::AccumulatePyramidOutputs::on(total_faces[i - 1], nms_p_faces[i]);
    }

    //Proposal post-processing
    cv::GArray<custom::Face> final_faces_pnet = custom::RunNMS::on(total_faces[pyramid_levels - 1], 0.7f, true);

    //Refinement part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_pnet = custom::R_O_NetPreProcGetROIs::on(final_faces_pnet, in_sz);
    cv::GArray<cv::GMat> regressionsRNet, scoresRNet;
    cv::GMat in_originalRGB_transposed = cv::gapi::transpose(in_originalRGB);
    std::tie(regressionsRNet, scoresRNet) = cv::gapi::infer<custom::MTCNNRefinement>(faces_roi_pnet, in_originalRGB_transposed);

    //Refinement post-processing
    cv::GArray<custom::Face> rnet_post_proc_faces = custom::RNetPostProc::on(final_faces_pnet, scoresRNet, regressionsRNet, conf_thresh_r);
    cv::GArray<custom::Face> nms07_r_faces_total = custom::RunNMS::on(rnet_post_proc_faces, 0.7f, false);
    cv::GArray<custom::Face> final_r_faces_for_bb2squares = custom::ApplyRegression::on(nms07_r_faces_total, true);
    cv::GArray<custom::Face> final_faces_rnet = custom::BBoxesToSquares::on(final_r_faces_for_bb2squares);

    //Output part of MTCNN graph
    cv::GArray<cv::Rect> faces_roi_rnet = custom::R_O_NetPreProcGetROIs::on(final_faces_rnet, in_sz);
    cv::GArray<cv::GMat> regressionsONet, scoresONet, landmarksONet;
    std::tie(regressionsONet, landmarksONet, scoresONet) = cv::gapi::infer<custom::MTCNNOutput>(faces_roi_rnet, in_originalRGB_transposed);

    //Output post-processing
    cv::GArray<custom::Face> onet_post_proc_faces = custom::ONetPostProc::on(final_faces_rnet, scoresONet, regressionsONet, landmarksONet, conf_thresh_o);
    cv::GArray<custom::Face> final_o_faces_for_nms07 = custom::ApplyRegression::on(onet_post_proc_faces, true);
    cv::GArray<custom::Face> nms07_o_faces_total = custom::RunNMS::on(final_o_faces_for_nms07, 0.7f, true);
    cv::GArray<custom::Face> final_faces_onet = custom::SwapFaces::on(nms07_o_faces_total);

    cv::GComputation graph_mtcnn(cv::GIn(in_original), cv::GOut(cv::gapi::copy(in_original), final_faces_onet));

    // MTCNN Refinement detection network
    auto mtcnnr_net = cv::gapi::ie::Params<custom::MTCNNRefinement>{
        model_path_r,                // path to topology IR
        weights_path(model_path_r),  // path to weights
        target_dev_r,                // device specifier
    }.cfgOutputLayers({ "conv5-2", "prob1" }).cfgInputLayers({ "data" });

    // MTCNN Output detection network
    auto mtcnno_net = cv::gapi::ie::Params<custom::MTCNNOutput>{
        model_path_o,                // path to topology IR
        weights_path(model_path_o),  // path to weights
        target_dev_o,                // device specifier
    }.cfgOutputLayers({ "conv6-2", "conv6-3", "prob1" }).cfgInputLayers({ "data" });

    auto networks_mtcnn = cv::gapi::networks(mtcnnr_net, mtcnno_net);

    // MTCNN Proposal detection network
    for (int i = 0; i < pyramid_levels; ++i)
    {
        std::string net_id = get_pnet_level_name(level_size[i]);
        std::vector<size_t> reshape_dims = { 1, 3, (size_t)level_size[i].width, (size_t)level_size[i].height };
        cv::gapi::ie::Params<cv::gapi::Generic> mtcnnp_net{
                    net_id,                      // tag
                    model_path_p,                // path to topology IR
                    weights_path(model_path_p),  // path to weights
                    target_dev_p,                // device specifier
        };
        mtcnnp_net.cfgInputReshape({ {"data", reshape_dims} });
        networks_mtcnn += cv::gapi::networks(mtcnnp_net);
    }

    auto kernels_mtcnn = cv::gapi::kernels< custom::OCVBuildFaces
                                          , custom::OCVRunNMS
                                          , custom::OCVAccumulatePyramidOutputs
                                          , custom::OCVApplyRegression
                                          , custom::OCVBBoxesToSquares
                                          , custom::OCVR_O_NetPreProcGetROIs
                                          , custom::OCVRNetPostProc
                                          , custom::OCVONetPostProc
                                          , custom::OCVSwapFaces
    >();
    auto mtcnn_args = cv::compile_args(networks_mtcnn, kernels_mtcnn);
    if (streaming_queue_capacity != 0)
        mtcnn_args += cv::compile_args(cv::gapi::streaming::queue_capacity{ streaming_queue_capacity });
    auto pipeline_mtcnn = graph_mtcnn.compileStreaming(std::move(mtcnn_args));

    std::cout << "Reading " << input_file_name << std::endl;
    // Input stream
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name);

    // Set the pipeline source & start the pipeline
    pipeline_mtcnn.setSource(cv::gin(in_src));
    pipeline_mtcnn.start();

    // Declare the output data & run the processing loop
    cv::TickMeter tm;
    cv::Mat image;
    std::vector<custom::Face> out_faces;

    tm.start();
    int frames = 0;
    while (pipeline_mtcnn.pull(cv::gout(image, out_faces))) {
        frames++;
        std::cout << "Final Faces Size " << out_faces.size() << std::endl;
        std::vector<vis::rectPoints> data;
        // show the image with faces in it
        for (const auto& out_face : out_faces) {
            std::vector<cv::Point> pts;
            for (size_t p = 0; p < NUM_PTS; ++p) {
                pts.push_back(
                    cv::Point(static_cast<int>(out_face.ptsCoords[2 * p]), static_cast<int>(out_face.ptsCoords[2 * p + 1])));
            }
            auto rect = out_face.bbox.getRect();
            auto d = std::make_pair(rect, pts);
            data.push_back(d);
        }
        // Visualize results on the frame
        auto resultImg = vis::drawRectsAndPoints(image, data);
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        cv::putText(resultImg, fps_str, { 0,32 }, cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
        cv::imshow("Out", resultImg);
        cv::waitKey(1);
        out_faces.clear();
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
        << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
