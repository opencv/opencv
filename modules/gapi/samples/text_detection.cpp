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
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

const std::string about =
    "This is an OpenCV-based version of OMZ Text Detection example";
const std::string keys =
    "{ h help |                           | Print this help message }"
    "{ input  |                           | Path to the input video file }"
    "{ tdm    | text-detection-0004.xml   | Path to OpenVINO text detection model (.xml), versions 0003 and 0004 work }"
    "{ tdd    | CPU                       | Target device for the text detector (e.g. CPU, GPU, VPU, ...) }"
    "{ trm    | text-recognition-0012.xml | Path to OpenVINO text recognition model (.xml) }"
    "{ trd    | CPU                       | Target device for the text recognition (e.g. CPU, GPU, VPU, ...) }"
    "{ bw     | 0                         | CTC beam search decoder bandwidth, if 0, a CTC greedy decoder is used}"
    "{ sset   | 0123456789abcdefghijklmnopqrstuvwxyz | Symbol set to use with text recognition decoder. Shouldn't contain symbol #. }"
    "{ thr    | 0.2                       | Text recognition confidence threshold}"
    ;

namespace {
std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    const auto ext = model_path.substr(sz - EXT_LEN);
    CV_Assert(cv::toLowerCase(ext) == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}

//////////////////////////////////////////////////////////////////////
// Taken from OMZ samples as-is
template<typename Iter>
void softmax_and_choose(Iter begin, Iter end, int *argmax, float *prob) {
    auto max_element = std::max_element(begin, end);
    *argmax = static_cast<int>(std::distance(begin, max_element));
    float max_val = *max_element;
    double sum = 0;
    for (auto i = begin; i != end; i++) {
       sum += std::exp((*i) - max_val);
    }
    if (std::fabs(sum) < std::numeric_limits<double>::epsilon()) {
        throw std::logic_error("sum can't be equal to zero");
    }
    *prob = 1.0f / static_cast<float>(sum);
}

template<typename Iter>
std::vector<float> softmax(Iter begin, Iter end) {
    std::vector<float> prob(end - begin, 0.f);
    std::transform(begin, end, prob.begin(), [](float x) { return std::exp(x); });
    float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
    for (int i = 0; i < static_cast<int>(prob.size()); i++)
        prob[i] /= sum;
    return prob;
}

struct BeamElement {
    std::vector<int> sentence;   //!< The sequence of chars that will be a result of the beam element

    float prob_blank;            //!< The probability that the last char in CTC sequence
                                 //!< for the beam element is the special blank char

    float prob_not_blank;        //!< The probability that the last char in CTC sequence
                                 //!< for the beam element is NOT the special blank char

    float prob() const {         //!< The probability of the beam element.
        return prob_blank + prob_not_blank;
    }
};

std::string CTCGreedyDecoder(const float *data,
                             const std::size_t sz,
                             const std::string &alphabet,
                             const char pad_symbol,
                             double *conf) {
    std::string res = "";
    bool prev_pad = false;
    *conf = 1;

    const auto num_classes = alphabet.length();
    for (auto it = data; it != (data+sz); it += num_classes) {
        int argmax = 0;
        float prob = 0.f;

        softmax_and_choose(it, it + num_classes, &argmax, &prob);
        (*conf) *= prob;

        auto symbol = alphabet[argmax];
        if (symbol != pad_symbol) {
            if (res.empty() || prev_pad || (!res.empty() && symbol != res.back())) {
                prev_pad = false;
                res += symbol;
            }
        } else {
            prev_pad = true;
        }
    }
    return res;
}

std::string CTCBeamSearchDecoder(const float *data,
                                 const std::size_t sz,
                                 const std::string &alphabet,
                                 double *conf,
                                 int bandwidth) {
    const auto num_classes = alphabet.length();

    std::vector<BeamElement> curr;
    std::vector<BeamElement> last;

    last.push_back(BeamElement{std::vector<int>(), 1.f, 0.f});

    for (auto it = data; it != (data+sz); it += num_classes) {
        curr.clear();

        std::vector<float> prob = softmax(it, it + num_classes);

        for(const auto& candidate: last) {
            float prob_not_blank = 0.f;
            const std::vector<int>& candidate_sentence = candidate.sentence;
            if (!candidate_sentence.empty()) {
                int n = candidate_sentence.back();
                prob_not_blank = candidate.prob_not_blank * prob[n];
            }
            float prob_blank = candidate.prob() * prob[num_classes - 1];

            auto check_res = std::find_if(curr.begin(),
                                          curr.end(),
                                          [&candidate_sentence](const BeamElement& n) {
                                              return n.sentence == candidate_sentence;
                                          });
            if (check_res == std::end(curr)) {
                curr.push_back(BeamElement{candidate.sentence, prob_blank, prob_not_blank});
            } else {
                check_res->prob_not_blank  += prob_not_blank;
                if (check_res->prob_blank != 0.f) {
                    throw std::logic_error("Probability that the last char in CTC-sequence "
                                           "is the special blank char must be zero here");
                }
                check_res->prob_blank = prob_blank;
            }

            for (int i = 0; i < static_cast<int>(num_classes) - 1; i++) {
                auto extend = candidate_sentence;
                extend.push_back(i);

                if (candidate_sentence.size() > 0 && candidate.sentence.back() == i) {
                    prob_not_blank = prob[i] * candidate.prob_blank;
                } else {
                    prob_not_blank = prob[i] * candidate.prob();
                }

                auto check_res2 = std::find_if(curr.begin(),
                                              curr.end(),
                                              [&extend](const BeamElement &n) {
                                                  return n.sentence == extend;
                                              });
                if (check_res2 == std::end(curr)) {
                    curr.push_back(BeamElement{extend, 0.f, prob_not_blank});
                } else {
                    check_res2->prob_not_blank += prob_not_blank;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](const BeamElement &a, const BeamElement &b) -> bool {
            return a.prob() > b.prob();
        });

        last.clear();
        int num_to_copy = std::min(bandwidth, static_cast<int>(curr.size()));
        for (int b = 0; b < num_to_copy; b++) {
            last.push_back(curr[b]);
        }
    }

    *conf = last[0].prob();
    std::string res="";
    for (const auto& idx: last[0].sentence) {
        res += alphabet[idx];
    }

    return res;
}

//////////////////////////////////////////////////////////////////////
} // anonymous namespace

namespace custom {
namespace {

//////////////////////////////////////////////////////////////////////
// Define networks for this sample
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(TextDetection,
          <GMat2(cv::GMat)>,
          "sample.custom.text_detect");

G_API_NET(TextRecognition,
          <cv::GMat(cv::GMat)>,
          "sample.custom.text_recogn");

// Define custom operations
using GSize = cv::GOpaque<cv::Size>;
using GRRects = cv::GArray<cv::RotatedRect>;
G_API_OP(PostProcess,
        <GRRects(cv::GMat,cv::GMat,GSize,float,float)>,
        "sample.custom.text.post_proc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  const cv::GMatDesc &,
                                  const cv::GOpaqueDesc &,
                                  float,
                                  float) {
        return cv::empty_array_desc();
    }
};

using GMats = cv::GArray<cv::GMat>;
G_API_OP(CropLabels,
         <GMats(cv::GMat,GRRects,GSize)>,
         "sample.custom.text.crop") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  const cv::GArrayDesc &,
                                  const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

//////////////////////////////////////////////////////////////////////
// Implement custom operations
GAPI_OCV_KERNEL(OCVPostProcess, PostProcess) {
    static void run(const cv::Mat &link,
                    const cv::Mat &segm,
                    const cv::Size &img_size,
                    const float link_threshold,
                    const float segm_threshold,
                    std::vector<cv::RotatedRect> &out) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int kMinArea = 300;
        const int kMinHeight = 10;

        const float *link_data_pointer = link.ptr<float>();
        std::vector<float> link_data(link_data_pointer, link_data_pointer + link.total());
        link_data = transpose4d(link_data, dimsToShape(link.size), {0, 2, 3, 1});
        softmax(link_data);
        link_data = sliceAndGetSecondChannel(link_data);
        std::vector<int> new_link_data_shape = {
            link.size[0],
            link.size[2],
            link.size[3],
            link.size[1]/2,
        };

        const float *cls_data_pointer = segm.ptr<float>();
        std::vector<float> cls_data(cls_data_pointer, cls_data_pointer + segm.total());
        cls_data = transpose4d(cls_data, dimsToShape(segm.size), {0, 2, 3, 1});
        softmax(cls_data);
        cls_data = sliceAndGetSecondChannel(cls_data);
        std::vector<int> new_cls_data_shape = {
            segm.size[0],
            segm.size[2],
            segm.size[3],
            segm.size[1]/2,
        };

        out = maskToBoxes(decodeImageByJoin(cls_data, new_cls_data_shape,
                                            link_data, new_link_data_shape,
                                            segm_threshold, link_threshold),
                          static_cast<float>(kMinArea),
                          static_cast<float>(kMinHeight),
                          img_size);
    }

    static std::vector<std::size_t> dimsToShape(const cv::MatSize &sz) {
        const int n_dims = sz.dims();
        std::vector<std::size_t> result;
        result.reserve(n_dims);

        // cv::MatSize is not iterable...
        for (int i = 0; i < n_dims; i++) {
            result.emplace_back(static_cast<std::size_t>(sz[i]));
        }
        return result;
    }

    static void softmax(std::vector<float> &rdata) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const size_t last_dim = 2;
        for (size_t i = 0 ; i < rdata.size(); i+=last_dim) {
            float m = std::max(rdata[i], rdata[i+1]);
            rdata[i] = std::exp(rdata[i] - m);
            rdata[i + 1] = std::exp(rdata[i + 1] - m);
            float s = rdata[i] + rdata[i + 1];
            rdata[i] /= s;
            rdata[i + 1] /= s;
        }
    }

    static std::vector<float> transpose4d(const std::vector<float> &data,
                                          const std::vector<size_t> &shape,
                                          const std::vector<size_t> &axes) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        if (shape.size() != axes.size())
            throw std::runtime_error("Shape and axes must have the same dimension.");

        for (size_t a : axes) {
            if (a >= shape.size())
                throw std::runtime_error("Axis must be less than dimension of shape.");
        }
        size_t total_size = shape[0]*shape[1]*shape[2]*shape[3];
        std::vector<size_t> steps {
            shape[axes[1]]*shape[axes[2]]*shape[axes[3]],
            shape[axes[2]]*shape[axes[3]],
            shape[axes[3]],
            1
         };

        size_t source_data_idx = 0;
        std::vector<float> new_data(total_size, 0);
        std::vector<size_t> ids(shape.size());
        for (ids[0] = 0; ids[0] < shape[0]; ids[0]++) {
            for (ids[1] = 0; ids[1] < shape[1]; ids[1]++) {
                for (ids[2] = 0; ids[2] < shape[2]; ids[2]++) {
                    for (ids[3]= 0; ids[3] < shape[3]; ids[3]++) {
                        size_t new_data_idx = ids[axes[0]]*steps[0] + ids[axes[1]]*steps[1] +
                            ids[axes[2]]*steps[2] + ids[axes[3]]*steps[3];
                        new_data[new_data_idx] = data[source_data_idx++];
                    }
                }
            }
        }
        return new_data;
    }

    static std::vector<float> sliceAndGetSecondChannel(const std::vector<float> &data) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::vector<float> new_data(data.size() / 2, 0);
        for (size_t i = 0; i < data.size() / 2; i++) {
            new_data[i] = data[2 * i + 1];
        }
        return new_data;
    }

    static void join(const int p1,
                     const int p2,
                     std::unordered_map<int, int> &group_mask) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int root1 = findRoot(p1, group_mask);
        const int root2 = findRoot(p2, group_mask);
        if (root1 != root2) {
            group_mask[root1] = root2;
        }
    }

    static cv::Mat decodeImageByJoin(const std::vector<float> &cls_data,
                                     const std::vector<int>   &cls_data_shape,
                                     const std::vector<float> &link_data,
                                     const std::vector<int>   &link_data_shape,
                                     float cls_conf_threshold,
                                     float link_conf_threshold) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        const int h = cls_data_shape[1];
        const int w = cls_data_shape[2];

        std::vector<uchar> pixel_mask(h * w, 0);
        std::unordered_map<int, int> group_mask;
        std::vector<cv::Point> points;
        for (int i = 0; i < static_cast<int>(pixel_mask.size()); i++) {
            pixel_mask[i] = cls_data[i] >= cls_conf_threshold;
            if (pixel_mask[i]) {
                points.emplace_back(i % w, i / w);
                group_mask[i] = -1;
            }
        }
        std::vector<uchar> link_mask(link_data.size(), 0);
        for (size_t i = 0; i < link_mask.size(); i++) {
            link_mask[i] = link_data[i] >= link_conf_threshold;
        }
        size_t neighbours = size_t(link_data_shape[3]);
        for (const auto &point : points) {
            size_t neighbour = 0;
            for (int ny = point.y - 1; ny <= point.y + 1; ny++) {
                for (int nx = point.x - 1; nx <= point.x + 1; nx++) {
                    if (nx == point.x && ny == point.y)
                        continue;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        uchar pixel_value = pixel_mask[size_t(ny) * size_t(w) + size_t(nx)];
                        uchar link_value = link_mask[(size_t(point.y) * size_t(w) + size_t(point.x))
                                                     *neighbours + neighbour];
                        if (pixel_value && link_value) {
                            join(point.x + point.y * w, nx + ny * w, group_mask);
                        }
                    }
                    neighbour++;
                }
            }
        }
        return get_all(points, w, h, group_mask);
    }

    static cv::Mat get_all(const std::vector<cv::Point> &points,
                           const int w,
                           const int h,
                           std::unordered_map<int, int> &group_mask) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::unordered_map<int, int> root_map;
        cv::Mat mask(h, w, CV_32S, cv::Scalar(0));
        for (const auto &point : points) {
            int point_root = findRoot(point.x + point.y * w, group_mask);
            if (root_map.find(point_root) == root_map.end()) {
                root_map.emplace(point_root, static_cast<int>(root_map.size() + 1));
            }
            mask.at<int>(point.x + point.y * w) = root_map[point_root];
        }
        return mask;
    }

    static int findRoot(const int point,
                        std::unordered_map<int, int> &group_mask) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        int root = point;
        bool update_parent = false;
        while (group_mask.at(root) != -1) {
            root = group_mask.at(root);
            update_parent = true;
        }
        if (update_parent) {
            group_mask[point] = root;
        }
        return root;
    }

    static std::vector<cv::RotatedRect> maskToBoxes(const cv::Mat &mask,
                                                    const float min_area,
                                                    const float min_height,
                                                    const cv::Size &image_size) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        std::vector<cv::RotatedRect> bboxes;
        double min_val = 0.;
        double max_val = 0.;
        cv::minMaxLoc(mask, &min_val, &max_val);
        int max_bbox_idx = static_cast<int>(max_val);
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, image_size, 0, 0, cv::INTER_NEAREST);

        for (int i = 1; i <= max_bbox_idx; i++) {
            cv::Mat bbox_mask = resized_mask == i;
            std::vector<std::vector<cv::Point>> contours;

            cv::findContours(bbox_mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty())
                continue;
            cv::RotatedRect r = cv::minAreaRect(contours[0]);
            if (std::min(r.size.width, r.size.height) < min_height)
                continue;
            if (r.size.area() < min_area)
                continue;
            bboxes.emplace_back(r);
        }
        return bboxes;
    }
}; // GAPI_OCV_KERNEL(PostProcess)

GAPI_OCV_KERNEL(OCVCropLabels, CropLabels) {
    static void run(const cv::Mat &image,
                    const std::vector<cv::RotatedRect> &detections,
                    const cv::Size &outSize,
                    std::vector<cv::Mat> &out) {
        out.clear();
        out.reserve(detections.size());
        cv::Mat crop(outSize, CV_8UC3, cv::Scalar(0));
        cv::Mat gray(outSize, CV_8UC1, cv::Scalar(0));
        std::vector<int> blob_shape = {1,1,outSize.height,outSize.width};

        for (auto &&rr : detections) {
            std::vector<cv::Point2f> points(4);
            rr.points(points.data());

            const auto top_left_point_idx = topLeftPointIdx(points);
            cv::Point2f point0 = points[static_cast<size_t>(top_left_point_idx)];
            cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
            cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

            std::vector<cv::Point2f> from{point0, point1, point2};
            std::vector<cv::Point2f> to{
                cv::Point2f(0.0f, 0.0f),
                cv::Point2f(static_cast<float>(outSize.width-1), 0.0f),
                cv::Point2f(static_cast<float>(outSize.width-1),
                            static_cast<float>(outSize.height-1))
            };
            cv::Mat M = cv::getAffineTransform(from, to);
            cv::warpAffine(image, crop, M, outSize);
            cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

            cv::Mat blob;
            gray.convertTo(blob, CV_32F);
            out.push_back(blob.reshape(1, blob_shape)); // pass as 1,1,H,W instead of H,W
        }
    }

    static int topLeftPointIdx(const std::vector<cv::Point2f> &points) {
        // NOTE: Taken from the OMZ text detection sample almost as-is
        cv::Point2f most_left(std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max());
        cv::Point2f almost_most_left(std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max());
        int most_left_idx = -1;
        int almost_most_left_idx = -1;

        for (size_t i = 0; i < points.size() ; i++) {
            if (most_left.x > points[i].x) {
                if (most_left.x < std::numeric_limits<float>::max()) {
                    almost_most_left = most_left;
                    almost_most_left_idx = most_left_idx;
                }
                most_left = points[i];
                most_left_idx = static_cast<int>(i);
            }
            if (almost_most_left.x > points[i].x && points[i] != most_left) {
                almost_most_left = points[i];
                almost_most_left_idx = static_cast<int>(i);
            }
        }

        if (almost_most_left.y < most_left.y) {
            most_left = almost_most_left;
            most_left_idx = almost_most_left_idx;
        }
        return most_left_idx;
    }

}; // GAPI_OCV_KERNEL(CropLabels)

} // anonymous namespace
} // namespace custom

namespace vis {
namespace {

void drawRotatedRect(cv::Mat &m, const cv::RotatedRect &rc) {
    std::vector<cv::Point2f> tmp_points(5);
    rc.points(tmp_points.data());
    tmp_points[4] = tmp_points[0];
    auto prev = tmp_points.begin(), it = prev+1;
    for (; it != tmp_points.end(); ++it) {
        cv::line(m, *prev, *it, cv::Scalar(50, 205, 50), 2);
        prev = it;
    }
}

void drawText(cv::Mat &m, const cv::RotatedRect &rc, const std::string &str) {
    const int    fface   = cv::FONT_HERSHEY_SIMPLEX;
    const double scale   = 0.7;
    const int    thick   = 1;
          int    base    = 0;
    const auto text_size = cv::getTextSize(str, fface, scale, thick, &base);

    std::vector<cv::Point2f> tmp_points(4);
    rc.points(tmp_points.data());
    const auto tl_point_idx = custom::OCVCropLabels::topLeftPointIdx(tmp_points);
    cv::Point text_pos = tmp_points[tl_point_idx];
    text_pos.x = std::max(0, text_pos.x);
    text_pos.y = std::max(text_size.height, text_pos.y);

    cv::rectangle(m,
                  text_pos + cv::Point{0, base},
                  text_pos + cv::Point{text_size.width, -text_size.height},
                  CV_RGB(50, 205, 50),
                  cv::FILLED);
    const auto white = CV_RGB(255, 255, 255);
    cv::putText(m, str, text_pos, fface, scale, white, thick, 8);
}

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
    const auto input_file_name = cmd.get<std::string>("input");
    const auto tdet_model_path = cmd.get<std::string>("tdm");
    const auto trec_model_path = cmd.get<std::string>("trm");
    const auto tdet_target_dev = cmd.get<std::string>("tdd");
    const auto trec_target_dev = cmd.get<std::string>("trd");
    const auto ctc_beam_dec_bw = cmd.get<int>("bw");
    const auto dec_conf_thresh = cmd.get<double>("thr");

    const auto pad_symbol      = '#';
    const auto symbol_set      = cmd.get<std::string>("sset") + pad_symbol;

    cv::GMat in;
    cv::GOpaque<cv::Size> in_rec_sz;
    cv::GMat link, segm;
    std::tie(link, segm) = cv::gapi::infer<custom::TextDetection>(in);
    cv::GOpaque<cv::Size> size = cv::gapi::streaming::size(in);
    cv::GArray<cv::RotatedRect> rrs = custom::PostProcess::on(link, segm, size, 0.8f, 0.8f);
    cv::GArray<cv::GMat> labels = custom::CropLabels::on(in, rrs, in_rec_sz);
    cv::GArray<cv::GMat> text = cv::gapi::infer2<custom::TextRecognition>(in, labels);

    cv::GComputation graph(cv::GIn(in, in_rec_sz),
                           cv::GOut(cv::gapi::copy(in), rrs, text));

    // Text detection network
    auto tdet_net = cv::gapi::ie::Params<custom::TextDetection> {
        tdet_model_path,                // path to topology IR
        weights_path(tdet_model_path),  // path to weights
        tdet_target_dev,                // device specifier
    }.cfgOutputLayers({"model/link_logits_/add", "model/segm_logits/add"});

    auto trec_net = cv::gapi::ie::Params<custom::TextRecognition> {
        trec_model_path,                // path to topology IR
        weights_path(trec_model_path),  // path to weights
        trec_target_dev,                // device specifier
    };
    auto networks = cv::gapi::networks(tdet_net, trec_net);

    auto kernels = cv::gapi::kernels< custom::OCVPostProcess
                                    , custom::OCVCropLabels
                                    >();
    auto pipeline = graph.compileStreaming(cv::compile_args(kernels, networks));

    std::cout << "Reading " << input_file_name << std::endl;

    // Input stream
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input_file_name);

    // Text recognition input size (also an input parameter to the graph)
    auto in_rsz = cv::Size{ 120, 32 };

    // Set the pipeline source & start the pipeline
    pipeline.setSource(cv::gin(in_src, in_rsz));
    pipeline.start();

    // Declare the output data & run the processing loop
    cv::TickMeter tm;
    cv::Mat image;
    std::vector<cv::RotatedRect> out_rcs;
    std::vector<cv::Mat> out_text;

    tm.start();
    int frames = 0;
    while (pipeline.pull(cv::gout(image, out_rcs, out_text))) {
        frames++;

        CV_Assert(out_rcs.size() == out_text.size());
        const auto num_labels = out_rcs.size();

        std::vector<cv::Point2f> tmp_points(4);
        for (std::size_t l = 0; l < num_labels; l++) {
            // Decode the recognized text in the rectangle
            const auto &blob = out_text[l];
            const float *data = blob.ptr<float>();
            const auto sz = blob.total();
            double conf = 1.0;
            const std::string res = ctc_beam_dec_bw == 0
                ? CTCGreedyDecoder(data, sz, symbol_set, pad_symbol, &conf)
                : CTCBeamSearchDecoder(data, sz, symbol_set, &conf, ctc_beam_dec_bw);

            // Draw a bounding box for this rotated rectangle
            const auto &rc = out_rcs[l];
            vis::drawRotatedRect(image, rc);

            // Draw text, if decoded
            if (conf >= dec_conf_thresh) {
                vis::drawText(image, rc, res);
            }
        }
        tm.stop();
        cv::imshow("Out", image);
        cv::waitKey(1);
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
