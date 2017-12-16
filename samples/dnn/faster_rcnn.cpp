// Faster-RCNN models use custom layer called 'Proposal' written in Python. To
// map it into OpenCV's layer replace a layer node with [type: 'Python'] to the
// following definition:
// layer {
//   name: 'proposal'
//   type: 'Proposal'
//   bottom: 'rpn_cls_prob_reshape'
//   bottom: 'rpn_bbox_pred'
//   bottom: 'im_info'
//   top: 'rois'
//   proposal_param {
//     ratio: 0.5
//     ratio: 1.0
//     ratio: 2.0
//     scale: 8
//     scale: 16
//     scale: 32
//   }
// }
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

const char* keys =
    "{ help  h |     | print help message  }"
    "{ proto p |     | path to .prototxt   }"
    "{ model m |     | path to .caffemodel }"
    "{ image i |     | path to input image }"
    "{ conf  c | 0.8 | minimal confidence  }";

const char* classNames[] = {
    "__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
};

static const int kInpWidth = 800;
static const int kInpHeight = 600;

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);
    parser.about( "This sample is used to run Faster-RCNN object detection with OpenCV.\n"
                  "You can get required models from https://github.com/rbgirshick/py-faster-rcnn" );

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String protoPath = parser.get<String>("proto");
    String modelPath = parser.get<String>("model");
    String imagePath = parser.get<String>("image");
    float confThreshold = parser.get<float>("conf");
    CV_Assert(!protoPath.empty(), !modelPath.empty(), !imagePath.empty());

    // Load a model.
    Net net = readNetFromCaffe(protoPath, modelPath);

    // Create a preprocessing layer that does final bounding boxes applying predicted
    // deltas to objects locations proposals and doing non-maximum suppression over it.
    LayerParams lp;
    lp.set("code_type", "CENTER_SIZE");               // An every bounding box is [xmin, ymin, xmax, ymax]
    lp.set("num_classes", 21);
    lp.set("share_location", (int)false);             // Separate predictions for different classes.
    lp.set("background_label_id", 0);
    lp.set("variance_encoded_in_target", (int)true);
    lp.set("keep_top_k", 100);
    lp.set("nms_threshold", 0.3);
    lp.set("normalized_bbox", (int)false);
    Ptr<Layer> detectionOutputLayer = DetectionOutputLayer::create(lp);

    Mat img = imread(imagePath);
    resize(img, img, Size(kInpWidth, kInpHeight));
    Mat blob = blobFromImage(img, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
    Mat imInfo = (Mat_<float>(1, 3) << img.rows, img.cols, 1.6f);

    net.setInput(blob, "data");
    net.setInput(imInfo, "im_info");

    std::vector<Mat> outs;
    std::vector<String> outNames(3);
    outNames[0] = "proposal";
    outNames[1] = "bbox_pred";
    outNames[2] = "cls_prob";
    net.forward(outs, outNames);

    Mat proposals = outs[0].colRange(1, 5).clone();  // Only last 4 columns.
    Mat& deltas = outs[1];
    Mat& scores = outs[2];

    // Reshape proposals from Nx4 to 1x1xN*4
    std::vector<int> shape(3, 1);
    shape[2] = (int)proposals.total();
    proposals = proposals.reshape(1, shape);

    // Run postprocessing layer.
    std::vector<Mat> layerInputs(3), layerOutputs(1), layerInternals;
    layerInputs[0] = deltas.reshape(1, 1);
    layerInputs[1] = scores.reshape(1, 1);
    layerInputs[2] = proposals;
    detectionOutputLayer->forward(layerInputs, layerOutputs, layerInternals);

    // Draw detections.
    Mat detections = layerOutputs[0];
    const float* data = (float*)detections.data;
    for (size_t i = 0; i < detections.total(); i += 7)
    {
        // An every detection is a vector [id, classId, confidence, left, top, right, bottom]
        float confidence = data[i + 2];
        if (confidence > confThreshold)
        {
            int classId = (int)data[i + 1];
            int left = max(0, min((int)data[i + 3], img.cols - 1));
            int top = max(0, min((int)data[i + 4], img.rows - 1));
            int right = max(0, min((int)data[i + 5], img.cols - 1));
            int bottom = max(0, min((int)data[i + 6], img.rows - 1));

            // Draw a bounding box.
            rectangle(img, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

            // Put a label with a class name and confidence.
            String label = cv::format("%s, %.3f", classNames[classId], confidence);
            int baseLine;
            Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            top = max(top, labelSize.height);
            rectangle(img, Point(left, top - labelSize.height),
                      Point(left + labelSize.width, top + baseLine),
                      Scalar(255, 255, 255), FILLED);
            putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }
    imshow("frame", img);
    waitKey();
    return 0;
}
