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
    parser.about("This sample is used to run Faster-RCNN and R-FCN object detection "
                 "models with OpenCV. You can get required models from "
                 "https://github.com/rbgirshick/py-faster-rcnn (Faster-RCNN) and from "
                 "https://github.com/YuwenXiong/py-R-FCN (R-FCN). Corresponding .prototxt "
                 "files may be found at https://github.com/opencv/opencv_extra/tree/master/testdata/dnn.");
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

    Mat img = imread(imagePath);
    resize(img, img, Size(kInpWidth, kInpHeight));
    Mat blob = blobFromImage(img, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
    Mat imInfo = (Mat_<float>(1, 3) << img.rows, img.cols, 1.6f);

    net.setInput(blob, "data");
    net.setInput(imInfo, "im_info");

    // Draw detections.
    Mat detections = net.forward();
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
