#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "opencv2/video/detail/tracking.detail.hpp"
#include <opencv2/highgui.hpp>
#include <fstream>
//#include "bytetracker.hpp"
#include "opencv2/video/tracking.hpp"


// Namespaces.
using namespace cv;
using namespace std;
using namespace dnn;
//using namespace cv::detail::tracking;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

string DETECTIONS_OUTPUT_PATH = "det.txt";
string TRACKINGS_OUTPUT_PATH = "tracked.txt";
string VIDEO_OUTPUT_PATH = "output.mp4";

int outputCodec = VideoWriter::fourcc('M', 'J', 'P', 'G');
double outputFps = 25;
Size outputSize(1920, 1080);

vector<Mat> preProcessImage(Mat&, Net&);
Mat formatYolov5(const Mat&);
Mat postProcessImage(Mat&, vector<Mat>&, const vector<string>&, vector<Detection>&);
void drawLabel(Mat&, string, int, int);
void writeDetectionsToFile(const vector<Detection>, const std::string&, const int);
void writeTracksToFile(const vector<Strack>, const std::string&, const int);

int example()
{
    // Load class list.

    vector<string> classList;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line))
    {
        classList.push_back(line);
    }

    string folderPath = "./imgs/%06d.jpg";
    // string outputPath = "output.txt";

    VideoCapture capture;
    VideoWriter writer(VIDEO_OUTPUT_PATH, outputCodec, outputFps, outputSize);


    capture.open(folderPath);

    if (!capture.isOpened())
    {
        std::cout << "failed to open the image sequence: " << folderPath << std::endl;
        return -1;
    }

    // Load image sequence.
    Mat frame;
    int frameNumber = 0;
    //int fps = capture.get(CAP_PROP_FPS);
    int fps = 25;
    ByteTrackerImpl tracker(fps, 30);
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            std::cout << "Failed to read the frame." << std::endl;
            break;
        }

        // Load model.
        Net net;
        net = readNet("../YOLOv5s.onnx");
        // net.setPreferableBackend(dnn::DNN_TARGET_CPU);
        // net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
        vector<Mat> detections; // Process the image.
        vector<Detection> objects;
        detections = preProcessImage(frame, net);
        Mat frameClone = frame.clone();
        Mat img = postProcessImage(frameClone, detections, classList,
                               objects);
        vector<Strack> trackedObjects = tracker.update(objects);
        for (const Strack track : trackedObjects)
        {
            Scalar color = tracker.getColor(track.getId());
            Rect tlwh_ = track.getTlwh();
            int id_ = track.getId();
            if (track.getState() == TrackState::Tracked)
            {
                rectangle(img, tlwh_, color, 2);
                putText(img, to_string(id_), Point(tlwh_.x, tlwh_.y - 5), FONT_FACE, FONT_SCALE, RED); // THICKNESS
            }
        }
        writeTracksToFile(trackedObjects, TRACKINGS_OUTPUT_PATH, frameNumber);
        writeDetectionsToFile(objects, DETECTIONS_OUTPUT_PATH, frameNumber);

        // Put efficiency information.
        // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time : %.2f ms", t);
        putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
        imshow("Output", img);
        writer.write(img);
        //waitKey(0);

        if (waitKey(1) == 27)
            break;

        frameNumber++;
    }
    writer.release();
    capture.release();
    

    return 0;
}

vector<Mat> preProcessImage(Mat &inputImage, Net &net)
{
    // Convert to blob.
    // Mat img = formatYolov5(inputImage);

    Mat blob;
    dnn::blobFromImage(inputImage, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat formatYolov5(const Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    int _min = MIN(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

Mat postProcessImage(Mat &inputImage, vector<Mat> &output, const vector<string> &className,       
    vector<Detection> &object)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float xFactor = inputImage.cols / INPUT_WIDTH;
    float yFactor = inputImage.rows / INPUT_HEIGHT;
    float *data = (float *)output[0].data;
    const int dimensions = 85;

    //  25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, className.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point classId;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
            // Continue if the class score is above the threshold.
            if (maxClassScore > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                classIds.push_back(classId.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * xFactor);
                int top = int((cy - 0.5 * h) * yFactor);
                int width = int(w * xFactor);
                int height = int(h * yFactor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
    std::vector<int> nmsResult;
    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nmsResult);
    for (int i = 0; i < nmsResult.size(); i++)
    {
        int idx = nmsResult[i];
        Rect box = boxes[idx];
        int top = box.y;
        int left = box.x;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(inputImage, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = className[classIds[idx]] + ":" + label;
        // Draw class labels.
        drawLabel(inputImage, label, left, top);
        Detection det;
        det.box = box;
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        object.push_back(det);
    }

    return inputImage;
}

void drawLabel(Mat &inputImage, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size labelSize = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, labelSize.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + labelSize.width, top + labelSize.height + baseLine);
    // Draw white rectangle.
    rectangle(inputImage, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(inputImage, label, Point(left, top + labelSize.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

void writeDetectionsToFile(const vector<Detection> objects, const std::string &outputPath, 
    const int frameNumber)
{
    // Open the output file for writing
    std::ofstream outputFile(outputPath, std::ios_base::app);
    if (!outputFile.is_open())
    {
        std::cout << "Failed to open the output file: " << outputPath << std::endl;
        return;
    }

    // Iterate over the detections and write the data to the output file
    for (const auto object : objects)
    {
        // Extract the detection data (frame, id, x1, y1, width, height, score)
        int y = object.box.y;
        int x = object.box.x;
        int width = object.box.width;
        int height = object.box.height;
        float score = object.confidence;

        // Write the data to the output file
        outputFile << frameNumber << ", " << -1 << ", " << x << ", " << y << ", " << width << ", " << height << ", " << score << ", -1, -1, -1" << std::endl;
    }

    // Close the output file
    outputFile.close();

    //std::cout << "\n Detection data saved to: " << outputPath << std::endl;
}

void writeTracksToFile(const vector<Strack> objects, const std::string &outputPath, 
    const int frameNumber)
{
    // Open the output file for writing
    std::ofstream outputFile(outputPath, std::ios_base::app);
    if (!outputFile.is_open())
    {
        std::cout << "Failed to open the output file: " << outputPath << std::endl;
        return;
    }

    // Iterate over the detections and write the data to the output file
    for (const auto object : objects)
    {
        // Extract the detection data (frame, id, x1, y1, width, height, score)
        int id = object.getId();
        Rect tlwh_ = object.getTlwh();
        int y = tlwh_.y;
        int x = tlwh_.x;
        int width = tlwh_.width;
        int height = tlwh_.height;
        float score = object.getScore();

        // Write the data to the output file
        outputFile << frameNumber << ", " << id << ", " << x << ", " << y << ", " << width << ", " << height << ", " << score << ", -1, -1, -1" << std::endl;
    }

    // Close the output file
    outputFile.close();

    //std::cout << "\n Detection data saved to: " << outputPath << std::endl;
}

Scalar getColor(const int idx)
{
    int value = idx + 3;
    return Scalar(37 * value % 255, 17 * value % 255, 29 * value % 255);
}