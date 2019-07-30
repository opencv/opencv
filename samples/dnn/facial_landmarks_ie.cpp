//  This sample demonstrates the use of pretrained face and facial landmarks detection OpenVINO
//  networks with opencv's dnn module.
//
//  The sample uses two pretrained OpenVINO networks so the OpenVINO package has to be preinstalled.
//  Please install topologies described below using downloader.py
//  (.../openvino/deployment_tools/tools/model_downloader) to run this sample.
//  Face detection model - face-detection-adas-0001:
//  https://github.com/opencv/open_model_zoo/tree/master/intel_models/face-detection-adas-0001
//  Facial landmarks detection model - facial-landmarks-35-adas-0002:
//  https://github.com/opencv/open_model_zoo/tree/master/intel_models/facial-landmarks-35-adas-0002

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>

int main(int argc, char** argv)
{
    const std::string winName = "FacialLandmarksDetector";
    const Scalar clrGreen(0, 255, 0);
    const Scalar clrYellow(0, 255, 255);
    const Scalar clrBlack(0, 0, 0);

    CommandLineParser parser(argc, argv,
     "{ help  h              | |     Print the help message. }"
     "{ facestruct           | |     Full path to a Face detection model structure file (for example, .xml file).}"
     "{ faceweights          | |     Full path to a Face detection model weights file (for example, .bin file).}"
     "{ landmstruct          | |     Full path to a Facial Landmarks detection model structure file (for example, .xml file).}"
     "{ landmweights         | |     Full path to a Facial Landmarks detection model weights file (for example, .bin file).}"
     "{ input                | |     Full path to an input image or a video file. Skip this argument to capture frames from a camera.}"
     );

    parser.about("Use this script to run the face and facial landmarks detection deep learning IE networks using dnn API.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    //Parsing input arguments
    const std::string faceXmlPath = parser.get<std::string>("facestruct");
    const std::string faceBinPath = parser.get<std::string>("faceweights");

    const std::string landmXmlPath = parser.get<std::string>("landmstruct");
    const std::string landmBinPath = parser.get<std::string>("landmweights");

    //Models' definition & initialization
    Net faceNet = readNet(faceXmlPath, faceBinPath);
    const unsigned int faceObjectSize = 7;
    const float faceConfThreshold = 0.7f;
    const unsigned int faceCols = 672;
    const unsigned int faceRows = 384;

    Net landmNet = readNet(landmXmlPath, landmBinPath);
    const unsigned int landmCols = 60;
    const unsigned int landmRows = 60;

    //Input
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else if (!cap.open(0))
    {
        std::cout << "No input available" << std::endl;
        return 1;
    }

    Mat img;
    while (waitKey(1) < 0)
    {
        cap >> img;
        if (img.empty())
        {
           waitKey();
           break;
        }

        Mat imgShow;         //Image to show
        img.copyTo(imgShow);

        //Infering Face detector
        faceNet.setInput(blobFromImage(img, 1.0, Size(faceCols, faceRows)));
        Mat faceOut = faceNet.forward();

            //Face boxes processing
        float* faceData = (float*)(faceOut.data);
        for (size_t i = 0ul; i < faceOut.total(); i += faceObjectSize)
        {
            float faceConfidence = faceData[i + 2];
            if (faceConfidence > faceConfThreshold)
            {
                int faceLeft = int(faceData[i + 3] * img.cols);
                faceLeft = std::max(faceLeft, 0);
                int faceTop = int(faceData[i + 4] * img.rows);
                faceTop = std::max(faceTop, 0);
                int faceRight  = int(faceData[i + 5] * img.cols);
                faceRight = std::min(faceRight, img.cols - 2);
                int faceBot = int(faceData[i + 6] * img.rows);
                faceBot = std::min(faceBot, img.rows - 2);
                int faceWidth  = faceRight - faceLeft + 1;
                int faceHeight = faceBot - faceTop + 1;

                rectangle(imgShow, Rect(faceLeft, faceTop, faceWidth, faceHeight), clrGreen, 1);

                //Postprocessing for landmarks
                int faceMaxSize = std::max(faceWidth, faceHeight);
                int faceWidthAdd = faceMaxSize - faceWidth;
                int faceHeightAdd = faceMaxSize - faceHeight;

                Mat imgCrop;
                cv::copyMakeBorder(img(Rect(faceLeft, faceTop, faceWidth, faceHeight)), imgCrop,
                                   faceHeightAdd / 2, (faceHeightAdd + 1) / 2,  faceWidthAdd / 2,
                                   (faceWidthAdd + 1) / 2, BORDER_CONSTANT | BORDER_ISOLATED ,
                                   clrBlack);

                //Infering Landmarks detector
                landmNet.setInput(blobFromImage(imgCrop, 1.0, Size(landmCols, landmRows)));
                Mat landmOut = landmNet.forward();

                //Landmarks processing
                float* landmData = (float*)(landmOut.data);
                size_t j = 0ul;
                for (; j < 18 * 2; j += 2)
                {
                    Point p (int(landmData[j] * imgCrop.cols + faceLeft - faceWidthAdd / 2),
                             int(landmData[j + 1] * imgCrop.rows + faceTop - faceHeightAdd / 2));
                     //Point drawing
                    line(imgShow, p, p, clrYellow, 3);
                }

                Point ptsJaw[17];
                for(; j < landmOut.total(); j += 2)
                {
                    ptsJaw[(j - 18 * 2) / 2] = Point(int(landmData[j] * imgCrop.cols + faceLeft -
                                                         faceWidthAdd / 2),
                                                     int(landmData[j + 1] * imgCrop.rows + faceTop -
                                                         faceHeightAdd / 2));
                }
                const Point* ptsJawPtr[1] = {ptsJaw};
                int ptsJawNumber[1] = {17};
                polylines(imgShow, ptsJawPtr, ptsJawNumber, 1, false, clrYellow, 1);
            }
            else
                break;
        }
        imshow(winName, imgShow);
    }
}
