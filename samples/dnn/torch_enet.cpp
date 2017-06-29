/*
Sample of using OpenCV dnn module with Torch ENet model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
using namespace std;

const String keys =
        "{help h    || Sample app for loading ENet Torch model. "
                       "The model and class names list can be downloaded here: "
                       "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AAD5YkUa8XgMpHs2gCRgmCVCa }"
        "{model m   || path to Torch .net model file (model_best.net) }"
        "{image i   || path to image file }"
        "{c_names c || path to file with classnames for channels (optional, categories.txt) }"
        "{result r  || path to save output blob (optional, binary format, NCHW order) }"
        "{show s    || whether to show all output channels or not}"
        "{o_blob    || output blob's name. If empty, last blob's name in net is used}"
        ;

static void colorizeSegmentation(const Mat &score, Mat &segm,
                                 Mat &legend, vector<String> &classNames, vector<Vec3b> &colors);
static vector<Vec3b> readColors(const String &filename, vector<String>& classNames);

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelFile = parser.get<String>("model");
    String imageFile = parser.get<String>("image");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    String classNamesFile = parser.get<String>("c_names");
    String resultFile = parser.get<String>("result");

    //! [Read model and initialize network]
    dnn::Net net = dnn::readNetFromTorch(modelFile);

    //! [Prepare blob]
    Mat img = imread(imageFile), input;
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    Size origSize = img.size();
    Size inputImgSize = cv::Size(1024, 512);

    if (inputImgSize != origSize)
        resize(img, img, inputImgSize);       //Resize image to input size

    Mat inputBlob = blobFromImage(img, 1./255);   //Convert Mat to image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob, "");        //set the network input
    //! [Set input blob]

    TickMeter tm;

    String oBlob = net.getLayerNames().back();
    if (!parser.get<String>("o_blob").empty())
    {
        oBlob = parser.get<String>("o_blob");
    }

    //! [Make forward pass]
    tm.start();
    Mat result = net.forward(oBlob);
    tm.stop();

    if (!resultFile.empty()) {
        CV_Assert(result.isContinuous());

        ofstream fout(resultFile.c_str(), ios::out | ios::binary);
        fout.write((char*)result.data, result.total() * sizeof(float));
        fout.close();
    }

    std::cout << "Output blob: " << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << "\n";
    std::cout << "Inference time, ms: " << tm.getTimeMilli()  << std::endl;

    if (parser.has("show"))
    {
        std::vector<String> classNames;
        vector<cv::Vec3b> colors;
        if(!classNamesFile.empty()) {
            colors = readColors(classNamesFile, classNames);
        }
        Mat segm, legend;
        colorizeSegmentation(result, segm, legend, classNames, colors);

        Mat show;
        addWeighted(img, 0.1, segm, 0.9, 0.0, show);

        cv::resize(show, show, origSize, 0, 0, cv::INTER_NEAREST);
        imshow("Result", show);
        if(classNames.size())
            imshow("Legend", legend);
        waitKey();
    }

    return 0;
} //main

static void colorizeSegmentation(const Mat &score, Mat &segm, Mat &legend, vector<String> &classNames, vector<Vec3b> &colors)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    cv::Mat maxCl(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1);
    for (int ch = 0; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }

    if (classNames.size() == colors.size())
    {
        int blockHeight = 30;
        legend.create(blockHeight*(int)classNames.size(), 200, CV_8UC3);
        for(int i = 0; i < (int)classNames.size(); i++)
        {
            cv::Mat block = legend.rowRange(i*blockHeight, (i+1)*blockHeight);
            block = colors[i];
            putText(block, classNames[i], Point(0, blockHeight/2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
        }
    }
}

static vector<Vec3b> readColors(const String &filename, vector<String>& classNames)
{
    vector<cv::Vec3b> colors;
    classNames.clear();

    ifstream fp(filename.c_str());
    if (!fp.is_open())
    {
        cerr << "File with colors not found: " << filename << endl;
        exit(-1);
    }

    string line;
    while (!fp.eof())
    {
        getline(fp, line);
        if (line.length())
        {
            stringstream ss(line);

            string name; ss >> name;
            int temp;
            cv::Vec3b color;
            ss >> temp; color[0] = (uchar)temp;
            ss >> temp; color[1] = (uchar)temp;
            ss >> temp; color[2] = (uchar)temp;
            classNames.push_back(name);
            colors.push_back(color);
        }
    }

    fp.close();
    return colors;
}
