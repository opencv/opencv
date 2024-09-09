#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/core/opengl.hpp"

#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdint.h>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout << "\nThis program demonstrates how to use MSER to detect extremal regions\n"
            "Usage:\n"
         << argv[0] << " <image1(without parameter a synthetic image is used as default)>\n"
            "Press esc key when image window is active to change descriptor parameter\n"
            "Press 2, 8, 4, 6, +, -, or 5 keys in openGL windows to change view or use mouse\n";
}

struct MSERParams
{
    MSERParams(int _delta = 5, int _min_area = 60, int _max_area = 14400,
               double _max_variation = 0.25, double _min_diversity = .2,
               int _max_evolution = 200, double _area_threshold = 1.01,
               double _min_margin = 0.003, int _edge_blur_size = 5)
    {
        delta = _delta;
        minArea = _min_area;
        maxArea = _max_area;
        maxVariation = _max_variation;
        minDiversity = _min_diversity;
        maxEvolution = _max_evolution;
        areaThreshold = _area_threshold;
        minMargin = _min_margin;
        edgeBlurSize = _edge_blur_size;
        pass2Only = false;
    }

    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    bool pass2Only;

    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};

static String Legende(const MSERParams &pAct)
{
    ostringstream ss;
    ss << "Area[" << pAct.minArea << "," << pAct.maxArea << "] ";
    ss << "del. [" << pAct.delta << "] ";
    ss << "var. [" << pAct.maxVariation << "] ";
    ss << "div. [" << (int)pAct.minDiversity << "] ";
    ss << "pas. [" << (int)pAct.pass2Only << "] ";
    ss << "RGb->evo. [" << pAct.maxEvolution << "] ";
    ss << "are. [" << (int)pAct.areaThreshold << "] ";
    ss << "mar. [" << (int)pAct.minMargin << "] ";
    ss << "siz. [" << pAct.edgeBlurSize << "]";

    return ss.str();
}

static void addNestedRectangles(Mat &img, Point p0, int* width, int *color, int n) {
    for (int i = 0; i<n; i++)
    {
        rectangle(img, Rect(p0, Size(width[i], width[i])), Scalar(color[i]), 1);
        p0 += Point((width[i] - width[i + 1]) / 2, (width[i] - width[i + 1]) / 2);
        floodFill(img, p0, Scalar(color[i]));
    }
}

static void addNestedCircles(Mat &img, Point p0, int *width, int *color, int n) {
    for (int i = 0; i<n; i++)
    {
        circle(img, p0, width[i] / 2, Scalar(color[i]), 1);
        floodFill(img, p0, Scalar(color[i]));
    }
}

static Mat MakeSyntheticImage()
{
    const int fond = 0;

    Mat img(800, 800, CV_8UC1);
    img = Scalar(fond);

    int width[] = { 390, 380, 300, 290, 280, 270, 260, 250, 210, 190, 150, 100, 80, 70 };

    int color1[] = { 80, 180, 160, 140, 120, 100, 90, 110, 170, 150, 140, 100, 220 };
    int color2[] = { 81, 181, 161, 141, 121, 101, 91, 111, 171, 151, 141, 101, 221 };
    int color3[] = { 175, 75, 95, 115, 135, 155, 165, 145, 85, 105, 115, 155, 35 };
    int color4[] = { 173, 73, 93, 113, 133, 153, 163, 143, 83, 103, 113, 153, 33 };

    addNestedRectangles(img, Point(10, 10), width, color1, 13);
    addNestedCircles(img, Point(200, 600), width, color2, 13);

    addNestedRectangles(img, Point(410, 10), width, color3, 13);
    addNestedCircles(img, Point(600, 600), width, color4, 13);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    Mat hist;

    // we compute the histogram
    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, histRange, true, false);

    cout << "****************Maximal region************************\n";
    for (int i = 0; i < hist.rows; i++)
    {
        if (hist.at<float>(i, 0)!=0)
        {
            cout << "h" << setw(3) << left << i << "\t=\t" << hist.at<float>(i, 0) << "\n";
        }
    }

    return img;
}

int main(int argc, char *argv[])
{
    Mat imgOrig, img;
    Size blurSize(5, 5);
    cv::CommandLineParser parser(argc, argv, "{ help h | | }{ @input | | }");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    string input = parser.get<string>("@input");
    if (!input.empty())
    {
        imgOrig = imread(samples::findFile(input), IMREAD_GRAYSCALE);
        blur(imgOrig, img, blurSize);
    }
    else
    {
        imgOrig = MakeSyntheticImage();
        img = imgOrig;
    }

    // Descriptor array MSER
    vector<String> typeDesc;
    // Param array for MSER
    vector<MSERParams> pMSER;

    // Color palette
    vector<Vec3b> palette;
    for (int i = 0; i<=numeric_limits<uint16_t>::max(); i++)
        palette.push_back(Vec3b((uchar)rand(), (uchar)rand(), (uchar)rand()));

    help(argv);

    MSERParams params;

    params.delta = 10;
    params.minArea = 100;
    params.maxArea = 5000;
    params.maxVariation = 2;
    params.minDiversity = 0;
    params.pass2Only = true;

    typeDesc.push_back("MSER");
    pMSER.push_back(params);

    params.pass2Only = false;
    typeDesc.push_back("MSER");
    pMSER.push_back(params);

    params.delta = 100;
    typeDesc.push_back("MSER");
    pMSER.push_back(params);

    vector<MSERParams>::iterator itMSER = pMSER.begin();
    Ptr<Feature2D> b;
    String label;
    // Descriptor loop
    vector<String>::iterator itDesc;
    Mat result(img.rows, img.cols, CV_8UC3);
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "MSER")
        {
            if (img.type() == CV_8UC3)
            {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity, itMSER->maxEvolution,
                                 itMSER->areaThreshold, itMSER->minMargin, itMSER->edgeBlurSize);
                label = Legende(*itMSER);
                ++itMSER;
            }
            else
            {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity);
                b.dynamicCast<MSER>()->setPass2Only(itMSER->pass2Only);
                label = Legende(*itMSER);
                ++itMSER;
            }
        }

        if (img.type()==CV_8UC3)
        {
            img.copyTo(result);
        }
        else
        {
            vector<Mat> plan;
            plan.push_back(img);
            plan.push_back(img);
            plan.push_back(img);
            merge(plan,result);
        }
        try
        {
            // We can detect regions using detectRegions method
            vector<KeyPoint> keyImg;
            vector<Rect> zone;
            vector<vector <Point> > region;
            Mat desc;

            if (b.dynamicCast<MSER>().get())
            {
                Ptr<MSER> sbd = b.dynamicCast<MSER>();
                sbd->detectRegions(img, region, zone);
                //result = Scalar(0, 0, 0);
                int nbPixelInMSER=0;
                for (vector<vector <Point> >::iterator itr = region.begin(); itr != region.end(); ++itr)
                {
                    for (vector <Point>::iterator itp = itr->begin(); itp != itr->end(); ++itp)
                    {
                        // all pixels belonging to region become blue
                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(128, 0, 0);
                        nbPixelInMSER++;
                    }
                }
                cout << "Number of MSER region: " << region.size() << "; Number of pixels in all MSER region: " << nbPixelInMSER << "\n";
            }

            const string winName = *itDesc + label;
            // namedWindow(winName, WINDOW_AUTOSIZE); // 注释掉图像显示部分
            // imshow(winName, result); // 注释掉图像显示部分
            // imshow("Original", img); // 注释掉图像显示部分

            // 保存处理后的图像
            string result_filename = *itDesc + label + "_result.png";
            imwrite(result_filename, result);
            cout << "Result image saved as: " << result_filename << endl;
        }
        catch (const Exception& e)
        {
            cout << "Feature: " << *itDesc << "\n";
            cout << e.msg << endl;
        }
        // waitKey(); // 注释掉等待按键部分
    }
    return 0;
}

