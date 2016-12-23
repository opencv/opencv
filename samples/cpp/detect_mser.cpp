#include <opencv2/opencv.hpp>
#include "opencv2/core/opengl.hpp"
#include "opencv2/cvconfig.h"

#include <vector>
#include <map>
#include <iostream>
#ifdef HAVE_OPENGL
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX 1
#include <windows.h>
#endif
#if defined(_WIN64)
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#endif


using namespace std;
using namespace cv;


static void help()
{
    cout << "\n This program demonstrates how to use MSER to detect extremal regions \n"
        "Usage: \n"
        "  ./detect_mser <image1(without parameter a syntehtic image is used as default)>\n"
        "Press esc key when image window is active to change  descriptor parameter\n"
        "Press 2, 8, 4, 6, +,- or 5 keys in openGL windows to change view or use mouse\n";
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

static String Legende(MSERParams &pAct)
{
    String s="";
    String inf = static_cast<const ostringstream&>(ostringstream() << pAct.minArea).str();
    String sup = static_cast<const ostringstream&>(ostringstream() << pAct.maxArea).str();
    s = " Area[" + inf + "," + sup + "]";

    inf = static_cast<const ostringstream&>(ostringstream() << pAct.delta).str();
    s += " del. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << pAct.maxVariation).str();
    s += " var. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.minDiversity).str();
    s += " div. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.pass2Only).str();
    s += " pas. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.maxEvolution).str();
    s += "RGb-> evo. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.areaThreshold).str();
    s += " are. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.minMargin).str();
    s += " mar. [" + inf + "]";
    inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.edgeBlurSize).str();
    s += " siz. [" + inf + "]";
    return s;
}


#ifdef HAVE_OPENGL
const int win_width = 800;
const int win_height = 640;
#endif
bool    rotateEnable=true;
bool    keyPressed=false;

Vec4f   rotAxis(1,0,1,0);
Vec3f  zoom(1,0,0);

float	obsX = (float)0, obsY = (float)0, obsZ = (float)-10, tx = (float)0, ty = (float)0;
float	thetaObs = (float)-1.570, phiObs = (float)1.570, rObs = (float)10;
int prevX=-1,prevY=-1,prevTheta=-1000,prevPhi=-1000;

#ifdef HAVE_OPENGL
struct DrawData

    {
    ogl::Arrays arr;
    ogl::Texture2D tex;
    ogl::Buffer indices;
    };


static void draw(void* userdata)
{
    DrawData* data = static_cast<DrawData*>(userdata);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(obsX, obsY, obsZ, 0, 0, .0, .0, 10.0, 0.0);
    glTranslatef(tx,ty,0);
    keyPressed = false;
    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

static void onMouse(int event, int x, int y, int flags, void*)
{
    if (event == EVENT_RBUTTONDOWN)
    {
        prevX = x;
        prevY = y;
    }
    if (event == EVENT_RBUTTONUP)
    {
        prevX = -1;
        prevY = -1;
    }
    if (prevX != -1)
    {
        tx += float((x - prevX) / 100.0);
        ty -= float((y - prevY) / 100.0);
        prevX = x;
        prevY = y;
    }
    if (event == EVENT_LBUTTONDOWN)
    {
        prevTheta = x;
        prevPhi = y;
    }
    if (event == EVENT_LBUTTONUP)
    {
        prevTheta = -1000;
        prevPhi = -1000;
    }
    if (prevTheta != -1000)
    {
        if (x - prevTheta<0)
        {
            thetaObs +=(float)0.02;
        }
        else if (x - prevTheta>0)
        {
            thetaObs -= (float)0.02;
        }
        if (y - prevPhi<0)
        {
            phiObs -= (float)0.02;
        }
        else if (y - prevPhi>0)
        {
            phiObs += (float)0.02;
        }
        prevTheta = x;
        prevPhi = y;
    }
    if (event==EVENT_MOUSEWHEEL)
    {
        if (getMouseWheelDelta(flags)>0)
            rObs += (float)0.1;
        else
            rObs -= (float)0.1;
    }
    float pi = static_cast<float>(CV_PI);
    if (thetaObs>pi)
    {
        thetaObs = -2 * pi + thetaObs;
    }
    if (thetaObs<-pi)
    {
        thetaObs = 2 * pi + thetaObs;
    }
    if (phiObs>pi / 2)
    {
        phiObs = pi / 2 - (float)0.0001;
    }
    if (phiObs<-pi / 2)
    {
        phiObs = -pi / 2 + (float)0.00001;
    }
    if (rObs<0)
    {
        rObs = 0;
    }

}
#endif

#ifdef HAVE_OPENGL
static void DrawOpenGLMSER(Mat img, Mat result)
{
    Mat imgGray;
    if (img.type() != CV_8UC1)
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
    else
        imgGray = img;
    namedWindow("OpenGL", WINDOW_OPENGL);
    setMouseCallback("OpenGL", onMouse, NULL);

    Mat_<Vec3f> vertex(1, img.cols*img.rows);
    Mat_<Vec2f> texCoords(1, img.cols*img.rows);
    for (int i = 0, nbPix = 0; i<img.rows; i++)
        {
        for (int j = 0; j<img.cols; j++, nbPix++)
            {
            float x = (j) / (float)img.cols;
            float y = (i) / (float)img.rows;
            vertex.at< Vec3f >(0, nbPix) = Vec3f(float(2 * (x - 0.5)), float(2 * (0.5 - y)), float(imgGray.at<uchar>(i, j) / 512.0));
            texCoords.at< Vec2f>(0, nbPix) = Vec2f(x, y);
            }
        }

    Mat_<int> indices(1, (img.rows - 1)*(6 * img.cols));
    for (int i = 1, nbPix = 0; i<img.rows; i++)
        {
        for (int j = 1; j<img.cols; j++)
            {
            int c = i*img.cols + j;
            indices.at<int>(0, nbPix++) = c ;
            indices.at<int>(0, nbPix++) = c - 1;
            indices.at<int>(0, nbPix++) = c- img.cols - 1;
            indices.at<int>(0, nbPix++) = c- img.cols - 1;
            indices.at<int>(0, nbPix++) = c - img.cols;
            indices.at<int>(0, nbPix++) = c ;
            }
        }

    DrawData *data = new DrawData;

    data->arr.setVertexArray(vertex);
    data->arr.setTexCoordArray(texCoords);
    data->indices.copyFrom(indices);
    data->tex.copyFrom(result);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)win_width / win_height, 0.0, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    data->tex.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_CULL_FACE);
    setOpenGlDrawCallback("OpenGL", draw, data);

    for (;;)
        {
        updateWindow("OpenGL");
        char key = (char)waitKey(40);
        if (key == 27)
            break;
        if (key == 0x20)
            rotateEnable = !rotateEnable;
        float pi = static_cast<float>(CV_PI);

        switch (key) {
            case '5':
                obsX = 0, obsY = 0, obsZ = -10;
                thetaObs = -pi/2, phiObs = pi/2, rObs = 10;
                tx=0;ty=0;
                break;
            case '4':
                thetaObs += (float)0.1;
                break;
            case '6':
                thetaObs -= (float)0.1;
                break;
            case '2':
                phiObs -= (float).1;
                break;
            case '8':
                phiObs += (float).1;
                break;
            case '+':
                rObs -= (float).1;
                break;
            case '-':
                rObs += (float).1;
                break;
        }
        if (thetaObs>pi)
        {
            thetaObs = -2 * pi + thetaObs;
        }
        if (thetaObs<-pi)
            thetaObs = 2 * pi + thetaObs;
        if (phiObs>pi / 2)
            phiObs = pi / 2 - (float)0.0001;
        if (phiObs<-pi / 2)
            phiObs = -pi / 2 + (float)0.00001;
        if (rObs<0)
            rObs = 0;
        obsX = rObs*cos(thetaObs)*cos(phiObs);
        obsY = rObs*sin(thetaObs)*cos(phiObs);
        obsZ = rObs*sin(phiObs);
    }
    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();
}
#endif

static Mat MakeSyntheticImage()
{
    Mat img(800, 800, CV_8UC1);
    map<int, char> val;
    int fond = 0;
    img = Scalar(fond);
    val[fond] = 1;
    int width1[] = { 390, 380, 300, 290, 280, 270, 260, 250, 210, 190, 150, 100, 80, 70 };
    int color1[] = { 80, 180, 160, 140, 120, 100, 90, 110, 170, 150, 140, 100, 220 };
    Point p0(10, 10);
    int *width, *color;

    width = width1;
    color = color1;
    for (int i = 0; i<13; i++)
        {
        rectangle(img, Rect(p0, Size(width[i], width[i])), Scalar(color[i]), 1);
        p0 += Point((width[i] - width[i + 1]) / 2, (width[i] - width[i + 1]) / 2);
        floodFill(img, p0, Scalar(color[i]));

        }
    int color2[] = { 81, 181, 161, 141, 121, 101, 91, 111, 171, 151, 141, 101, 221 };
    color = color2;
    p0 = Point(200, 600);
    for (int i = 0; i<13; i++)
        {
        circle(img, p0, width[i] / 2, Scalar(color[i]), 1);
        floodFill(img, p0, Scalar(color[i]));

        }
    int color3[] = { 175,75,95,115,135,155,165,145,85,105,115,156 };
    color = color3;
    p0 = Point(410, 10);
    for (int i = 0; i<13; i++)
        {
        rectangle(img, Rect(p0, Size(width[i], width[i])), Scalar(color[i]), 1);
        p0 += Point((width[i] - width[i + 1]) / 2, (width[i] - width[i + 1]) / 2);
        floodFill(img, p0, Scalar(color[i]));

        }
    int color4[] = { 173,73,93,113,133,153,163,143,83,103,114,154 };
    color = color4;

    p0 = Point(600, 600);
    for (int i = 0; i<13; i++)
    {
        circle(img, p0, width[i] / 2, Scalar(color[i]), 1);
        floodFill(img, p0, Scalar(color[i]));
    }
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    Mat hist;
    // we compute the histogram
    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, histRange, true, false);
    cout << "****************Maximal region************************\n";
    for (int i = 0; i < hist.rows ; i++)
    {
        if (hist.at<float>(i, 0)!=0)
        {
            cout << "h" << i << "=\t" << hist.at<float>(i, 0) <<  "\n";
        }
    }

    return img;
}

int main(int argc, char *argv[])
{
    vector<String> fileName;
    Mat imgOrig,img;
    Size blurSize(5,5);
    cv::CommandLineParser parser(argc, argv, "{ help h | | }{ @input | | }");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string input = parser.get<string>("@input");
    if (!input.empty())
    {
        fileName.push_back(input);
        imgOrig = imread(fileName[0], IMREAD_GRAYSCALE);
        blur(imgOrig, img, blurSize);
    }
    else
    {
        fileName.push_back("SyntheticImage.bmp");
        imgOrig = MakeSyntheticImage();
        img=imgOrig;
    }

    MSERParams pDefaultMSER;
    // Descriptor array MSER
    vector<String> typeDesc;
    // Param array for MSER
    vector<MSERParams> pMSER;
    vector<MSERParams>::iterator itMSER;

    // Color palette
    vector<Vec3b>  palette;
    for (int i = 0; i<65536; i++)
        palette.push_back(Vec3b((uchar)rand(), (uchar)rand(), (uchar)rand()));
    help();

    typeDesc.push_back("MSER");
    pMSER.push_back(pDefaultMSER);
    pMSER.back().delta = 10;
    pMSER.back().minArea = 100;
    pMSER.back().maxArea = 5000;
    pMSER.back().maxVariation = 2;
    pMSER.back().minDiversity = 0;
    pMSER.back().pass2Only = true;
    typeDesc.push_back("MSER");
    pMSER.push_back(pDefaultMSER);
    pMSER.back().delta = 10;
    pMSER.back().minArea = 100;
    pMSER.back().maxArea = 5000;
    pMSER.back().maxVariation = 2;
    pMSER.back().minDiversity = 0;
    pMSER.back().pass2Only = false;
    typeDesc.push_back("MSER");
    pMSER.push_back(pDefaultMSER);
    pMSER.back().delta = 100;
    pMSER.back().minArea = 100;
    pMSER.back().maxArea = 5000;
    pMSER.back().maxVariation = 2;
    pMSER.back().minDiversity = 0;
    pMSER.back().pass2Only = false;
    itMSER = pMSER.begin();
    vector<double> desMethCmp;
    Ptr<Feature2D> b;
    String label;
    // Descriptor loop
    vector<String>::iterator itDesc;
    Mat result(img.rows, img.cols, CV_8UC3);
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "MSER"){
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
            vector<KeyPoint>  keyImg;
            vector<Rect>  zone;
            vector<vector <Point> >  region;
            Mat     desc;

            if (b.dynamicCast<MSER>() != NULL)
            {
                Ptr<MSER> sbd = b.dynamicCast<MSER>();
                sbd->detectRegions(img, region, zone);
                int i = 0;
                //result = Scalar(0, 0, 0);
                int nbPixelInMSER=0;
                for (vector<vector <Point> >::iterator itr = region.begin(); itr != region.end(); ++itr, ++i)
                {
                    for (vector <Point>::iterator itp = region[i].begin(); itp != region[i].end(); ++itp)
                    {
                        // all pixels belonging to region become blue
                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(128, 0, 0);
                        nbPixelInMSER++;
                    }
                }
                cout << "Number of MSER region " << region.size()<<" Number of pixels in all MSER region : "<<nbPixelInMSER<<"\n";
            }
            namedWindow(*itDesc + label, WINDOW_AUTOSIZE);
            imshow(*itDesc + label, result);
            imshow("Original", img);
        }
        catch (Exception& e)
        {
            cout << "Feature : " << *itDesc << "\n";
            cout << e.msg << endl;
        }
#ifdef HAVE_OPENGL
        DrawOpenGLMSER(img, result);
#endif
        waitKey();
    }
    return 0;
}
