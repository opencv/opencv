#include <opencv2/opencv.hpp>
#include "opencv2/core/opengl.hpp"

#include <vector>
#include <map>
#include <iostream>
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

using namespace std;
using namespace cv;


void Example_MSER(vector<String> &fileName);

static void help()
{
    cout << "\n This program demonstrates how to use BLOB and MSER to detect region \n"
        "Usage: \n"
        "  ./BLOB_MSER <image1(../data/forme2.jpg as default)>\n"
        "Press a key when image window is active to change descriptor";
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

String Legende(SimpleBlobDetector::Params &pAct)
{
    String s="";
    if (pAct.filterByArea)
        {
        String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minArea))->str();
        String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxArea))->str();
        s = " Area range [" + inf + " to  " + sup + "]";
        }
    if (pAct.filterByCircularity)
        {
        String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minCircularity))->str();
        String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxCircularity))->str();
        if (s.length()==0)
            s = " Circularity range [" + inf + " to  " + sup + "]";
        else
            s += " AND Circularity range [" + inf + " to  " + sup + "]";
        }
    if (pAct.filterByColor)
        {
        String inf = static_cast<ostringstream*>(&(ostringstream() << (int)pAct.blobColor))->str();
        if (s.length() == 0)
            s = " Blob color " + inf;
        else
            s += " AND Blob color " + inf;
        }
    if (pAct.filterByConvexity)
        {
        String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minConvexity))->str();
        String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxConvexity))->str();
        if (s.length() == 0)
            s = " Convexity range[" + inf + " to  " + sup + "]";
        else
            s += " AND  Convexity range[" + inf + " to  " + sup + "]";
        }
    if (pAct.filterByInertia)
        {
        String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.minInertiaRatio))->str();
        String sup = static_cast<ostringstream*>(&(ostringstream() << pAct.maxInertiaRatio))->str();
        if (s.length() == 0)
            s = " Inertia ratio range [" + inf + " to  " + sup + "]";
        else
            s += " AND  Inertia ratio range [" + inf + " to  " + sup + "]";
        }
    return s;
}


const int win_width = 800;
const int win_height = 640;

struct DrawData
    {
    ogl::Arrays arr;
    ogl::Texture2D tex;
    ogl::Buffer indices;
    };

void draw(void* userdata);

void draw(void* userdata)
    {
    DrawData* data = static_cast<DrawData*>(userdata);

    glRotated(0.6, 0, 1, 0);

    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
    }

int main(int argc, char *argv[])
{

Mat imgcol = imread("../data/lena.jpg");
namedWindow("OpenGL", WINDOW_OPENGL);
//resizeWindow("OpenGL", win_width, win_height);

Mat_<Vec3f> vertex(1, 4);
vertex << Vec3f(-1, 1,0), Vec3f(-1, -1,0), Vec3f(1, -1,1), Vec3f(1, 1,-1);

Mat_<Vec2f> texCoords(1, 4);
texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

Mat_<int> indices(1, 6);
indices << 0, 1, 2,2, 3, 0;

DrawData *data = new DrawData;

data->arr.setVertexArray(vertex);
data->arr.setTexCoordArray(texCoords);
data->indices.copyFrom(indices);
data->tex.copyFrom(imgcol);

glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0, (double)win_width / win_height, 0.1, 100.0);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);

glEnable(GL_TEXTURE_2D);
data->tex.bind();

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

glDisable(GL_CULL_FACE);

setOpenGlDrawCallback("OpenGL", draw, data);

for (;;)
    {
    updateWindow("OpenGL");
    int key = waitKey(40);
    if ((key & 0xff) == 27)
        break;
    }

setOpenGlDrawCallback("OpenGL", 0, 0);
destroyAllWindows();






    vector<String> fileName;
    Example_MSER(fileName);
    Mat img(600,800,CV_8UC1);
    if (argc == 1)
    {
        fileName.push_back("../data/BLOB_MSER.bmp");
    }
    else if (argc == 2)
        {
        fileName.push_back(argv[1]);
        }
    else
        {
        help();
        return(0);
        }
    img = imread(fileName[0], IMREAD_UNCHANGED);
    if (img.rows*img.cols <= 0)
        {
        cout << "Image " << fileName[0] << " is empty or cannot be found\n";
        return(0);
        }

    SimpleBlobDetector::Params pDefaultBLOB;
    MSERParams pDefaultMSER;
    // This is default parameters for SimpleBlobDetector
    pDefaultBLOB.thresholdStep = 10;
    pDefaultBLOB.minThreshold = 10;
    pDefaultBLOB.maxThreshold = 220;
    pDefaultBLOB.minRepeatability = 2;
    pDefaultBLOB.minDistBetweenBlobs = 10;
    pDefaultBLOB.filterByColor = false;
    pDefaultBLOB.blobColor = 0;
    pDefaultBLOB.filterByArea = false;
    pDefaultBLOB.minArea = 25;
    pDefaultBLOB.maxArea = 5000;
    pDefaultBLOB.filterByCircularity = false;
    pDefaultBLOB.minCircularity = 0.9f;
    pDefaultBLOB.maxCircularity = std::numeric_limits<float>::max();
    pDefaultBLOB.filterByInertia = false;
    pDefaultBLOB.minInertiaRatio = 0.1f;
    pDefaultBLOB.maxInertiaRatio = std::numeric_limits<float>::max();
    pDefaultBLOB.filterByConvexity = false;
    pDefaultBLOB.minConvexity = 0.95f;
    pDefaultBLOB.maxConvexity = std::numeric_limits<float>::max();
    // Descriptor array (BLOB or MSER)
    vector<String> typeDesc;
    // Param array for BLOB
    vector<SimpleBlobDetector::Params> pBLOB;
    vector<SimpleBlobDetector::Params>::iterator itBLOB;
    // Param array for MSER
    vector<MSERParams> pMSER;
    vector<MSERParams>::iterator itMSER;

    // Color palette
    vector<Vec3b>  palette;
    for (int i=0;i<65536;i++)
        palette.push_back(Vec3b((uchar)rand(), (uchar)rand(), (uchar)rand()));
    help();

/*    typeDesc.push_back("MSER");
    pMSER.push_back(pDefaultMSER);
    pMSER.back().delta = 1;
    pMSER.back().minArea = 1;
    pMSER.back().maxArea = 180000;
    pMSER.back().maxVariation= 500;
    pMSER.back().minDiversity = 0;
    pMSER.back().pass2Only = false;*/
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByColor = true;
    pBLOB.back().blobColor = 0;

    // This descriptor are going to be detect and compute 4 BLOBS with 4 differents params
    // Param for first BLOB detector we want all
    typeDesc.push_back("BLOB");    // see http://docs.opencv.org/trunk/d0/d7a/classcv_1_1SimpleBlobDetector.html
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().minArea = 1;
    pBLOB.back().maxArea = int(img.rows*img.cols);
    // Param for second BLOB detector we want area between 500 and 2900 pixels
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().minArea = 500;
    pBLOB.back().maxArea = 2900;
    // Param for third BLOB detector we want only circular object
    typeDesc.push_back("BLOB");    
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByCircularity = true;
    // Param for Fourth BLOB detector we want ratio inertia
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByInertia = true;
    pBLOB.back().minInertiaRatio = 0;
    pBLOB.back().maxInertiaRatio = (float)0.2;
    // Param for Fourth BLOB detector we want ratio inertia
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByConvexity = true;
    pBLOB.back().minConvexity = 0.;
    pBLOB.back().maxConvexity = (float)0.9;


    itBLOB = pBLOB.begin();
    itMSER = pMSER.begin();
    vector<double> desMethCmp;
    Ptr<Feature2D> b;
    String label;
    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++)
    {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "BLOB"){
            b = SimpleBlobDetector::create(*itBLOB);
            label=Legende(*itBLOB);

            itBLOB++;
        }
        if (*itDesc == "MSER"){
            if(img.type()==CV_8UC3)
            {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity, itMSER->maxEvolution,
                                itMSER->areaThreshold, itMSER->minMargin, itMSER->edgeBlurSize);
                b.dynamicCast<MSER>()->setPass2Only(itMSER->pass2Only);
            }
            else
            {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity);
            }
            //b = MSER::create();
            //b = MSER::create();
            }
        try {
            // We can detect keypoint with detect method
            vector<KeyPoint>  keyImg;
            vector<Rect>  zone;
            vector<vector <Point>>  region;
            Mat     desc, result(img.rows,img.cols,CV_8UC3);
                

            if (b.dynamicCast<SimpleBlobDetector>() != NULL)
            {
                Ptr<SimpleBlobDetector> sbd = b.dynamicCast<SimpleBlobDetector>();
                sbd->detect(img, keyImg, Mat());
                drawKeypoints(img,keyImg,result);
                int i=0;
                for (vector<KeyPoint>::iterator k=keyImg.begin();k!=keyImg.end();k++,i++)
                    circle(result,k->pt,k->size,palette[i%65536]);
            }
            if (b.dynamicCast<MSER>() != NULL)
            {
                Ptr<MSER> sbd = b.dynamicCast<MSER>();
                sbd->detectRegions(img, region, zone);
                int i = 0;
                result=Scalar(0,0,0);
                for (vector<Rect>::iterator r = zone.begin(); r != zone.end();r++,i++)
                {
                    // we draw a white rectangle which include all region pixels
                    rectangle(result, *r, Vec3b(255, 0, 0), 2);
                }
                i=0;
                for (vector<vector <Point>>::iterator itr = region.begin(); itr != region.end(); itr++, i++)
                {
                    for (vector <Point>::iterator itp = region[i].begin(); itp != region[i].end(); itp++)
                    {
                        // all pixels belonging to region are red
                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(0,0,128);
                    }
                }
            }
            namedWindow(*itDesc+label , WINDOW_AUTOSIZE);
            imshow(*itDesc + label, result);
            imshow("Original", img);
            FileStorage fs(*itDesc + "_" + fileName[0] + ".xml", FileStorage::WRITE);
            fs<<*itDesc<<keyImg;
                waitKey();
        }
        catch (Exception& e)
        {
            cout << "Feature : " << *itDesc << "\n";
            cout<<e.msg<<endl;
        }
    }
    return 0;
}




void Example_MSER(vector<String> &fileName)
{
    Mat img(800, 800, CV_8UC1);
    fileName.push_back("SyntheticImage.bmp");
    map<int, char> val;
    int fond = 0;
    img = Scalar(fond);
    val[fond] = 1;
    int width1[]={390,380,300,290,280,270,260,250,210,190,150,100, 80,70};
    int color1[]={ 80,180,160,140,120,100, 90,110,170,150,140,100,220};
    Point p0(10, 10);
    int *width,*color;

    width = width1;
    color = color1;
    for (int i = 0; i<13; i++)
        {
        rectangle(img, Rect(p0, Size(width[i], width[i])), Scalar(color[i]), 1);
        p0 += Point((width[i] - width[i + 1]) / 2, (width[i] - width[i + 1]) / 2);
        floodFill(img, p0, Scalar(color[i]));

        }
    p0 = Point(200, 600);
    for (int i = 0; i<13; i++)
        {
        circle(img, p0, width[i] / 2, Scalar(color[i]), 1);
        floodFill(img, p0, Scalar(color[i]));

        }
    for (int i = 0; i<13; i++)
        color1[i] =  255 - color1[i];
    p0 = Point(410, 10);
    for (int i = 0; i<13; i++)
        {
        rectangle(img, Rect(p0, Size(width[i], width[i])), Scalar(color[i]), 1);
        p0 += Point((width[i] - width[i + 1]) / 2, (width[i] - width[i + 1]) / 2);
        floodFill(img, p0, Scalar(color[i]));

        }

    p0 = Point(600, 600);
    for (int i = 0; i<13; i++)
        {
        circle(img, p0, width[i]/2,Scalar(color[i]), 1);
        floodFill(img, p0 , Scalar(color[i]));

        }






    int channel = 1;
    int histSize =  256 ;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    Mat hist;
    // we compute the histogram from the 0-th and 1-st channels

    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, histRange, true, false);
    Mat cumHist(hist.size(), hist.type());
    cumHist.at<float>(0, 0) = hist.at<float>(0, 0);
    for (int i = 1; i < hist.rows; i++)
        cumHist.at<float>(i, 0) = cumHist.at<float>(i - 1, 0) + hist.at<float>(i, 0);
    imwrite(fileName[0], img);
    cout << "****************Maximal region************************\n";
    cout << "i\th\t\tsh\t\tq\n";
    cout << 0 << "\t" << hist.at<float>(0, 0) << "\t\t" << cumHist.at<float>(0, 0) << "\t\t\n";
    for (int i = 1; i < hist.rows-1 ; i++)
        {
        if (cumHist.at<float>(i, 0)>0)
            {
            cout << i << "\t" << hist.at<float>(i, 0) << "\t\t" << cumHist.at<float>(i, 0) << "\t\t" << (cumHist.at<float>(i + 1, 0) - cumHist.at<float>(i, 0)) / cumHist.at<float>(i, 0);
            }
        else
            cout << i << "\t" << hist.at<float>(i, 0) << "\t\t" << cumHist.at<float>(i, 0) << "\t\t";
        cout << endl;
        }
    cout << 255 << "\t" << hist.at<float>(255, 0) << "\t\t" << cumHist.at<float>(255, 0) << "\t\t\n";
    cout << "****************Minimal region************************\n";
    cumHist.at<float>(255, 0) = hist.at<float>(255, 0);
    for (int i = 254; i >= 0; i--)
        cumHist.at<float>(i, 0) = cumHist.at<float>(i + 1, 0) + hist.at<float>(i, 0);
    cout << "Minimal region\ni\th\t\tsh\t\tq\n";
    cout << 255-255 << "\t" << hist.at<float>(255, 0) << "\t\t" << cumHist.at<float>(255, 0) << "\t\t\n";
    for (int i = 254; i>=0; i--)
        {
        if (cumHist.at<float>(i, 0)>0)
            {
            cout << 255 - i << "\t" << i << "\t" << hist.at<float>(i, 0) << "\t\t" << cumHist.at<float>(i, 0) << "\t\t" << (cumHist.at<float>(i + 1, 0) - cumHist.at<float>(i, 0)) / cumHist.at<float>(i, 0);
            }
        else
            cout << 255 - i << "\t" << i << "\t" << hist.at<float>(i, 0) << "\t\t" << cumHist.at<float>(i, 0) << "\t\t";
        cout << endl;
        }
    // img = imread("C:/Users/laurent_2/Pictures/basketball1.png", IMREAD_GRAYSCALE);

    MSERParams pDefaultMSER;
    // Descriptor array (BLOB or MSER)
    vector<String> typeDesc;
    // Param array for BLOB
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
    pMSER.back().delta = 1000;
    pMSER.back().minArea = 1;
    pMSER.back().maxArea = 180000;
    pMSER.back().maxVariation = 1.701;
    pMSER.back().minDiversity = 0;
    pMSER.back().pass2Only = true;
    itMSER = pMSER.begin();
    vector<double> desMethCmp;
    Ptr<Feature2D> b;
    String label;
    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++)
        {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "MSER"){
            if (img.type() == CV_8UC3)
                {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity, itMSER->maxEvolution,
                                 itMSER->areaThreshold, itMSER->minMargin, itMSER->edgeBlurSize);
                }
            else
                {
                b = MSER::create(itMSER->delta, itMSER->minArea, itMSER->maxArea, itMSER->maxVariation, itMSER->minDiversity);
                b.dynamicCast<MSER>()->setPass2Only(itMSER->pass2Only);
                }
            }
        try {
            // We can detect keypoint with detect method
            vector<KeyPoint>  keyImg;
            vector<Rect>  zone;
            vector<vector <Point>>  region;
            Mat     desc, result(img.rows, img.cols, CV_8UC3);
            int nb = img.channels();

            if (b.dynamicCast<MSER>() != NULL)
                {
                Ptr<MSER> sbd = b.dynamicCast<MSER>();
                sbd->detectRegions(img, region, zone);
                int i = 0;
                result = Scalar(0, 0, 0);
                for (vector<vector <Point>>::iterator itr = region.begin(); itr != region.end(); itr++, i++)
                {
                    for (vector <Point>::iterator itp = region[i].begin(); itp != region[i].end(); itp+=2)
                    {
                        // all pixels belonging to region are red
                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(0, 0, 128);
                    }
                }
                i = 0;
                 for (vector<Rect>::iterator r = zone.begin(); r != zone.end(); r++, i++)
                {
                    // we draw a white rectangle which include all region pixels
                    rectangle(result, *r, Vec3b(255, 0, 0), 2);
                }
               }
            namedWindow(*itDesc + label, WINDOW_AUTOSIZE);
            imshow(*itDesc + label, result);
            imshow("Original", img);
            FileStorage fs(*itDesc + "_" + fileName[0] + ".xml", FileStorage::WRITE);
            fs << *itDesc << keyImg;
            waitKey();
            }
        catch (Exception& e)
            {
            cout << "Feature : " << *itDesc << "\n";
            cout << e.msg << endl;
            }
        }
    return;
    }
