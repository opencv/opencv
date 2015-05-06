#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

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
        String inf = static_cast<ostringstream*>(&(ostringstream() << pAct.blobColor))->str();
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



int main(int argc, char *argv[])
{
    vector<String> fileName;
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
    Mat imgOrig = imread(fileName[0], IMREAD_UNCHANGED),img;
    if (imgOrig.rows*imgOrig.cols <= 0)
        {
        cout << "Image " << fileName[0] << " is empty or cannot be found\n";
        return(0);
        }
    GaussianBlur(imgOrig,img,Size(11,11),0.1,0.1);

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
        palette.push_back(Vec3b(rand(),rand(),rand()));
    help();
    typeDesc.push_back("MSER");
    pMSER.push_back(pDefaultMSER);
    pMSER.back().minArea = 1;
    pMSER.back().maxArea = img.rows*img.cols;

    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByColor = true;
    pBLOB.back().blobColor = 255;
    // This descriptor are going to be detect and compute 4 BLOBS with 4 differents params
    // Param for first BLOB detector we want all
    typeDesc.push_back("BLOB");    // see http://docs.opencv.org/trunk/d0/d7a/classcv_1_1SimpleBlobDetector.html
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().minArea = 1;
    pBLOB.back().maxArea = img.rows*img.cols;
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
    pBLOB.back().maxInertiaRatio = 0.2;
    // Param for Fourth BLOB detector we want ratio inertia
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByConvexity = true;
    pBLOB.back().minConvexity = 0.;
    pBLOB.back().maxConvexity = 0.9;


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
            b = MSER::create(itMSER->delta, itMSER->minArea,itMSER->maxArea,itMSER->maxVariation,itMSER->minDiversity,itMSER->maxEvolution,
                itMSER->areaThreshold,itMSER->minMargin,itMSER->edgeBlurSize);
            }
        try {
            // We can detect keypoint with detect method
            vector<KeyPoint>  keyImg;
            vector<Rect>  zone;
            vector<vector <Point>>  region;
            Mat     desc, result;
            int nb = img.channels();
            if (img.channels() == 3)
            {
                img.copyTo(result);
            }
            else
                {
                vector<Mat> plan;
                plan.push_back(img);
                plan.push_back(img);
                plan.push_back(img);
                merge(plan, result);
                }
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
                sbd->detectRegions(img,  region, zone);
                int i = 0;
                
                for (vector<Rect>::iterator r = zone.begin(); r != zone.end();r++,i++)
                {
                rectangle(result, *r, palette[i % 65536],2);
                }
                i=0;
                for (vector<vector <Point>>::iterator itr = region.begin(); itr != region.end(); itr++, i++)
                    {
                    for (vector <Point>::iterator itp = region[i].begin(); itp != region[i].end(); itp++)
                        {

                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(0,0,0);
                        }
                    }
                i = 0;
                for (vector<vector <Point>>::iterator itr = region.begin(); itr != region.end(); itr++, i++)
                    {
                    for (vector <Point>::iterator itp = region[i].begin(); itp != region[i].end(); itp++)
                        {

                        result.at<Vec3b>(itp->y, itp->x) = Vec3b(0,255,255);
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
