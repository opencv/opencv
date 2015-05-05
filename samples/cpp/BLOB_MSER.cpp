#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\n This program demonstrates how to use BLOB and MSER to detect region \n"
        "Usage: \n"
        "  ./BLOB_MSER <image1(../data/basketball1.png as default)>\n"
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



int main(int argc, char *argv[])
{
    vector<String> fileName;
    if (argc == 1)
        {
        fileName.push_back("../data/forme.jpg");
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
    GaussianBlur(imgOrig,img,Size(11,11),5,5);
    SimpleBlobDetector::Params pDefaultBLOB;
    // This is default parameters for SimpleBlobDetector
    pDefaultBLOB.thresholdStep = 10;
    pDefaultBLOB.minThreshold = 1;
    pDefaultBLOB.maxThreshold = 220;
    pDefaultBLOB.minRepeatability = 2;
    pDefaultBLOB.minDistBetweenBlobs = 10;
    pDefaultBLOB.filterByColor = false;
    pDefaultBLOB.blobColor = 0;
    pDefaultBLOB.filterByArea = false;
    pDefaultBLOB.minArea = 25;
    pDefaultBLOB.maxArea = 5000;
    pDefaultBLOB.filterByCircularity = false;
    pDefaultBLOB.minCircularity = 0.8f;
    pDefaultBLOB.maxCircularity = std::numeric_limits<float>::max();
    pDefaultBLOB.filterByInertia = false;
    pDefaultBLOB.minInertiaRatio = 0.1f;
    pDefaultBLOB.maxInertiaRatio = std::numeric_limits<float>::max();
    pDefaultBLOB.filterByConvexity = false;
    pDefaultBLOB.minConvexity = 0.95f;
    pDefaultBLOB.maxConvexity = std::numeric_limits<float>::max();
    MSERParams pDefaultMSER;
    // Descriptor array (BLOB or MSER)
    vector<String> typeDesc;
    // Param array for BLOB
    vector<SimpleBlobDetector::Params> pBLOB;
    vector<SimpleBlobDetector::Params>::iterator itBLOB;
    // Param array for MSER
    vector<MSERParams> pMSER;

    // Color palette
    vector<Vec3b>  palette;
    for (int i=0;i<65536;i++)
        palette.push_back(Vec3b(rand(),rand(),rand()));
    help();
    // This descriptor are going to be detect and compute 4 BLOBS with 4 differents params
    typeDesc.push_back("BLOB");    // see http://docs.opencv.org/trunk/d0/d7a/classcv_1_1SimpleBlobDetector.html
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().minArea = 1;
    pBLOB.back().maxArea = img.rows*img.cols;
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().maxArea = img.rows*img.cols;
    pBLOB.back().filterByCircularity = true;
    typeDesc.push_back("BLOB");    
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByInertia = true;
    typeDesc.push_back("BLOB");    
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByColor = true;
    pBLOB.back().blobColor = 60;
    typeDesc.push_back("MSER");    
    itBLOB=pBLOB.begin();
    vector<double> desMethCmp;
    Ptr<Feature2D> b;

    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++)
    {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "BLOB"){
            b = SimpleBlobDetector::create(*itBLOB);
            itBLOB++;
        }
        if (*itDesc == "MSER"){
            b = MSER::create();
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

                        result.at<Vec3b>(itp->y, itp->x) += palette[i % 65536];
                        }
                    }
                }
            namedWindow(*itDesc , WINDOW_AUTOSIZE);
            imshow(*itDesc, result);
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
