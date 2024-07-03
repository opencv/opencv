#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <map>
#include <iostream>

using namespace std;
using namespace cv;


static void help(char** argv)
{
    cout << "\n This program demonstrates how to use BLOB to detect and filter region \n"
         << "Usage: \n"
         << argv[0]
         << " <image1(detect_blob.png as default)>\n"
         << "Press a key when image window is active to change descriptor";
}


static String Legende(SimpleBlobDetector::Params &pAct)
{
    String s = "";
    if (pAct.filterByArea)
    {
        String inf = static_cast<const ostringstream&>(ostringstream() << pAct.minArea).str();
        String sup = static_cast<const ostringstream&>(ostringstream() << pAct.maxArea).str();
        s = " Area range [" + inf + " to  " + sup + "]";
    }
    if (pAct.filterByCircularity)
    {
        String inf = static_cast<const ostringstream&>(ostringstream() << pAct.minCircularity).str();
        String sup = static_cast<const ostringstream&>(ostringstream() << pAct.maxCircularity).str();
        if (s.length() == 0)
            s = " Circularity range [" + inf + " to  " + sup + "]";
        else
            s += " AND Circularity range [" + inf + " to  " + sup + "]";
    }
    if (pAct.filterByColor)
    {
        String inf = static_cast<const ostringstream&>(ostringstream() << (int)pAct.blobColor).str();
        if (s.length() == 0)
            s = " Blob color " + inf;
        else
            s += " AND Blob color " + inf;
    }
    if (pAct.filterByConvexity)
    {
        String inf = static_cast<const ostringstream&>(ostringstream() << pAct.minConvexity).str();
        String sup = static_cast<const ostringstream&>(ostringstream() << pAct.maxConvexity).str();
        if (s.length() == 0)
            s = " Convexity range[" + inf + " to  " + sup + "]";
        else
            s += " AND  Convexity range[" + inf + " to  " + sup + "]";
    }
    if (pAct.filterByInertia)
    {
        String inf = static_cast<const ostringstream&>(ostringstream() << pAct.minInertiaRatio).str();
        String sup = static_cast<const ostringstream&>(ostringstream() << pAct.maxInertiaRatio).str();
        if (s.length() == 0)
            s = " Inertia ratio range [" + inf + " to  " + sup + "]";
        else
            s += " AND  Inertia ratio range [" + inf + " to  " + sup + "]";
    }
    return s;
}



int main(int argc, char *argv[])
{
    String fileName;
    cv::CommandLineParser parser(argc, argv, "{@input |detect_blob.png| }{h help | | }");
    if (parser.has("h"))
    {
        help(argv);
        return 0;
    }
    fileName = parser.get<string>("@input");
    Mat img = imread(samples::findFile(fileName), IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Image " << fileName << " is empty or cannot be found\n";
        return 1;
    }

    SimpleBlobDetector::Params pDefaultBLOB;
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
    pDefaultBLOB.maxCircularity = (float)1e37;
    pDefaultBLOB.filterByInertia = false;
    pDefaultBLOB.minInertiaRatio = 0.1f;
    pDefaultBLOB.maxInertiaRatio = (float)1e37;
    pDefaultBLOB.filterByConvexity = false;
    pDefaultBLOB.minConvexity = 0.95f;
    pDefaultBLOB.maxConvexity = (float)1e37;
    // Descriptor array for BLOB
    vector<String> typeDesc;
    // Param array for BLOB
    vector<SimpleBlobDetector::Params> pBLOB;
    vector<SimpleBlobDetector::Params>::iterator itBLOB;
    // Color palette
    vector< Vec3b >  palette;
    for (int i = 0; i<65536; i++)
    {
        uchar c1 = (uchar)rand();
        uchar c2 = (uchar)rand();
        uchar c3 = (uchar)rand();
        palette.push_back(Vec3b(c1, c2, c3));
    }
    help(argv);


    // These descriptors are going to be detecting and computing BLOBS with 6 different params
    // Param for first BLOB detector we want all
    typeDesc.push_back("BLOB");    // see http://docs.opencv.org/5.x/d0/d7a/classcv_1_1SimpleBlobDetector.html
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByArea = true;
    pBLOB.back().minArea = 1;
    pBLOB.back().maxArea = float(img.rows*img.cols);
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
    // Param for fifth BLOB detector we want ratio inertia
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByConvexity = true;
    pBLOB.back().minConvexity = 0.;
    pBLOB.back().maxConvexity = (float)0.9;
    // Param for six BLOB detector we want blob with gravity center color equal to 0
    typeDesc.push_back("BLOB");
    pBLOB.push_back(pDefaultBLOB);
    pBLOB.back().filterByColor = true;
    pBLOB.back().blobColor = 0;

    itBLOB = pBLOB.begin();
    vector<double> desMethCmp;
    Ptr<Feature2D> b;
    String label;
    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        vector<KeyPoint> keyImg1;
        if (*itDesc == "BLOB")
        {
            b = SimpleBlobDetector::create(*itBLOB);
            label = Legende(*itBLOB);
            ++itBLOB;
        }
        try
        {
            // We can detect keypoint with detect method
            vector<KeyPoint>  keyImg;
            vector<Rect>  zone;
            vector<vector <Point> >  region;
            Mat     desc, result(img.rows, img.cols, CV_8UC3);
            if (b.dynamicCast<SimpleBlobDetector>().get())
            {
                Ptr<SimpleBlobDetector> sbd = b.dynamicCast<SimpleBlobDetector>();
                sbd->detect(img, keyImg, Mat());
                drawKeypoints(img, keyImg, result);
                int i = 0;
                for (vector<KeyPoint>::iterator k = keyImg.begin(); k != keyImg.end(); ++k, ++i)
                    circle(result, k->pt, (int)k->size, palette[i % 65536]);
            }
            namedWindow(*itDesc + label, WINDOW_AUTOSIZE);
            imshow(*itDesc + label, result);
            imshow("Original", img);
            waitKey();
        }
        catch (const Exception& e)
        {
            cout << "Feature : " << *itDesc << "\n";
            cout << e.msg << endl;
        }
    }
    return 0;
}
