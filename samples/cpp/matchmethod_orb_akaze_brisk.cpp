#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static void help(char* argv[])
{
    cout << "\n This program demonstrates how to detect compute and match ORB BRISK and AKAZE descriptors \n"
            "Usage: \n  "
         << argv[0] << " --image1=<image1(basketball1.png as default)> --image2=<image2(basketball2.png as default)>\n"
        "Press a key when image window is active to change algorithm or descriptor";
}



int main(int argc, char *argv[])
{
    vector<String> typeDesc;
    vector<String> typeAlgoMatch;
    vector<String> fileName;
    // This descriptor are going to be detect and compute
    typeDesc.push_back("AKAZE-DESCRIPTOR_KAZE_UPRIGHT");    // see https://docs.opencv.org/master/d8/d30/classcv_1_1AKAZE.html
    typeDesc.push_back("AKAZE");    // see http://docs.opencv.org/master/d8/d30/classcv_1_1AKAZE.html
    typeDesc.push_back("ORB");      // see http://docs.opencv.org/master/de/dbf/classcv_1_1BRISK.html
    typeDesc.push_back("BRISK");    // see http://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
    // This algorithm would be used to match descriptors see http://docs.opencv.org/master/db/d39/classcv_1_1DescriptorMatcher.html#ab5dc5036569ecc8d47565007fa518257
    typeAlgoMatch.push_back("BruteForce");
    typeAlgoMatch.push_back("BruteForce-L1");
    typeAlgoMatch.push_back("BruteForce-Hamming");
    typeAlgoMatch.push_back("BruteForce-Hamming(2)");
    cv::CommandLineParser parser(argc, argv,
        "{ @image1 | basketball1.png | }"
        "{ @image2 | basketball2.png | }"
        "{help h ||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    fileName.push_back(samples::findFile(parser.get<string>(0)));
    fileName.push_back(samples::findFile(parser.get<string>(1)));
    Mat img1 = imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img2 = imread(fileName[1], IMREAD_GRAYSCALE);
    if (img1.empty())
    {
        cerr << "Image " << fileName[0] << " is empty or cannot be found" << endl;
        return 1;
    }
    if (img2.empty())
    {
        cerr << "Image " << fileName[1] << " is empty or cannot be found" << endl;
        return 1;
    }

    vector<double> desMethCmp;
    Ptr<Feature2D> b;

    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        Ptr<DescriptorMatcher> descriptorMatcher;
        // Match between img1 and img2
        vector<DMatch> matches;
        // keypoint  for img1 and img2
        vector<KeyPoint> keyImg1, keyImg2;
        // Descriptor for img1 and img2
        Mat descImg1, descImg2;
        vector<String>::iterator itMatcher = typeAlgoMatch.end();
        if (*itDesc == "AKAZE-DESCRIPTOR_KAZE_UPRIGHT"){
            b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
        }
        if (*itDesc == "AKAZE"){
            b = AKAZE::create();
        }
        if (*itDesc == "ORB"){
            b = ORB::create();
        }
        else if (*itDesc == "BRISK"){
            b = BRISK::create();
        }
        try
        {
            // We can detect keypoint with detect method
            b->detect(img1, keyImg1, Mat());
            // and compute their descriptors with method  compute
            b->compute(img1, keyImg1, descImg1);
            // or detect and compute descriptors in one step
            b->detectAndCompute(img2, Mat(),keyImg2, descImg2,false);
            // Match method loop
            for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher){
                descriptorMatcher = DescriptorMatcher::create(*itMatcher);
                if ((*itMatcher == "BruteForce-Hamming" || *itMatcher == "BruteForce-Hamming(2)") && (b->descriptorType() == CV_32F || b->defaultNorm() <= NORM_L2SQR))
                {
                    cout << "**************************************************************************\n";
                    cout << "It's strange. You should use Hamming distance only for a binary descriptor\n";
                    cout << "**************************************************************************\n";
                }
                if ((*itMatcher == "BruteForce" || *itMatcher == "BruteForce-L1") && (b->defaultNorm() >= NORM_HAMMING))
                {
                    cout << "**************************************************************************\n";
                    cout << "It's strange. You shouldn't use L1 or L2 distance for a binary descriptor\n";
                    cout << "**************************************************************************\n";
                }
                try
                {
                    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
                    // Keep best matches only to have a nice drawing.
                    // We sort distance between descriptor matches
                    Mat index;
                    int nbMatch=int(matches.size());
                    Mat tab(nbMatch, 1, CV_32F);
                    for (int i = 0; i<nbMatch; i++)
                    {
                        tab.at<float>(i, 0) = matches[i].distance;
                    }
                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                    vector<DMatch> bestMatches;
                    for (int i = 0; i<30; i++)
                    {
                        bestMatches.push_back(matches[index.at<int>(i, 0)]);
                    }
                    Mat result;
                    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
                    namedWindow(*itDesc+": "+*itMatcher, WINDOW_AUTOSIZE);
                    imshow(*itDesc + ": " + *itMatcher, result);
                    // Saved result could be wrong due to bug 4308
                    FileStorage fs(*itDesc + "_" + *itMatcher + ".yml", FileStorage::WRITE);
                    fs<<"Matches"<<matches;
                    vector<DMatch>::iterator it;
                    cout<<"**********Match results**********\n";
                    cout << "Index \tIndex \tdistance\n";
                    cout << "in img1\tin img2\n";
                    // Use to compute distance between keyPoint matches and to evaluate match algorithm
                    double cumSumDist2=0;
                    for (it = bestMatches.begin(); it != bestMatches.end(); ++it)
                    {
                        cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
                        Point2d p=keyImg1[it->queryIdx].pt-keyImg2[it->trainIdx].pt;
                        cumSumDist2=p.x*p.x+p.y*p.y;
                    }
                    desMethCmp.push_back(cumSumDist2);
                    waitKey();
                }
                catch (const Exception& e)
                {
                    cout << e.msg << endl;
                    cout << "Cumulative distance cannot be computed." << endl;
                    desMethCmp.push_back(-1);
                }
            }
        }
        catch (const Exception& e)
        {
            cerr << "Exception: " << e.what() << endl;
            cout << "Feature : " << *itDesc << "\n";
            if (itMatcher != typeAlgoMatch.end())
            {
                cout << "Matcher : " << *itMatcher << "\n";
            }
        }
    }
    int i=0;
    cout << "Cumulative distance between keypoint match for different algorithm and feature detector \n\t";
    cout << "We cannot say which is the best but we can say results are different! \n\t";
    for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher)
    {
        cout<<*itMatcher<<"\t";
    }
    cout << "\n";
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        cout << *itDesc << "\t";
        for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher, ++i)
        {
            cout << desMethCmp[i]<<"\t";
        }
        cout<<"\n";
    }
    return 0;
}
