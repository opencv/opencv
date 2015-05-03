#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


int main(void)
{
    vector<String> typeAlgoMatch;
    typeAlgoMatch.push_back("BruteForce");
    typeAlgoMatch.push_back("BruteForce-Hamming");
    typeAlgoMatch.push_back("BruteForce-Hamming(2)");

    vector<String> typeDesc;
    typeDesc.push_back("AKAZE");
    typeDesc.push_back("ORB");
    typeDesc.push_back("BRISK");

    String  dataFolder("../data/");
    vector<String> fileName;
    fileName.push_back("basketball1.png");
    fileName.push_back("basketball2.png");

    Mat img1 = imread(dataFolder+fileName[0], IMREAD_GRAYSCALE);
    Mat img2 = imread(dataFolder+fileName[1], IMREAD_GRAYSCALE);

    Ptr<Feature2D> b;

    vector<String>::iterator itDesc;
// Descriptor loop
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); itDesc++){
        Ptr<DescriptorMatcher> descriptorMatcher;
        vector<DMatch> matches;	        /*<! Match between img and img2*/
        vector<KeyPoint> keyImg1;		/*<! keypoint  for img1 */
        vector<KeyPoint> keyImg2;		/*<! keypoint  for img2 */
        Mat descImg1, descImg2;         /*<! Descriptor for img1 and img2 */
        vector<String>::iterator itMatcher = typeAlgoMatch.end();
        if (*itDesc == "AKAZE"){
            b = AKAZE::create();
        }
        if (*itDesc == "ORB"){
            b = ORB::create();
        }
        else if (*itDesc == "BRISK"){
            b = BRISK::create();
        }
        try {
            b->detect(img1, keyImg1, Mat());
            b->compute(img1, keyImg1, descImg1);
            b->detectAndCompute(img2, Mat(),keyImg2, descImg2,false);
 // Match method loop
             for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); itMatcher++){
                descriptorMatcher = DescriptorMatcher::create(*itMatcher);
                descriptorMatcher->match(descImg1, descImg2, matches, Mat());
 // Keep best matches only to have a nice drawing
                Mat index;
                int nbMatch=int(matches.size());
                Mat tab(nbMatch, 1, CV_32F);
                for (int i = 0; i<nbMatch; i++)
                    tab.at<float>(i, 0) = matches[i].distance;
                sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                vector<DMatch> bestMatches;
                for (int i = 0; i<30; i++)
                    bestMatches.push_back(matches[index.at<int>(i, 0)]);
                Mat result;
                drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
                namedWindow(*itDesc+": "+*itMatcher, WINDOW_AUTOSIZE);
                imshow(*itDesc + ": " + *itMatcher, result);
                FileStorage fs(*itDesc+"_"+*itMatcher+"_"+fileName[0]+"_"+fileName[1]+".xml", FileStorage::WRITE);
                fs<<"Matches"<<matches;
                vector<DMatch>::iterator it;
                cout << "Index \tIndex \tindex \tdistance\n";
                cout << "in img1\tin img2\timage\t\n";
                for (it = matches.begin(); it != matches.end(); it++)
                    cout << it->queryIdx << "\t" << it->trainIdx << "\t" << it->imgIdx << "\t" << it->distance<<"\n";
                waitKey();
            }
        }
        catch (Exception& e){
            cout << "Feature : " << *itDesc << "\n";
            if (itMatcher != typeAlgoMatch.end())
                cout << "Matcher : " << *itMatcher << "\n";
            cout<<e.msg<<endl;
        }
    }
    return 0;
}
