#include <highgui.h>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const char dlmtr = '/';

void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask );
void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames );

int main(int argc, char** argv)
{
    Mat queryImg;
    vector<KeyPoint> queryPoints;
    Mat queryDescs;

    vector<Mat> trainImgCollection;
    vector<vector<KeyPoint> > trainPointCollection;
    vector<Mat> trainDescCollection;

    vector<DMatch> matches;

    if( argc != 7 )
    {
        cout << "Format:" << endl;
        cout << argv[0] << "[detectorType] [descriptorType] [matcherType] [queryImage] [fileWithTrainImages] [dirToSaveResImages]" << endl;
        return -1;
    }

    cout << "< 1.) Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
    Ptr<FeatureDetector> detector = createFeatureDetector( argv[1] );
    Ptr<DescriptorExtractor> descriptorExtractor = createDescriptorExtractor( argv[2] );
    Ptr<DescriptorMatcher> descriptorMatcher = createDescriptorMatcher( argv[3] );
    cout << ">" << endl;
    if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "Can not create feature detector or descriptor exstractor or descriptor matcher of given types." << endl << ">" << endl;
        return -1;
	}

    cout << "< 2.) Reading the images..." << endl;
    queryImg = imread( argv[4], CV_LOAD_IMAGE_GRAYSCALE);
    if( queryImg.empty() )
    {
        cout << "Query image can not be read." << endl << ">" << endl;
        return -1;
    }
    string trainDirName;
    vector<string> trainFilenames;
    vector<int> usedTrainImgIdxs;
    readTrainFilenames( argv[5], trainDirName, trainFilenames );
    if( trainFilenames.empty() )
    {
        cout << "Train image filenames can not be read." << endl << ">" << endl;
        return -1;
    }
    for( size_t i = 0; i < trainFilenames.size(); i++ )
    {
        Mat img = imread( trainDirName + trainFilenames[i], CV_LOAD_IMAGE_GRAYSCALE );
        if( img.empty() ) cout << "Train image " << trainDirName + trainFilenames[i] << " can not be read." << endl;
        trainImgCollection.push_back( img );
        usedTrainImgIdxs.push_back( i );
    }
    if( trainImgCollection.empty() )
    {
        cout << "All train images can not be read." << endl << ">" << endl;
        return -1;
    }
    else
        cout << trainImgCollection.size() << " train images were read." << endl;
    cout << ">" << endl;

    cout << endl << "< 3.) Extracting keypoints from images..." << endl;
    detector->detect( queryImg, queryPoints );
    detector->detect( trainImgCollection, trainPointCollection );
    cout << ">" << endl;

    cout << "< 4.) Computing descriptors for keypoints..." << endl;
    descriptorExtractor->compute( queryImg, queryPoints, queryDescs );
    descriptorExtractor->compute( trainImgCollection, trainPointCollection, trainDescCollection );
    cout << ">" << endl;

    cout << "< 5.) Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    descriptorMatcher->add( trainDescCollection );
    descriptorMatcher->match( queryDescs, matches );
    CV_Assert( queryPoints.size() == matches.size() );
    cout << ">" << endl;

    Mat drawImg;
    vector<char> mask;
    for( size_t i = 0; i < trainImgCollection.size(); i++ )
    {
        maskMatchesByTrainImgIdx( matches, i, mask );
        drawMatches( queryImg, queryPoints, trainImgCollection[i], trainPointCollection[i],
                     matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask );

        imwrite( string(argv[6]) + "/res_" + trainFilenames[usedTrainImgIdxs[i]] + ".png", drawImg );
    }
    return 0;
}


void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
{
    mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
            mask[i] = 1;
    }
}

void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind(dlmtr);
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}
