#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/features2d/features2d.hpp"

namespace cv
{

class CV_EXPORTS_AS(FeatureDetector) javaFeatureDetector : public FeatureDetector
{
public:
#if 0
    CV_WRAP void detect( const Mat& image, CV_OUT vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    CV_WRAP void detect( const vector<Mat>& images, CV_OUT vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const;
    CV_WRAP virtual bool empty() const;
#endif

    enum
    {
        FAST          = 1,
        STAR          = 2,
        SIFT          = 3,
        SURF          = 4,
        ORB           = 5,
        MSER          = 6,
        GFTT          = 7,
        HARRIS        = 8,


        GRIDRETECTOR = 1000,

        GRID_FAST      = GRIDRETECTOR + FAST,
        GRID_STAR      = GRIDRETECTOR + STAR,
        GRID_SIFT      = GRIDRETECTOR + SIFT,
        GRID_SURF      = GRIDRETECTOR + SURF,
        GRID_ORB       = GRIDRETECTOR + ORB,
        GRID_MSER      = GRIDRETECTOR + MSER,
        GRID_GFTT      = GRIDRETECTOR + GFTT,
        GRID_HARRIS    = GRIDRETECTOR + HARRIS,


        PYRAMIDDETECTOR = 2000,

        PYRAMID_FAST   = PYRAMIDDETECTOR + FAST,
        PYRAMID_STAR   = PYRAMIDDETECTOR + STAR,
        PYRAMID_SIFT   = PYRAMIDDETECTOR + SIFT,
        PYRAMID_SURF   = PYRAMIDDETECTOR + SURF,
        PYRAMID_ORB    = PYRAMIDDETECTOR + ORB,
        PYRAMID_MSER   = PYRAMIDDETECTOR + MSER,
        PYRAMID_GFTT   = PYRAMIDDETECTOR + GFTT,
        PYRAMID_HARRIS = PYRAMIDDETECTOR + HARRIS,

        DYNAMICDETECTOR = 3000,

        DYNAMIC_FAST   = DYNAMICDETECTOR + FAST,
        DYNAMIC_STAR   = DYNAMICDETECTOR + STAR,
        DYNAMIC_SIFT   = DYNAMICDETECTOR + SIFT,
        DYNAMIC_SURF   = DYNAMICDETECTOR + SURF,
        DYNAMIC_ORB    = DYNAMICDETECTOR + ORB,
        DYNAMIC_MSER   = DYNAMICDETECTOR + MSER,
        DYNAMIC_GFTT   = DYNAMICDETECTOR + GFTT,
        DYNAMIC_HARRIS = DYNAMICDETECTOR + HARRIS
    };

    //supported: FAST STAR SIFT SURF ORB MSER GFTT HARRIS Grid(XXXX) Pyramid(XXXX) Dynamic(XXXX)
    //not supported: SimpleBlob, Dense
    CV_WRAP static javaFeatureDetector* create( int detectorType )
    {
        string name;
        if (detectorType > DYNAMICDETECTOR)
        {
            name = "Dynamic";
            detectorType -= DYNAMICDETECTOR;
        }
        if (detectorType > PYRAMIDDETECTOR)
        {
            name = "Pyramid";
            detectorType -= PYRAMIDDETECTOR;
        }
        if (detectorType > GRIDRETECTOR)
        {
            name = "Grid";
            detectorType -= GRIDRETECTOR;
        }

        switch(detectorType)
        {
        case FAST:
            name += "FAST";
            break;
        case STAR:
            name += "STAR";
            break;
        case SIFT:
            name += "SIFT";
            break;
        case SURF:
            name += "SURF";
            break;
        case ORB:
            name += "ORB";
            break;
        case MSER:
            name += "MSER";
            break;
        case GFTT:
            name += "GFTT";
            break;
        case HARRIS:
            name += "HARRIS";
            break;
        default:
            CV_Error( CV_StsBadArg, "Specified feature detector type is not supported." );
            break;
        }

        Ptr<FeatureDetector> detector = FeatureDetector::create(name);
        detector.addref();
        return (javaFeatureDetector*)((FeatureDetector*) detector);
    }

    CV_WRAP void write( const string& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        ((FeatureDetector*)this)->write(fs);
        fs.release();
    }

    CV_WRAP void read( const string& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        ((FeatureDetector*)this)->read(fs.root());
        fs.release();
    }
};

class CV_EXPORTS_AS(DescriptorMatcher) javaDescriptorMatcher : public DescriptorMatcher
{
public:
#if 0
    CV_WRAP virtual bool isMaskSupported() const;
    CV_WRAP virtual void add( const vector<Mat>& descriptors );
    CV_WRAP const vector<Mat>& getTrainDescriptors() const;
    CV_WRAP virtual void clear();
    CV_WRAP virtual bool empty() const;
    CV_WRAP virtual void train();
    CV_WRAP void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                CV_OUT vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    CV_WRAP void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                   CV_OUT vector<vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                      CV_OUT vector<vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void match( const Mat& queryDescriptors, CV_OUT vector<DMatch>& matches,
                const vector<Mat>& masks=vector<Mat>() );
    CV_WRAP void knnMatch( const Mat& queryDescriptors, CV_OUT vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    CV_WRAP void radiusMatch( const Mat& queryDescriptors, CV_OUT vector<vector<DMatch> >& matches, float maxDistance,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
#endif

    enum
    {
        FLANNBASED            = 1,
        BRUTEFORCE            = 2,
        BRUTEFORCE_L1         = 3,
        BRUTEFORCE_HAMMING    = 4,
        BRUTEFORCE_HAMMINGLUT = 5
    };

    CV_WRAP_AS(clone) javaDescriptorMatcher* jclone( bool emptyTrainData=false ) const
    {
        Ptr<DescriptorMatcher> matcher = this->clone(emptyTrainData);
        matcher.addref();
        return (javaDescriptorMatcher*)((DescriptorMatcher*) matcher);
    }
    
    //supported: FlannBased, BruteForce, BruteForce-L1, BruteForce-Hamming, BruteForce-HammingLUT
    CV_WRAP static javaDescriptorMatcher* create( int matcherType )
    {
        string name;

        switch(matcherType)
        {
        case FLANNBASED:
            name = "FlannBased";
            break;
        case BRUTEFORCE:
            name = "BruteForce";
            break;
        case BRUTEFORCE_L1:
            name = "BruteForce-L1";
            break;
        case BRUTEFORCE_HAMMING:
            name = "BruteForce-Hamming";
            break;
        case BRUTEFORCE_HAMMINGLUT:
            name = "BruteForce-HammingLUT";
            break;
        default:
            CV_Error( CV_StsBadArg, "Specified descriptor matcher type is not supported." );
            break;

        }

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(name);
        matcher.addref();
        return (javaDescriptorMatcher*)((DescriptorMatcher*) matcher);
    }

    CV_WRAP void write( const string& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        ((DescriptorMatcher*)this)->write(fs);
        fs.release();
    }

    CV_WRAP void read( const string& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        ((DescriptorMatcher*)this)->read(fs.root());
        fs.release();
    }
};

class CV_EXPORTS_AS(DescriptorExtractor) javaDescriptorExtractor : public DescriptorExtractor
{
public:
#if 0
    CV_WRAP void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    CV_WRAP void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, CV_OUT vector<Mat>& descriptors ) const;
    CV_WRAP virtual int descriptorSize() const;
    CV_WRAP virtual int descriptorType() const;

    CV_WRAP virtual bool empty() const;
#endif

    enum
    {
        SIFT  = 1,
        SURF  = 2,
        ORB   = 3,
        BRIEF = 4,


        OPPONENTEXTRACTOR = 1000,

        OPPENENT_SIFT  = OPPONENTEXTRACTOR + SIFT,
        OPPENENT_SURF  = OPPONENTEXTRACTOR + SURF,
        OPPENENT_ORB   = OPPONENTEXTRACTOR + ORB,
        OPPENENT_BRIEF = OPPONENTEXTRACTOR + BRIEF
    };

    //supported SIFT, SURF, ORB, BRIEF, Opponent(XXXX)
    //not supported: Calonder
    CV_WRAP static javaDescriptorExtractor* create( int extractorType )
    {
        string name;

        if (extractorType > OPPONENTEXTRACTOR)
        {
            name = "Opponent";
            extractorType -= OPPONENTEXTRACTOR;
        }

        switch(extractorType)
        {
        case SIFT:
            name += "SIFT";
            break;
        case SURF:
            name += "SURF";
            break;
        case ORB:
            name += "ORB";
            break;
        case BRIEF:
            name += "BRIEF";
            break;
        default:
            CV_Error( CV_StsBadArg, "Specified descriptor extractor type is not supported." );
            break;
        }

        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(name);
        extractor.addref();
        return (javaDescriptorExtractor*)((DescriptorExtractor*) extractor);
    }

    CV_WRAP void write( const string& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        ((DescriptorExtractor*)this)->write(fs);
        fs.release();
    }

    CV_WRAP void read( const string& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        ((DescriptorExtractor*)this)->read(fs.root());
        fs.release();
    }
};

#if 0
enum
{ 
          DRAW_OVER_OUTIMG = 1, // Output image matrix will not be created (Mat::create).
                                // Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4 // For each keypoint the circle around keypoint with keypoint size and
                                  // orientation will be drawn.
};

// Draw keypoints.
CV_EXPORTS_W void drawKeypoints( const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImage,
                               const Scalar& color=Scalar::all(-1), int flags=0 );

// Draws matches of keypints from two images on output image.
CV_EXPORTS_W void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                             const Mat& img2, const vector<KeyPoint>& keypoints2,
                             const vector<DMatch>& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const vector<char>& matchesMask=vector<char>(), int flags=0 );

CV_EXPORTS_W void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                             const Mat& img2, const vector<KeyPoint>& keypoints2,
                             const vector<vector<DMatch> >& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const vector<vector<char> >& matchesMask=vector<vector<char> >(), int flags=0);
                             
#endif

} //cv

#endif // __OPENCV_FEATURES_2D_MANUAL_HPP__
