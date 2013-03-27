#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d/features2d.hpp"

#undef SIMPLEBLOB // to solve conflict with wincrypt.h on windows

namespace cv
{

class CV_EXPORTS_AS(FeatureDetector) javaFeatureDetector : public FeatureDetector
{
public:
#if 0
    //DO NOT REMOVE! The block is required for sources parser
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
        SIMPLEBLOB    = 9,
        DENSE         = 10,
        BRISK         = 11,


        GRIDDETECTOR = 1000,
        GRIDRETECTOR = 1000,

        GRID_FAST          = GRIDDETECTOR + FAST,
        GRID_STAR          = GRIDDETECTOR + STAR,
        GRID_SIFT          = GRIDDETECTOR + SIFT,
        GRID_SURF          = GRIDDETECTOR + SURF,
        GRID_ORB           = GRIDDETECTOR + ORB,
        GRID_MSER          = GRIDDETECTOR + MSER,
        GRID_GFTT          = GRIDDETECTOR + GFTT,
        GRID_HARRIS        = GRIDDETECTOR + HARRIS,
        GRID_SIMPLEBLOB    = GRIDDETECTOR + SIMPLEBLOB,
        GRID_DENSE         = GRIDDETECTOR + DENSE,
        GRID_BRISK         = GRIDDETECTOR + BRISK,


        PYRAMIDDETECTOR = 2000,

        PYRAMID_FAST       = PYRAMIDDETECTOR + FAST,
        PYRAMID_STAR       = PYRAMIDDETECTOR + STAR,
        PYRAMID_SIFT       = PYRAMIDDETECTOR + SIFT,
        PYRAMID_SURF       = PYRAMIDDETECTOR + SURF,
        PYRAMID_ORB        = PYRAMIDDETECTOR + ORB,
        PYRAMID_MSER       = PYRAMIDDETECTOR + MSER,
        PYRAMID_GFTT       = PYRAMIDDETECTOR + GFTT,
        PYRAMID_HARRIS     = PYRAMIDDETECTOR + HARRIS,
        PYRAMID_SIMPLEBLOB = PYRAMIDDETECTOR + SIMPLEBLOB,
        PYRAMID_DENSE      = PYRAMIDDETECTOR + DENSE,
        PYRAMID_BRISK      = PYRAMIDDETECTOR + BRISK,

        DYNAMICDETECTOR = 3000,

        DYNAMIC_FAST       = DYNAMICDETECTOR + FAST,
        DYNAMIC_STAR       = DYNAMICDETECTOR + STAR,
        DYNAMIC_SIFT       = DYNAMICDETECTOR + SIFT,
        DYNAMIC_SURF       = DYNAMICDETECTOR + SURF,
        DYNAMIC_ORB        = DYNAMICDETECTOR + ORB,
        DYNAMIC_MSER       = DYNAMICDETECTOR + MSER,
        DYNAMIC_GFTT       = DYNAMICDETECTOR + GFTT,
        DYNAMIC_HARRIS     = DYNAMICDETECTOR + HARRIS,
        DYNAMIC_SIMPLEBLOB = DYNAMICDETECTOR + SIMPLEBLOB,
        DYNAMIC_DENSE      = DYNAMICDETECTOR + DENSE,
        DYNAMIC_BRISK      = DYNAMICDETECTOR + BRISK
    };

    //supported: FAST STAR SIFT SURF ORB MSER GFTT HARRIS BRISK Grid(XXXX) Pyramid(XXXX) Dynamic(XXXX)
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
        if (detectorType > GRIDDETECTOR)
        {
            name = "Grid";
            detectorType -= GRIDDETECTOR;
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
        case SIMPLEBLOB:
            name += "SimpleBlob";
            break;
        case DENSE:
            name += "Dense";
            break;
        case BRISK:
            name += "BRISK";
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
    //DO NOT REMOVE! The block is required for sources parser
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
        BRUTEFORCE_HAMMINGLUT = 5,
        BRUTEFORCE_SL2        = 6
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
        case BRUTEFORCE_SL2:
            name = "BruteForce-SL2";
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
    //DO NOT REMOVE! The block is required for sources parser
    CV_WRAP void compute( const Mat& image, CV_IN_OUT vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    CV_WRAP void compute( const vector<Mat>& images, CV_IN_OUT vector<vector<KeyPoint> >& keypoints, CV_OUT vector<Mat>& descriptors ) const;
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
        BRISK = 5,
        FREAK = 6,


        OPPONENTEXTRACTOR = 1000,



        OPPONENT_SIFT  = OPPONENTEXTRACTOR + SIFT,
        OPPONENT_SURF  = OPPONENTEXTRACTOR + SURF,
        OPPONENT_ORB   = OPPONENTEXTRACTOR + ORB,
        OPPONENT_BRIEF = OPPONENTEXTRACTOR + BRIEF,
        OPPONENT_BRISK = OPPONENTEXTRACTOR + BRISK,
        OPPONENT_FREAK = OPPONENTEXTRACTOR + FREAK
    };

    //supported SIFT, SURF, ORB, BRIEF, BRISK, FREAK, Opponent(XXXX)
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
        case BRISK:
            name += "BRISK";
            break;
        case FREAK:
            name += "FREAK";
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

class CV_EXPORTS_AS(GenericDescriptorMatcher) javaGenericDescriptorMatcher : public GenericDescriptorMatcher
{
public:
#if 0
    //DO NOT REMOVE! The block is required for sources parser
    CV_WRAP virtual void add( const vector<Mat>& images,
                      vector<vector<KeyPoint> >& keypoints );
    CV_WRAP const vector<Mat>& getTrainImages() const;
    CV_WRAP const vector<vector<KeyPoint> >& getTrainKeypoints() const;
    CV_WRAP virtual void clear();
    CV_WRAP virtual bool isMaskSupported();
    CV_WRAP virtual void train();
    CV_WRAP void classify( const Mat& queryImage, CV_IN_OUT vector<KeyPoint>& queryKeypoints,
                           const Mat& trainImage, vector<KeyPoint>& trainKeypoints ) const;
    CV_WRAP void classify( const Mat& queryImage, CV_IN_OUT vector<KeyPoint>& queryKeypoints );
    CV_WRAP void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                CV_OUT vector<DMatch>& matches, const Mat& mask=Mat() ) const;
    CV_WRAP void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                   const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                   CV_OUT vector<vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      const Mat& trainImage, vector<KeyPoint>& trainKeypoints,
                      CV_OUT vector<vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void match( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                CV_OUT vector<DMatch>& matches, const vector<Mat>& masks=vector<Mat>() );
    CV_WRAP void knnMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                   CV_OUT vector<vector<DMatch> >& matches, int k,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    CV_WRAP void radiusMatch( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                      CV_OUT vector<vector<DMatch> >& matches, float maxDistance,
                      const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    CV_WRAP virtual bool empty() const;
#endif

    enum
    {
        ONEWAY = 1,
        FERN   = 2
    };

    CV_WRAP_AS(clone) javaGenericDescriptorMatcher* jclone( bool emptyTrainData=false ) const
    {
        Ptr<GenericDescriptorMatcher> matcher = this->clone(emptyTrainData);
        matcher.addref();
        return (javaGenericDescriptorMatcher*)((GenericDescriptorMatcher*) matcher);
    }

    //supported: OneWay, Fern
    //unsupported: Vector
    CV_WRAP static javaGenericDescriptorMatcher* create( int matcherType )
    {
        string name;

        switch(matcherType)
        {
        case ONEWAY:
            name = "ONEWAY";
            break;
        case FERN:
            name = "FERN";
            break;
        default:
            CV_Error( CV_StsBadArg, "Specified generic descriptor matcher type is not supported." );
            break;
        }

        Ptr<GenericDescriptorMatcher> matcher = GenericDescriptorMatcher::create(name);
        matcher.addref();
        return (javaGenericDescriptorMatcher*)((GenericDescriptorMatcher*) matcher);
    }

    CV_WRAP void write( const string& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        ((GenericDescriptorMatcher*)this)->write(fs);
        fs.release();
    }

    CV_WRAP void read( const string& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        ((GenericDescriptorMatcher*)this)->read(fs.root());
        fs.release();
    }
};

#if 0
//DO NOT REMOVE! The block is required for sources parser
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

CV_EXPORTS_AS(drawMatches2) void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
                             const Mat& img2, const vector<KeyPoint>& keypoints2,
                             const vector<vector<DMatch> >& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const vector<vector<char> >& matchesMask=vector<vector<char> >(), int flags=0);

#endif

} //cv

#endif // HAVE_OPENCV_FEATURES2D

#endif // __OPENCV_FEATURES_2D_MANUAL_HPP__
