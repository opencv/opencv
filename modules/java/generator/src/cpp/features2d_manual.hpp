#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d.hpp"

#undef SIMPLEBLOB // to solve conflict with wincrypt.h on windows

namespace cv
{

class CV_EXPORTS_AS(FeatureDetector) javaFeatureDetector
{
public:
    CV_WRAP void detect( const Mat& image, CV_OUT std::vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const
    { return wrapped->detect(image, keypoints, mask); }

    CV_WRAP void detect( const std::vector<Mat>& images, CV_OUT std::vector<std::vector<KeyPoint> >& keypoints, const std::vector<Mat>& masks=std::vector<Mat>() ) const
    { return wrapped->detect(images, keypoints, masks); }

    CV_WRAP bool empty() const
    { return wrapped->empty(); }

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
        String name;
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
            name = name + "FAST";
            break;
        case STAR:
            name = name + "STAR";
            break;
        case SIFT:
            name = name + "SIFT";
            break;
        case SURF:
            name = name + "SURF";
            break;
        case ORB:
            name = name + "ORB";
            break;
        case MSER:
            name = name + "MSER";
            break;
        case GFTT:
            name = name + "GFTT";
            break;
        case HARRIS:
            name = name + "HARRIS";
            break;
        case SIMPLEBLOB:
            name = name + "SimpleBlob";
            break;
        case DENSE:
            name = name + "Dense";
            break;
        case BRISK:
            name = name + "BRISK";
            break;
        default:
            CV_Error( Error::StsBadArg, "Specified feature detector type is not supported." );
            break;
        }

        return new javaFeatureDetector(FeatureDetector::create(name));
    }

    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        wrapped->write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        wrapped->read(fs.root());
    }

private:
    javaFeatureDetector(Ptr<FeatureDetector> _wrapped) : wrapped(_wrapped)
    {}

    Ptr<FeatureDetector> wrapped;
};

class CV_EXPORTS_AS(DescriptorMatcher) javaDescriptorMatcher
{
public:
    CV_WRAP bool isMaskSupported() const
    { return wrapped->isMaskSupported(); }

    CV_WRAP void add( const std::vector<Mat>& descriptors )
    { return wrapped->add(descriptors); }

    CV_WRAP const std::vector<Mat>& getTrainDescriptors() const
    { return wrapped->getTrainDescriptors(); }

    CV_WRAP void clear()
    { return wrapped->clear(); }

    CV_WRAP bool empty() const
    { return wrapped->empty(); }

    CV_WRAP void train()
    { return wrapped->train(); }

    CV_WRAP void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                CV_OUT std::vector<DMatch>& matches, const Mat& mask=Mat() ) const
    { return wrapped->match(queryDescriptors, trainDescriptors, matches, mask); }

    CV_WRAP void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const
    { return wrapped->knnMatch(queryDescriptors, trainDescriptors, matches, k, mask, compactResult); }

    CV_WRAP void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                      CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const
    { return wrapped->radiusMatch(queryDescriptors, trainDescriptors, matches, maxDistance, mask, compactResult); }

    CV_WRAP void match( const Mat& queryDescriptors, CV_OUT std::vector<DMatch>& matches,
                const std::vector<Mat>& masks=std::vector<Mat>() )
    { return wrapped->match(queryDescriptors, matches, masks); }

    CV_WRAP void knnMatch( const Mat& queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
           const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )
    { return wrapped->knnMatch(queryDescriptors, matches, k, masks, compactResult); }

    CV_WRAP void radiusMatch( const Mat& queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                   const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )
    { return wrapped->radiusMatch(queryDescriptors, matches, maxDistance, masks, compactResult); }

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
        return new javaDescriptorMatcher(wrapped->clone(emptyTrainData));
    }

    //supported: FlannBased, BruteForce, BruteForce-L1, BruteForce-Hamming, BruteForce-HammingLUT
    CV_WRAP static javaDescriptorMatcher* create( int matcherType )
    {
        String name;

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
            CV_Error( Error::StsBadArg, "Specified descriptor matcher type is not supported." );
            break;
        }

        return new javaDescriptorMatcher(DescriptorMatcher::create(name));
    }

    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        wrapped->write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        wrapped->read(fs.root());
    }

private:
    javaDescriptorMatcher(Ptr<DescriptorMatcher> _wrapped) : wrapped(_wrapped)
    {}

    Ptr<DescriptorMatcher> wrapped;
};

class CV_EXPORTS_AS(DescriptorExtractor) javaDescriptorExtractor
{
public:
    CV_WRAP void compute( const Mat& image, CV_IN_OUT std::vector<KeyPoint>& keypoints, Mat& descriptors ) const
    { return wrapped->compute(image, keypoints, descriptors); }

    CV_WRAP void compute( const std::vector<Mat>& images, CV_IN_OUT std::vector<std::vector<KeyPoint> >& keypoints, CV_OUT std::vector<Mat>& descriptors ) const
    { return wrapped->compute(images, keypoints, descriptors); }

    CV_WRAP int descriptorSize() const
    { return wrapped->descriptorSize(); }

    CV_WRAP int descriptorType() const
    { return wrapped->descriptorType(); }

    CV_WRAP bool empty() const
    { return wrapped->empty(); }

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
        String name;

        if (extractorType > OPPONENTEXTRACTOR)
        {
            name = "Opponent";
            extractorType -= OPPONENTEXTRACTOR;
        }

        switch(extractorType)
        {
        case SIFT:
            name = name + "SIFT";
            break;
        case SURF:
            name = name + "SURF";
            break;
        case ORB:
            name = name + "ORB";
            break;
        case BRIEF:
            name = name + "BRIEF";
            break;
        case BRISK:
            name = name + "BRISK";
            break;
        case FREAK:
            name = name + "FREAK";
            break;
        default:
            CV_Error( Error::StsBadArg, "Specified descriptor extractor type is not supported." );
            break;
        }

        return new javaDescriptorExtractor(DescriptorExtractor::create(name));
    }

    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        wrapped->write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        wrapped->read(fs.root());
    }

private:
    javaDescriptorExtractor(Ptr<DescriptorExtractor> _wrapped) : wrapped(_wrapped)
    {}

    Ptr<DescriptorExtractor> wrapped;
};

class CV_EXPORTS_AS(GenericDescriptorMatcher) javaGenericDescriptorMatcher
{
public:
    CV_WRAP void add( const std::vector<Mat>& images,
                      std::vector<std::vector<KeyPoint> >& keypoints )
    { return wrapped->add(images, keypoints); }

    CV_WRAP const std::vector<Mat>& getTrainImages() const
    { return wrapped->getTrainImages(); }

    CV_WRAP const std::vector<std::vector<KeyPoint> >& getTrainKeypoints() const
    { return wrapped->getTrainKeypoints(); }

    CV_WRAP void clear()
    { return wrapped->clear(); }

    CV_WRAP bool isMaskSupported()
    { return wrapped->isMaskSupported(); }

    CV_WRAP void train()
    { return wrapped->train(); }

    CV_WRAP void classify( const Mat& queryImage, CV_IN_OUT std::vector<KeyPoint>& queryKeypoints,
                           const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints ) const
    { return wrapped->classify(queryImage, queryKeypoints, trainImage, trainKeypoints); }

    CV_WRAP void classify( const Mat& queryImage, CV_IN_OUT std::vector<KeyPoint>& queryKeypoints )
    { return wrapped->classify(queryImage, queryKeypoints); }

    CV_WRAP void match( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                CV_OUT std::vector<DMatch>& matches, const Mat& mask=Mat() ) const
    { return wrapped->match(queryImage, queryKeypoints, trainImage, trainKeypoints, matches, mask); }

    CV_WRAP void knnMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                   const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const
    { return wrapped->knnMatch(queryImage, queryKeypoints, trainImage, trainKeypoints,
                               matches, k, mask, compactResult); }

    CV_WRAP void radiusMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                      const Mat& trainImage, std::vector<KeyPoint>& trainKeypoints,
                      CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const
    { return wrapped->radiusMatch(queryImage, queryKeypoints, trainImage, trainKeypoints,
                                   matches, maxDistance, mask, compactResult); }

    CV_WRAP void match( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                CV_OUT std::vector<DMatch>& matches, const std::vector<Mat>& masks=std::vector<Mat>() )
    { return wrapped->match(queryImage, queryKeypoints, matches, masks); }

    CV_WRAP void knnMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )
    { return wrapped->knnMatch(queryImage, queryKeypoints, matches, k, masks, compactResult); }

    CV_WRAP void radiusMatch( const Mat& queryImage, std::vector<KeyPoint>& queryKeypoints,
                      CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      const std::vector<Mat>& masks=std::vector<Mat>(), bool compactResult=false )
    { return wrapped->radiusMatch(queryImage, queryKeypoints, matches, maxDistance, masks, compactResult); }

    CV_WRAP bool empty() const
    { return wrapped->empty(); }


    enum
    {
        ONEWAY = 1,
        FERN   = 2
    };

    CV_WRAP_AS(clone) javaGenericDescriptorMatcher* jclone( bool emptyTrainData=false ) const
    {
        return new javaGenericDescriptorMatcher(wrapped->clone(emptyTrainData));
    }

    //supported: OneWay, Fern
    //unsupported: Vector
    CV_WRAP static javaGenericDescriptorMatcher* create( int matcherType )
    {
        String name;

        switch(matcherType)
        {
        case ONEWAY:
            name = "ONEWAY";
            break;
        case FERN:
            name = "FERN";
            break;
        default:
            CV_Error( Error::StsBadArg, "Specified generic descriptor matcher type is not supported." );
            break;
        }

        return new javaGenericDescriptorMatcher(GenericDescriptorMatcher::create(name));
    }

    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        wrapped->write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        wrapped->read(fs.root());
    }

private:
    javaGenericDescriptorMatcher(Ptr<GenericDescriptorMatcher> _wrapped) : wrapped(_wrapped)
    {}

    Ptr<GenericDescriptorMatcher> wrapped;
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
CV_EXPORTS_W void drawKeypoints( const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& outImage,
                               const Scalar& color=Scalar::all(-1), int flags=0 );

// Draws matches of keypints from two images on output image.
CV_EXPORTS_W void drawMatches( const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                             const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), int flags=0 );

CV_EXPORTS_AS(drawMatches2) void drawMatches( const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                             const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=0);

#endif

} //cv

#endif // HAVE_OPENCV_FEATURES2D

#endif // __OPENCV_FEATURES_2D_MANUAL_HPP__
