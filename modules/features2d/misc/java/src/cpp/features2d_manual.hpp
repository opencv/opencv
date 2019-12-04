#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d.hpp"
#include "features2d_converters.hpp"

#undef SIMPLEBLOB // to solve conflict with wincrypt.h on windows

namespace cv
{

/**
 * @deprecated Please use direct instantiation of Feature2D classes
 */
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
        AKAZE         = 12,


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
        GRID_AKAZE         = GRIDDETECTOR + AKAZE,


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
        PYRAMID_AKAZE      = PYRAMIDDETECTOR + AKAZE,

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
        DYNAMIC_BRISK      = DYNAMICDETECTOR + BRISK,
        DYNAMIC_AKAZE      = DYNAMICDETECTOR + AKAZE
    };

    /**
     * supported: FAST STAR SIFT SURF ORB MSER GFTT HARRIS BRISK AKAZE Grid(XXXX) Pyramid(XXXX) Dynamic(XXXX)
     * not supported: SimpleBlob, Dense
     * @deprecated
     */
    CV_WRAP static Ptr<javaFeatureDetector> create( int detectorType )
    {
        //String name;
        if (detectorType > DYNAMICDETECTOR)
        {
            //name = "Dynamic";
            detectorType -= DYNAMICDETECTOR;
        }
        if (detectorType > PYRAMIDDETECTOR)
        {
            //name = "Pyramid";
            detectorType -= PYRAMIDDETECTOR;
        }
        if (detectorType > GRIDDETECTOR)
        {
            //name = "Grid";
            detectorType -= GRIDDETECTOR;
        }

        Ptr<FeatureDetector> fd;
        switch(detectorType)
        {
        case FAST:
            fd = FastFeatureDetector::create();
            break;
        //case STAR:
        //    fd = xfeatures2d::StarDetector::create();
        //    break;
        //case SIFT:
        //    name = name + "SIFT";
        //    break;
        //case SURF:
        //    name = name + "SURF";
        //    break;
        case ORB:
            fd = ORB::create();
            break;
        case MSER:
            fd = MSER::create();
            break;
        case GFTT:
            fd = GFTTDetector::create();
            break;
        case HARRIS:
            {
            Ptr<GFTTDetector> gftt = GFTTDetector::create();
            gftt->setHarrisDetector(true);
            fd = gftt;
            }
            break;
        case SIMPLEBLOB:
            fd = SimpleBlobDetector::create();
            break;
        //case DENSE:
        //    name = name + "Dense";
        //    break;
        case BRISK:
            fd = BRISK::create();
            break;
        case AKAZE:
            fd = AKAZE::create();
            break;
        default:
            CV_Error( Error::StsBadArg, "Specified feature detector type is not supported." );
            break;
        }

        return makePtr<javaFeatureDetector>(fd);
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

    javaFeatureDetector(Ptr<FeatureDetector> _wrapped) : wrapped(_wrapped)
    {}

private:

    Ptr<FeatureDetector> wrapped;
};

/**
 * @deprecated
 */
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
        AKAZE = 7,


        OPPONENTEXTRACTOR = 1000,



        OPPONENT_SIFT  = OPPONENTEXTRACTOR + SIFT,
        OPPONENT_SURF  = OPPONENTEXTRACTOR + SURF,
        OPPONENT_ORB   = OPPONENTEXTRACTOR + ORB,
        OPPONENT_BRIEF = OPPONENTEXTRACTOR + BRIEF,
        OPPONENT_BRISK = OPPONENTEXTRACTOR + BRISK,
        OPPONENT_FREAK = OPPONENTEXTRACTOR + FREAK,
        OPPONENT_AKAZE = OPPONENTEXTRACTOR + AKAZE
    };

    //supported SIFT, SURF, ORB, BRIEF, BRISK, FREAK, AKAZE, Opponent(XXXX)
    //not supported: Calonder
    CV_WRAP static Ptr<javaDescriptorExtractor> create( int extractorType )
    {
        //String name;

        if (extractorType > OPPONENTEXTRACTOR)
        {
            //name = "Opponent";
            extractorType -= OPPONENTEXTRACTOR;
        }

        Ptr<DescriptorExtractor> de;
        switch(extractorType)
        {
        //case SIFT:
        //    name = name + "SIFT";
        //    break;
        //case SURF:
        //    name = name + "SURF";
        //    break;
        case ORB:
            de = ORB::create();
            break;
        //case BRIEF:
        //    name = name + "BRIEF";
        //    break;
        case BRISK:
            de = BRISK::create();
            break;
        //case FREAK:
        //    name = name + "FREAK";
        //    break;
        case AKAZE:
            de = AKAZE::create();
            break;
        default:
            CV_Error( Error::StsBadArg, "Specified descriptor extractor type is not supported." );
            break;
        }

        return makePtr<javaDescriptorExtractor>(de);
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

    javaDescriptorExtractor(Ptr<DescriptorExtractor> _wrapped) : wrapped(_wrapped)
    {}

private:

    Ptr<DescriptorExtractor> wrapped;
};

#ifdef OPENCV_BINDINGS_PARSER
//DO NOT REMOVE! The block is required for sources parser
enum
{
          DRAW_OVER_OUTIMG = 1, // Output image matrix will not be created (Mat::create).
                                // Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4 // For each keypoint the circle around keypoint with keypoint size and
                                  // orientation will be drawn.
};

CV_EXPORTS_AS(drawMatches2) void drawMatches( const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                             const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, Mat& outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=0);

#endif

} //cv

#endif // HAVE_OPENCV_FEATURES2D

#endif // __OPENCV_FEATURES_2D_MANUAL_HPP__
