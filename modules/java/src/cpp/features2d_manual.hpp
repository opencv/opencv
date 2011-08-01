#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/features2d/features2d.hpp"

namespace cv
{

class CV_EXPORTS_AS(FeatureDetector) javaFeatureDetector : public FeatureDetector
{
public:
#if 0
    void detect( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    void detect( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const;
    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;
    virtual bool empty() const;
#endif

	//supported: FAST STAR SIFT SURF ORB MSER GFTT HARRIS Grid(XXXX) Pyramid(XXXX) Dynamic(XXXX)
	//not supported: SimpleBlob, Dense
	CV_WRAP_AS(create) static javaFeatureDetector* jcreate( const string& detectorType )
	{
	    Ptr<FeatureDetector> detector = FeatureDetector::create(detectorType);
		detector.addref();
		return (javaFeatureDetector*)((FeatureDetector*) detector);
	}
};

class CV_EXPORTS_AS(DescriptorMatcher) javaDescriptorMatcher : public DescriptorMatcher
{
public:
#if 0
    CV_WRAP virtual bool isMaskSupported() const;
	CV_WRAP virtual void add( const vector<Mat>& descriptors );
	//CV_WRAP const vector<Mat>& getTrainDescriptors() const;
	CV_WRAP virtual void clear();
	CV_WRAP virtual bool empty() const;
	CV_WRAP virtual void train();
	CV_WRAP void match( const Mat& queryDescriptors, const Mat& trainDescriptors,
                vector<DMatch>& matches, const Mat& mask=Mat() ) const;
	CV_WRAP void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                   vector<vector<DMatch> >& matches, int k,
                   const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors,
                      vector<vector<DMatch> >& matches, float maxDistance,
                      const Mat& mask=Mat(), bool compactResult=false ) const;
    CV_WRAP void match( const Mat& queryDescriptors, vector<DMatch>& matches,
                const vector<Mat>& masks=vector<Mat>() );
    CV_WRAP void knnMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    CV_WRAP void radiusMatch( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
                   const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    CV_WRAP virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    CV_WRAP virtual void write( FileStorage& ) const;
#endif

    CV_WRAP_AS(clone) javaDescriptorMatcher* jclone( bool emptyTrainData=false ) const
	{
	    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::clone(emptyTrainData);
		matcher.addref();
		return (javaDescriptorMatcher*)((DescriptorMatcher*) matcher);
	}
	
	//supported: FlannBased, BruteForce, BruteForce-L1, BruteForce-Hamming, BruteForce-HammingLUT
	CV_WRAP_AS(create) static javaDescriptorMatcher* jcreate( const string& descriptorMatcherType )
	{
	    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(descriptorMatcherType);
		matcher.addref();
		return (javaDescriptorMatcher*)((DescriptorMatcher*) matcher);
	}
};

class CV_EXPORTS_AS(DescriptorExtractor) javaDescriptorExtractor : public DescriptorExtractor
{
public:
#if 0
    CV_WRAP void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    CV_WRAP void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const;
    CV_WRAP virtual void read( const FileNode& );
    CV_WRAP virtual void write( FileStorage& ) const;
    CV_WRAP virtual int descriptorSize() const = 0;
    CV_WRAP virtual int descriptorType() const = 0;

    CV_WRAP virtual bool empty() const;
#endif

    //supported SIFT, SURF, ORB, BRIEF, Opponent(XXXX)
	//not supported: Calonder
	CV_WRAP_AS(create) static javaDescriptorExtractor* jcreate( const string& descriptorExtractorType )
	{
	    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptorExtractorType);
		extractor.addref();
		return (javaDescriptorExtractor*)((DescriptorExtractor*) extractor);
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