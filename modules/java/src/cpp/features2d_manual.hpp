#ifndef __OPENCV_FEATURES_2D_MANUAL_HPP__
#define __OPENCV_FEATURES_2D_MANUAL_HPP__

#include "opencv2/features2d/features2d.hpp"

namespace cv
{

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
	
	CV_WRAP_AS(create) static javaDescriptorMatcher* jcreate( const string& descriptorMatcherType )
	{
	    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(descriptorMatcherType);
		matcher.addref();
		return (javaDescriptorMatcher*)((DescriptorMatcher*) matcher);
	}
};

} //cv

#endif // __OPENCV_FEATURES_2D_MANUAL_HPP__