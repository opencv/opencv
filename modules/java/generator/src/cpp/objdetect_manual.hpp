#ifndef __OPENCV_OBJDETECT_MANUAL_HPP__
#define __OPENCV_OBJDETECT_MANUAL_HPP__

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_OBJDETECT
#include "opencv2/objdetect.hpp"
namespace cv
{

    class CV_EXPORTS_AS(CascadeClassifier) javaCascadeClassifier
    {
    public:
        CV_WRAP javaCascadeClassifier() {}
        CV_WRAP javaCascadeClassifier(const String& filename)
        {
            cc = createCascadeClassifier(filename);
        }

        CV_WRAP bool load(const String& filename)
        {
            cc = createCascadeClassifier(filename);
            return !cc.empty();
        }

        CV_WRAP bool empty() const
        {
            return cc.empty();
        }

        CV_WRAP void detectMultiScale( const Mat& image,
                                       CV_OUT std::vector<Rect>& objects,
                                       double scaleFactor = 1.1,
                                       int minNeighbors = 3, int flags = 0,
                                       Size minSize = Size(),
                                       Size maxSize = Size() )
        {
            cc->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize);
        }

        CV_WRAP void detectMultiScale( InputArray image,
                                       CV_OUT std::vector<Rect>& objects,
                                       CV_OUT std::vector<int>& numDetections,
                                       double scaleFactor=1.1,
                                       int minNeighbors=3, int flags=0,
                                       Size minSize=Size(),
                                       Size maxSize=Size() )
        {
            cc->detectMultiScale(image, objects, numDetections,
                    scaleFactor, minNeighbors, flags, minSize, maxSize);
        }

        CV_WRAP void detectMultiScale( InputArray image,
                                       CV_OUT std::vector<Rect>& objects,
                                       CV_OUT std::vector<int>& rejectLevels,
                                       CV_OUT std::vector<double>& levelWeights,
                                       double scaleFactor = 1.1,
                                       int minNeighbors = 3, int flags = 0,
                                       Size minSize = Size(),
                                       Size maxSize = Size(),
                                       bool outputRejectLevels = false )
        {
            cc->detectMultiScale(image, objects, rejectLevels, levelWeights,
                                 scaleFactor, minNeighbors, flags,
                                 minSize, maxSize, outputRejectLevels);
        }

    private:
        Ptr<CascadeClassifier> cc;
    };

}
#endif // HAVE_OPENCV_OBJDETECT

#endif // __OPENCV_OBJDETECT_MANUAL_HPP__
