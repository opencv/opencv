//
//  DetectionBasedTracker.mm
//
//  Created by Giles Payne on 2020/04/05.
//

#import "DetectionBasedTracker.h"
#import "Mat.h"
#import "Rect2i.h"
#import "CVObjcUtil.h"

class CascadeDetectorAdapter: public cv::DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):IDetector(), Detector(detector) {}

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }

    virtual ~CascadeDetectorAdapter() {}

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};


struct DetectorAgregator
{
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;
    cv::Ptr<cv::DetectionBasedTracker> tracker;
    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter>& _mainDetector, cv::Ptr<CascadeDetectorAdapter>& _trackingDetector):mainDetector(_mainDetector), trackingDetector(_trackingDetector) {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);
        cv::DetectionBasedTracker::Parameters DetectorParams;
        tracker = cv::makePtr<cv::DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

@implementation DetectionBasedTracker {
    DetectorAgregator* agregator;
}

- (instancetype)initWithCascadeName:(NSString*)cascadeName minFaceSize:(int)faceSize {
    self = [super init];
    if (self) {
        auto mainDetector = cv::makePtr<CascadeDetectorAdapter>(cv::makePtr<cv::CascadeClassifier>(cascadeName.UTF8String));
        auto trackingDetector = cv::makePtr<CascadeDetectorAdapter>(
            cv::makePtr<cv::CascadeClassifier>(cascadeName.UTF8String));
        agregator = new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0) {
            agregator->mainDetector->setMinObjectSize(cv::Size(faceSize, faceSize));
        }
    }
    return self;
}

- (void)dealloc
{
    delete agregator;
}

- (void)start {
    agregator->tracker->run();
}

- (void)stop {
    agregator->tracker->stop();
}

- (void)setFaceSize:(int)size {
    agregator->mainDetector->setMinObjectSize(cv::Size(size, size));
}

- (void)detect:(Mat*)imageGray faces:(NSMutableArray<Rect2i*>*)faces {
    std::vector<cv::Rect> rectFaces;
    agregator->tracker->process(*((cv::Mat*)imageGray.nativePtr));
    agregator->tracker->getObjects(rectFaces);
    CV2OBJC(cv::Rect, Rect2i, rectFaces, faces);
}

@end
