//
//  Aruco2DetectionParameters.mm
//

#import "Aruco2DetectionParameters.h"

@implementation Aruco2DetectionParameters {
    cv::aruco2::DetectionParameters native;
}

- (int)boxFilterSize { return native.boxFilterSize; }
- (void)setBoxFilterSize:(int)v { native.boxFilterSize = v; }

- (int)thres { return native.thres; }
- (void)setThres:(int)v { native.thres = v; }

- (int)minSize { return native.minSize; }
- (void)setMinSize:(int)v { native.minSize = v; }

- (int)maxAttemptsPerCandidate { return native.maxAttemptsPerCandidate; }
- (void)setMaxAttemptsPerCandidate:(int)v { native.maxAttemptsPerCandidate = v; }

- (float)maxTimesRevisited { return native.maxTimesRevisited; }
- (void)setMaxTimesRevisited:(float)v { native.maxTimesRevisited = v; }

- (int)markerBorderBits { return native.markerBorderBits; }
- (void)setMarkerBorderBits:(int)v { native.markerBorderBits = v; }

- (double)errorCorrectionRate { return native.errorCorrectionRate; }
- (void)setErrorCorrectionRate:(double)v { native.errorCorrectionRate = v; }

- (double)maxErroneousBitsInBorderRate { return native.maxErroneousBitsInBorderRate; }
- (void)setMaxErroneousBitsInBorderRate:(double)v { native.maxErroneousBitsInBorderRate = v; }

- (BOOL)detectInvertedMarker { return native.detectInvertedMarker ? YES : NO; }
- (void)setDetectInvertedMarker:(BOOL)v { native.detectInvertedMarker = (bool)v; }

- (cv::aruco2::DetectionParameters&)nativeRef { return native; }

- (instancetype)init {
    self = [super init];
    if (self) { native = cv::aruco2::DetectionParameters(); }
    return self;
}

+ (instancetype)fromNative:(cv::aruco2::DetectionParameters&)params {
    Aruco2DetectionParameters* obj = [[Aruco2DetectionParameters alloc] init];
    obj->native = params;
    return obj;
}

- (NSString*)description {
    return [NSString stringWithFormat:
        @"Aruco2DetectionParameters { boxFilterSize:%d thres:%d minSize:%d errorCorrectionRate:%f }",
        native.boxFilterSize, native.thres, native.minSize, native.errorCorrectionRate];
}

@end
