//
//  Aruco2DetectionParameters.h
//
//  Hand-written wrapper for cv::aruco2::DetectionParameters.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/objdetect/aruco2.hpp>
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Detection parameters for aruco2 marker detection.
 *
 * All parameters have defaults that work well for standard printed markers under normal
 * lighting.  Tune only when detection fails or produces false positives.
 */
CV_EXPORTS @interface Aruco2DetectionParameters : NSObject

/** Size of the box filter kernel used for adaptive thresholding (pixels, must be odd). Default: 15. */
@property int boxFilterSize;

/** Threshold offset applied after the box filter subtraction. Default: 3. */
@property int thres;

/** Minimum side length (pixels) for a contour to be considered a marker candidate. Default: 10. */
@property int minSize;

/** Number of attempts to identify a candidate by slightly perturbing its corners. Default: 5. */
@property int maxAttemptsPerCandidate;

/** Controls how aggressively the contour tracer prunes revisited paths [0,1]. Default: 0.05. */
@property float maxTimesRevisited;

/** Width of the mandatory black border around each marker, in bits. Default: 1. */
@property int markerBorderBits;

/** Fraction of maxCorrectionBits to use when matching a candidate. Default: 0. */
@property double errorCorrectionRate;

/** Maximum fraction of border bits allowed to be wrong before rejecting a candidate. Default: 0. */
@property double maxErroneousBitsInBorderRate;

/** Set to true to detect markers printed white-on-black (inverted polarity). Default: false. */
@property BOOL detectInvertedMarker;

#ifdef __cplusplus
@property(readonly) cv::aruco2::DetectionParameters& nativeRef;
#endif

- (instancetype)init;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::aruco2::DetectionParameters&)params;
#endif

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
