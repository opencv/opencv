//
//  Aruco2FractalMarker.h
//
//  Hand-written wrapper for cv::aruco2::FractalMarker.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/objdetect/aruco2.hpp>
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import "Point2f.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A detected fractal marker.
 *
 * Fractal markers are nested ArUco-like markers.
 * `corners` holds the 4 outer corners, clockwise from top-left.
 * `type` is the fractal configuration (Aruco2FractalType as int).
 * `id` is the id of the outer (external) marker.
 */
CV_EXPORTS @interface Aruco2FractalMarker : NSObject

/** 4 outer corners, clockwise from top-left. */
@property NSMutableArray<Point2f*>* corners;

/** Fractal configuration used for detection (Aruco2FractalType as int). */
@property int type;

/** Id of the outer (external) marker; -1 if unidentified. */
@property int id;

#ifdef __cplusplus
@property(readonly) cv::aruco2::FractalMarker& nativeRef;
#endif

- (instancetype)init;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::aruco2::FractalMarker&)fractal;
#endif

/** Returns the i-th corner point. */
- (Point2f*)getCorner:(int)i NS_SWIFT_NAME(getCorner(_:));

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
