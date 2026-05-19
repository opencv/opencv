//
//  Aruco2FiducialMarker.h
//
//  Hand-written wrapper for cv::aruco2::FiducialMarker.
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
 * A single detected ArUco fiducial marker.
 *
 * `corners` holds the four image-plane corner points in clockwise order.
 * `id` is the marker identifier within its dictionary.
 * `dict` is the DictionaryType as an integer (see Aruco2 enum constants).
 */
CV_EXPORTS @interface Aruco2FiducialMarker : NSObject

/** Four corner points in clockwise order from top-left. */
@property NSMutableArray<Point2f*>* corners;

/** Marker id; -1 if unidentified. */
@property int id;

/** Dictionary this marker belongs to (Aruco2DictionaryType as int). */
@property int dictionary;

#ifdef __cplusplus
@property(readonly) cv::aruco2::FiducialMarker& nativeRef;
#endif

- (instancetype)init;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::aruco2::FiducialMarker&)marker;
#endif

/** Returns the i-th corner point. */
- (Point2f*)getCorner:(int)i NS_SWIFT_NAME(getCorner(_:));

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
