//
//  Aruco2Diamond.h
//
//  Hand-written wrapper for cv::aruco2::Diamond.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/objdetect/aruco2.hpp>
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import "Aruco2FiducialMarker.h"
#import "Int4.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A detected ChArUco2-style diamond marker.
 *
 * A diamond is a 2×2 block of ArUco markers.
 * `id` holds the four constituent marker ids (clockwise from top-left).
 * `markers` holds the four detected FiducialMarker objects.
 */
CV_EXPORTS @interface Aruco2Diamond : NSObject

/** Ids of the 4 constituent markers (clockwise from top-left). */
@property Int4* id;

/** Dictionary used for the 4 markers (Aruco2DictionaryType as int). */
@property int dict;

/** The 4 detected markers forming the diamond. */
@property NSArray<Aruco2FiducialMarker*>* markers;

#ifdef __cplusplus
@property(readonly) cv::aruco2::Diamond& nativeRef;
#endif

- (instancetype)init;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::aruco2::Diamond&)diamond;
#endif

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
