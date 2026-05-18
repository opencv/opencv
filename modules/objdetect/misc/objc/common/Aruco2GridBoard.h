//
//  Aruco2GridBoard.h
//
//  Hand-written wrapper for cv::aruco2::GridBoard.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/objdetect/aruco2.hpp>
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import "Aruco2FiducialMarker.h"
#import "Size2i.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Result of detecting a ChArUco2-style grid board.
 *
 * `markers` holds the detected ArUco markers.
 * `gridSize` is the board layout in columns × rows.
 * `dict` is the DictionaryType as an integer.
 */
CV_EXPORTS @interface Aruco2GridBoard : NSObject

/** Board dimensions: width × height in markers. */
@property Size2i* gridSize;

/** Dictionary used for all markers on the board (Aruco2DictionaryType as int). */
@property int dict;

/** Detected markers (subset of the full board when partially occluded). */
@property NSArray<Aruco2FiducialMarker*>* markers;

#ifdef __cplusplus
@property(readonly) cv::aruco2::GridBoard& nativeRef;
#endif

- (instancetype)init;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::aruco2::GridBoard&)board;
#endif

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
