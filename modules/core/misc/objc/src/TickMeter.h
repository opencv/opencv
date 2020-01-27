//
// This file is auto-generated. Please don't modify it!
//
#pragma once

#ifdef __cplusplus
#import "opencv2/opencv.hpp"
#import "opencv2/opencv_modules.hpp"
#endif

#import <Foundation/Foundation.h>





NS_ASSUME_NONNULL_BEGIN

@interface TickMeter : NSObject

#ifdef __cplusplus
@property(readonly) cv::TickMeter* nativePtr;
@property(readonly) cv::TickMeter& nativeRef;
#endif

- (void)dealloc;

#ifdef __cplusplus
- (instancetype)initWithNativePtr:(cv::TickMeter*)nativePtr;
+ (instancetype)fromNative:(cv::TickMeter*)nativePtr;
#endif


//
//   cv::TickMeter::TickMeter()
//
- (instancetype)init;


//
//  double cv::TickMeter::getTimeMicro()
//
/**
 * returns passed time in microseconds.
 */
- (double)getTimeMicro;


//
//  double cv::TickMeter::getTimeMilli()
//
/**
 * returns passed time in milliseconds.
 */
- (double)getTimeMilli;


//
//  double cv::TickMeter::getTimeSec()
//
/**
 * returns passed time in seconds.
 */
- (double)getTimeSec;


//
//  int64 cv::TickMeter::getCounter()
//
/**
 * returns internal counter value.
 */
- (long)getCounter;


//
//  int64 cv::TickMeter::getTimeTicks()
//
/**
 * returns counted ticks.
 */
- (long)getTimeTicks;


//
//  void cv::TickMeter::reset()
//
/**
 * resets internal values.
 */
- (void)reset;


//
//  void cv::TickMeter::start()
//
/**
 * starts counting ticks.
 */
- (void)start;


//
//  void cv::TickMeter::stop()
//
/**
 * stops counting ticks.
 */
- (void)stop;



@end

NS_ASSUME_NONNULL_END



