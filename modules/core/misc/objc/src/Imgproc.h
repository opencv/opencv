//
//  ImgProc.h
//
//  Created by Giles Payne on 2020/01/28.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#endif

@class Mat;
@class CVPoint;
@class Scalar;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// C++: enum LineTypes
typedef NS_ENUM(int, LineTypes) {
    FILLED = -1,
    LINE_4 = 4,
    LINE_8 = 8,
    LINE_AA = 16
};

@interface Imgproc : NSObject

+ (void)fillConvexPoly:(Mat*)img pointArray:(NSArray<CVPoint*>*)pointArray color:(Scalar*)color lineType:(int)lineType shift:(int)shift;

+ (void)fillConvexPoly:(Mat*)img pointArray:(NSArray<CVPoint*>*)pointArray color:(Scalar*)color;

+ (void)fillPoly:(Mat*)img pointArrayArray:(NSArray<NSArray<CVPoint*>*>*)pointArrayArray color:(Scalar*)color;

+ (void)fillPoly:(Mat*)img pointArrayArray:(NSArray<NSArray<CVPoint*>*>*)pointArrayArray color:(Scalar*)color lineType:(int)lineType shift:(int)shift offset:(CVPoint*)offset;

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness lineType:(int)lineType shift:(int)shift;

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness lineType:(int)lineType;

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness;

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar;

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift;

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType;

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness;

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color;

@end

NS_ASSUME_NONNULL_END
