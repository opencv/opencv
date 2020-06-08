//
//  Converters.h
//
//  Created by Giles Payne on 2020/03/03.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>
#import "Mat.h"
#import "CvType.h"
#import "Point2i.h"
#import "Point2f.h"
#import "Point2d.h"
#import "Point3i.h"
#import "Point3f.h"
#import "Point3d.h"
#import "Rect2i.h"
#import "Rect2d.h"
#import "KeyPoint.h"
#import "DMatch.h"
#import "RotatedRect.h"

NS_ASSUME_NONNULL_BEGIN

@interface Converters : NSObject

+ (Mat*)vector_Point_to_Mat:(NSArray<Point2i*>*)pts NS_SWIFT_NAME(vector_Point_to_Mat(_:));

+ (NSArray<Point2i*>*)Mat_to_vector_Point:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point(_:));

+ (Mat*)vector_Point2f_to_Mat:(NSArray<Point2f*>*)pts NS_SWIFT_NAME(vector_Point2f_to_Mat(_:));

+ (NSArray<Point2f*>*)Mat_to_vector_Point2f:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point2f(_:));

+ (Mat*)vector_Point2d_to_Mat:(NSArray<Point2d*>*)pts NS_SWIFT_NAME(vector_Point2d_to_Mat(_:));

+ (NSArray<Point2f*>*)Mat_to_vector_Point2d:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point2d(_:));

+ (Mat*)vector_Point3i_to_Mat:(NSArray<Point3i*>*)pts NS_SWIFT_NAME(vector_Point3i_to_Mat(_:));

+ (NSArray<Point3i*>*)Mat_to_vector_Point3i:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point3i(_:));

+ (Mat*)vector_Point3f_to_Mat:(NSArray<Point3f*>*)pts NS_SWIFT_NAME(vector_Point3f_to_Mat(_:));

+ (NSArray<Point3f*>*)Mat_to_vector_Point3f:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point3f(_:));

+ (Mat*)vector_Point3d_to_Mat:(NSArray<Point3d*>*)pts NS_SWIFT_NAME(vector_Point3d_to_Mat(_:));

+ (NSArray<Point3d*>*)Mat_to_vector_Point3d:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Point3d(_:));

+ (Mat*)vector_float_to_Mat:(NSArray<NSNumber*>*)fs NS_SWIFT_NAME(vector_float_to_Mat(_:));

+ (NSArray<NSNumber*>*)Mat_to_vector_float:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_float(_:));

+ (Mat*)vector_uchar_to_Mat:(NSArray<NSNumber*>*)us NS_SWIFT_NAME(vector_uchar_to_Mat(_:));

+ (NSArray<NSNumber*>*)Mat_to_vector_uchar:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_uchar(_:));

+ (Mat*)vector_char_to_Mat:(NSArray<NSNumber*>*)cs NS_SWIFT_NAME(vector_char_to_Mat(_:));

+ (NSArray<NSNumber*>*)Mat_to_vector_char:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_char(_:));

+ (Mat*)vector_int_to_Mat:(NSArray<NSNumber*>*)is NS_SWIFT_NAME(vector_int_to_Mat(_:));

+ (NSArray<NSNumber*>*)Mat_to_vector_int:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_int(_:));

+ (Mat*)vector_Rect_to_Mat:(NSArray<Rect2i*>*)rs NS_SWIFT_NAME(vector_Rect_to_Mat(_:));

+ (NSArray<Rect2i*>*)Mat_to_vector_Rect:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Rect(_:));

+ (Mat*)vector_Rect2d_to_Mat:(NSArray<Rect2d*>*)rs NS_SWIFT_NAME(vector_Rect2d_to_Mat(_:));

+ (NSArray<Rect2d*>*)Mat_to_vector_Rect2d:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_Rect2d(_:));

+ (Mat*)vector_KeyPoint_to_Mat:(NSArray<KeyPoint*>*)kps NS_SWIFT_NAME(vector_KeyPoint_to_Mat(_:));

+ (NSArray<KeyPoint*>*)Mat_to_vector_KeyPoint:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_KeyPoint(_:));

+ (Mat*)vector_double_to_Mat:(NSArray<NSNumber*>*)ds NS_SWIFT_NAME(vector_double_to_Mat(_:));

+ (NSArray<NSNumber*>*)Mat_to_vector_double:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_double(_:));

+ (Mat*)vector_DMatch_to_Mat:(NSArray<DMatch*>*)matches NS_SWIFT_NAME(vector_DMatch_to_Mat(_:));

+ (NSArray<DMatch*>*)Mat_to_vector_DMatch:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_DMatch(_:));

+ (Mat*)vector_RotatedRect_to_Mat:(NSArray<RotatedRect*>*)rs NS_SWIFT_NAME(vector_RotatedRect_to_Mat(_:));

+ (NSArray<RotatedRect*>*)Mat_to_vector_RotatedRect:(Mat*)mat NS_SWIFT_NAME(Mat_to_vector_RotatedRect(_:));

@end

NS_ASSUME_NONNULL_END
