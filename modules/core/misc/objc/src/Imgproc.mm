//
//  ImgProc.mm
//
//  Created by Giles Payne on 2020/01/28.
//

#import "Imgproc.h"
#import "Mat.h"
#import "CVPoint.h"
#import "Scalar.h"
#import "CVObjcUtil.h"

@implementation Imgproc

+ (void)fillConvexPoly:(Mat*)img pointArray:(NSArray<CVPoint*>*)pointArray color:(Scalar*)color lineType:(int)lineType shift:(int)shift {
    OBJC2CV_C(cv::Point, cv::Point2d, CVPoint, pointVector, pointArray);
    cv::fillConvexPoly(img.nativeRef, pointVector, color.nativeRef, lineType, shift);
}

+ (void)fillConvexPoly:(Mat*)img pointArray:(NSArray<CVPoint*>*)pointArray color:(Scalar*)color {
    OBJC2CV_C(cv::Point, cv::Point2d, CVPoint, pointVector, pointArray);
    cv::fillConvexPoly(img.nativeRef, pointVector, color.nativeRef);
}

+ (void)fillPoly:(Mat*)img pointArrayArray:(NSArray<NSArray<CVPoint*>*>*)pointArrayArray color:(Scalar*)color lineType:(int)lineType shift:(int)shift offset:(CVPoint*)offset  {
    OBJC2CV2_C(cv::Point, cv::Point2d, CVPoint, pointVectorVector, pointArrayArray);
    cv::fillPoly(img.nativeRef, pointVectorVector, color.nativeRef, lineType, shift, offset.nativeRef);
}

+ (void)fillPoly:(Mat*)img pointArrayArray:(NSArray<NSArray<CVPoint*>*>*)pointArrayArray color:(Scalar*)color {
    OBJC2CV2_C(cv::Point, cv::Point2d, CVPoint, pointVectorVector, pointArrayArray);
    cv::fillPoly(img.nativeRef, pointVectorVector, color.nativeRef);
}

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::line(img.nativeRef, point1.nativeRef, point2.nativeRef, scalar.nativeRef, thickness, lineType, shift);
}

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness lineType:(int)lineType {
    cv::line(img.nativeRef, point1.nativeRef, point2.nativeRef, scalar.nativeRef, thickness, lineType);
}

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar thickness:(int)thickness {
    cv::line(img.nativeRef, point1.nativeRef, point2.nativeRef, scalar.nativeRef, thickness);
}

+ (void)line:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 scalar:(Scalar*)scalar {
    cv::line(img.nativeRef, point1.nativeRef, point2.nativeRef, scalar.nativeRef);
}

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType shift:(int)shift {
    cv::rectangle(img.nativeRef, point1.nativeRef, point2.nativeRef, color.nativeRef, thickness, lineType, shift);
}

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness lineType:(int)lineType {
    cv::rectangle(img.nativeRef, point1.nativeRef, point2.nativeRef, color.nativeRef, thickness, lineType);
}

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color thickness:(int)thickness {
    cv::rectangle(img.nativeRef, point1.nativeRef, point2.nativeRef, color.nativeRef, thickness);
}

+ (void)rectangle:(Mat*)img point1:(CVPoint*)point1 point2:(CVPoint*)point2 color:(Scalar*)color {
    cv::rectangle(img.nativeRef, point1.nativeRef, point2.nativeRef, color.nativeRef);
}

@end
