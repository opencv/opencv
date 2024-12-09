//
//  Mat.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

#ifdef AVAILABLE_IMGCODECS
#if TARGET_OS_IPHONE || TARGET_OS_VISION
#import <UIKit/UIKit.h>
#elif TARGET_OS_MAC
#import <AppKit/AppKit.h>
#endif
#endif

@class Size2i;
@class Scalar;
@class Range;
@class Rect2i;
@class Point2i;

NS_ASSUME_NONNULL_BEGIN

/**
 The class Mat represents an n-dimensional dense numerical single-channel or multi-channel array.
 ####Swift Example
 ```swift
 let mat = Mat(rows: 2, cols: 3, type: CvType.CV_8U)
 try! mat.put(row: 0, col: 0, data: [2, 3, 4, 4, 5, 6] as [Int8])
 print("mat: \(mat.dump())")
 ```
 ####Objective-C Example
 ```objc
 Mat* mat = [[Mat alloc] initWithRows:2 cols:3 type: CV_8U];
 [m1 put:0 col:0 data:@[@2, @3, @4, @3, @4, @5]];
 NSLog(@"mat: %@", [m1 dump]);
 ```
*/
CV_EXPORTS @interface Mat : NSObject

#ifdef __cplusplus
@property(readonly) cv::Ptr<cv::Mat> nativePtr;
@property(readonly) cv::Mat& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Ptr<cv::Mat>)nativeMat;
+ (instancetype)fromNativePtr:(cv::Ptr<cv::Mat>)nativePtr;
+ (instancetype)fromNative:(cv::Mat&)nativeRef;
#endif
/**
 Creates a Mat object with the specified number of rows and columns and Mat type
 @param rows Number of rows
 @param cols Number of columns
 @param type Mat type (refer: `CvType`)
*/
- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type;
- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type data:(NSData*)data;
- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type data:(NSData*)data step:(long)step;
- (instancetype)initWithSize:(Size2i*)size type:(int)type;
- (instancetype)initWithSizes:(NSArray<NSNumber*>*)sizes type:(int)type;
- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type scalar:(Scalar*)scalar;
- (instancetype)initWithSize:(Size2i*)size type:(int)type scalar:(Scalar*)scalar;
- (instancetype)initWithSizes:(NSArray<NSNumber*>*)sizes type:(int)type scalar:(Scalar*)scalar;
- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange colRange:(Range*)colRange;
- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange;
- (instancetype)initWithMat:(Mat*)mat ranges:(NSArray<Range*>*)ranges;
- (instancetype)initWithMat:(Mat*)mat rect:(Rect2i*)roi;

#pragma mark - Mat operations

- (Mat*)adjustRoiTop:(int)dtop bottom:(int)dbottom left:(int)dleft right:(int)dright NS_SWIFT_NAME(adjustRoi(top:bottom:left:right:));
- (void)assignTo:(Mat*)mat type:(int)type;
- (void)assignTo:(Mat*)mat;
- (BOOL)isSameMat:(Mat*)mat;
- (int)channels;
- (int)checkVector:(int)elemChannels depth:(int)depth requireContinuous:(BOOL) requireContinuous NS_SWIFT_NAME(checkVector(elemChannels:depth:requireContinuous:));
- (int)checkVector:(int)elemChannels depth:(int)depth NS_SWIFT_NAME(checkVector(elemChannels:depth:));
- (int)checkVector:(int)elemChannels NS_SWIFT_NAME(checkVector(elemChannels:));
- (Mat*)clone;
- (Mat*)col:(int)x;
- (Mat*)colRange:(int)start end:(int)end NS_SWIFT_NAME(colRange(start:end:));
- (Mat*)colRange:(Range*)range;
- (int)dims;
- (int)cols;
- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha beta:(double)beta;
- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha;
- (void)convertTo:(Mat*)mat rtype:(int)rtype;
- (void)copyTo:(Mat*)mat;
- (void)copyTo:(Mat*)mat mask:(Mat*)mask;
- (void)create:(int)rows cols:(int)cols type:(int)type NS_SWIFT_NAME(create(rows:cols:type:));
- (void)create:(Size2i*)size type:(int)type NS_SWIFT_NAME(create(size:type:));
- (void)createEx:(NSArray<NSNumber*>*)sizes type:(int)type  NS_SWIFT_NAME(create(sizes:type:));
- (void)copySize:(Mat*)mat;
- (Mat*)cross:(Mat*)mat;
- (unsigned char*)dataPtr NS_SWIFT_NAME(dataPointer());
- (int)depth;
- (Mat*)diag:(int)diagonal;
- (Mat*)diag;
+ (Mat*)diag:(Mat*)diagonal;
- (double)dot:(Mat*)mat;
- (long)elemSize;
- (long)elemSize1;
- (BOOL)empty;
+ (Mat*)eye:(int)rows cols:(int)cols type:(int)type NS_SWIFT_NAME(eye(rows:cols:type:));
+ (Mat*)eye:(Size2i*)size type:(int)type NS_SWIFT_NAME(eye(size:type:));
- (Mat*)inv:(int)method;
- (Mat*)inv;
- (BOOL)isContinuous;
- (BOOL)isSubmatrix;
- (void)locateROI:(Size2i*)wholeSize ofs:(Point2i*)offset NS_SWIFT_NAME(locateROI(wholeSize:offset:));
- (Mat*)mul:(Mat*)mat scale:(double)scale;
/**
 Performs element-wise multiplication
 @param mat operand with with which to perform element-wise multiplication
*/
- (Mat*)mul:(Mat*)mat;
/**
 Performs matrix multiplication
 @param mat operand with with which to perform matrix multiplication
 @see `Core.gemm(...)`
*/
- (Mat*)matMul:(Mat*)mat;
+ (Mat*)ones:(int)rows cols:(int)cols type:(int)type NS_SWIFT_NAME(ones(rows:cols:type:));
+ (Mat*)ones:(Size2i*)size type:(int)type NS_SWIFT_NAME(ones(size:type:));
+ (Mat*)onesEx:(NSArray<NSNumber*>*)sizes type:(int)type NS_SWIFT_NAME(ones(sizes:type:));
- (void)push_back:(Mat*)mat;
- (Mat*)reshape:(int)channels rows:(int)rows NS_SWIFT_NAME(reshape(channels:rows:));
- (Mat*)reshape:(int)channels NS_SWIFT_NAME(reshape(channels:));
- (Mat*)reshape:(int)channels newshape:(NSArray<NSNumber*>*)newshape NS_SWIFT_NAME(reshape(channels:newshape:));
- (Mat*)row:(int)y;
- (Mat*)rowRange:(int)start end:(int)end NS_SWIFT_NAME(rowRange(start:end:));
- (Mat*)rowRange:(Range*)range;
- (int)rows;
- (Mat*)setToScalar:(Scalar*)scalar NS_SWIFT_NAME(setTo(scalar:));
- (Mat*)setToScalar:(Scalar*)scalar mask:(Mat*)mask NS_SWIFT_NAME(setTo(scalar:mask:));
- (Mat*)setToValue:(Mat*)value mask:(Mat*)mask NS_SWIFT_NAME(setTo(value:mask:));
- (Mat*)setToValue:(Mat*)value NS_SWIFT_NAME(setTo(value:));
- (Size2i*)size;
- (int)size:(int)dim;
- (long)step1:(int)dim;
- (long)step1;
- (Mat*)submat:(int)rowStart rowEnd:(int)rowEnd colStart:(int)colStart colEnd:(int)colEnd NS_SWIFT_NAME(submat(rowStart:rowEnd:colStart:colEnd:));
- (Mat*)submat:(Range*)rowRange colRange:(Range*)colRange NS_SWIFT_NAME(submat(rowRange:colRange:));
- (Mat*)submat:(NSArray<Range*>*)ranges NS_SWIFT_NAME(submat(ranges:));
- (Mat*)submatRoi:(Rect2i*)roi NS_SWIFT_NAME(submat(roi:));
- (Mat*)t;
- (long)total;
- (int)type;
+ (Mat*)zeros:(int)rows cols:(int)cols type:(int)type;
+ (Mat*)zeros:(Size2i*)size type:(int)type;
+ (Mat*)zerosEx:(NSArray<NSNumber*>*)sizes type:(int)type NS_SWIFT_NAME(zeros(sizes:type:));
- (NSString*)description;
- (NSString*)dump;
- (int)height;
- (int)width;

#pragma mark - Accessors

- (int)put:(int)row col:(int)col data:(NSArray<NSNumber*>*)data NS_REFINED_FOR_SWIFT;
- (int)put:(NSArray<NSNumber*>*)indices data:(NSArray<NSNumber*>*)data NS_REFINED_FOR_SWIFT;
- (int)get:(int)row col:(int)col data:(NSMutableArray<NSNumber*>*)data NS_REFINED_FOR_SWIFT;
- (int)get:(NSArray<NSNumber*>*)indices data:(NSMutableArray<NSNumber*>*)data NS_REFINED_FOR_SWIFT;

- (NSArray<NSNumber*>*)get:(int)row col:(int)col NS_REFINED_FOR_SWIFT;
- (NSArray<NSNumber*>*)get:(NSArray<NSNumber*>*)indices NS_REFINED_FOR_SWIFT;

- (int)get:(NSArray<NSNumber*>*)indices count:(int)count byteBuffer:(char*)buffer NS_REFINED_FOR_SWIFT;
- (int)get:(NSArray<NSNumber*>*)indices count:(int)count doubleBuffer:(double*)buffer NS_REFINED_FOR_SWIFT;
- (int)get:(NSArray<NSNumber*>*)indices count:(int)count floatBuffer:(float*)buffer NS_REFINED_FOR_SWIFT;
- (int)get:(NSArray<NSNumber*>*)indices count:(int)count intBuffer:(int*)buffer NS_REFINED_FOR_SWIFT;
- (int)get:(NSArray<NSNumber*>*)indices count:(int)count shortBuffer:(short*)buffer NS_REFINED_FOR_SWIFT;

- (int)put:(NSArray<NSNumber*>*)indices count:(int)count byteBuffer:(const char*)buffer NS_REFINED_FOR_SWIFT;
- (int)put:(NSArray<NSNumber*>*)indices count:(int)count doubleBuffer:(const double*)buffer NS_REFINED_FOR_SWIFT;
- (int)put:(NSArray<NSNumber*>*)indices count:(int)count floatBuffer:(const float*)buffer NS_REFINED_FOR_SWIFT;
- (int)put:(NSArray<NSNumber*>*)indices count:(int)count intBuffer:(const int*)buffer NS_REFINED_FOR_SWIFT;
- (int)put:(NSArray<NSNumber*>*)indices count:(int)count shortBuffer:(const short*)buffer NS_REFINED_FOR_SWIFT;

#pragma mark - Converters

#ifdef AVAILABLE_IMGCODECS

- (CGImageRef)toCGImage CF_RETURNS_RETAINED;
- (instancetype)initWithCGImage:(CGImageRef)image;
- (instancetype)initWithCGImage:(CGImageRef)image alphaExist:(BOOL)alphaExist;

#if TARGET_OS_IPHONE || TARGET_OS_VISION

- (UIImage*)toUIImage;
- (instancetype)initWithUIImage:(UIImage*)image;
- (instancetype)initWithUIImage:(UIImage*)image alphaExist:(BOOL)alphaExist;

#elif TARGET_OS_MAC

- (NSImage*)toNSImage;
- (instancetype)initWithNSImage:(NSImage*)image;
- (instancetype)initWithNSImage:(NSImage*)image alphaExist:(BOOL)alphaExist;

#endif

#endif

#pragma mark - QuickLook

#ifdef AVAILABLE_IMGCODECS

- (id)debugQuickLookObject;

#endif

@end

NS_ASSUME_NONNULL_END
