//
//  CVType.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/13.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef CVType_h
#define CVType_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

#import <Foundation/Foundation.h>

@interface CVType : NSObject

//@property (class, nonatomic, assign, readonly) int CV_8U;
//@property (class, nonatomic, assign, readonly) int CV_8S;
//@property (class, nonatomic, assign, readonly) int CV_16U;
//@property (class, nonatomic, assign, readonly) int CV_16S;
//@property (class, nonatomic, assign, readonly) int CV_32S;
//@property (class, nonatomic, assign, readonly) int CV_32F;
//@property (class, nonatomic, assign, readonly) int CV_64F;
//@property (class, nonatomic, assign, readonly) int CV_16F;

//@property (class, nonatomic, assign, readonly) int CV_8UC1;
//@property (class, nonatomic, assign, readonly) int CV_8UC2;
//@property (class, nonatomic, assign, readonly) int CV_8UC3;
//@property (class, nonatomic, assign, readonly) int CV_8UC4;
//@property (class, nonatomic, assign, readonly) int CV_8SC1;
//@property (class, nonatomic, assign, readonly) int CV_8SC2;
//@property (class, nonatomic, assign, readonly) int CV_8SC3;
//@property (class, nonatomic, assign, readonly) int CV_8SC4;

//@property (class, nonatomic, assign, readonly) int CV_16UC1;
//@property (class, nonatomic, assign, readonly) int CV_16UC2;
//@property (class, nonatomic, assign, readonly) int CV_16UC3;
//@property (class, nonatomic, assign, readonly) int CV_16UC4;
//@property (class, nonatomic, assign, readonly) int CV_16SC1;
//@property (class, nonatomic, assign, readonly) int CV_16SC2;
//@property (class, nonatomic, assign, readonly) int CV_16SC3;
//@property (class, nonatomic, assign, readonly) int CV_16SC4;

//@property (class, nonatomic, assign, readonly) int CV_32SC1;
//@property (class, nonatomic, assign, readonly) int CV_32SC2;
//@property (class, nonatomic, assign, readonly) int CV_32SC3;
//@property (class, nonatomic, assign, readonly) int CV_32SC4;
//@property (class, nonatomic, assign, readonly) int CV_32FC1;
//@property (class, nonatomic, assign, readonly) int CV_32FC2;
//@property (class, nonatomic, assign, readonly) int CV_32FC3;
//@property (class, nonatomic, assign, readonly) int CV_32FC4;

//@property (class, nonatomic, assign, readonly) int CV_64FC1;
//@property (class, nonatomic, assign, readonly) int CV_64FC2;
//@property (class, nonatomic, assign, readonly) int CV_64FC3;
//@property (class, nonatomic, assign, readonly) int CV_64FC4;

//@property (class, nonatomic, assign, readonly) int CV_16FC1;
//@property (class, nonatomic, assign, readonly) int CV_16FC2;
//@property (class, nonatomic, assign, readonly) int CV_16FC3;
//@property (class, nonatomic, assign, readonly) int CV_16FC4;

//@property (class, nonatomic, assign, readonly) int CV_CN_MAX;
//@property (class, nonatomic, assign, readonly) int CV_CN_SHIFT;
//@property (class, nonatomic, assign, readonly) int CV_DEPTH_MAX;

//+ (int)CV_8UC:(int)channels;
//+ (int)CV_8SC:(int)channels;
//+ (int)CV_16UC:(int)channels;
//+ (int)CV_16SC:(int)channels;
//+ (int)CV_32SC:(int)channels;
//+ (int)CV_32FC:(int)channels;
//+ (int)CV_64FC:(int)channels;
//+ (int)CV_16FC:(int)channels;

+ (int)makeType:(int)depth channels:(int)channels;
+ (int)channels:(int)type;
+ (int)depth:(int)type;
+ (int)rawTypeSize:(int)type;
+ (BOOL)isInteger:(int)type;
+ (int)ELEM_SIZE:(int)type;
+ (NSString*)typeToString:(int)type;

@end

#endif /* CVType_h */
