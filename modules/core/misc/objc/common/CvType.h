//
//  CvType.h
//
//  Created by Giles Payne on 2019/10/13.
//

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Utility functions for handling CvType values
*/
CV_EXPORTS @interface CvType : NSObject

#pragma mark - Type Utility functions

/**
* Create CvType value from depth and channel values
* @param depth Depth value. One of CV_8U, CV_8S, CV_16U, CV_16S,  CV_32S, CV_32F or CV_64F
* @param channels Number of channels (from 1 to  (CV_CN_MAX - 1))
*/
+ (int)makeType:(int)depth channels:(int)channels;

/**
* Get number of channels for type
* @param type  Type value
*/
+ (int)channels:(int)type;

/**
* Get depth for type
* @param type  Type value
*/
+ (int)depth:(int)type;

/**
* Get raw type size in bytes for type
* @param type  Type value
*/
+ (int)rawTypeSize:(int)type;

/**
* Returns true if the raw type is an integer type (if depth is CV_8U, CV_8S, CV_16U, CV_16S or CV_32S)
* @param type  Type value
*/
+ (BOOL)isInteger:(int)type;

/**
* Get element size in bytes for type
* @param type  Type value
*/
+ (int)ELEM_SIZE:(int)type NS_SWIFT_NAME(elemSize(_:));

/**
* Get the string name for type
* @param type  Type value
*/
+ (NSString*)typeToString:(int)type;

@end

NS_ASSUME_NONNULL_END
