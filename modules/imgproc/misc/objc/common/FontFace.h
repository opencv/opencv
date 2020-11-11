//
//  FontFace.h
//
//  Created by VP in 2020
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface FontFace : NSObject

@property(readonly) NSString* name;

#ifdef __cplusplus
@property(readonly) cv::FontFace& nativeRef;
#endif

-(instancetype)initWith:(const NSString*)name;
-(instancetype)init;

#ifdef __cplusplus
+(instancetype)fromNative:(cv::FontFace&)fface;
#endif

-(NSString *)description;

@end

NS_ASSUME_NONNULL_END
