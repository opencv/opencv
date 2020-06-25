//
//  Mat+UIImage.h
//
//  Created by Giles Payne on 2020/03/03.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface Mat (Converters)

-(UIImage*)toUIImage;
-(instancetype)initWithUIImage:(UIImage*)image;
-(instancetype)initWithUIImage:(UIImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
