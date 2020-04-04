//
//  FloatVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

NS_ASSUME_NONNULL_BEGIN
@interface FloatVector : NSObject

-(instancetype)initWithData:(NSData*)data;
-(instancetype)initWithVector:(FloatVector*)src;

@property(readonly) NSInteger length;
#ifdef __OBJC__
@property(readonly) float* nativeArray;
-(instancetype)initWithNativeArray:(float*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
@property(readonly) std::vector<float>& nativeRef;
-(instancetype)initWithStdVector:(std::vector<float>&)src;
+(instancetype)fromNative:(std::vector<float>&)src;
#endif

-(float)get:(NSInteger)index;
@property(readonly) NSData* data;

@end
NS_ASSUME_NONNULL_END
