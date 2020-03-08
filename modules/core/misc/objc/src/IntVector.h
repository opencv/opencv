//
//  IntVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

NS_ASSUME_NONNULL_BEGIN
@interface IntVector : NSObject

-(instancetype)initWithData:(NSData*)data;
-(instancetype)initWithVector:(IntVector*)src;

@property(readonly) NSInteger length;
#ifdef __OBJC__
@property(readonly) int* nativeArray;
-(instancetype)initWithNativeArray:(int*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
@property(readonly) std::vector<int>& nativeRef;
-(instancetype)initWithStdVector:(std::vector<int>&)src;
+(instancetype)fromNative:(std::vector<int>&)src;
#endif

-(int)get:(NSInteger)index;
@property(readonly) NSData* data;

@end
NS_ASSUME_NONNULL_END
