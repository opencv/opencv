//
//  ByteVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

NS_ASSUME_NONNULL_BEGIN
@interface ByteVector : NSObject

-(instancetype)initWithData:(NSData*)data;
-(instancetype)initWithVector:(ByteVector*)src;

@property(readonly) NSInteger length;
#ifdef __OBJC__
@property(readonly) SInt8* nativeArray;
-(instancetype)initWithNativeArray:(SInt8*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
@property(readonly) std::vector<SInt8>& vector;
-(instancetype)initWithStdVector:(std::vector<SInt8>&)src;
+(instancetype)fromNative:(std::vector<SInt8>&)src;
#endif

-(SInt8)get:(NSInteger)index;
@property(readonly) NSData* data;

@end
NS_ASSUME_NONNULL_END
