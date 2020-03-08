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
@property(readonly) char* nativeArray;
-(instancetype)initWithNativeArray:(char*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
@property(readonly) std::vector<char>& nativeRef;
-(instancetype)initWithStdVector:(std::vector<char>&)src;
+(instancetype)fromNative:(std::vector<char>&)src;
#endif

-(char)get:(NSInteger)index;
@property(readonly) NSData* data;

@end
NS_ASSUME_NONNULL_END
