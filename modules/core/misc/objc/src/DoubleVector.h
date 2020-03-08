//
//  DoubleVector.h
//  InteropTest
//
//  Created by Giles Payne on 2020/01/04.
//  Copyright © 2020 Xtravision. All rights reserved.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

#ifdef __cplusplus
template <typename T> std::vector<T*> arrayToVector(NSArray<T*>* _Nonnull array) {
    std::vector<T*> ret;
    for (T* t in array) {
        ret.push_back(t);
    }
    return ret;
}
#endif

NS_ASSUME_NONNULL_BEGIN
@interface DoubleVector : NSObject

-(instancetype)initWithData:(NSData*)data;
-(instancetype)initWithVector:(DoubleVector*)src;

@property(readonly) size_t length;
#ifdef __OBJC__
@property(readonly) double* nativeArray;
-(instancetype)initWithNativeArray:(double*)array elements:(int)elements;
#endif

#ifdef __cplusplus
@property(readonly) std::vector<double>& nativeRef;
-(instancetype)initWithStdVector:(std::vector<double>&)src;
+(instancetype)fromNative:(std::vector<double>&)src;
#endif

-(double)get:(NSInteger)index;
@property(readonly) NSData* data;

@end
NS_ASSUME_NONNULL_END
