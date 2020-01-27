//
//  Scalar.m
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "Scalar.h"

@implementation Scalar {
    cv::Scalar native;
}

#ifdef __cplusplus
- (cv::Scalar&)nativeRef {
    native.val[0] = self.val[0].doubleValue;
    native.val[1] = self.val[1].doubleValue;
    native.val[2] = self.val[2].doubleValue;
    native.val[3] = self.val[3].doubleValue;
    return native;
}
#endif

- (instancetype)initWithVals:(NSArray<NSNumber*> *)vals {
    self = [super init];
    if (self != nil) {
        self.val = [NSMutableArray arrayWithArray:vals];
        while (self.val.count < 4) {
            [self.val addObject:@0];
        }
    }
    return self;
}

- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3 {
    self = [super init];
    if (self != nil) {
        self.val =  [NSMutableArray arrayWithObjects:[NSNumber numberWithDouble:v0], [NSNumber numberWithDouble:v1], [NSNumber numberWithDouble:v2], [NSNumber numberWithDouble:v3], nil];
    }
    return self;
}

- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 {
    return [self initWithV0:v0 v1:v1 v2:v2 v3:0];
}

- (instancetype)initWithV0:(double)v0 v1:(double)v1 {
    return [self initWithV0:v0 v1:v1 v2:0 v3:0];
}

- (instancetype)initWithV0:(double)v0 {
    return [self initWithV0:v0 v1:0 v2:0 v3:0];
}

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Scalar&)nativeScalar {
    return [[Scalar alloc] initWithV0:nativeScalar.val[0] v1:nativeScalar.val[1] v2:nativeScalar.val[2] v3:nativeScalar.val[3]];
}
#endif

- (void)set:(NSArray<NSNumber*> *)vals {
    if (vals != nil) {
        self.val[0] = [vals count] > 0 ? vals[0] : @0;
        self.val[1] = [vals count] > 1 ? vals[1] : @0;
        self.val[2] = [vals count] > 2 ? vals[2] : @0;
        self.val[3] = [vals count] > 3 ? vals[3] : @0;
    } else {
        self.val[0] = self.val[1] = self.val[2] = self.val[3] = @0;
    }
}

+ (Scalar*)all:(double)v {
    return [[Scalar alloc] initWithV0:v v1:v v2:v v3:v];
}

- (Scalar*)clone {
    return [[Scalar alloc] initWithVals:self.val];
}

- (Scalar*)mul:(Scalar*)it scale:(double)scale {
    return [[Scalar alloc] initWithV0:it.val[0].doubleValue*scale v1:it.val[1].doubleValue*scale v2:it.val[2].doubleValue*scale v3:it.val[3].doubleValue*scale];
}

- (Scalar*)mul:(Scalar*)it {
    return [self mul:it scale:1];
}

- (Scalar*)conj {
    return [[Scalar alloc] initWithV0:self.val[0].doubleValue v1:-self.val[1].doubleValue v2:-self.val[2].doubleValue v3:-self.val[3].doubleValue];
}

- (BOOL)isReal {
    return self.val[1].doubleValue == self.val[2].doubleValue == self.val[3].doubleValue == 0;
}

- (BOOL)isEqual:(id)other
{
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Scalar class]]) {
        return NO;
    } else {
        Scalar* it = (Scalar*) other;
        return [it.val isEqual:self.val];
    }
}

- (NSUInteger)hash
{
    return self.val.hash;
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"[%f-%f-%f-%f]", self.val[0].doubleValue, self.val[0].doubleValue, self.val[0].doubleValue, self.val[0].doubleValue];
}

@end
