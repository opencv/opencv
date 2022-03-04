//
//  Scalar.mm
//
//  Created by Giles Payne on 2019/10/06.
//

#import "Scalar.h"

double getVal(NSArray<NSNumber*>* vals, int index) {
    return [vals count] > index ? vals[index].doubleValue : 0;
}

@implementation Scalar {
    cv::Scalar native;
}

- (NSArray<NSNumber*>*)val {
    return @[[NSNumber numberWithDouble:native.val[0]], [NSNumber numberWithDouble:native.val[1]], [NSNumber numberWithDouble:native.val[2]], [NSNumber numberWithDouble:native.val[3]]];
}

#ifdef __cplusplus
- (cv::Scalar&)nativeRef {
    return native;
}
#endif

- (instancetype)initWithVals:(NSArray<NSNumber*> *)vals {
    return [self initWithV0:getVal(vals, 0) v1:getVal(vals, 1) v2:getVal(vals, 2) v3:getVal(vals, 3)];
}

- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3 {
    self = [super init];
    if (self != nil) {
        native.val[0] = v0;
        native.val[1] = v1;
        native.val[2] = v2;
        native.val[3] = v3;
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

+ (Scalar*)all:(double)v {
    return [[Scalar alloc] initWithV0:v v1:v v2:v v3:v];
}

- (Scalar*)clone {
    return [Scalar fromNative:self.nativeRef];
}

- (Scalar*)mul:(Scalar*)it scale:(double)scale {
    return [[Scalar alloc] initWithV0:self.nativeRef.val[0]*it.nativeRef.val[0]*scale v1:self.nativeRef.val[1]*it.nativeRef.val[1]*scale v2:self.nativeRef.val[2]*it.nativeRef.val[2]*scale v3:self.nativeRef.val[3]*it.nativeRef.val[3]*scale];
}

- (Scalar*)mul:(Scalar*)it {
    return [self mul:it scale:1];
}

- (Scalar*)conj {
    return [[Scalar alloc] initWithV0:self.nativeRef.val[0] v1:-self.nativeRef.val[1] v2:-self.nativeRef.val[2] v3:-self.nativeRef.val[3]];
}

- (BOOL)isReal {
    return self.nativeRef.val[1] == self.nativeRef.val[2] == self.nativeRef.val[3] == 0;
}

- (BOOL)isEqual:(id)other
{
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Scalar class]]) {
        return NO;
    } else {
        Scalar* it = (Scalar*) other;
        return it.nativeRef.val[0] == self.nativeRef.val[0] && it.nativeRef.val[1] == self.nativeRef.val[1] && it.nativeRef.val[2] == self.nativeRef.val[2] && it.nativeRef.val[3] == self.nativeRef.val[3];
    }
}

#define DOUBLE_TO_BITS(x)  ((Cv64suf){ .f = x }).i

- (NSUInteger)hash
{
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.nativeRef.val[0]);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.nativeRef.val[1]);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.nativeRef.val[2]);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.nativeRef.val[3]);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"Scalar [%lf, %lf, %lf, %lf]", self.nativeRef.val[0], self.nativeRef.val[1], self.nativeRef.val[2], self.nativeRef.val[3]];
}

@end
