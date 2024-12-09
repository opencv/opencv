//
//  Double3.mm
//
//  Created by Giles Payne on 2020/05/22.
//

#import "Double3.h"
#import "Mat.h"

@implementation Double3 {
    cv::Vec3d native;
}

-(double)v0 {
    return native[0];
}

-(void)setV0:(double)v {
    native[0] = v;
}

-(double)v1 {
    return native[1];
}

-(void)setV1:(double)v {
    native[1] = v;
}

-(double)v2 {
    return native[2];
}

-(void)setV2:(double)v {
    native[2] = v;
}

-(instancetype)init {
    return [self initWithV0:0 v1:0 v2:0];
}

-(instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 {
    self = [super init];
    if (self) {
        self.v0 = v0;
        self.v1 = v1;
        self.v2 = v2;
    }
    return self;
}

-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+(instancetype)fromNative:(cv::Vec3d&)vec3d {
    return [[Double3 alloc] initWithV0:vec3d[0] v1:vec3d[1] v2:vec3d[2]];
}

-(void)set:(NSArray<NSNumber*>*)vals {
    self.v0 = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.v1 = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
    self.v2 = (vals != nil && vals.count > 2) ? vals[2].doubleValue : 0;
}

-(NSArray<NSNumber*>*)get {
    return @[[NSNumber numberWithFloat:native[0]], [NSNumber numberWithFloat:native[1]], [NSNumber numberWithFloat:native[2]]];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Double3 class]]) {
        return NO;
    } else {
        Double3* d3 = (Double3*)other;
        return self.v0 == d3.v0 && self.v1 == d3.v1 && self.v2 == d3.v2;
    }
}

@end
