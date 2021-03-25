//
//  Float4.mm
//
//  Created by Giles Payne on 2020/02/05.
//

#import "Float4.h"
#import "Mat.h"

@implementation Float4 {
    cv::Vec4f native;
}

-(float)v0 {
    return native[0];
}

-(void)setV0:(float)v {
    native[0] = v;
}

-(float)v1 {
    return native[1];
}

-(void)setV1:(float)v {
    native[1] = v;
}

-(float)v2 {
    return native[2];
}

-(void)setV2:(float)v {
    native[2] = v;
}

-(float)v3 {
    return native[3];
}

-(void)setV3:(float)v {
    native[3] = v;
}

-(instancetype)init {
    return [self initWithV0:0.0 v1:0.0 v2:0.0 v3:0.0];
}

-(instancetype)initWithV0:(float)v0 v1:(float)v1 v2:(float)v2 v3:(float)v3 {
    self = [super init];
    if (self) {
        self.v0 = v0;
        self.v1 = v1;
        self.v2 = v2;
        self.v3 = v3;
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

+(instancetype)fromNative:(cv::Vec4f&)vec4f {
    return [[Float4 alloc] initWithV0:vec4f[0] v1:vec4f[1] v2:vec4f[2] v3:vec4f[3]];
}

-(void)set:(NSArray<NSNumber*>*)vals {
    self.v0 = (vals != nil && vals.count > 0) ? vals[0].floatValue : 0;
    self.v1 = (vals != nil && vals.count > 1) ? vals[1].floatValue : 0;
    self.v2 = (vals != nil && vals.count > 2) ? vals[2].floatValue : 0;
    self.v3 = (vals != nil && vals.count > 3) ? vals[3].floatValue : 0;
}

-(NSArray<NSNumber*>*)get {
    return @[[NSNumber numberWithFloat:native[0]], [NSNumber numberWithFloat:native[1]], [NSNumber numberWithFloat:native[2]], [NSNumber numberWithFloat:native[3]]];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Float4 class]]) {
        return NO;
    } else {
        Float4* point = (Float4*)other;
        return self.v0 == point.v0 && self.v1 == point.v1 && self.v2 == point.v2 && self.v3 == point.v3;
    }
}

@end
