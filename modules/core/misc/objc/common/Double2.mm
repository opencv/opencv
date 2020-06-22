//
//  Double2.mm
//
//  Created by Giles Payne on 2020/05/22.
//

#import "Double2.h"
#import "Mat.h"

@implementation Double2 {
    cv::Vec2d native;
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

-(instancetype)init {
    return [self initWithV0:0 v1:0];
}

-(instancetype)initWithV0:(double)v0 v1:(double)v1 {
    self = [super init];
    if (self) {
        self.v0 = v0;
        self.v1 = v1;
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

+(instancetype)fromNative:(cv::Vec2d&)vec2d {
    return [[Double2 alloc] initWithV0:vec2d[0] v1:vec2d[1]];
}

-(void)set:(NSArray<NSNumber*>*)vals {
    self.v0 = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.v1 = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
}

-(NSArray<NSNumber*>*)get {
    return @[[NSNumber numberWithFloat:native[0]], [NSNumber numberWithFloat:native[1]]];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Double2 class]]) {
        return NO;
    } else {
        Double2* d2 = (Double2*)other;
        return self.v0 == d2.v0 && self.v1 == d2.v1;
    }
}

@end
