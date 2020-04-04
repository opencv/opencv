//
//  Int4.mm
//
//  Created by Giles Payne on 2020/02/05.
//

#import "Int4.h"
#import "Mat.h"

@implementation Int4 {
    cv::Vec4i native;
}

-(int)v0 {
    return native[0];
}

-(void)setV0:(int)v {
    native[0] = v;
}

-(int)v1 {
    return native[1];
}

-(void)setV1:(int)v {
    native[1] = v;
}

-(int)v2 {
    return native[2];
}

-(void)setV2:(int)v {
    native[2] = v;
}

-(int)v3 {
    return native[3];
}

-(void)setV3:(int)v {
    native[3] = v;
}

-(instancetype)init {
    return [self initWithV0:0 v1:0 v2:0 v3:0];
}

-(instancetype)initWithV0:(int)v0 v1:(int)v1 v2:(int)v2 v3:(int)v3 {
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

+(instancetype)fromNative:(cv::Vec4i&)vec4i {
    return [[Int4 alloc] initWithV0:vec4i[0] v1:vec4i[1] v2:vec4i[2] v3:vec4i[3]];
}

-(void)set:(NSArray<NSNumber*>*)vals {
    self.v0 = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.v1 = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
    self.v2 = (vals != nil && vals.count > 2) ? vals[2].intValue : 0;
    self.v3 = (vals != nil && vals.count > 3) ? vals[3].intValue : 0;
}

-(NSArray<NSNumber*>*)get {
    return @[[NSNumber numberWithFloat:native[0]], [NSNumber numberWithFloat:native[1]], [NSNumber numberWithFloat:native[2]], [NSNumber numberWithFloat:native[3]]];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Int4 class]]) {
        return NO;
    } else {
        Int4* point = (Int4*)other;
        return self.v0 == point.v0 && self.v1 == point.v1 && self.v2 == point.v2 && self.v3 == point.v3;
    }
}

@end
