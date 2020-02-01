//
//  CVSize.mm
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "CVSize.h"
#import "CVPoint.h"
#import "CVObjcUtil.h"

@implementation CVSize {
    cv::Size2d native;
}

- (double)width {
    return native.width;
}

- (void)setWidth:(double)val {
    native.width = val;
}

- (double)height {
    return native.height;
}

- (void)setHeight:(double)val {
    native.height = val;
}

- (cv::Size2d&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithWidth:0 height:0];
}

- (instancetype)initWithWidth:(double)width height:(double)height {
    self = [super init];
    if (self) {
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)initWithPoint:(CVPoint*)point {
    return [self initWithWidth:point.x height:point.y];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}
 
+ (instancetype)fromNative:(cv::Size2d&)size {
    return [[CVSize alloc] initWithWidth:size.width height:size.height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.width = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.height = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
}

- (CVSize*)clone {
    return [[CVSize alloc] initWithWidth:self.width height:self.height];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[CVSize class]]) {
        return NO;
    } else {
        CVSize* it = (CVSize*)other;
        return self.width == it.width && self.height == it.height;
    }
}

- (NSUInteger)hash
{
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.height);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.width);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Size {%lf,%lf}", self.width, self.height];
}

@end
