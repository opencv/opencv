//
//  CVRect.m
//  StitchApp
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "CVRect.h"
#import "CVPoint.h"
#import "CVSize.h"

@implementation CVRect {
    cv::Rect native;
}

- (int)x {
    return native.x;
}

- (void)setX:(int)val {
    native.x = val;
}

- (int)y {
    return native.y;
}

- (void)setY:(int)val {
    native.y = val;
}

- (int)width {
    return native.width;
}

- (void)setWidth:(int)val {
    native.width = val;
}

- (int)height {
    return native.height;
}

- (void)setHeight:(int)val {
    native.height = val;
}

- (cv::Rect&)nativeRef {
    return native;
}

- (instancetype)initWithX:(int)x y:(int)y width:(int)width height:(int)height {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)init {
    return [self initWithX:0 y:0 width:0 height:0];
}

- (instancetype)initWithPoint:(CVPoint*)point1 point:(CVPoint*)point2 {
    int x = (point1.x < point2.x ? point1.x : point2.x);
    int y = (point1.y < point2.y ? point1.y : point2.y);
    int width = (point1.x > point2.x ? point1.x : point2.x) - x;
    int height = (point1.y > point2.y ? point1.y : point2.y) - y;
    return [self initWithX:x y:y width:width height:height];
}

- (instancetype)initWithPoint:(CVPoint*)point size:(CVSize*)size {
    return [self initWithX:point.x y:point.y width:size.width height:size.height];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Rect&)rect {
    return [[CVRect alloc] initWithX:rect.x y:rect.y width:rect.width height:rect.height];
}

- (CVRect*)clone {
    return [[CVRect alloc] initWithX:self.x y:self.y width:self.width height:self.height];
}

- (CVPoint *)tl {
    return [[CVPoint alloc] initWithX:self.x y:self.y];
}

- (CVPoint *)br {
    return [[CVPoint alloc] initWithX:self.x + self.width y:self.y + self.height];
}

- (CVSize *)size {
    return [[CVSize alloc] initWithWidth:self.width height:self.height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (BOOL)contains:(CVPoint*)point {
    return self.x <= point.x && point.x < self.x + self.width && self.y <= point.y && point.y < self.y + self.height;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
    self.width = (vals != nil && vals.count > 2) ? vals[2].intValue : 0;
    self.height = (vals != nil && vals.count > 3) ? vals[3].intValue : 0;
}

- (BOOL)isEqual:(id)other{
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[CVRect class]]) {
        return NO;
    } else {
        CVRect* rect = (CVRect*)other;
        return self.x == rect.x && self.y == rect.y && self.width == rect.width && self.height == rect.height;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.x;
    result = prime * result + self.y;
    result = prime * result + self.width;
    result = prime * result + self.height;
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"CVRect {%d,%d,%d,%d}", self.x, self.y, self.width, self.height];
}

@end
