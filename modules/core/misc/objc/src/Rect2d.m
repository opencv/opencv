//
//  CVRect.m
//  StitchApp
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "Rect2d.h"
#import "CVPoint.h"
#import "CVSize.h"
#import "CVObjcUtil.h"

@implementation Rect2d

- (instancetype)init {
    return [self initWithX:0 y:0 width:0 height:0];
}

- (instancetype)initWithX:(double)x y:(double)y width:(double)width height:(double)height {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    if (vals != nil) {
        self.x = vals.count > 0 ? vals[0].doubleValue : 0.0;
        self.y = vals.count > 1 ? vals[1].doubleValue : 0.0;
        self.width = vals.count > 2 ? vals[2].doubleValue : 0.0;
        self.height = vals.count > 3 ? vals[3].doubleValue : 0.0;
    } else {
        self.x = 0.0;
        self.y = 0.0;
        self.width = 0.0;
        self.height = 0.0;
    }
}

- (Rect2d*)clone {
    return [[Rect2d alloc] initWithX:self.x y:self.y width:self.width height:self.height];
}

- (CVPoint*)tl {
    return [[CVPoint alloc] initWithX:self.x y:self.y];
}

- (CVPoint*)br {
    return [[CVPoint alloc] initWithX:self.x + self.width y:self.y + self.height];
}

- (CVSize*)size {
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

- (BOOL)isEqual:(id)other{
    if (other == self) {
        return YES;
    } else if (![super isEqual:other] || ![other isKindOfClass:[Rect2d class]]) {
        return NO;
    } else {
        Rect2d* rect = (Rect2d*)other;
        return self.x == rect.x && self.y == rect.y && self.width == rect.width && self.height == rect.height;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.x);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.y);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.width);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.height);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Rect2d {%lf,%lf,%lf,%lf}", self.x, self.y, self.width, self.height];
}

@end
