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

@implementation CVRect

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

- (BOOL)isEqual:(id)other{
    if (other == self) {
        return YES;
    } else if (![super isEqual:other] || ![other isKindOfClass:[CVRect class]]) {
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
    return [NSString stringWithFormat:@"CVRect2d {%d,%d,%d,%d}", self.x, self.y, self.width, self.height];
}

@end
