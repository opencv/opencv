//
//  RotatedRect.m
//  StitchApp
//
//  Created by Giles Payne on 2019/12/26.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "RotatedRect.h"
#import "CVPoint.h"
#import "CVSize.h"
#import "CVRect.h"
#import "CVObjcUtil.h"

#include <math.h>

@implementation RotatedRect

- (instancetype)init {
    return [self initWithCenter:[CVPoint new] size:[CVSize new] angle:0.0];
}

- (instancetype)initWithCenter:(CVPoint*)center size:(CVSize*)size angle:(double)angle {
    self = [super init];
    if (self) {
        self.center = center;
        self.size = size;
        self.angle = angle;
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
        self.center.x = vals.count > 0 ? vals[0].doubleValue : 0.0;
        self.center.y = vals.count > 1 ? vals[1].doubleValue : 0.0;
        self.size.width = vals.count > 2 ? vals[2].doubleValue : 0.0;
        self.size.height = vals.count > 3 ? vals[3].doubleValue : 0.0;
        self.angle = vals.count > 4 ? vals[4].doubleValue : 0.0;
    } else {
        self.center.x = 0.0;
        self.center.y = 0.0;
        self.size.width = 0.0;
        self.size.height = 0.0;
        self.angle = 0.0;
    }
}

- (NSArray<CVPoint*>*)points {
    double _angle = self.angle * M_PI / 180.0;
    double b = cos(_angle) * 0.5;
    double a = sin(_angle) * 0.5f;

    CVPoint* p0 = [[CVPoint alloc] initWithX:self.center.x - a * self.size.height - b * self.size.width y:self.center.y + b * self.size.height - a * self.size.width];
    CVPoint* p1 = [[CVPoint alloc] initWithX:self.center.x + a * self.size.height - b * self.size.width y:self.center.y - b * self.size.height - a * self.size.width];
    CVPoint* p2 = [[CVPoint alloc] initWithX:2 * self.center.x - p0.x y:2 * self.center.y - p0.y];
    CVPoint* p3 = [[CVPoint alloc] initWithX:2 * self.center.x - p1.x y:2 * self.center.y - p1.y];
    return [NSArray arrayWithObjects:p0, p1, p2, p3, nil];
}

- (CVRect*)boundingRect {
    NSArray<CVPoint*>* pts = [self points];
    CVRect* rect = [[CVRect alloc] initWithX:(int)floor(MIN(MIN(MIN(pts[0].x, pts[1].x), pts[2].x), pts[3].x)) y:(int)floor(MIN(MIN(MIN(pts[0].y, pts[1].y), pts[2].y), pts[3].y)) width:(int)ceil(MAX(MAX(MAX(pts[0].x, pts[1].x), pts[2].x), pts[3].x)) height:(int)ceil(MAX(MAX(MAX(pts[0].y, pts[1].y), pts[2].y), pts[3].y))];
    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;
    return rect;
}

- (RotatedRect*)clone {
    return [[RotatedRect alloc] initWithCenter:self.center size:self.size angle:self.angle];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[RotatedRect class]]) {
        return NO;
    } else {
        RotatedRect* rect = (RotatedRect*)other;
        return [self.center isEqual:rect.center] && [self.size isEqual:rect.size] && self.angle == rect.angle;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.center.x);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.center.y);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.size.width);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.size.height);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.angle);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"RotatedRect {%@,%@,%lf}", self.center.description, self.size.description, self.angle];
}

@end
