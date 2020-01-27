//
//  CVPoint.m
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "CVPoint.h"
#import "CVRect.h"
#import "CVObjcUtil.h"

@implementation CVPoint {
    cv::Point native;
}

- (double)x {
    return native.x;
}

- (void)setX:(double)val {
    native.x = val;
}

- (double)y {
    return native.y;
}

- (void)setY:(double)val {
    native.y = val;
}

#ifdef __cplusplus
- (cv::Point&)nativeRef {
    return native;
}
#endif

- (instancetype)init {
    return [self initWithX:0 y:0];
}

- (instancetype)initWithX:(double)x y:(double)y {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
    }
    return self;
}

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point*)point {
    return [[CVPoint alloc] initWithX:point->x y:point->y];
}
#endif

- (CVPoint*) clone {
    return [[CVPoint alloc] initWithX:self.x y:self.y];
}

- (double)dot:(CVPoint*)point {
    return self.x * point.x + self.y * point.y;
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[CVPoint class]]) {
        return NO;
    } else {
        CVPoint* point = (CVPoint*)other;
        return self.x == point.x && self.y == point.y;
    }
}

- (BOOL)inside:(CVRect *)rect {
    return [rect contains:self];
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.x);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.y);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point {%lf,%lf}", self.x, self.y];
}

@end
