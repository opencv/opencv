//
//  CVPoint.m
//  StitchApp
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "CVPoint.h"
#import "CVRect.h"

@implementation CVPoint

- (instancetype)initWithX:(double)x y:(double)y {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
    }
    return self;
}

- (instancetype)init {
    return [self initWithX:0 y:0];
}

- (CVPoint*) clone {
    return [[CVPoint alloc] initWithX:self.x y:self.y];
}

- (double)dot:(CVPoint*)point {
    return self.x * point.x + self.y * point.y;
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![super isEqual:other] || ![other isKindOfClass:[CVPoint class]]) {
        return NO;
    } else {
        CVPoint* point = (CVPoint*)other;
        return self.x == point.x && self.y == point.y;
    }
}

- (BOOL)inside:(CVRect *)rect {
    return [rect contains:self];
}

static int64_t doubleToBits(double x) {
    const union { double d; int64_t l; } xUnion = { .d = x };
    return xUnion.l;
}

- (NSUInteger)hash
{
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = doubleToBits(self.x);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = doubleToBits(self.y);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"[%f,%f]", self.x, self.y];
}

@end
