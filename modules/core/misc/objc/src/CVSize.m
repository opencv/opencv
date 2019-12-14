//
//  CVSize.mm
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "CVSize.h"

@implementation CVSize

- (instancetype)init {
    return [self initWithWidth:0 height:0];
}

- (instancetype)initWithWidth:(double)width height:(double)height {
    self = [super init];
    if (self) {
        self.width = 0;
        self.height = 0;
    }
    return self;
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (CVSize*)clone {
    return [[CVSize alloc] initWithWidth:self.width height:self.height];
}

- (BOOL)isEqual:(id)other
{
    if (other == self) {
        return YES;
    } else if (![super isEqual:other] || ![other isKindOfClass:[CVSize class]]) {
        return NO;
    } else {
        CVSize* it = (CVSize*)other;
        return self.width == it.width && self.height == it.height;
    }
}

static int64_t doubleToBits(double x) {
    const union { double d; int64_t l; } xUnion = { .d = x };
    return xUnion.l;
}

- (NSUInteger)hash
{
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = doubleToBits(self.height);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = doubleToBits(self.width);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"%fx%f", self.width, self.height];
}

@end
