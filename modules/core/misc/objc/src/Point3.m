//
//  CVPoint.m
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "Point3.h"
#import "CVPoint.h"
#import "CVObjcUtil.h"

@implementation Point3

- (instancetype)init {
    return [self initWithX:0 y:0 z:0];
}

- (instancetype)initWithX:(double)x y:(double)y z:(double)z {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.z = z;
    }
    return self;
}

- (instancetype)initWithPoint:(CVPoint*)point {
    return [self initWithX:point.x y:point.y z:0];
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
        self.z = vals.count > 2 ? vals[2].doubleValue : 0.0;
    } else {
        self.x = 0.0;
        self.y = 0.0;
        self.z = 0.0;
    }
}

- (Point3*) clone {
    return [[Point3 alloc] initWithX:self.x y:self.y z:self.z];
}

- (double)dot:(Point3*)point {
    return self.x * point.x + self.y * point.y + self.z * point.z;
}

- (Point3*)cross:(Point3*)point {
    return [[Point3 alloc] initWithX:(self.y * point.z - self.z * point.y) y:(self.z * point.x - self.x * point.z) z:(self.x * point.y - self.y * point.x)];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point3 class]]) {
        return NO;
    } else {
        Point3* point = (Point3*)other;
        return self.x == point.x && self.y == point.y && self.z == point.z;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    int64_t temp = DOUBLE_TO_BITS(self.x);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.y);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.z);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point3 {%lf,%lf,%lf}", self.x, self.y, self.z];
}

@end
