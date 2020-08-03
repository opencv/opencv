//
//  Point3d.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point3d.h"
#import "Point2d.h"

@implementation Point3d {
    cv::Point3d native;
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

- (double)z {
    return native.z;
}

- (void)setZ:(double)val {
    native.z = val;
}

- (cv::Point3d&)nativeRef {
    return native;
}

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

- (instancetype)initWithPoint:(Point2d*)point {
    return [self initWithX:point.x y:point.y z:0];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Point3d&)point {
    return [[Point3d alloc] initWithX:point.x y:point.y z:point.z];
}

- (void)update:(cv::Point3d&)point {
    self.x = point.x;
    self.y = point.y;
    self.z = point.z;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0.0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0.0;
    self.z = (vals != nil && vals.count > 2) ? vals[2].doubleValue : 0.0;
}

- (Point3d*) clone {
    return [[Point3d alloc] initWithX:self.x y:self.y z:self.z];
}

- (double)dot:(Point3d*)point {
    return self.x * point.x + self.y * point.y + self.z * point.z;
}

- (Point3d*)cross:(Point3d*)point {
    return [[Point3d alloc] initWithX:(self.y * point.z - self.z * point.y) y:(self.z * point.x - self.x * point.z) z:(self.x * point.y - self.y * point.x)];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point3d class]]) {
        return NO;
    } else {
        Point3d* point = (Point3d*)other;
        return self.x == point.x && self.y == point.y && self.z == point.z;
    }
}

#define DOUBLE_TO_BITS(x)  ((Cv64suf){ .f = x }).i

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
