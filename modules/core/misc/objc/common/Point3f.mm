//
//  Point3f.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point3f.h"
#import "Point2f.h"

@implementation Point3f {
    cv::Point3f native;
}

- (float)x {
    return native.x;
}

- (void)setX:(float)val {
    native.x = val;
}

- (float)y {
    return native.y;
}

- (void)setY:(float)val {
    native.y = val;
}

- (float)z {
    return native.z;
}

- (void)setZ:(float)val {
    native.z = val;
}

- (cv::Point3f&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithX:0 y:0 z:0];
}

- (instancetype)initWithX:(float)x y:(float)y z:(float)z {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.z = z;
    }
    return self;
}

- (instancetype)initWithPoint:(Point2f*)point {
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
    self.x = (vals != nil && vals.count > 0) ? vals[0].floatValue : 0.0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].floatValue : 0.0;
    self.z = (vals != nil && vals.count > 2) ? vals[2].floatValue : 0.0;
}

- (Point3f*) clone {
    return [[Point3f alloc] initWithX:self.x y:self.y z:self.z];
}

- (double)dot:(Point3f*)point {
    return self.x * point.x + self.y * point.y + self.z * point.z;
}

- (Point3f*)cross:(Point3f*)point {
    return [[Point3f alloc] initWithX:(self.y * point.z - self.z * point.y) y:(self.z * point.x - self.x * point.z) z:(self.x * point.y - self.y * point.x)];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point3f class]]) {
        return NO;
    } else {
        Point3f* point = (Point3f*)other;
        return self.x == point.x && self.y == point.y && self.z == point.z;
    }
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + FLOAT_TO_BITS(self.x);
    result = prime * result + FLOAT_TO_BITS(self.y);
    result = prime * result + FLOAT_TO_BITS(self.z);
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point3f {%f,%f,%f}", self.x, self.y, self.z];
}

@end
