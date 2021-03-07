//
//  Point3i.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point3i.h"
#import "Point2i.h"
#import "CVObjcUtil.h"

@implementation Point3i {
    cv::Point3i native;
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

- (int)z {
    return native.z;
}

- (void)setZ:(int)val {
    native.z = val;
}

- (cv::Point3i&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithX:0 y:0 z:0];
}

- (instancetype)initWithX:(int)x y:(int)y z:(int)z {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.z = z;
    }
    return self;
}

- (instancetype)initWithPoint:(Point2i*)point {
    return [self initWithX:point.x y:point.y z:0];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Point3i&)point {
    return [[Point3i alloc] initWithX:point.x y:point.y z:point.z];
}

- (void)update:(cv::Point3i&)point {
    self.x = point.x;
    self.y = point.y;
    self.z = point.z;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
    self.z = (vals != nil && vals.count > 2) ? vals[2].intValue : 0;
}

- (Point3i*) clone {
    return [[Point3i alloc] initWithX:self.x y:self.y z:self.z];
}

- (double)dot:(Point3i*)point {
    return self.x * point.x + self.y * point.y + self.z * point.z;
}

- (Point3i*)cross:(Point3i*)point {
    return [[Point3i alloc] initWithX:(self.y * point.z - self.z * point.y) y:(self.z * point.x - self.x * point.z) z:(self.x * point.y - self.y * point.x)];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point3i class]]) {
        return NO;
    } else {
        Point3i* point = (Point3i*)other;
        return self.x == point.x && self.y == point.y && self.z == point.z;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.x;
    result = prime * result + self.y;
    result = prime * result + self.z;
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point3i {%d,%d,%d}", self.x, self.y, self.z];
}

@end
