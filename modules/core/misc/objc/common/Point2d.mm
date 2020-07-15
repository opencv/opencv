//
//  Point2d.m
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point2d.h"
#import "Rect2d.h"

@implementation Point2d {
    cv::Point2d native;
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

- (cv::Point2d&)nativeRef {
    return native;
}

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

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Point2d&)point {
    return [[Point2d alloc] initWithX:point.x y:point.y];
}

- (void)update:(cv::Point2d&)point {
    self.x = point.x;
    self.y = point.y;
}

- (Point2d*) clone {
    return [[Point2d alloc] initWithX:self.x y:self.y];
}

- (double)dot:(Point2d*)point {
    return self.x * point.x + self.y * point.y;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point2d class]]) {
        return NO;
    } else {
        Point2d* point = (Point2d*)other;
        return self.x == point.x && self.y == point.y;
    }
}

- (BOOL)inside:(Rect2d*)rect {
    return [rect contains:self];
}

#define DOUBLE_TO_BITS(x)  ((Cv64suf){ .f = x }).i

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
    return [NSString stringWithFormat:@"Point2d {%lf,%lf}", self.x, self.y];
}

@end
