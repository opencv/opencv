//
//  Point2f.m
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point2f.h"
#import "Rect2f.h"

@implementation Point2f {
    cv::Point2f native;
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

- (cv::Point2f&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithX:0 y:0];
}

- (instancetype)initWithX:(float)x y:(float)y {
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

+ (instancetype)fromNative:(cv::Point2f&)point {
    return [[Point2f alloc] initWithX:point.x y:point.y];
}

- (void)update:(cv::Point2f&)point {
    self.x = point.x;
    self.y = point.y;
}

- (Point2f*) clone {
    return [[Point2f alloc] initWithX:self.x y:self.y];
}

- (double)dot:(Point2f*)point {
    return self.x * point.x + self.y * point.y;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point2f class]]) {
        return NO;
    } else {
        Point2f* point = (Point2f*)other;
        return self.x == point.x && self.y == point.y;
    }
}

- (BOOL)inside:(Rect2f *)rect {
    return [rect contains:self];
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + FLOAT_TO_BITS(self.x);
    result = prime * result + FLOAT_TO_BITS(self.x);
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point2f {%f,%f}", self.x, self.y];
}

@end
