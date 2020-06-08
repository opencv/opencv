//
//  Point2i.m
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Point2i.h"
#import "Rect2i.h"
#import "CVObjcUtil.h"

@implementation Point2i {
    cv::Point2i native;
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

- (cv::Point2i&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithX:0 y:0];
}

- (instancetype)initWithX:(int)x y:(int)y {
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

+ (instancetype)fromNative:(cv::Point2i&)point {
    return [[Point2i alloc] initWithX:point.x y:point.y];
}

- (void)update:(cv::Point2i&)point {
    self.x = point.x;
    self.y = point.y;
}

- (Point2i*) clone {
    return [[Point2i alloc] initWithX:self.x y:self.y];
}

- (double)dot:(Point2i*)point {
    return self.x * point.x + self.y * point.y;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].doubleValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].doubleValue : 0;
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Point2i class]]) {
        return NO;
    } else {
        Point2i* point = (Point2i*)other;
        return self.x == point.x && self.y == point.y;
    }
}

- (BOOL)inside:(Rect2i*)rect {
    return [rect contains:self];
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.x;
    result = prime * result + self.y;
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Point2i {%d,%d}", self.x, self.y];
}

@end
