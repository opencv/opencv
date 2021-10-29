//
//  Rect2d.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Rect2d.h"
#import "Point2d.h"
#import "Size2d.h"

@implementation Rect2d {
    cv::Rect2d native;
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

- (double)width {
    return native.width;
}

- (void)setWidth:(double)val {
    native.width = val;
}

- (double)height {
    return native.height;
}

- (void)setHeight:(double)val {
    native.height = val;
}

- (cv::Rect2d&)nativeRef {
    return native;
}

- (instancetype)initWithX:(double)x y:(double)y width:(double)width height:(double)height {
    self = [super init];
    if (self) {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)init {
    return [self initWithX:0 y:0 width:0 height:0];
}

- (instancetype)initWithPoint:(Point2d*)point1 point:(Point2d*)point2 {
    double x = (point1.x < point2.x ? point1.x : point2.x);
    double y = (point1.y < point2.y ? point1.y : point2.y);
    double width = (point1.x > point2.x ? point1.x : point2.x) - x;
    double height = (point1.y > point2.y ? point1.y : point2.y) - y;
    return [self initWithX:x y:y width:width height:height];
}

- (instancetype)initWithPoint:(Point2d*)point size:(Size2d*)size {
    return [self initWithX:point.x y:point.y width:size.width height:size.height];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Rect2d&)rect {
    return [[Rect2d alloc] initWithX:rect.x y:rect.y width:rect.width height:rect.height];
}

- (Rect2d*)clone {
    return [[Rect2d alloc] initWithX:self.x y:self.y width:self.width height:self.height];
}

- (Point2d*)tl {
    return [[Point2d alloc] initWithX:self.x y:self.y];
}

- (Point2d*)br {
    return [[Point2d alloc] initWithX:self.x + self.width y:self.y + self.height];
}

- (Size2d*)size {
    return [[Size2d alloc] initWithWidth:self.width height:self.height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (BOOL)contains:(Point2d*)point {
    return self.x <= point.x && point.x < self.x + self.width && self.y <= point.y && point.y < self.y + self.height;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
    self.width = (vals != nil && vals.count > 2) ? vals[2].intValue : 0;
    self.height = (vals != nil && vals.count > 3) ? vals[3].intValue : 0;
}

- (BOOL)isEqual:(id)other{
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Rect2d class]]) {
        return NO;
    } else {
        Rect2d* rect = (Rect2d*)other;
        return self.x == rect.x && self.y == rect.y && self.width == rect.width && self.height == rect.height;
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
    temp = DOUBLE_TO_BITS(self.width);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    temp = DOUBLE_TO_BITS(self.height);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Rect2d {%lf,%lf,%lf,%lf}", self.x, self.y, self.width, self.height];
}

@end
