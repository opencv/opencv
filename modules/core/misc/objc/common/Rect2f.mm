//
//  Rect2d.mm
//
//  Created by Giles Payne on 2019/10/09.
//

#import "Rect2f.h"
#import "Point2f.h"
#import "Size2f.h"

@implementation Rect2f {
    cv::Rect2f native;
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

- (float)width {
    return native.width;
}

- (void)setWidth:(float)val {
    native.width = val;
}

- (float)height {
    return native.height;
}

- (void)setHeight:(float)val {
    native.height = val;
}

- (cv::Rect2f&)nativeRef {
    return native;
}

- (instancetype)initWithX:(float)x y:(float)y width:(float)width height:(float)height {
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

- (instancetype)initWithPoint:(Point2f*)point1 point:(Point2f*)point2 {
    float x = (point1.x < point2.x ? point1.x : point2.x);
    float y = (point1.y < point2.y ? point1.y : point2.y);
    float width = (point1.x > point2.x ? point1.x : point2.x) - x;
    float height = (point1.y > point2.y ? point1.y : point2.y) - y;
    return [self initWithX:x y:y width:width height:height];
}

- (instancetype)initWithPoint:(Point2f*)point size:(Size2f*)size {
    return [self initWithX:point.x y:point.y width:size.width height:size.height];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Rect2f&)rect {
    return [[Rect2f alloc] initWithX:rect.x y:rect.y width:rect.width height:rect.height];
}

- (Rect2f*)clone {
    return [[Rect2f alloc] initWithX:self.x y:self.y width:self.width height:self.height];
}

- (Point2f*)tl {
    return [[Point2f alloc] initWithX:self.x y:self.y];
}

- (Point2f*)br {
    return [[Point2f alloc] initWithX:self.x + self.width y:self.y + self.height];
}

- (Size2f*)size {
    return [[Size2f alloc] initWithWidth:self.width height:self.height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (BOOL)contains:(Point2f*)point {
    return self.x <= point.x && point.x < self.x + self.width && self.y <= point.y && point.y < self.y + self.height;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.x = (vals != nil && vals.count > 0) ? vals[0].floatValue : 0;
    self.y = (vals != nil && vals.count > 1) ? vals[1].floatValue : 0;
    self.width = (vals != nil && vals.count > 2) ? vals[2].floatValue : 0;
    self.height = (vals != nil && vals.count > 3) ? vals[3].floatValue : 0;
}

- (BOOL)isEqual:(id)other{
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Rect2f class]]) {
        return NO;
    } else {
        Rect2f* rect = (Rect2f*)other;
        return self.x == rect.x && self.y == rect.y && self.width == rect.width && self.height == rect.height;
    }
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + FLOAT_TO_BITS(self.x);
    result = prime * result + FLOAT_TO_BITS(self.y);
    result = prime * result + FLOAT_TO_BITS(self.width);
    result = prime * result + FLOAT_TO_BITS(self.height);
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Rect2f {%lf,%lf,%lf,%lf}", self.x, self.y, self.width, self.height];
}

@end
