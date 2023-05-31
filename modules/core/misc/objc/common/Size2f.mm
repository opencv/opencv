//
//  Size2f.mm
//
//  Created by Giles Payne on 2019/10/06.
//

#import "Size2f.h"
#import "Point2f.h"

@implementation Size2f {
    cv::Size2f native;
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

- (cv::Size2f&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithWidth:0 height:0];
}

- (instancetype)initWithWidth:(float)width height:(float)height {
    self = [super init];
    if (self) {
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)initWithPoint:(Point2f*)point {
    return [self initWithWidth:point.x height:point.y];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Size2f&)size {
    return [[Size2f alloc] initWithWidth:size.width height:size.height];
}

+ (instancetype)width:(float)width height:(float)height {
    return [[Size2f alloc] initWithWidth:width height:height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.width = (vals != nil && vals.count > 0) ? vals[0].floatValue : 0;
    self.height = (vals != nil && vals.count > 1) ? vals[1].floatValue : 0;
}

- (Size2f*)clone {
    return [[Size2f alloc] initWithWidth:self.width height:self.height];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Size2f class]]) {
        return NO;
    } else {
        Size2f* it = (Size2f*)other;
        return self.width == it.width && self.height == it.height;
    }
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + FLOAT_TO_BITS(self.height);
    result = prime * result + FLOAT_TO_BITS(self.width);
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Size2f {%f,%f}", self.width, self.height];
}

@end
