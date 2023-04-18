//
//  Size2i.mm
//
//  Created by Giles Payne on 2019/10/06.
//

#import "Size2i.h"
#import "Point2i.h"
#import "CVObjcUtil.h"

@implementation Size2i {
    cv::Size2i native;
}

- (int)width {
    return native.width;
}

- (void)setWidth:(int)val {
    native.width = val;
}

- (int)height {
    return native.height;
}

- (void)setHeight:(int)val {
    native.height = val;
}

- (cv::Size2i&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithWidth:0 height:0];
}

- (instancetype)initWithWidth:(int)width height:(int)height {
    self = [super init];
    if (self) {
        self.width = width;
        self.height = height;
    }
    return self;
}

- (instancetype)initWithPoint:(Point2i*)point {
    return [self initWithWidth:point.x height:point.y];
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Size2i&)size {
    return [[Size2i alloc] initWithWidth:size.width height:size.height];
}

+ (instancetype)width:(int)width height:(int)height {
    return [[Size2i alloc] initWithWidth:width height:height];
}

- (double)area {
    return self.width * self.height;
}

- (BOOL)empty {
    return self.width <= 0 || self.height <= 0;
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.width = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.height = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
}

- (Size2i*)clone {
    return [[Size2i alloc] initWithWidth:self.width height:self.height];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Size2i class]]) {
        return NO;
    } else {
        Size2i* it = (Size2i*)other;
        return self.width == it.width && self.height == it.height;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.height;
    result = prime * result + self.width;
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Size2i {%d,%d}", self.width, self.height];
}

@end
