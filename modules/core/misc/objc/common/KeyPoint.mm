//
//  KeyPoint.m
//
//  Created by Giles Payne on 2019/12/25.
//

#import "KeyPoint.h"
#import "Point2f.h"

@implementation KeyPoint {
    cv::KeyPoint native;
}

- (cv::KeyPoint&)nativeRef {
    native.pt.x = self.pt.x;
    native.pt.y = self.pt.y;
    native.size = self.size;
    native.angle = self.angle;
    native.response = self.response;
    native.octave = self.octave;
    native.class_id = self.classId;
    return native;
}

- (instancetype)init {
    return [self initWithX:0 y:0 size:0];
}

- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave classId:(int)classId {
    self = [super init];
    if (self != nil) {
        self.pt = [[Point2f alloc] initWithX:x y:y];
        self.size = size;
        self.angle = angle;
        self.response = response;
        self.octave = octave;
        self.classId = classId;
    }
    return self;
}

- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave {
    return [self initWithX:x y:y size:size angle:angle response:response octave:octave classId:-1];
}

- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response {
    return [self initWithX:x y:y size:size angle:angle response:response octave:0];
}

- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle {
    return [self initWithX:x y:y size:size angle:angle response:0];
}

- (instancetype)initWithX:(float)x y:(float)y size:(float)size {
    return [self initWithX:x y:y size:size angle:-1];
}

+ (instancetype)fromNative:(cv::KeyPoint&)keyPoint {
    return [[KeyPoint alloc] initWithX:keyPoint.pt.x y:keyPoint.pt.y size:keyPoint.size angle:keyPoint.angle response:keyPoint.response octave:keyPoint.octave classId:keyPoint.class_id];
}

- (KeyPoint*)clone {
    return [[KeyPoint alloc] initWithX:self.pt.x y:self.pt.y size:self.size angle:self.angle response:self.response octave:self.octave classId:self.classId];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[KeyPoint class]]) {
        return NO;
    } else {
        KeyPoint* keyPoint = (KeyPoint*)other;
        return [self.pt isEqual:keyPoint.pt] && self.size == keyPoint.size && self.angle == keyPoint.angle && self.response == keyPoint.response && self.octave == keyPoint.octave && self.classId == keyPoint.classId;
    }
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + FLOAT_TO_BITS(self.pt.x);
    result = prime * result + FLOAT_TO_BITS(self.pt.y);
    result = prime * result + FLOAT_TO_BITS(self.size);
    result = prime * result + FLOAT_TO_BITS(self.angle);
    result = prime * result + FLOAT_TO_BITS(self.response);
    result = prime * result + self.octave;
    result = prime * result + self.classId;
    return result;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"KeyPoint { pt: %@, size: %f, angle: %f, response: %f, octave: %d, classId: %d}", self.pt.description, self.size, self.angle, self.response, self.octave, self.classId];
}

@end
