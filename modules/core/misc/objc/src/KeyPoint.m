//
//  KeyPoint.m
//
//  Created by Giles Payne on 2019/12/25.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "KeyPoint.h"
#import "Point2f.h"

@implementation KeyPoint

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

- (NSString*)description {
    return [NSString stringWithFormat:@"KeyPoint { pt: %@, size: %f, angle: %f, response: %f, octave: %d, classId: %d}", self.pt.description, self.size, self.angle, self.response, self.octave, self.classId];
}

@end
