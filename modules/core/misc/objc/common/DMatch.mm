//
//  DMatch.m
//
//  Created by Giles Payne on 2019/12/25.
//

#import "DMatch.h"

@implementation DMatch {
    cv::DMatch native;
}

- (int)queryIdx {
    return native.queryIdx;
}

- (void)setQueryIdx:(int)queryIdx {
    native.queryIdx = queryIdx;
}

- (int)trainIdx {
    return native.trainIdx;
}

- (void)setTrainIdx:(int)trainIdx {
    native.trainIdx = trainIdx;
}

- (int)imgIdx {
    return native.imgIdx;
}

- (void)setImgIdx:(int)imgIdx {
    native.imgIdx = imgIdx;
}

- (float)distance {
    return native.distance;
}

- (void)setDistance:(float)distance {
    native.distance = distance;
}

- (cv::DMatch&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithQueryIdx:-1 trainIdx:-1 distance:FLT_MAX];
}

- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx distance:(float)distance {
    return [self initWithQueryIdx:queryIdx trainIdx:trainIdx imgIdx:-1 distance:distance];
}

- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx imgIdx:(int)imgIdx distance:(float)distance {
    self = [super init];
    if (self != nil) {
        self.queryIdx = queryIdx;
        self.trainIdx = trainIdx;
        self.imgIdx = imgIdx;
        self.distance = distance;
    }
    return self;
}

+ (instancetype)fromNative:(cv::DMatch&)dMatch {
    return [[DMatch alloc] initWithQueryIdx:dMatch.queryIdx trainIdx:dMatch.trainIdx imgIdx:dMatch.imgIdx distance:dMatch.distance];
}

- (BOOL)lessThan:(DMatch*)it {
    return self.distance < it.distance;
}


- (DMatch*)clone {
    return [[DMatch alloc] initWithQueryIdx:self.queryIdx trainIdx:self.trainIdx imgIdx:self.imgIdx distance:self.distance];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[DMatch class]]) {
        return NO;
    } else {
        DMatch* dMatch = (DMatch*)other;
        return self.queryIdx == dMatch.queryIdx && self.trainIdx == dMatch.trainIdx && self.imgIdx == dMatch.imgIdx && self.distance == dMatch.distance;
    }
}

#define FLOAT_TO_BITS(x)  ((Cv32suf){ .f = x }).i

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.queryIdx;
    result = prime * result + self.trainIdx;
    result = prime * result + self.imgIdx;
    result = prime * result + FLOAT_TO_BITS(self.distance);
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"DMatch { queryIdx: %d, trainIdx: %d, imgIdx: %d, distance: %f}", self.queryIdx, self.trainIdx, self.imgIdx, self.distance];
}

@end
