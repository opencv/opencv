//
//  DMatch.m
//
//  Created by Giles Payne on 2019/12/25.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "DMatch.h"

@implementation DMatch

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

- (BOOL)lessThan:(DMatch*)it {
    return self.distance < it.distance;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"DMatch { queryIdx: %d, trainIdx: %d, imgIdx: %d, distance: %f}", self.queryIdx, self.trainIdx, self.imgIdx, self.distance];
}

@end
