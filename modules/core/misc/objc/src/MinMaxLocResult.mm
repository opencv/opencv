//
//  MinMaxLocResult.m
//  StitchApp
//
//  Created by Giles Payne on 2019/12/28.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "MinMaxLocResult.h"
#import "CVPoint.h"

@implementation MinMaxLocResult

- (instancetype)init {
    return [self initWithMinval:0 maxVal:0 minLoc:[CVPoint new] maxLoc:[CVPoint new]];
}

- (instancetype)initWithMinval:(double)minVal maxVal:(double)maxVal minLoc:(CVPoint*)minLoc maxLoc:(CVPoint*)maxLoc {
    self = [super init];
    if (self) {
        self.minVal = minVal;
        self.maxVal = maxVal;
        self.minLoc = minLoc;
        self.maxLoc = maxLoc;
    }
    return self;
}

@end
