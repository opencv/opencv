//
//  MinMaxLocResult.m
//
//  Created by Giles Payne on 2019/12/28.
//

#import "MinMaxLocResult.h"
#import "Point2i.h"

@implementation MinMaxLocResult

- (instancetype)init {
    return [self initWithMinval:0 maxVal:0 minLoc:[Point2i new] maxLoc:[Point2i new]];
}

- (instancetype)initWithMinval:(double)minVal maxVal:(double)maxVal minLoc:(Point2i*)minLoc maxLoc:(Point2i*)maxLoc {
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
