//
//  TermCriteria.m
//
//  Created by Giles Payne on 2019/12/25.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "TermCriteria.h"
#import "CVObjcUtil.h"

@implementation TermCriteria {
    cv::TermCriteria native;
}


- (int)type {
    return native.type;
}

- (void)setType:(int)val {
    native.type = val;
}

- (int)maxCount {
    return native.maxCount;
}

- (void)setMaxCount:(int)val {
    native.maxCount = val;
}

- (double)epsilon {
    return native.epsilon;
}

- (void)setEpsilon:(double)val {
    native.epsilon = val;
}

#ifdef __cplusplus
- (cv::TermCriteria&)nativeRef {
    return native;
}
#endif

- (instancetype)init {
    return [self initWithType:0 maxCount:0 epsilon:0.0];
}

- (instancetype)initWithType:(int)type maxCount:(int)maxCount epsilon:(double)epsilon {
    self = [super init];
    if (self != nil) {
        self.type = type;
        self.maxCount = maxCount;
        self.epsilon = epsilon;
    }
    return self;
}

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::TermCriteria&)nativeTermCriteria {
    return [[TermCriteria alloc] initWithType:nativeTermCriteria.type maxCount:nativeTermCriteria.maxCount epsilon:nativeTermCriteria.epsilon];
}
#endif

- (void)set:(NSArray<NSNumber*>*)vals {
    if (vals != nil) {
        self.type = vals.count > 0 ? vals[0].intValue : 0;
        self.maxCount = vals.count > 1 ? vals[1].intValue : 0;
        self.epsilon = vals.count > 2 ? vals[2].doubleValue : 0;
    } else {
        self.type = 0;
        self.maxCount = 0;
        self.epsilon = 0.0f;
    }
}

- (TermCriteria*)clone {
    return [[TermCriteria alloc] initWithType:self.type maxCount:self.maxCount epsilon:self.epsilon];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[TermCriteria class]]) {
        return NO;
    } else {
        TermCriteria* it = (TermCriteria*)other;
        return self.type == it.type && self.maxCount == it.maxCount && self.epsilon == it.epsilon;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.type;
    result = prime * result + self.maxCount;
    int64_t temp = DOUBLE_TO_BITS(self.epsilon);
    result = prime * result + (int32_t) (temp ^ (temp >> 32));
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"TermCriteria { type: %d, maxCount: %d, epsilon: %lf}", self.type, self.maxCount, self.epsilon];
}

@end
