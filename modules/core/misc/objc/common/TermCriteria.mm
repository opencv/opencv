//
//  TermCriteria.m
//
//  Created by Giles Payne on 2019/12/25.
//

#import "TermCriteria.h"

@implementation TermCriteria {
    cv::TermCriteria native;
}

+ (int)COUNT {
    return 1;
}

+ (int)EPS {
    return 2;
}

+ (int)MAX_ITER {
    return 1;
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
    if (self) {
        self.type = type;
        self.maxCount = maxCount;
        self.epsilon = epsilon;
    }
    return self;
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [super init];
    if (self) {
        [self set:vals];
    }
    return self;
}

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::TermCriteria&)nativeTermCriteria {
    return [[TermCriteria alloc] initWithType:nativeTermCriteria.type maxCount:nativeTermCriteria.maxCount epsilon:nativeTermCriteria.epsilon];
}
#endif

- (void)set:(NSArray<NSNumber*>*)vals {
    self.type = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.maxCount = (vals != nil && vals.count > 1) ? vals[1].intValue : 0;
    self.epsilon = (vals != nil && vals.count > 2) ? vals[2].doubleValue : 0.0;
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

#define DOUBLE_TO_BITS(x)  ((Cv64suf){ .f = x }).i

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
