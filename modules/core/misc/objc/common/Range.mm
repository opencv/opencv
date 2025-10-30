//
//  Range.mm
//
//  Created by Giles Payne on 2019/10/08.
//

#import "Range.h"

@implementation Range {
    cv::Range native;
}

- (int)start {
    return native.start;
}

- (void)setStart:(int)val {
    native.start = val;
}

- (int)end {
    return native.end;
}

- (void)setEnd:(int)val {
    native.end = val;
}

- (cv::Range&)nativeRef {
    return native;
}

- (instancetype)init {
    return [self initWithStart:0 end: 0];
}

- (instancetype)initWithStart:(int)start end:(int)end {
    self = [super init];
    if (self != nil) {
        self.start = start;
        self.end = end;
    }
    return self;
}

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals {
    self = [self init];
    if (self != nil) {
        [self set:vals];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Range&)range {
    return [[Range alloc] initWithStart:range.start end:range.end];
}

- (void)set:(NSArray<NSNumber*>*)vals {
    self.start = (vals != nil && vals.count > 0) ? vals[0].intValue : 0;
    self.end = (vals != nil && vals.count > 1 ) ? vals[1].intValue : 0;
}

- (int)size {
    return [self empty] ? 0 : self.end - self.start;
}

- (BOOL)empty {
    return self.end <= self.start;
}

+ (Range*)all {
    return [[Range alloc] initWithStart:INT_MIN end:INT_MAX];
}

- (Range*)intersection:(Range*)r1 {
    Range* out = [[Range alloc] initWithStart:MAX(r1.start, self.start) end:MIN(r1.end, self.end)];
    out.end = MAX(out.end, out.start);
    return out;
}

- (Range*)shift:(int)delta {
    return [[Range alloc] initWithStart:self.start + delta end:self.end + delta];
}

- (Range*)clone {
    return [[Range alloc] initWithStart:self.start end:self.end];
}

- (BOOL)isEqual:(id)other {
    if (other == self) {
        return YES;
    } else if (![other isKindOfClass:[Range class]]) {
        return NO;
    } else {
        Range* it = (Range*)other;
        return self.start == it.start && self.end == it.end;
    }
}

- (NSUInteger)hash {
    int prime = 31;
    uint32_t result = 1;
    result = prime * result + self.start;
    result = prime * result + self.end;
    return result;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"Range {%d, %d}", self.start, self.end];
}

@end
