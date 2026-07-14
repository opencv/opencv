//
//  DoubleVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "DoubleVector.h"
#import <vector>

@implementation DoubleVector {
    std::vector<double> v;
}

-(instancetype)initWithData:(NSData*)data {
    self = [super init];
    if (self) {
        if (data.length % sizeof(double) != 0) {
            @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Invalid data length" userInfo:nil];
        }
        v.insert(v.begin(), (double*)data.bytes, (double*)data.bytes + data.length/sizeof(double));
    }
    return self;
}

-(instancetype)initWithVector:(DoubleVector*)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.nativeRef.begin(), src.nativeRef.end());
    }
    return self;
}

-(size_t)length {
    return v.size();
}

-(double*)nativeArray {
    return (double*)v.data();
}

-(instancetype)initWithNativeArray:(double*)array elements:(int)elements {
    self = [super init];
    if (self) {
        v.insert(v.begin(), array, array + elements);
    }
    return self;
}

- (std::vector<double>&)nativeRef {
    return v;
}

-(instancetype)initWithStdVector:(std::vector<double>&)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.begin(), src.end());
    }
    return self;
}

+(instancetype)fromNative:(std::vector<double>&)src {
    return [[DoubleVector alloc] initWithStdVector:src];
}

-(double)get:(NSInteger)index {
    if (index < 0 || index >= (long)v.size()) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return v[index];
}

-(NSData*)data {
    return [NSData dataWithBytesNoCopy:v.data() length:(v.size() * sizeof(double)) freeWhenDone:NO];
}

@end
