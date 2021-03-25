//
//  FloatVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "FloatVector.h"
#import <vector>

@implementation FloatVector {
    std::vector<float> v;
}

-(instancetype)initWithData:(NSData*)data {
    self = [super init];
    if (self) {
        if (data.length % sizeof(float) != 0) {
            @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Invalid data length" userInfo:nil];
        }
        v.insert(v.begin(), (float*)data.bytes, (float*)data.bytes + data.length/sizeof(float));
    }
    return self;
}

-(instancetype)initWithVector:(FloatVector *)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.nativeRef.begin(), src.nativeRef.end());
    }
    return self;
}

-(NSInteger)length {
    return v.size();
}

-(float*)nativeArray {
    return (float*)v.data();
}

-(instancetype)initWithNativeArray:(float*)array elements:(NSInteger)elements {
    self = [super init];
    if (self) {
        v.insert(v.begin(), array, array + elements);
    }
    return self;
}

- (std::vector<float>&)nativeRef {
    return v;
}

-(instancetype)initWithStdVector:(std::vector<float>&)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.begin(), src.end());
    }
    return self;
}

+(instancetype)fromNative:(std::vector<float>&)src {
    return [[FloatVector alloc] initWithStdVector:src];
}

-(float)get:(NSInteger)index {
    if (index < 0 || index >= (long)v.size()) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return v[index];
}

-(NSData*)data {
    return [NSData dataWithBytesNoCopy:v.data() length:(v.size() * sizeof(float)) freeWhenDone:NO];
}

@end
