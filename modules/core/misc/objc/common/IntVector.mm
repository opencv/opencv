//
//  IntVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "IntVector.h"
#import <vector>

@implementation IntVector {
    std::vector<int> v;
}

-(instancetype)initWithData:(NSData*)data {
    self = [super init];
    if (self) {
        if (data.length % sizeof(int) != 0) {
            @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Invalid data length" userInfo:nil];
        }
        v.insert(v.begin(), (int*)data.bytes, (int*)data.bytes + data.length/sizeof(int));
    }
    return self;
}

-(instancetype)initWithVector:(IntVector*)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.nativeRef.begin(), src.nativeRef.end());
    }
    return self;
}

-(NSInteger)length {
    return v.size();
}

-(int*)nativeArray {
    return (int*)v.data();
}

-(instancetype)initWithNativeArray:(int*)array elements:(NSInteger)elements {
    self = [super init];
    if (self) {
        v.insert(v.begin(), array, array + elements);
    }
    return self;
}

- (std::vector<int>&)nativeRef {
    return v;
}

-(instancetype)initWithStdVector:(std::vector<int>&)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.begin(), src.end());
    }
    return self;
}

+(instancetype)fromNative:(std::vector<int>&)src {
    return [[IntVector alloc] initWithStdVector:src];
}

-(int)get:(NSInteger)index {
    if (index < 0 || index >= (long)v.size()) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return v[index];
}

-(NSData*)data {
    return [NSData dataWithBytesNoCopy:v.data() length:(v.size() * sizeof(int)) freeWhenDone:NO];
}

@end
