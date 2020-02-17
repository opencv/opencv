//
//  ByteVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "ByteVector.h"
#import <vector>

@implementation ByteVector {
    std::vector<SInt8> v;
}

-(instancetype)initWithData:(NSData*)data {
    self = [super init];
    if (self) {
        if (data.length % sizeof(SInt8) != 0) {
            @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Invalid data length" userInfo:nil];
        }
        v.insert(v.begin(), (SInt8*)data.bytes, (SInt8*)data.bytes + data.length/sizeof(SInt8));
    }
    return self;
}

-(instancetype)initWithVector:(ByteVector*)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.vector.begin(), src.vector.end());
    }
    return self;
}

-(NSInteger)length {
    return v.size();
}

-(SInt8*)nativeArray {
    return (SInt8*)v.data();
}

-(instancetype)initWithNativeArray:(SInt8*)array elements:(NSInteger)elements {
    self = [super init];
    if (self) {
        v.insert(v.begin(), array, array + elements);
    }
    return self;
}

- (std::vector<SInt8>&)vector {
    return v;
}

-(instancetype)initWithStdVector:(std::vector<SInt8>&)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.begin(), src.end());
    }
    return self;
}

+(instancetype)fromNative:(std::vector<SInt8>&)src {
    return [[ByteVector alloc] initWithStdVector:src];
}

-(SInt8)get:(NSInteger)index {
    if (index < 0 || index >= (long)v.size()) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return v[index];
}

-(NSData*)data {
    return [NSData dataWithBytesNoCopy:v.data() length:(v.size() * sizeof(SInt8)) freeWhenDone:NO];
}

@end
