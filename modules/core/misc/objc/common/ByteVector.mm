//
//  ByteVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "ByteVector.h"
#import <vector>

@implementation ByteVector {
    std::vector<char> v;
}

-(instancetype)initWithData:(NSData*)data {
    self = [super init];
    if (self) {
        if (data.length % sizeof(char) != 0) {
            @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Invalid data length" userInfo:nil];
        }
        v.insert(v.begin(), (char*)data.bytes, (char*)data.bytes + data.length/sizeof(char));
    }
    return self;
}

-(instancetype)initWithVector:(ByteVector*)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.nativeRef.begin(), src.nativeRef.end());
    }
    return self;
}

-(NSInteger)length {
    return v.size();
}

-(char*)nativeArray {
    return (char*)v.data();
}

-(instancetype)initWithNativeArray:(char*)array elements:(NSInteger)elements {
    self = [super init];
    if (self) {
        v.insert(v.begin(), array, array + elements);
    }
    return self;
}

- (std::vector<char>&)nativeRef {
    return v;
}

-(instancetype)initWithStdVector:(std::vector<char>&)src {
    self = [super init];
    if (self) {
        v.insert(v.begin(), src.begin(), src.end());
    }
    return self;
}

+(instancetype)fromNative:(std::vector<char>&)src {
    return [[ByteVector alloc] initWithStdVector:src];
}

-(char)get:(NSInteger)index {
    if (index < 0 || index >= (long)v.size()) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return v[index];
}

-(NSData*)data {
    return [NSData dataWithBytesNoCopy:v.data() length:(v.size() * sizeof(char)) freeWhenDone:NO];
}

@end
