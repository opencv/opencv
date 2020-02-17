//
//  IntVector.m
//
//  Created by Giles Payne on 2020/01/04.
//

#import "ObjectVector.h"
#import <vector>

@implementation ObjectVector {
    NSArray<id>* _array;
}

-(instancetype)initWithVector:(ObjectVector<id>*)src {
    self = [super init];
    if (self) {
        _array = src.array;
    }
    return self;
}

-(NSInteger)length {
    return _array.count;
}

-(NSArray<id>*)array {
    return _array;
}

-(instancetype)initWithArray:(NSArray<id>*)array {
    self = [super init];
    if (self) {
        _array = array;
    }
    return self;
}

-(id)get:(NSInteger)index {
    if (index < 0 || index >= (long)_array.count) {
        @throw [NSException exceptionWithName:NSRangeException reason:@"Invalid data length" userInfo:nil];
    }
    return _array[index];
}

-(id)objectAtIndexedSubscript:(NSInteger)index {
    return [self get:index];
}

- (NSUInteger)countByEnumeratingWithState:(nonnull NSFastEnumerationState *)state objects:(__unsafe_unretained id  _Nullable * _Nonnull)buffer count:(NSUInteger)len {
    return [_array countByEnumeratingWithState:state objects:buffer count:len];
}

-(void)updateArray:(NSArray<id>*)array {
    _array = array;
}

@end
