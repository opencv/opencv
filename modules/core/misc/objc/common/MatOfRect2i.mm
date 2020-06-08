//
//  MatOfRect2i.m
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfRect2i.h"
#import "Range.h"
#import "Rect2i.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfRect2i

static const int _depth = CV_32S;
static const int _channels = 4;

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat {
    self = [super initWithNativeMat:nativeMat];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}
#endif

- (instancetype)initWithMat:(Mat*)mat {
    self = [super initWithMat:mat rowRange:[Range all]];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}

- (instancetype)initWithArray:(NSArray<Rect2i*>*)array {
    self = [super init];
    if (self) {
        [self fromArray:array];
    }
    return self;
}

- (void)alloc:(int)elemNumber {
    if (elemNumber>0) {
        [super create:elemNumber cols:1 type:[CvType makeType:_depth channels:_channels]];
    }
}

- (void)fromArray:(NSArray<Rect2i*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithInt:array[index].x];
        data[_channels * index + 1] = [NSNumber numberWithInt:array[index].y];
        data[_channels * index + 2] = [NSNumber numberWithInt:array[index].width];
        data[_channels * index + 3] = [NSNumber numberWithInt:array[index].height];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<Rect2i*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<Rect2i*>* ret = createArrayWithSize(length, [Rect2i new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[Rect2i alloc] initWithX:data[index * _channels].intValue y:data[index * _channels + 1].intValue width:data[index * _channels + 2].intValue height:data[index * _channels + 3].intValue];
        }
    }
    return ret;
}

- (int)length {
    int num = [self checkVector:_channels depth:_depth];
    if (num < 0) {
        @throw  [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Incompatible Mat" userInfo:nil];
    }
    return num * _channels;
}

@end
