//
//  MatOfKeyPoint.m
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfKeyPoint.h"
#import "Range.h"
#import "Point2f.h"
#import "KeyPoint.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfKeyPoint

static const int _depth = CV_32F;
static const int _channels = 7;

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

- (instancetype)initWithArray:(NSArray<KeyPoint*>*)array {
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

- (void)fromArray:(NSArray<KeyPoint*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithFloat:array[index].pt.x];
        data[_channels * index + 1] = [NSNumber numberWithFloat:array[index].pt.y];
        data[_channels * index + 2] = [NSNumber numberWithFloat:array[index].size];
        data[_channels * index + 3] = [NSNumber numberWithFloat:array[index].angle];
        data[_channels * index + 4] = [NSNumber numberWithFloat:array[index].response];
        data[_channels * index + 5] = [NSNumber numberWithFloat:array[index].octave];
        data[_channels * index + 6] = [NSNumber numberWithFloat:array[index].classId];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<KeyPoint*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<KeyPoint*>* ret = createArrayWithSize(length, [KeyPoint new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[KeyPoint alloc] initWithX:data[index * _channels].floatValue y:data[index * _channels + 1].floatValue size:data[index * _channels + 2].floatValue angle:data[index * _channels + 3].floatValue response:data[index * _channels + 4].floatValue octave:data[index * _channels + 5].intValue classId:data[index * _channels + 6].intValue];
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
