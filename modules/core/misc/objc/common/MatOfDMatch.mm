//
//  MatOfDMatch.m
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfDMatch.h"
#import "Range.h"
#import "DMatch.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfDMatch

static const int _depth = CV_32F;
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

- (instancetype)initWithArray:(NSArray<DMatch*>*)array {
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

- (void)fromArray:(NSArray<DMatch*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithFloat:array[index].queryIdx];
        data[_channels * index + 1] = [NSNumber numberWithFloat:array[index].trainIdx];
        data[_channels * index + 2] = [NSNumber numberWithFloat:array[index].imgIdx];
        data[_channels * index + 3] = [NSNumber numberWithFloat:array[index].distance];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<DMatch*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<DMatch*>* ret = createArrayWithSize(length, [DMatch new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[DMatch alloc] initWithQueryIdx:data[index * _channels].intValue trainIdx:data[index * _channels + 1].intValue imgIdx:data[index * _channels + 2].intValue distance:data[index * _channels + 3].floatValue];
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
