//
//  MatOfRotatedRect.mm
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfRotatedRect.h"
#import "Range.h"
#import "RotatedRect.h"
#import "Point2f.h"
#import "Size2f.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfRotatedRect

static const int _depth = CV_32F;
static const int _channels = 5;

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

- (instancetype)initWithArray:(NSArray<RotatedRect*>*)array {
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

- (void)fromArray:(NSArray<RotatedRect*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithFloat:array[index].center.x];
        data[_channels * index + 1] = [NSNumber numberWithFloat:array[index].center.y];
        data[_channels * index + 2] = [NSNumber numberWithFloat:array[index].size.width];
        data[_channels * index + 3] = [NSNumber numberWithFloat:array[index].size.height];
        data[_channels * index + 4] = [NSNumber numberWithFloat:array[index].angle];

    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<RotatedRect*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<RotatedRect*>* ret = createArrayWithSize(length, [RotatedRect new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[RotatedRect alloc] initWithCenter:[[Point2f alloc] initWithX:data[index * _channels].floatValue y:data[index * _channels + 1].floatValue] size:[[Size2f alloc] initWithWidth:data[index * _channels + 2].floatValue height:data[index * _channels + 3].floatValue] angle:data[index * _channels + 4].floatValue];
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
