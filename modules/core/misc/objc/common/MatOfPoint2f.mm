//
//  MatOfPoint2f.mm
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfPoint2f.h"
#import "Range.h"
#import "Point2f.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfPoint2f

static const int _depth = CV_32F;
static const int _channels = 2;

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

- (instancetype)initWithArray:(NSArray<Point2f*>*)array {
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

- (void)fromArray:(NSArray<Point2f*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithFloat:array[index].x];
        data[_channels * index + 1] = [NSNumber numberWithFloat:array[index].y];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<Point2f*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<Point2f*>* ret = createArrayWithSize(length, [Point2f new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[Point2f alloc] initWithX:data[index * _channels].floatValue y:data[index * _channels + 1].floatValue];
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
