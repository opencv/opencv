//
//  MatOfRect2d.mm
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfRect2d.h"
#import "Range.h"
#import "Rect2d.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfRect2d

static const int _depth = CV_64F;
static const int _channels = 4;

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat {
    self = [super initWithNativeMat:nativeMat];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}

+ (instancetype)fromNative:(cv::Mat*)nativeMat {
    return [[MatOfRect2d alloc] initWithNativeMat:nativeMat];
}

#endif

- (instancetype)initWithMat:(Mat*)mat {
    self = [super initWithMat:mat rowRange:[Range all]];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}

- (instancetype)initWithArray:(NSArray<Rect2d*>*)array {
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

- (void)fromArray:(NSArray<Rect2d*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < (int)array.count; index++) {
        data[_channels * index] = [NSNumber numberWithDouble:array[index].x];
        data[_channels * index + 1] = [NSNumber numberWithDouble:array[index].y];
        data[_channels * index + 2] = [NSNumber numberWithDouble:array[index].width];
        data[_channels * index + 3] = [NSNumber numberWithDouble:array[index].height];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<Rect2d*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<Rect2d*>* ret = createArrayWithSize(length, [Rect2d new]);
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = createArrayWithSize([self length], @0.0);
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[Rect2d alloc] initWithX:data[index * _channels].doubleValue y:data[index * _channels + 1].doubleValue width:data[index * _channels + 2].doubleValue height:data[index * _channels + 3].doubleValue];
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
