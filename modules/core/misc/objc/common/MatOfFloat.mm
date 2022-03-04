//
//  MatOfFloat.mm
//
//  Created by Giles Payne on 2019/12/26.
//

#import "MatOfFloat.h"
#import "Range.h"
#import "CvType.h"
#import "ArrayUtil.h"

@implementation MatOfFloat

static const int _depth = CV_32F;
static const int _channels = 1;

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

- (instancetype)initWithArray:(NSArray<NSNumber*>*)array {
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

- (void)fromArray:(NSArray<NSNumber*>*)array {
    [self alloc:(int)array.count / _channels];
    [self put:0 col:0 data:array];
}

- (NSArray<NSNumber*>*)toArray {
    int length = [self length];
    NSMutableArray<NSNumber*>* data = createArrayWithSize(length, @0.0);
    [self get:0 col:0 data:data];
    return data;
}

- (int)length {
    int num = [self checkVector:_channels depth:_depth];
    if (num < 0) {
        @throw  [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Incompatible Mat" userInfo:nil];
    }
    return num * _channels;
}

@end
