//
//  OMat.m
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#import "Mat.h"
#import "CVSize.h"
#import "Scalar.h"
#import "Range.h"
#import "CVRect.h"
#import "CVPoint.h"
#import "CVType.h"

// returns true if final index was reached
static bool updateIdx(cv::Mat* mat, std::vector<int>& indices, int inc) {
    for (int index = mat->dims-1; index>=0; index--) {
        if (inc == 0) return false;
        indices[index] = (indices[index] + 1) % mat->size[index];
        inc--;
    }
    return true;
}

@implementation Mat

- (instancetype)init {
    self = [super init];
    if (self) {
        _nativeMat = new cv::Mat();
    }
    return self;
}

- (void)dealloc {
    if (_nativeMat != NULL) {
        _nativeMat->release();
    }
}

- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat {
    self = [super init];
    if (self) {
        _nativeMat = nativeMat;
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type {
    self = [super init];
    if (self) {
        _nativeMat = new cv::Mat(rows, cols, type);
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type data:(NSData*)data {
    self = [super init];
    if (self) {
        _nativeMat = new cv::Mat(rows, cols, type, (void*)data.bytes);
    }
    return self;
}

- (instancetype)initWithSize:(CVSize *)size type:(int)type {
    self = [super init];
    if (self) {
        _nativeMat = new cv::Mat(size.width, size.height, type);
    }
    return self;
}

- (instancetype)initWithSizes:(NSArray<NSNumber*>*)sizes type:(int)type {
    self = [super init];
    if (self) {
        std::vector<int> vSizes;
        for (NSNumber* size in sizes) {
            vSizes.push_back(size.intValue);
        }
        _nativeMat = new cv::Mat((int)sizes.count, vSizes.data(), type);
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type scalar:(Scalar*)scalar {
    self = [super init];
    if (self) {
        cv::Scalar scalerTemp(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
        _nativeMat = new cv::Mat(rows, cols, type, scalerTemp);
    }
    return self;
}

- (instancetype)initWithSize:(CVSize *)size type:(int)type scalar:(Scalar *)scalar {
    self = [super init];
    if (self) {
        cv::Scalar scalerTemp(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
        _nativeMat = new cv::Mat(size.width, size.height, type, scalerTemp);
    }
    return self;
}

- (instancetype)initWithSizes:(NSArray<NSNumber*>*)sizes type:(int)type scalar:(Scalar *)scalar {
    self = [super init];
    if (self) {
        cv::Scalar scalerTemp(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
        std::vector<int> vSizes;
        for (NSNumber* size in sizes) {
            vSizes.push_back(size.intValue);
        }
        _nativeMat = new cv::Mat((int)sizes.count, vSizes.data(), type, scalerTemp);
    }
    return self;
}

- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange colRange:(Range*)colRange {
    self = [super init];
    if (self) {
        cv::Range rows(rowRange.start, rowRange.end);
        cv::Range cols(colRange.start, colRange.end);
        _nativeMat = new cv::Mat(*(cv::Mat*)mat.nativeMat, rows, cols);
    }
    return self;
}

- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange {
    self = [super init];
    if (self) {
        cv::Range rows(rowRange.start, rowRange.end);
        _nativeMat = new cv::Mat(*(cv::Mat*)mat.nativeMat, rows);
    }
    return self;
}

- (instancetype)initWithMat:(Mat*)mat ranges:(NSArray<Range*>*)ranges {
    self = [super init];
    if (self) {
        std::vector<cv::Range> tempRanges;
        for (Range* range in ranges) {
            tempRanges.push_back(cv::Range(range.start, range.end));
        }
        _nativeMat = new cv::Mat(_nativeMat->operator()(tempRanges));
    }
    return self;
}

- (instancetype)initWithMat:(Mat *)mat rect:(CVRect *)roi {
    self = [super init];
    if (self) {
        cv::Range rows(roi.y, roi.y + roi.height);
        cv::Range cols(roi.x, roi.x + roi.width);
        _nativeMat = new cv::Mat(*(cv::Mat*)mat.nativeMat, rows, cols);
    }
    return self;
}

- (Mat*)adjustRoiTop:(int)dtop bottom:(int)dbottom left:(int)dleft right:(int)dright {
    cv::Mat adjusted = _nativeMat->adjustROI(dtop, dbottom, dleft, dright);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(adjusted)];
}

- (void)assignTo:(Mat*)mat type:(int)type {
    _nativeMat->assignTo(*(cv::Mat*)mat.nativeMat, type);
}

- (void)assignTo:(Mat*)mat {
    _nativeMat->assignTo(*(cv::Mat*)mat.nativeMat);
}

- (int)channels {
    return _nativeMat->channels();
}

- (int)checkVector:(int)elemChannels depth:(int)depth requireContinuous:(BOOL) requireContinuous {
    return _nativeMat->checkVector(elemChannels, depth, requireContinuous);
}

- (int)checkVector:(int)elemChannels depth:(int)depth {
    return _nativeMat->checkVector(elemChannels, depth);
}

- (int)checkVector:(int)elemChannels {
    return _nativeMat->checkVector(elemChannels);
}

- (Mat*)clone {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativeMat->clone()))];
}

- (Mat*)col:(int)x {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativeMat->col(x)))];
}

- (Mat*)colRange:(int)start end:(int)end {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativeMat->colRange(start, end)))];
}

- (Mat*)colRange:(Range*)range {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativeMat->colRange(range.start, range.end)))];
}

- (int)dims {
    return _nativeMat->dims;
}

- (int)cols {
    return _nativeMat->cols;
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha beta:(double)beta {
    _nativeMat->convertTo(*(cv::Mat*)mat->_nativeMat, rtype, alpha, beta);
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha {
    _nativeMat->convertTo(*(cv::Mat*)mat->_nativeMat, rtype, alpha);
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype {
    _nativeMat->convertTo(*(cv::Mat*)mat->_nativeMat, rtype);
}

- (void)copyTo:(Mat*)mat {
    _nativeMat->copyTo(*(cv::Mat*)mat->_nativeMat);
}

- (void)copyTo:(Mat*)mat mask:(Mat*)mask {
    _nativeMat->copyTo(*(cv::Mat*)mat->_nativeMat, *(cv::Mat*)mask->_nativeMat);
}

- (void)create:(int)rows cols:(int)cols type:(int)type {
    _nativeMat->create(rows, cols, type);
}

- (void)create:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    _nativeMat->create(tempSize, type);
}

- (void)createEx:(NSArray<NSNumber*>*)sizes type:(int)type {
    std::vector<int> tempSizes;
    for (NSNumber* size in sizes) {
        tempSizes.push_back(size.intValue);
    }
    _nativeMat->create((int)tempSizes.size(), tempSizes.data(), type);
}

- (void)copySize:(Mat*)mat {
    _nativeMat->copySize(*(cv::Mat*)mat.nativeMat);
}

- (Mat*)cross:(Mat*)mat {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->cross(*(cv::Mat*)mat.nativeMat))];
}

- (int)depth {
    return _nativeMat->depth();
}

- (Mat*)diag:(int)diagonal {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->diag(diagonal))];
}

- (Mat*)diag {
    return [self diag:0];
}

+ (Mat*)diag:(Mat*)diagonal {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::diag(*(cv::Mat*)diagonal.nativeMat))];
}

- (double)dot:(Mat*)mat {
    return _nativeMat->dot(*(cv::Mat*)mat.nativeMat);
}

- (long)elemSize {
    return _nativeMat->elemSize();
}

- (long)elemSize1 {
    return _nativeMat->elemSize1();
}

- (BOOL)empty {
    return _nativeMat->empty();
}

+ (Mat*)eye:(int)rows cols:(int)cols type:(int)type {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::eye(rows, cols, type))];
}

+ (Mat*)eye:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::eye(tempSize, type))];
}

- (Mat*)inv:(int)method {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->inv(method))];
}

- (Mat*)inv {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->inv())];
}

- (BOOL)isContinuous {
    return _nativeMat->isContinuous();
}

- (BOOL)isSubmatrix {
    return _nativeMat->isSubmatrix();
}

- (void)locateROI:(CVSize*)wholeSize ofs:(CVPoint*)ofs {
    cv::Size tempWholeSize;
    cv::Point tempOfs;
    _nativeMat->locateROI(tempWholeSize, tempOfs);
    if (wholeSize != nil) {
        wholeSize.width = tempWholeSize.width;
        wholeSize.height = tempWholeSize.height;
    }
    if (ofs != nil) {
        ofs.x = tempOfs.x;
        ofs.y = tempOfs.y;
    }
}

- (Mat*)mul:(Mat*)mat scale:(double)scale {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->mul(*(cv::Mat*)mat.nativeMat, scale))];
}

- (Mat*)mul:(Mat*)mat {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->mul(*(cv::Mat*)mat.nativeMat))];
}

+ (Mat*)ones:(int)rows cols:(int)cols type:(int)type {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::ones(rows, cols, type))];
}

+ (Mat*)ones:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::ones(tempSize, type))];
}

+ (Mat*)onesEx:(NSArray<NSNumber*>*)sizes type:(int)type {
    std::vector<int> tempSizes;
    for (NSNumber* size in sizes) {
        tempSizes.push_back(size.intValue);
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::ones((int)tempSizes.size(), tempSizes.data(), type))];
}

- (void)push_back:(Mat*)mat {
    _nativeMat->push_back(*(cv::Mat*)mat.nativeMat);
}

- (Mat*)reshape:(int)channels rows:(int)rows {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->reshape(channels, rows))];
}

- (Mat*)reshape:(int)channels {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->reshape(channels))];
}

- (Mat*)reshape:(int)channels newshape:(NSArray<NSNumber*>*)newshape {
    std::vector<int> tempNewshape;
    for (NSNumber* size in newshape) {
        tempNewshape.push_back(size.intValue);
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->reshape(channels, tempNewshape))];
}

- (Mat*)row:(int)y {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->row(y))];
}

- (Mat*)rowRange:(int)start end:(int)end {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->rowRange(start, end))];
}

- (Mat*)rowRange:(Range*)range {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->rowRange(range.start, range.end))];
}

- (int)rows {
    return _nativeMat->rows;
}

- (Mat*)setToScalar:(Scalar*)scalar {
    cv::Scalar tempScalar(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->operator=(tempScalar))];
}

- (Mat*)setToScalar:(Scalar*)scalar mask:(Mat*)mask {
    cv::Scalar tempScalar(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->setTo(tempScalar, *(cv::Mat*)mask.nativeMat))];
}

- (Mat*)setToValue:(Mat*)value mask:(Mat*)mask {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->setTo(*(cv::Mat*)value.nativeMat, *(cv::Mat*)mask.nativeMat))];
}

- (Mat*)setToValue:(Mat*)value {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->setTo(*(cv::Mat*)value.nativeMat))];
}

- (CVSize*)size {
    return [[CVSize alloc] initWithWidth:_nativeMat->size().width height:_nativeMat->size().height];
}

- (int)size:(int)dimIndex {
    return _nativeMat->size[dimIndex];
}

- (long)step1:(int)dimIndex {
    return _nativeMat->step1(dimIndex);
}

- (long)step1 {
    return _nativeMat->step1();
}

- (Mat*)submat:(int)rowStart rowEnd:(int)rowEnd colStart:(int)colStart colEnd:(int)colEnd {
    Range* rowRange = [[Range alloc] initWithStart:rowStart end:rowEnd];
    Range* colRange = [[Range alloc] initWithStart:colStart end:colEnd];
    return [self submat:rowRange colRange:colRange];
}

- (Mat*)submat:(Range*)rowRange colRange:(Range*)colRange {
    cv::Range tempRowRange(rowRange.start, rowRange.end);
    cv::Range tempColRange(colRange.start, colRange.end);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->operator()(tempRowRange, tempColRange))];
}

- (Mat*)submat:(NSArray<Range*>*)ranges {
    std::vector<cv::Range> tempRanges;
    for (Range* range in ranges) {
        tempRanges.push_back(cv::Range(range.start, range.end));
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->operator()(tempRanges))];
}

- (Mat*)submatRoi:(CVRect*)roi {
    cv::Rect tempRoi(roi.x, roi.y, roi.width, roi.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->operator()(tempRoi))];
}

- (Mat*)t {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativeMat->t())];
}

- (long)total {
    return _nativeMat->total();
}

- (int)type {
    return _nativeMat->type();
}

+ (Mat*)zeros:(int)rows cols:(int)cols type:(int)type {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::zeros(rows, cols, type))];
}

+ (Mat*)zeros:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::zeros(tempSize, type))];
}

+ (Mat*)zerosEx:(NSArray<NSNumber*>*)sizes type:(int)type {
    std::vector<int> tempSizes;
    for (NSNumber* size in sizes) {
        tempSizes.push_back(size.intValue);
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::zeros((int)tempSizes.size(), tempSizes.data(), type))];
}

- (NSString*)dimsDescription {
    if (_nativeMat->dims <= 0) {
        return @"-1*-1*";
    } else {
        NSMutableString* ret = [NSMutableString string];
        for (int index=0; index<_nativeMat->dims; index++) {
            [ret appendFormat:@"%d*", _nativeMat->size[index]];
        }
        return ret;
    }
}

- (NSString*)description {
    NSString* dimDesc = [self dimsDescription];
    return [NSString stringWithFormat:@"Mat [ %@%@, isCont=%s, isSubmat=%s, nativeObj=0x%p, dataAddr=0x%p ]", dimDesc, [CVType typeToString:_nativeMat->type()], _nativeMat->isContinuous()?"YES":"NO", _nativeMat->isSubmatrix()?"YES":"NO", (void*)_nativeMat, (void*)_nativeMat->data];
}

- (NSString*)dump {
    NSMutableString* ret = [NSMutableString string];
    cv::Ptr<cv::Formatted> formatted = cv::Formatter::get()->format(*_nativeMat);
    for(const char* format = formatted->next(); format; format = formatted->next()) {
        [ret appendFormat:@"%s", format];
    }
    return ret;
}

template<typename T> void putData(uchar* dataDest, int dataLength, T (^readData)(int)) {
    T* tDataDest = (T*)dataDest;
    for (int index = 0; index < dataLength; index++) {
        tDataDest[index] = readData(index);
    }
}

- (void)put:(uchar*)dest data:(NSArray<NSNumber*>*)data dataOffset:(int)dataOffset dataLength:(int)dataLength {
    int depth = _nativeMat->depth();
    if (depth == CV_8U) {
        putData(dest, dataLength, ^uchar (int index) { return data[dataOffset + index].unsignedCharValue;} );
    } else if (depth == CV_8S) {
        putData(dest, dataLength, ^char (int index) { return data[dataOffset + index].charValue;} );
    } else if (depth == CV_16U) {
        putData(dest, dataLength, ^ushort (int index) { return data[dataOffset + index].unsignedShortValue;} );
    } else if (depth == CV_16S || depth == CV_16F) {
        putData(dest, dataLength, ^short (int index) { return data[dataOffset + index].shortValue;} );
    } else if (depth == CV_32S) {
        putData(dest, dataLength, ^int32_t (int index) { return data[dataOffset + index].intValue;} );
    } else if (depth == CV_32F) {
        putData(dest, dataLength, ^float (int index) { return data[dataOffset + index].floatValue;} );
    } else if (depth == CV_64F) {
        putData(dest, dataLength, ^double (int index) { return data[dataOffset + index].doubleValue;} );
    }
}

- (int)put:(NSArray<NSNumber*>*)indices data:(NSArray<NSNumber*>*)data {
    int type = _nativeMat->type();
    if (data == nil || data.count % [CVType channels:type] != 0) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                                  reason:[NSString stringWithFormat:@"Provided data element number (%lu) should be multiple of the Mat channels count (%d)", (unsigned long)(data == nil ? 0 : data.count), [CVType channels:type]]
                userInfo:nil];
        @throw exception;
    }
    std::vector<int> tempIndices;
    for (NSNumber* index in indices) {
        tempIndices.push_back(index.intValue);
    }
    for (int index = 0; index < _nativeMat->dims; index++) {
        if (_nativeMat->size[index]<=tempIndices[index]) {
            return 0; // indexes out of range
        }
    }

    int available = (int)data.count;
    int copyOffset = 0;
    int copyCount = _nativeMat->channels();
    for (int index = 0; index < _nativeMat->dims; index++) {
        copyCount *= (_nativeMat->size[index] - tempIndices[index]);
    }
    if (available < copyCount) {
        copyCount = available;
    }
    int result = copyCount;
    uchar* dest = _nativeMat->ptr(tempIndices.data());
    while (available > 0) {
        [self put:dest data:data dataOffset:(int)copyOffset dataLength:copyCount];
        updateIdx(_nativeMat, tempIndices, copyCount / (int)_nativeMat->elemSize());
        available -= copyCount;
        copyOffset += copyCount;
        copyCount = _nativeMat->size[_nativeMat->dims-1] * (int)_nativeMat->elemSize();
        if (available < copyCount) {
            copyCount = available;
        }
        dest = _nativeMat->ptr(tempIndices.data());
    }
    return result;
}

- (int)put:(int)row col:(int)col data:(NSArray<NSNumber*>*)data {
    NSArray<NSNumber*>* indices = @[[NSNumber numberWithInt:row], [NSNumber numberWithInt:col]];
    return [self put:indices data:data];
}

template<typename T> void getData(uchar* dataSource, int dataLength, void (^writeData)(int,T)) {
    T* tDataSource = (T*)dataSource;
    for (int index = 0; index < dataLength; index++) {
        writeData(index, tDataSource[index]);
    }
}

- (void)get:(uchar*)source data:(NSMutableArray<NSNumber*>*)data dataOffset:(int)dataOffset dataLength:(int)dataLength {
    int depth = _nativeMat->depth();
    if (depth == CV_8U) {
        getData(source, dataLength, ^void (int index, uchar value) { data[dataOffset + index] = [[NSNumber alloc] initWithUnsignedChar:value]; } );
    } else if (depth == CV_8S) {
        getData(source, dataLength, ^void (int index, char value) { data[dataOffset + index] = [[NSNumber alloc] initWithChar:value]; } );
    } else if (depth == CV_16U) {
        getData(source, dataLength, ^void (int index, ushort value) { data[dataOffset + index] = [[NSNumber alloc] initWithUnsignedShort:value]; } );
    } else if (depth == CV_16S || depth == CV_16F) {
        getData(source, dataLength, ^void (int index, short value) { data[dataOffset + index] = [[NSNumber alloc] initWithShort:value]; } );
    } else if (depth == CV_32S) {
        getData(source, dataLength, ^void (int index, int32_t value) { data[dataOffset + index] = [[NSNumber alloc] initWithInt:value]; } );
    } else if (depth == CV_32F) {
        getData(source, dataLength, ^void (int index, float value) { data[dataOffset + index] = [[NSNumber alloc] initWithFloat:value]; } );
    } else if (depth == CV_64F) {
        getData(source, dataLength, ^void (int index, double value) { data[dataOffset + index] = [[NSNumber alloc] initWithDouble:value]; } );
    }
}

- (int)get:(NSArray<NSNumber*>*)indices data:(NSMutableArray<NSNumber*>*)data {
    int type = _nativeMat->type();
    if (data == nil || data.count % [CVType channels:type] != 0) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                                  reason:[NSString stringWithFormat:@"Provided data element number (%lu) should be multiple of the Mat channels count (%d)", (unsigned long)(data == nil ? 0 : data.count), [CVType channels:type]]
                userInfo:nil];
        @throw exception;
    }
    std::vector<int> tempIndices;
    for (NSNumber* index in indices) {
        tempIndices.push_back(index.intValue);
    }
    for (int index = 0; index < _nativeMat->dims; index++) {
        if (_nativeMat->size[index]<=tempIndices[index]) {
            return 0; // indexes out of range
        }
    }

    int available = (int)data.count;
    int copyOffset = 0;
    int copyCount = _nativeMat->channels();
    for (int index = 0; index < _nativeMat->dims; index++) {
        copyCount *= (_nativeMat->size[index] - tempIndices[index]);
    }
    if (available < copyCount) {
        copyCount = available;
    }
    int result = copyCount;
    uchar* source = _nativeMat->ptr(tempIndices.data());
    while (available > 0) {
        [self get:source data:data dataOffset:(int)copyOffset dataLength:copyCount];
        updateIdx(_nativeMat, tempIndices, copyCount / (int)_nativeMat->elemSize());
        available -= copyCount;
        copyOffset += copyCount;
        copyCount = _nativeMat->size[_nativeMat->dims-1] * (int)_nativeMat->elemSize();
        if (available < copyCount) {
            copyCount = available;
        }
        source = _nativeMat->ptr(tempIndices.data());
    }
    return result;
}

- (int)get:(int)row col:(int)col data:(NSMutableArray<NSNumber*>*)data {
    NSArray<NSNumber*>* indices = @[[NSNumber numberWithInt:row], [NSNumber numberWithInt:col]];
    return [self get:indices data:data];
}

@end
