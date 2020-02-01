//
//  Mat.m
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
#import "CvType.h"
#import "CVObjcUtil.h"

// returns true if final index was reached
static bool updateIdx(cv::Mat* mat, std::vector<int>& indices, int inc) {
    for (int index = mat->dims-1; index>=0; index--) {
        if (inc == 0) return false;
        indices[index] = (indices[index] + 1) % mat->size[index];
        inc--;
    }
    return true;
}

@implementation Mat {
    NSData* _nsdata;
}

- (cv::Mat&)nativeRef {
    return *(cv::Mat*)_nativePtr;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat();
    }
    return self;
}

- (void)dealloc {
    if (_nativePtr != NULL) {
        _nativePtr->release();
        delete _nativePtr;
    }
    _nsdata = NULL;
}

- (instancetype)initWithNativeMat:(cv::Mat*)nativePtr {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat(*nativePtr);
    }
    return self;
}

+ (instancetype)fromNativePtr:(cv::Mat*)nativePtr {
    return [[Mat alloc] initWithNativeMat:nativePtr];
}

+ (instancetype)fromNative:(cv::Mat&)nativeRef {
    return [[Mat alloc] initWithNativeMat:&nativeRef];
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat(rows, cols, type);
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type data:(NSData*)data {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat(rows, cols, type, (void*)data.bytes);
        _nsdata = data; // hold onto a reference otherwise this object might be deallocated
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type data:(NSData*)data step:(long)step {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat(rows, cols, type, (void*)data.bytes, step);
        _nsdata = data; // hold onto a reference otherwise this object might be deallocated
    }
    return self;
}

- (instancetype)initWithSize:(CVSize *)size type:(int)type {
    self = [super init];
    if (self) {
        _nativePtr = new cv::Mat(size.width, size.height, type);
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
        _nativePtr = new cv::Mat((int)sizes.count, vSizes.data(), type);
    }
    return self;
}

- (instancetype)initWithRows:(int)rows cols:(int)cols type:(int)type scalar:(Scalar*)scalar {
    self = [super init];
    if (self) {
        cv::Scalar scalerTemp(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
        _nativePtr = new cv::Mat(rows, cols, type, scalerTemp);
    }
    return self;
}

- (instancetype)initWithSize:(CVSize *)size type:(int)type scalar:(Scalar *)scalar {
    self = [super init];
    if (self) {
        cv::Scalar scalerTemp(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
        _nativePtr = new cv::Mat(size.width, size.height, type, scalerTemp);
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
        _nativePtr = new cv::Mat((int)sizes.count, vSizes.data(), type, scalerTemp);
    }
    return self;
}

- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange colRange:(Range*)colRange {
    self = [super init];
    if (self) {
        cv::Range rows(rowRange.start, rowRange.end);
        cv::Range cols(colRange.start, colRange.end);
        _nativePtr = new cv::Mat(*(cv::Mat*)mat.nativePtr, rows, cols);
    }
    return self;
}

- (instancetype)initWithMat:(Mat*)mat rowRange:(Range*)rowRange {
    self = [super init];
    if (self) {
        cv::Range rows(rowRange.start, rowRange.end);
        _nativePtr = new cv::Mat(*(cv::Mat*)mat.nativePtr, rows);
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
        _nativePtr = new cv::Mat(mat.nativePtr->operator()(tempRanges));
    }
    return self;
}

- (instancetype)initWithMat:(Mat *)mat rect:(CVRect *)roi {
    self = [super init];
    if (self) {
        cv::Range rows(roi.y, roi.y + roi.height);
        cv::Range cols(roi.x, roi.x + roi.width);
        _nativePtr = new cv::Mat(*(cv::Mat*)mat.nativePtr, rows, cols);
    }
    return self;
}

- (BOOL)isSameMat:(Mat*)mat {
    return self.nativePtr == mat.nativePtr;
}

- (Mat*)adjustRoiTop:(int)dtop bottom:(int)dbottom left:(int)dleft right:(int)dright {
    cv::Mat adjusted = _nativePtr->adjustROI(dtop, dbottom, dleft, dright);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(adjusted)];
}

- (void)assignTo:(Mat*)mat type:(int)type {
    _nativePtr->assignTo(*(cv::Mat*)mat.nativePtr, type);
}

- (void)assignTo:(Mat*)mat {
    _nativePtr->assignTo(*(cv::Mat*)mat.nativePtr);
}

- (int)channels {
    return _nativePtr->channels();
}

- (int)checkVector:(int)elemChannels depth:(int)depth requireContinuous:(BOOL) requireContinuous {
    return _nativePtr->checkVector(elemChannels, depth, requireContinuous);
}

- (int)checkVector:(int)elemChannels depth:(int)depth {
    return _nativePtr->checkVector(elemChannels, depth);
}

- (int)checkVector:(int)elemChannels {
    return _nativePtr->checkVector(elemChannels);
}

- (Mat*)clone {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativePtr->clone()))];
}

- (Mat*)col:(int)x {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativePtr->col(x)))];
}

- (Mat*)colRange:(int)start end:(int)end {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativePtr->colRange(start, end)))];
}

- (Mat*)colRange:(Range*)range {
    return [[Mat alloc] initWithNativeMat:(new cv::Mat(_nativePtr->colRange(range.start, range.end)))];
}

- (int)dims {
    return _nativePtr->dims;
}

- (int)cols {
    return _nativePtr->cols;
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha beta:(double)beta {
    _nativePtr->convertTo(*(cv::Mat*)mat->_nativePtr, rtype, alpha, beta);
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype alpha:(double)alpha {
    _nativePtr->convertTo(*(cv::Mat*)mat->_nativePtr, rtype, alpha);
}

- (void)convertTo:(Mat*)mat rtype:(int)rtype {
    _nativePtr->convertTo(*(cv::Mat*)mat->_nativePtr, rtype);
}

- (void)copyTo:(Mat*)mat {
    _nativePtr->copyTo(*(cv::Mat*)mat->_nativePtr);
}

- (void)copyTo:(Mat*)mat mask:(Mat*)mask {
    _nativePtr->copyTo(*(cv::Mat*)mat->_nativePtr, *(cv::Mat*)mask->_nativePtr);
}

- (void)create:(int)rows cols:(int)cols type:(int)type {
    _nativePtr->create(rows, cols, type);
}

- (void)create:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    _nativePtr->create(tempSize, type);
}

- (void)createEx:(NSArray<NSNumber*>*)sizes type:(int)type {
    std::vector<int> tempSizes;
    for (NSNumber* size in sizes) {
        tempSizes.push_back(size.intValue);
    }
    _nativePtr->create((int)tempSizes.size(), tempSizes.data(), type);
}

- (void)copySize:(Mat*)mat {
    _nativePtr->copySize(*(cv::Mat*)mat.nativePtr);
}

- (Mat*)cross:(Mat*)mat {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->cross(*(cv::Mat*)mat.nativePtr))];
}

- (int)depth {
    return _nativePtr->depth();
}

- (Mat*)diag:(int)diagonal {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->diag(diagonal))];
}

- (Mat*)diag {
    return [self diag:0];
}

+ (Mat*)diag:(Mat*)diagonal {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::diag(*(cv::Mat*)diagonal.nativePtr))];
}

- (double)dot:(Mat*)mat {
    return _nativePtr->dot(*(cv::Mat*)mat.nativePtr);
}

- (long)elemSize {
    return _nativePtr->elemSize();
}

- (long)elemSize1 {
    return _nativePtr->elemSize1();
}

- (BOOL)empty {
    return _nativePtr->empty();
}

+ (Mat*)eye:(int)rows cols:(int)cols type:(int)type {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::eye(rows, cols, type))];
}

+ (Mat*)eye:(CVSize*)size type:(int)type {
    cv::Size tempSize(size.width, size.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(cv::Mat::eye(tempSize, type))];
}

- (Mat*)inv:(int)method {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->inv(method))];
}

- (Mat*)inv {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->inv())];
}

- (BOOL)isContinuous {
    return _nativePtr->isContinuous();
}

- (BOOL)isSubmatrix {
    return _nativePtr->isSubmatrix();
}

- (void)locateROI:(CVSize*)wholeSize ofs:(CVPoint*)ofs {
    cv::Size tempWholeSize;
    cv::Point tempOfs;
    _nativePtr->locateROI(tempWholeSize, tempOfs);
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
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->mul(*(cv::Mat*)mat.nativePtr, scale))];
}

- (Mat*)mul:(Mat*)mat {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->mul(*(cv::Mat*)mat.nativePtr))];
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
    _nativePtr->push_back(*(cv::Mat*)mat.nativePtr);
}

- (Mat*)reshape:(int)channels rows:(int)rows {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->reshape(channels, rows))];
}

- (Mat*)reshape:(int)channels {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->reshape(channels))];
}

- (Mat*)reshape:(int)channels newshape:(NSArray<NSNumber*>*)newshape {
    std::vector<int> tempNewshape;
    for (NSNumber* size in newshape) {
        tempNewshape.push_back(size.intValue);
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->reshape(channels, tempNewshape))];
}

- (Mat*)row:(int)y {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->row(y))];
}

- (Mat*)rowRange:(int)start end:(int)end {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->rowRange(start, end))];
}

- (Mat*)rowRange:(Range*)range {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->rowRange(range.start, range.end))];
}

- (int)rows {
    return _nativePtr->rows;
}

- (Mat*)setToScalar:(Scalar*)scalar {
    cv::Scalar tempScalar(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->operator=(tempScalar))];
}

- (Mat*)setToScalar:(Scalar*)scalar mask:(Mat*)mask {
    cv::Scalar tempScalar(scalar.val[0].doubleValue, scalar.val[1].doubleValue, scalar.val[2].doubleValue, scalar.val[3].doubleValue);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->setTo(tempScalar, *(cv::Mat*)mask.nativePtr))];
}

- (Mat*)setToValue:(Mat*)value mask:(Mat*)mask {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->setTo(*(cv::Mat*)value.nativePtr, *(cv::Mat*)mask.nativePtr))];
}

- (Mat*)setToValue:(Mat*)value {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->setTo(*(cv::Mat*)value.nativePtr))];
}

- (CVSize*)size {
    return [[CVSize alloc] initWithWidth:_nativePtr->size().width height:_nativePtr->size().height];
}

- (int)size:(int)dimIndex {
    return _nativePtr->size[dimIndex];
}

- (long)step1:(int)dimIndex {
    return _nativePtr->step1(dimIndex);
}

- (long)step1 {
    return _nativePtr->step1();
}

- (Mat*)submat:(int)rowStart rowEnd:(int)rowEnd colStart:(int)colStart colEnd:(int)colEnd {
    Range* rowRange = [[Range alloc] initWithStart:rowStart end:rowEnd];
    Range* colRange = [[Range alloc] initWithStart:colStart end:colEnd];
    return [self submat:rowRange colRange:colRange];
}

- (Mat*)submat:(Range*)rowRange colRange:(Range*)colRange {
    cv::Range tempRowRange(rowRange.start, rowRange.end);
    cv::Range tempColRange(colRange.start, colRange.end);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->operator()(tempRowRange, tempColRange))];
}

- (Mat*)submat:(NSArray<Range*>*)ranges {
    std::vector<cv::Range> tempRanges;
    for (Range* range in ranges) {
        tempRanges.push_back(cv::Range(range.start, range.end));
    }
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->operator()(tempRanges))];
}

- (Mat*)submatRoi:(CVRect*)roi {
    cv::Rect tempRoi(roi.x, roi.y, roi.width, roi.height);
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->operator()(tempRoi))];
}

- (Mat*)t {
    return [[Mat alloc] initWithNativeMat:new cv::Mat(_nativePtr->t())];
}

- (long)total {
    return _nativePtr->total();
}

- (int)type {
    return _nativePtr->type();
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
    if (_nativePtr->dims <= 0) {
        return @"-1*-1*";
    } else {
        NSMutableString* ret = [NSMutableString string];
        for (int index=0; index<_nativePtr->dims; index++) {
            [ret appendFormat:@"%d*", _nativePtr->size[index]];
        }
        return ret;
    }
}

- (NSString*)description {
    NSString* dimDesc = [self dimsDescription];
    return [NSString stringWithFormat:@"Mat [ %@%@, isCont=%s, isSubmat=%s, nativeObj=0x%p, dataAddr=0x%p ]", dimDesc, [CvType typeToString:_nativePtr->type()], _nativePtr->isContinuous()?"YES":"NO", _nativePtr->isSubmatrix()?"YES":"NO", (void*)_nativePtr, (void*)_nativePtr->data];
}

- (NSString*)dump {
    NSMutableString* ret = [NSMutableString string];
    cv::Ptr<cv::Formatted> formatted = cv::Formatter::get()->format(*_nativePtr);
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
    int depth = _nativePtr->depth();
    if (depth == CV_8U) {
        putData(dest, dataLength, ^uchar (int index) { return cv::saturate_cast<uchar>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_8S) {
        putData(dest, dataLength, ^char (int index) { return cv::saturate_cast<char>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_16U) {
        putData(dest, dataLength, ^ushort (int index) { return cv::saturate_cast<ushort>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_16S || depth == CV_16F) {
        putData(dest, dataLength, ^short (int index) { return cv::saturate_cast<short>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_32S) {
        putData(dest, dataLength, ^int32_t (int index) { return cv::saturate_cast<int32_t>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_32F) {
        putData(dest, dataLength, ^float (int index) { return cv::saturate_cast<float>(data[dataOffset + index].doubleValue);} );
    } else if (depth == CV_64F) {
        putData(dest, dataLength, ^double (int index) { return data[dataOffset + index].doubleValue;} );
    }
}

- (int)put:(NSArray<NSNumber*>*)indices data:(NSArray<NSNumber*>*)data {
    int type = _nativePtr->type();
    if (data == nil || data.count % [CvType channels:type] != 0) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                                  reason:[NSString stringWithFormat:@"Provided data element number (%lu) should be multiple of the Mat channels count (%d)", (unsigned long)(data == nil ? 0 : data.count), [CvType channels:type]]
                userInfo:nil];
        @throw exception;
    }
    std::vector<int> tempIndices;
    for (NSNumber* index in indices) {
        tempIndices.push_back(index.intValue);
    }
    for (int index = 0; index < _nativePtr->dims; index++) {
        if (_nativePtr->size[index]<=tempIndices[index]) {
            return 0; // indexes out of range
        }
    }

    int available = (int)data.count;
    int copyOffset = 0;
    int copyCount = _nativePtr->channels();
    for (int index = 0; index < _nativePtr->dims; index++) {
        copyCount *= (_nativePtr->size[index] - tempIndices[index]);
    }
    if (available < copyCount) {
        copyCount = available;
    }
    int result = 0;
    uchar* dest = _nativePtr->ptr(tempIndices.data());
    while (available > 0) {
        [self put:dest data:data dataOffset:(int)copyOffset dataLength:copyCount];
        result += copyCount * (int)(_nativePtr->elemSize()/_nativePtr->channels());
        if (!updateIdx(_nativePtr, tempIndices, copyCount / (int)_nativePtr->elemSize())) {
            break;
        }
        available -= copyCount;
        copyOffset += copyCount;
        copyCount = _nativePtr->size[_nativePtr->dims-1] * (int)_nativePtr->elemSize();
        if (available < copyCount) {
            copyCount = available;
        }
        dest = _nativePtr->ptr(tempIndices.data());
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
    int depth = _nativePtr->depth();
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
    int type = _nativePtr->type();
    if (data == nil || data.count % [CvType channels:type] != 0) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                                  reason:[NSString stringWithFormat:@"Provided data element number (%lu) should be multiple of the Mat channels count (%d)", (unsigned long)(data == nil ? 0 : data.count), [CvType channels:type]]
                userInfo:nil];
        @throw exception;
    }
    std::vector<int> tempIndices;
    for (NSNumber* index in indices) {
        tempIndices.push_back(index.intValue);
    }
    for (int index = 0; index < _nativePtr->dims; index++) {
        if (_nativePtr->size[index]<=tempIndices[index]) {
            return 0; // indexes out of range
        }
    }

    int available = (int)data.count;
    int copyOffset = 0;
    int copyCount = _nativePtr->channels();
    for (int index = 0; index < _nativePtr->dims; index++) {
        copyCount *= (_nativePtr->size[index] - tempIndices[index]);
    }
    if (available < copyCount) {
        copyCount = available;
    }
    int result = 0;
    uchar* source = _nativePtr->ptr(tempIndices.data());
    while (available > 0) {
        [self get:source data:data dataOffset:(int)copyOffset dataLength:copyCount];
        result += copyCount * (int)(_nativePtr->elemSize()/_nativePtr->channels());
        if (!updateIdx(_nativePtr, tempIndices, copyCount / (int)_nativePtr->elemSize())) {
            break;
        }
        available -= copyCount;
        copyOffset += copyCount;
        copyCount = _nativePtr->size[_nativePtr->dims-1] * (int)_nativePtr->elemSize();
        if (available < copyCount) {
            copyCount = available;
        }
        source = _nativePtr->ptr(tempIndices.data());
    }
    return result;
}

- (int)get:(int)row col:(int)col data:(NSMutableArray<NSNumber*>*)data {
    NSArray<NSNumber*>* indices = @[[NSNumber numberWithInt:row], [NSNumber numberWithInt:col]];
    return [self get:indices data:data];
}

- (NSArray<NSNumber*>*)get:(int)row col:(int)col {
    NSMutableArray<NSNumber*>* result = [NSMutableArray new];
    for (int index = 0; index<_nativePtr->channels(); index++) {
        [result addObject:@0];
    }
    [self get:row col:col data:result];
    return result;
}

- (NSArray<NSNumber*>*)get:(NSArray<NSNumber*>*)indices {
    NSMutableArray<NSNumber*>* result = [NSMutableArray new];
    for (int index = 0; index<_nativePtr->channels(); index++) {
        [result addObject:@0];
    }
    [self get:indices data:result];
    return result;
}

- (int)height {
    return [self rows];
}

- (int)width {
    return [self cols];
}

@end
