//
// This file is auto-generated. Please don't modify it!
//

#import "TickMeter.h"
#import "CVObjcUtil.h"



@implementation TickMeter

- (cv::TickMeter&)nativeRef {
    return *(cv::TickMeter*)_nativePtr;
}

- (void)dealloc {
    if (_nativePtr != NULL) {
        delete _nativePtr;
    }
}

- (instancetype)initWithNativePtr:(cv::TickMeter*)nativePtr {
    self = [super init];
    if (self) {
        _nativePtr = nativePtr;
    }
    return self;
}

+ (instancetype)fromNative:(cv::TickMeter*)nativePtr {
    return [[TickMeter alloc] initWithNativePtr:nativePtr];
}


//
//   cv::TickMeter::TickMeter()
//
- (instancetype)init {
    return [self initWithNativePtr:new cv::TickMeter()];
}


//
//  double cv::TickMeter::getTimeMicro()
//
- (double)getTimeMicro {
    double retVal = _nativePtr->getTimeMicro();
    return retVal;
}


//
//  double cv::TickMeter::getTimeMilli()
//
- (double)getTimeMilli {
    double retVal = _nativePtr->getTimeMilli();
    return retVal;
}


//
//  double cv::TickMeter::getTimeSec()
//
- (double)getTimeSec {
    double retVal = _nativePtr->getTimeSec();
    return retVal;
}


//
//  int64 cv::TickMeter::getCounter()
//
- (long)getCounter {
    int64 retVal = _nativePtr->getCounter();
    return retVal;
}


//
//  int64 cv::TickMeter::getTimeTicks()
//
- (long)getTimeTicks {
    int64 retVal = _nativePtr->getTimeTicks();
    return retVal;
}


//
//  void cv::TickMeter::reset()
//
- (void)reset {
    _nativePtr->reset();
}


//
//  void cv::TickMeter::start()
//
- (void)start {
    _nativePtr->start();
}


//
//  void cv::TickMeter::stop()
//
- (void)stop {
    _nativePtr->stop();
}



@end


