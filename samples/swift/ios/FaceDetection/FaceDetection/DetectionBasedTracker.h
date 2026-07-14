//
//  DetectionBasedTracker.h
//
//  Created by Giles Payne on 2020/04/05.
//

#pragma once

#ifdef __cplusplus
#import <OpenCV/OpenCV.h>
#endif

#import <Foundation/Foundation.h>

@class Rect2i;
@class Mat;

@interface DetectionBasedTracker : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithCascadeName:(NSString*)cascadeName minFaceSize:(int)minFaceSize;

- (void)start;

- (void)stop;

- (void)setFaceSize:(int)size;

- (void)detect:(Mat*)imageGray faces:(NSMutableArray<Rect2i*>*)faces;

@end
