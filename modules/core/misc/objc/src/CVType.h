//
//  CvType.h
//
//  Created by Giles Payne on 2019/10/13.
//

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CvType : NSObject

+ (int)makeType:(int)depth channels:(int)channels;
+ (int)channels:(int)type;
+ (int)depth:(int)type;
+ (int)rawTypeSize:(int)type;
+ (BOOL)isInteger:(int)type;
+ (int)ELEM_SIZE:(int)type NS_SWIFT_NAME(elemSize(_:));
+ (NSString*)typeToString:(int)type;

@end

NS_ASSUME_NONNULL_END
