//
//  CvType.m
//
//  Created by Giles Payne on 2019/10/13.
//

#import "CvType.h"

@implementation CvType

+ (int)makeType:(int)depth channels:(int)channels {
    if (channels <= 0 || channels >= CV_CN_MAX) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                reason:[NSString stringWithFormat:@"Channels count should be 1..%d", CV_CN_MAX - 1]
                userInfo:nil];
        @throw exception;
    }
    if (depth < 0 || depth >= CV_DEPTH_MAX) {
        NSException* exception = [NSException
                exceptionWithName:@"UnsupportedOperationException"
                reason:[NSString stringWithFormat:@"Data type depth should be 0..%d", CV_DEPTH_MAX - 1]
                userInfo:nil];
        @throw exception;
    }
    return (depth & (CV_DEPTH_MAX - 1)) + ((channels - 1) << CV_CN_SHIFT);
}

+ (int)channels:(int)type {
    return (type >> CV_CN_SHIFT) + 1;
}

+ (int)depth:(int)type {
    return type & (CV_DEPTH_MAX - 1);
}

+ (BOOL)isInteger:(int)type {
    return [CvType depth:type] < CV_32F;
}

+ (int)typeSizeBits:(int)type {
    int depth = [CvType depth:type];
    switch (depth) {
        case CV_8U:
        case CV_8S:
            return 8;
        case CV_16U:
        case CV_16S:
        case CV_16F:
            return 16;
        case CV_32S:
        case CV_32F:
            return 32;
        case CV_64F:
            return 64;
        default:
            NSException* exception = [NSException
                    exceptionWithName:@"UnsupportedOperationException"
                    reason:[NSString stringWithFormat:@"Unsupported CvType value: %d", type]
                    userInfo:nil];
            @throw exception;
    }
}

+ (int)rawTypeSize:(int)type {
    return [CvType typeSizeBits:type] >> 3;
}

+ (char)typeMnenomic:(int)type {
    int depth = [CvType depth:type];
    switch (depth) {
        case CV_8U:
        case CV_16U:
            return 'U';
        case CV_8S:
        case CV_16S:
        case CV_32S:
            return 'S';
        case CV_16F:
        case CV_32F:
        case CV_64F:
            return 'F';
        default:
            NSException* exception = [NSException
                    exceptionWithName:@"UnsupportedOperationException"
                    reason:[NSString stringWithFormat:@"Unsupported CvType value: %d", type]
                    userInfo:nil];
            @throw exception;
    }
}

+ (int)ELEM_SIZE:(int)type {
    int typeSizeBytes = [CvType rawTypeSize:type];
    return typeSizeBytes * [CvType channels:type];
}

+ (NSString*)typeToString:(int)type {
    int typeSizeBits = [CvType typeSizeBits:type];
    char typeMnenomic = [CvType typeMnenomic:type];
    int channels = [CvType channels:type];
    NSString* channelsSuffix = [NSString stringWithFormat:(channels <= 4)?@"%d":@"(%d)", channels];
    return [NSString stringWithFormat:@"CV_%d%cC%@", typeSizeBits, typeMnenomic, channelsSuffix];
}

@end
