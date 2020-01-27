//
//  Imgcodecs.m
//
//  Created by Giles Payne on 2020/01/22.
//

#import "Imgcodecs.h"
#import "Mat.h"

@implementation Imgcodecs

+ (Mat*)imread:(NSString*)filename flags:(int)flags {
    cv::Mat ret = cv::imread(cv::String(filename.UTF8String), flags);
    return [Mat fromNativeRef:ret];
}

+ (Mat*)imread:(NSString*)filename {
    cv::Mat ret = cv::imread(cv::String(filename.UTF8String));
    return [Mat fromNativeRef:ret];
}

@end
