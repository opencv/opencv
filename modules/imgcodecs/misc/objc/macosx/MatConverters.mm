//
//  MatConverters.mm
//
//  Created by Masaya Tsuruta on 2020/10/08.
//

#import "MatConverters.h"
#import <opencv2/imgcodecs/macosx.h>

@implementation MatConverters

+(CGImageRef)convertMatToCGImageRef:(Mat*)mat {
    return MatToCGImage(mat.nativeRef);
}

+(Mat*)convertCGImageRefToMat:(CGImageRef)image {
    return [MatConverters convertCGImageRefToMat:image alphaExist:NO];
}

+(Mat*)convertCGImageRefToMat:(CGImageRef)image alphaExist:(BOOL)alphaExist {
    Mat* mat = [Mat new];
    CGImageToMat(image, mat.nativeRef, (bool)alphaExist);
    return mat;
}

+(NSImage*)converMatToNSImage:(Mat*)mat {
    return MatToNSImage(mat.nativeRef);
}

+(Mat*)convertNSImageToMat:(NSImage*)image {
    return [MatConverters convertNSImageToMat:image alphaExist:NO];
}

+(Mat*)convertNSImageToMat:(NSImage*)image alphaExist:(BOOL)alphaExist {
    Mat* mat = [Mat new];
    NSImageToMat(image, mat.nativeRef, (bool)alphaExist);
    return mat;
}

@end
