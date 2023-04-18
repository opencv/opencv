//
//  MatConverters.mm
//
//  Created by Giles Payne on 2020/03/03.
//

#import "MatConverters.h"
#import <opencv2/imgcodecs/ios.h>

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

+(UIImage*)converMatToUIImage:(Mat*)mat {
    return MatToUIImage(mat.nativeRef);
}

+(Mat*)convertUIImageToMat:(UIImage*)image {
    return [MatConverters convertUIImageToMat:image alphaExist:NO];
}

+(Mat*)convertUIImageToMat:(UIImage*)image alphaExist:(BOOL)alphaExist {
    Mat* mat = [Mat new];
    UIImageToMat(image, mat.nativeRef, (bool)alphaExist);
    return mat;
}

@end
