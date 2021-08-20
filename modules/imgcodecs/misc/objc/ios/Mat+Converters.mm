//
//  Mat+Converters.mm
//
//  Created by Giles Payne on 2020/03/03.
//

#import "Mat+Converters.h"
#import <opencv2/imgcodecs/ios.h>

@implementation Mat (Converters)

-(CGImageRef)toCGImage {
    return MatToCGImage(self.nativeRef);
}

-(instancetype)initWithCGImage:(CGImageRef)image {
    return [self initWithCGImage:image alphaExist:NO];
}

-(instancetype)initWithCGImage:(CGImageRef)image alphaExist:(BOOL)alphaExist {
    self = [self init];
    if (self) {
        CGImageToMat(image, self.nativeRef, (bool)alphaExist);
    }
    return self;
}

-(UIImage*)toUIImage {
    return MatToUIImage(self.nativeRef);
}

-(instancetype)initWithUIImage:(UIImage*)image {
    return [self initWithUIImage:image alphaExist:NO];
}

-(instancetype)initWithUIImage:(UIImage*)image alphaExist:(BOOL)alphaExist {
    self = [self init];
    if (self) {
        UIImageToMat(image, self.nativeRef, (bool)alphaExist);
    }
    return self;
}

@end
