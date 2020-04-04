//
//  Mat+UIImage.mm
//
//  Created by Giles Payne on 2020/03/03.
//

#import "Mat+Converters.h"
#import <opencv2/imgcodecs/ios.h>

@implementation Mat (Converters)

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
