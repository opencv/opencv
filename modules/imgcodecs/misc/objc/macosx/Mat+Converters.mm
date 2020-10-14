//
//  Mat+Converters.mm
//
//  Created by Masaya Tsuruta on 2020/10/08.
//

#import "Mat+Converters.h"
#import <opencv2/imgcodecs/macosx.h>

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

-(NSImage*)toNSImage {
    return MatToNSImage(self.nativeRef);
}

-(instancetype)initWithNSImage:(NSImage*)image {
    return [self initWithNSImage:image alphaExist:NO];
}

-(instancetype)initWithNSImage:(NSImage*)image alphaExist:(BOOL)alphaExist {
    self = [self init];
    if (self) {
        NSImageToMat(image, self.nativeRef, (bool)alphaExist);
    }
    return self;
}

@end
