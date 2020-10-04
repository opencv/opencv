//
//  FontFace.mm
//
//  Created by VP in 2020.
//

#import "FontFace.h"

@implementation FontFace {
    cv::FontFace native;
}

-(cv::FontFace&)nativeRef {
    return native;
}

- (NSString*)name {
    return [NSString stringWithUTF8String:native.getName().c_str()];
}

- (double)scaleFactor {
    return native.getScaleFactor();
}

-(instancetype)initWith:(NSString*)name scaleFactor:(double)sf {
    self = [super init];
    if (self) {
        self.nativeRef.set(std::string(name.UTF8String), sf);
    }
    return self;
}
-(instancetype)init {
    return [super init];
}

-(instancetype)initWith:(NSString*)name {
    self = [super init];
    if (self) {
        self.nativeRef.set(std::string(name.UTF8String), 1.0);
    }
    return self;
}

+(instancetype)fromNative:(cv::FontFace&)fface {
    FontFace* ff = [[FontFace alloc] init];
    ff.nativeRef = fface;
    return ff;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"FontFace [ \nname=%s, \nscaleFactor=%lf\n]", self.nativeRef.getName().c_str(), self.scaleFactor];
}

@end
