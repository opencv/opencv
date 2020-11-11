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

-(instancetype)init {
    return [super init];
}

-(instancetype)initWith:(NSString*)name {
    self = [super init];
    if (self) {
        self.nativeRef.set(std::string(name.UTF8String));
    }
    return self;
}

+(instancetype)fromNative:(cv::FontFace&)fface {
    FontFace* ff = [[FontFace alloc] init];
    ff.nativeRef = fface;
    return ff;
}

- (NSString *)description {
    return [NSString stringWithFormat:@"FontFace [name=%s]", self.nativeRef.getName().c_str()];
}

@end
