#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device == nil;
    }
}
