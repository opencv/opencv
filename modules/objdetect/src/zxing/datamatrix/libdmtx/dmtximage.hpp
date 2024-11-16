//
//  dmtximage.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef dmtximage_hpp
#define dmtximage_hpp

#include <stdio.h>

namespace dmtx {

class DmtxImage {
public:
    DmtxImage() {}
    ~DmtxImage();
    
    int dmtxImageCreate(unsigned char *pxl_, int width_, int height_);
    int dmtxImageGetProp(int prop);
    int dmtxImageGetByteOffset(int x, int y);
    unsigned int dmtxImageGetPixelValue(int x, int y, /*@out@*/ int *value);
    unsigned int dmtxImageContainsInt(int margin, int x, int y);
    
private:
    int             width;
    int             height;
    int             bitsPerPixel;
    int             bytesPerPixel;
    int             rowPadBytes;
    int             rowSizeBytes;
    int             imageFlip;
    
    unsigned char  *pxl;
};

}  // namespace dmtx
#endif  // QBAR_AI_QBAR_ZXING_DATAMATRIX_LIBDMTX_DMTXIMAGE_H_ 
