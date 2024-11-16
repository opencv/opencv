#pragma once
#include <stdint.h>
#include <vector>
#include "counted.hpp"
#include "byte_matrix.hpp"

namespace zxing
{

typedef struct _ImageCutResult
{
    ArrayRef<uint8_t> arrImage;
    int iWidth;
    int iHeight;
}ImageCutResult;

class ImageCut
{
public:
    ImageCut();
    ~ImageCut();
    
    static int Cut(uint8_t * poImageData, int iWidth, int iHeight, int iTopLeftX, int iTopLeftY, int iBottomRightX, int iBottomRightY, ImageCutResult & result);
    static int Cut( Ref<ByteMatrix> matrix , float fRatio, ImageCutResult & result);
};

}  // namespace zxing
