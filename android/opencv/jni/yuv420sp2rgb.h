//yuv420sp2rgb.h
#ifndef YUV420SP2RGB_H
#define YUV420SP2RGB_H

#ifdef __cplusplus
extern "C" {
#endif

void color_convert_common(
    unsigned char *pY, unsigned char *pUV,
    int width, int height, unsigned char *buffer,
    int grey);

#ifdef __cplusplus
}
#endif

#endif
