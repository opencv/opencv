#include <string.h>
#include <jni.h>

#include <yuv420sp2rgb.h>
#include <yuv2rgb.h>

/*
 YUV 4:2:0 image with a plane of 8 bit Y samples followed by an interleaved
 U/V plane containing 8 bit 2x2 subsampled chroma samples.
 except the interleave order of U and V is reversed.

 H V
 Y Sample Period      1 1
 U (Cb) Sample Period 2 2
 V (Cr) Sample Period 2 2
 */

/*
 size of a char:
 find . -name limits.h -exec grep CHAR_BIT {} \;
 */

#ifndef max
#define max(a,b) (a > b ? a : b )
#define min(a,b) (a < b ? a : b )
#endif
enum
{
    FLAGS         = 0x40080100
};

#define READUV(U,V) (tables[256 + (U)] + tables[512 + (V)])
#define READY(Y)    tables[Y]
#define FIXUP(Y)                 \
do {                             \
    int tmp = (Y) & FLAGS;       \
    if (tmp != 0)                \
    {                            \
        tmp  -= tmp>>8;          \
        (Y)  |= tmp;             \
        tmp   = FLAGS & ~(Y>>1); \
        (Y)  += tmp>>8;          \
    }                            \
} while (0 == 1)

#define STORE(Y,DSTPTR)           \
do {                              \
    uint32_t Y2       = (Y);      \
    uint8_t  *DSTPTR2 = (DSTPTR); \
    (DSTPTR2)[2] = (Y2);          \
    (DSTPTR2)[1] = (Y2)>>22;      \
    (DSTPTR2)[0] = (Y2)>>11;      \
} while (0 == 1)

typedef unsigned char byte;
const int bytes_per_pixel = 2;
void color_convert_common(const unsigned char *pY, const unsigned char *pUV, int width, int height,
                          unsigned char *buffer, int grey)
{
#define LOOKUP 1
#if ! LOOKUP
  int nR, nG, nB;
#endif
  int dest_span = 3 * width;
  unsigned char *out = buffer;
  if (grey)
  {
    memcpy(out, pY, width * height * sizeof(unsigned char));
  }
  else
  {

#if LOOKUP
    const uint32_t* tables = yuv2rgb565_table;
    const byte* nY = pY;
    const byte* nUV = pUV;
    int idx = 0;
    while (nY+width < pUV)
    {
      int y = (idx / width);
      int x = (idx % width);
      byte Y = *nY;
      byte Y2 = nY[width];
      byte V = *nUV;
      byte U = *(nUV + 1);
      /* Do 2 row pairs */
      uint32_t uv, y0, y1;

      uv = READUV(U,V);
      y1 = uv + READY(Y);
      y0 = uv + READY(Y2);
      FIXUP(y1);
      FIXUP(y0);
      STORE(y1, &out[dest_span]);
      STORE(y0, out);
      out += 3;
      Y = *(++nY);
      Y2 = nY[width];
      y1 = uv + READY(Y);
      y0 = uv + READY(Y2);
      FIXUP(y1);
      FIXUP(y0);
      STORE(y1, &out[dest_span]);
      STORE(y0, out);
      out += 3;
      height += (2 << 16);
      ++nY;
      nUV = pUV + (y / 2) * width + 2 * (x / 2);
      idx+=2;
    }
#else
    const byte* nY = pY;
    const byte* nUV = pUV;
    int idx = 0;
    while (nY < pUV)
    {

      int y = (idx / width);
      int x = (idx % width);
      int Y = *nY;
      int V = *nUV;
      int U = *(nUV + 1);

      Y -= 16;
      V -= 128;
      U -= 128;
      if (y < 0)
      y = 0;

      nB = (int)(1192 * Y + 2066 * U);
      nG = (int)(1192 * Y - 833 * V - 400 * U);
      nR = (int)(1192 * Y + 1634 * V);

      nR = min(262143, max(0, nR));
      nG = min(262143, max(0, nG));
      nB = min(262143, max(0, nB));

      nR >>= 10;
      nR &= 0xff;
      nG >>= 10;
      nG &= 0xff;
      nB >>= 10;
      nB &= 0xff;

      *(out++) = (unsigned char)nR;
      *(out++) = (unsigned char)nG;
      *(out++) = (unsigned char)nB;
      nY += 1;
      nUV = pUV + (y / 2) * width + 2 * (x / 2);
      ++idx;
    }
#endif
  }


}
