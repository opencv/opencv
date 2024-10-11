/*
 * Copyright (C)2009-2019, 2021-2024 D. R. Commander.  All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the libjpeg-turbo Project nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#if !defined(_MSC_VER) || _MSC_VER > 1600
#include <stdint.h>
#endif
#include <cdjpeg.h>
#include "./tjutil.h"
#include "./turbojpeg.h"


#define THROW(op, err) { \
  printf("ERROR in line %d while %s:\n%s\n", __LINE__, op, err); \
  retval = -1;  goto bailout; \
}
#define THROW_UNIX(m)  THROW(m, strerror(errno))

static char tjErrorStr[JMSG_LENGTH_MAX] = "\0";
static int tjErrorLine = -1, tjErrorCode = -1;

#define THROW_TJG() { \
  printf("ERROR in line %d\n%s\n", __LINE__, tj3GetErrorStr(NULL)); \
  retval = -1;  goto bailout; \
}

#define THROW_TJ() { \
  int _tjErrorCode = tj3GetErrorCode(handle); \
  char *_tjErrorStr = tj3GetErrorStr(handle); \
  \
  if (!tj3Get(handle, TJPARAM_STOPONWARNING) && \
      _tjErrorCode == TJERR_WARNING) { \
    if (strncmp(tjErrorStr, _tjErrorStr, JMSG_LENGTH_MAX) || \
        tjErrorCode != _tjErrorCode || tjErrorLine != __LINE__) { \
      strncpy(tjErrorStr, _tjErrorStr, JMSG_LENGTH_MAX); \
      tjErrorStr[JMSG_LENGTH_MAX - 1] = '\0'; \
      tjErrorCode = _tjErrorCode; \
      tjErrorLine = __LINE__; \
      printf("WARNING in line %d:\n%s\n", __LINE__, _tjErrorStr); \
    } \
  } else { \
    printf("%s in line %d:\n%s\n", \
           _tjErrorCode == TJERR_WARNING ? "WARNING" : "ERROR", __LINE__, \
           _tjErrorStr); \
    retval = -1;  goto bailout; \
  } \
}

#define IS_CROPPED(cr)  (cr.x != 0 || cr.y != 0 || cr.w != 0 || cr.h != 0)

#define CROPPED_WIDTH(width) \
  (IS_CROPPED(cr) ? (cr.w != 0 ? cr.w : TJSCALED(width, sf) - cr.x) : \
                    TJSCALED(width, sf))

#define CROPPED_HEIGHT(height) \
  (IS_CROPPED(cr) ? (cr.h != 0 ? cr.h : TJSCALED(height, sf) - cr.y) : \
                    TJSCALED(height, sf))

static int stopOnWarning = 0, bottomUp = 0, noRealloc = 1, fastUpsample = 0,
  fastDCT = 0, optimize = 0, progressive = 0, limitScans = 0, maxMemory = 0,
  maxPixels = 0, arithmetic = 0, lossless = 0, restartIntervalBlocks = 0,
  restartIntervalRows = 0;
static int precision = 8, sampleSize, compOnly = 0, decompOnly = 0, doYUV = 0,
  quiet = 0, doTile = 0, pf = TJPF_BGR, yuvAlign = 1, doWrite = 1;
static char *ext = "ppm";
static const char *pixFormatStr[TJ_NUMPF] = {
  "RGB", "BGR", "RGBX", "BGRX", "XBGR", "XRGB", "GRAY", "", "", "", "", "CMYK"
};
static const char *subNameLong[TJ_NUMSAMP] = {
  "4:4:4", "4:2:2", "4:2:0", "GRAY", "4:4:0", "4:1:1", "4:4:1"
};
static const char *csName[TJ_NUMCS] = {
  "RGB", "YCbCr", "GRAY", "CMYK", "YCCK"
};
static const char *subName[TJ_NUMSAMP] = {
  "444", "422", "420", "GRAY", "440", "411", "441"
};
static tjscalingfactor *scalingFactors = NULL, sf = { 1, 1 };
static tjregion cr = { 0, 0, 0, 0 };
static int nsf = 0, xformOp = TJXOP_NONE, xformOpt = 0;
static int (*customFilter) (short *, tjregion, tjregion, int, int,
                            tjtransform *);
static double benchTime = 5.0, warmup = 1.0;


static char *formatName(int subsamp, int cs, char *buf)
{
  if (quiet == 1) {
    if (lossless)
      SNPRINTF(buf, 80, "%-2d/LOSSLESS   ", precision);
    else if (subsamp == TJSAMP_UNKNOWN)
      SNPRINTF(buf, 80, "%-2d/%-5s      ", precision, csName[cs]);
    else
      SNPRINTF(buf, 80, "%-2d/%-5s/%-5s", precision, csName[cs],
               subNameLong[subsamp]);
    return buf;
  } else {
    if (lossless)
      return (char *)"Lossless";
    else if (subsamp == TJSAMP_UNKNOWN)
      return (char *)csName[cs];
    else {
      SNPRINTF(buf, 80, "%s %s", csName[cs], subNameLong[subsamp]);
      return buf;
    }
  }
}


static char *sigfig(double val, int figs, char *buf, int len)
{
  char format[80];
  int digitsAfterDecimal = figs - (int)ceil(log10(fabs(val)));

  if (digitsAfterDecimal < 1)
    SNPRINTF(format, 80, "%%.0f");
  else
    SNPRINTF(format, 80, "%%.%df", digitsAfterDecimal);
  SNPRINTF(buf, len, format, val);
  return buf;
}


/* Custom DCT filter which produces a negative of the image */
static int dummyDCTFilter(short *coeffs, tjregion arrayRegion,
                          tjregion planeRegion, int componentIndex,
                          int transformIndex, tjtransform *transform)
{
  int i;

  for (i = 0; i < arrayRegion.w * arrayRegion.h; i++)
    coeffs[i] = -coeffs[i];
  return 0;
}


/* Decompression test */
static int decomp(unsigned char **jpegBufs, size_t *jpegSizes, void *dstBuf,
                  int w, int h, int subsamp, int jpegQual, char *fileName,
                  int tilew, int tileh)
{
  char tempStr[1024], sizeStr[24] = "\0", qualStr[16] = "\0";
  FILE *file = NULL;
  tjhandle handle = NULL;
  int i, row, col, iter = 0, dstBufAlloc = 0, retval = 0;
  double elapsed, elapsedDecode;
  int ps = tjPixelSize[pf];
  int scaledw, scaledh, pitch;
  int ntilesw = (w + tilew - 1) / tilew, ntilesh = (h + tileh - 1) / tileh;
  unsigned char *dstPtr, *dstPtr2, *yuvBuf = NULL;

  if (lossless) sf = TJUNSCALED;

  scaledw = TJSCALED(w, sf);
  scaledh = TJSCALED(h, sf);

  if (jpegQual > 0) {
    SNPRINTF(qualStr, 16, "_%s%d", lossless ? "PSV" : "Q", jpegQual);
    qualStr[15] = 0;
  }

  if ((handle = tj3Init(TJINIT_DECOMPRESS)) == NULL)
    THROW_TJG();
  if (tj3Set(handle, TJPARAM_STOPONWARNING, stopOnWarning) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_BOTTOMUP, bottomUp) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_FASTUPSAMPLE, fastUpsample) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_FASTDCT, fastDCT) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_SCANLIMIT, limitScans ? 500 : 0) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_MAXMEMORY, maxMemory) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_MAXPIXELS, maxPixels) == -1)
    THROW_TJ();

  if (IS_CROPPED(cr)) {
    if (tj3DecompressHeader(handle, jpegBufs[0], jpegSizes[0]) == -1)
      THROW_TJ();
  }
  if (tj3SetScalingFactor(handle, sf) == -1)
    THROW_TJ();
  if (tj3SetCroppingRegion(handle, cr) == -1)
    THROW_TJ();
  if (IS_CROPPED(cr)) {
    scaledw = cr.w ? cr.w : scaledw - cr.x;
    scaledh = cr.h ? cr.h : scaledh - cr.y;
  }
  pitch = scaledw * ps;

  if (dstBuf == NULL) {
#if ULLONG_MAX > SIZE_MAX
    if ((unsigned long long)pitch * (unsigned long long)scaledh *
        (unsigned long long)sampleSize > (unsigned long long)((size_t)-1))
      THROW("allocating destination buffer", "Image is too large");
#endif
    if ((dstBuf = malloc((size_t)pitch * scaledh * sampleSize)) == NULL)
      THROW_UNIX("allocating destination buffer");
    dstBufAlloc = 1;
  }

  /* Set the destination buffer to gray so we know whether the decompressor
     attempted to write to it */
  if (precision == 8)
    memset((unsigned char *)dstBuf, 127, (size_t)pitch * scaledh);
  else if (precision == 12) {
    for (i = 0; i < pitch * scaledh; i++)
      ((short *)dstBuf)[i] = (short)2047;
  } else {
    for (i = 0; i < pitch * scaledh; i++)
      ((unsigned short *)dstBuf)[i] = (unsigned short)32767;
  }

  if (doYUV) {
    int width = doTile ? tilew : scaledw;
    int height = doTile ? tileh : scaledh;
    size_t yuvSize = tj3YUVBufSize(width, yuvAlign, height, subsamp);

    if (yuvSize == 0)
      THROW_TJG();
    if ((yuvBuf = (unsigned char *)malloc(yuvSize)) == NULL)
      THROW_UNIX("allocating YUV buffer");
    memset(yuvBuf, 127, yuvSize);
  }

  /* Benchmark */
  iter = -1;
  elapsed = elapsedDecode = 0.;
  while (1) {
    int tile = 0;
    double start = getTime();

    for (row = 0, dstPtr = dstBuf; row < ntilesh;
         row++, dstPtr += (size_t)pitch * tileh * sampleSize) {
      for (col = 0, dstPtr2 = dstPtr; col < ntilesw;
           col++, tile++, dstPtr2 += ps * tilew * sampleSize) {
        int width = doTile ? min(tilew, w - col * tilew) : scaledw;
        int height = doTile ? min(tileh, h - row * tileh) : scaledh;

        if (doYUV) {
          double startDecode;

          if (tj3DecompressToYUV8(handle, jpegBufs[tile], jpegSizes[tile],
                                  yuvBuf, yuvAlign) == -1)
            THROW_TJ();
          startDecode = getTime();
          if (tj3DecodeYUV8(handle, yuvBuf, yuvAlign, dstPtr2, width, pitch,
                            height, pf) == -1)
            THROW_TJ();
          if (iter >= 0) elapsedDecode += getTime() - startDecode;
        } else {
          if (precision == 8) {
            if (tj3Decompress8(handle, jpegBufs[tile], jpegSizes[tile],
                               dstPtr2, pitch, pf) == -1)
              THROW_TJ();
          } else if (precision == 12) {
            if (tj3Decompress12(handle, jpegBufs[tile], jpegSizes[tile],
                                (short *)dstPtr2, pitch, pf) == -1)
              THROW_TJ();
          } else {
            if (tj3Decompress16(handle, jpegBufs[tile], jpegSizes[tile],
                                (unsigned short *)dstPtr2, pitch, pf) == -1)
              THROW_TJ();
          }
        }
      }
    }
    elapsed += getTime() - start;
    if (iter >= 0) {
      iter++;
      if (elapsed >= benchTime) break;
    } else if (elapsed >= warmup) {
      iter = 0;
      elapsed = elapsedDecode = 0.;
    }
  }
  if (doYUV) elapsed -= elapsedDecode;

  if (quiet) {
    printf("%-6s%s",
           sigfig((double)(w * h) / 1000000. * (double)iter / elapsed, 4,
                  tempStr, 1024),
           quiet == 2 ? "\n" : "  ");
    if (doYUV)
      printf("%s\n",
             sigfig((double)(w * h) / 1000000. * (double)iter / elapsedDecode,
                    4, tempStr, 1024));
    else if (quiet != 2) printf("\n");
  } else {
    printf("%s --> Frame rate:         %f fps\n",
           doYUV ? "Decomp to YUV" : "Decompress   ", (double)iter / elapsed);
    printf("                  Throughput:         %f Megapixels/sec\n",
           (double)(w * h) / 1000000. * (double)iter / elapsed);
    if (doYUV) {
      printf("YUV Decode    --> Frame rate:         %f fps\n",
             (double)iter / elapsedDecode);
      printf("                  Throughput:         %f Megapixels/sec\n",
             (double)(w * h) / 1000000. * (double)iter / elapsedDecode);
    }
  }

  if (!doWrite) goto bailout;

  if (sf.num != 1 || sf.denom != 1)
    SNPRINTF(sizeStr, 24, "%d_%d", sf.num, sf.denom);
  else if (tilew != w || tileh != h)
    SNPRINTF(sizeStr, 24, "%dx%d", tilew, tileh);
  else SNPRINTF(sizeStr, 24, "full");
  if (decompOnly)
    SNPRINTF(tempStr, 1024, "%s_%s.%s", fileName, sizeStr, ext);
  else
    SNPRINTF(tempStr, 1024, "%s_%s%s_%s.%s", fileName,
             lossless ? "LOSSLS" : subName[subsamp], qualStr, sizeStr, ext);

  if (precision == 8) {
    if (tj3SaveImage8(handle, tempStr, (unsigned char *)dstBuf, scaledw, 0,
                      scaledh, pf) == -1)
      THROW_TJ();
  } else if (precision == 12) {
    if (tj3SaveImage12(handle, tempStr, (short *)dstBuf, scaledw, 0, scaledh,
                       pf) == -1)
      THROW_TJ();
  } else {
    if (tj3SaveImage16(handle, tempStr, (unsigned short *)dstBuf, scaledw, 0,
                      scaledh, pf) == -1)
      THROW_TJ();
  }

bailout:
  if (file) fclose(file);
  tj3Destroy(handle);
  if (dstBufAlloc) free(dstBuf);
  free(yuvBuf);
  return retval;
}


static int fullTest(tjhandle handle, void *srcBuf, int w, int h, int subsamp,
                    int jpegQual, char *fileName)
{
  char tempStr[1024], tempStr2[80];
  FILE *file = NULL;
  unsigned char **jpegBufs = NULL, *yuvBuf = NULL, *srcPtr, *srcPtr2;
  void *tmpBuf = NULL;
  double start, elapsed, elapsedEncode;
  int row, col, i, tilew = w, tileh = h, retval = 0;
  int iter;
  size_t totalJpegSize = 0, *jpegSizes = NULL, yuvSize = 0;
  int ps = tjPixelSize[pf];
  int ntilesw = 1, ntilesh = 1, pitch = w * ps;
  const char *pfStr = pixFormatStr[pf];

#if ULLONG_MAX > SIZE_MAX
  if ((unsigned long long)pitch * (unsigned long long)h *
      (unsigned long long)sampleSize > (unsigned long long)((size_t)-1))
    THROW("allocating temporary image buffer", "Image is too large");
#endif
  if ((tmpBuf = malloc((size_t)pitch * h * sampleSize)) == NULL)
    THROW_UNIX("allocating temporary image buffer");

  if (!quiet)
    printf(">>>>>  %s (%s) <--> %d-bit JPEG (%s %s%d)  <<<<<\n", pfStr,
           bottomUp ? "Bottom-up" : "Top-down", precision,
           lossless ? "Lossless" : subNameLong[subsamp],
           lossless ? "PSV" : "Q", jpegQual);

  for (tilew = doTile ? 8 : w, tileh = doTile ? 8 : h; ;
       tilew *= 2, tileh *= 2) {
    if (tilew > w) tilew = w;
    if (tileh > h) tileh = h;
    ntilesw = (w + tilew - 1) / tilew;
    ntilesh = (h + tileh - 1) / tileh;

    if ((jpegBufs = (unsigned char **)malloc(sizeof(unsigned char *) *
                                             ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG tile array");
    memset(jpegBufs, 0, sizeof(unsigned char *) * ntilesw * ntilesh);
    if ((jpegSizes = (size_t *)malloc(sizeof(size_t) * ntilesw *
                                      ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG size array");
    memset(jpegSizes, 0, sizeof(size_t) * ntilesw * ntilesh);

    if (noRealloc) {
      for (i = 0; i < ntilesw * ntilesh; i++) {
        size_t jpegBufSize = tj3JPEGBufSize(tilew, tileh, subsamp);

        if (jpegBufSize == 0)
          THROW_TJG();
        if ((jpegBufs[i] = tj3Alloc(jpegBufSize)) == NULL)
          THROW_UNIX("allocating JPEG tiles");
      }
    }

    /* Compression test */
    if (quiet == 1)
      printf("%-4s(%s)  %-2d/%-6s %-3d   ", pfStr, bottomUp ? "BU" : "TD",
             precision, lossless ? "LOSSLS" : subNameLong[subsamp], jpegQual);
    if (precision == 8) {
      for (i = 0; i < h; i++)
        memcpy(&((unsigned char *)tmpBuf)[pitch * i],
               &((unsigned char *)srcBuf)[w * ps * i], w * ps);
    } else {
      for (i = 0; i < h; i++)
        memcpy(&((unsigned short *)tmpBuf)[pitch * i],
               &((unsigned short *)srcBuf)[w * ps * i], w * ps * sampleSize);
    }

    if (tj3Set(handle, TJPARAM_NOREALLOC, noRealloc) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_SUBSAMP, subsamp) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_FASTDCT, fastDCT) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_OPTIMIZE, optimize) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_PROGRESSIVE, progressive) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_ARITHMETIC, arithmetic) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_LOSSLESS, lossless) == -1)
      THROW_TJ();
    if (lossless) {
      if (tj3Set(handle, TJPARAM_LOSSLESSPSV, jpegQual) == -1)
        THROW_TJ();
    } else {
      if (tj3Set(handle, TJPARAM_QUALITY, jpegQual) == -1)
        THROW_TJ();
    }
    if (tj3Set(handle, TJPARAM_RESTARTBLOCKS, restartIntervalBlocks) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_RESTARTROWS, restartIntervalRows) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_MAXMEMORY, maxMemory) == -1)
      THROW_TJ();

    if (doYUV) {
      yuvSize = tj3YUVBufSize(tilew, yuvAlign, tileh, subsamp);
      if (yuvSize == 0)
        THROW_TJG();
      if ((yuvBuf = (unsigned char *)malloc(yuvSize)) == NULL)
        THROW_UNIX("allocating YUV buffer");
      memset(yuvBuf, 127, yuvSize);
    }

    /* Benchmark */
    iter = -1;
    elapsed = elapsedEncode = 0.;
    while (1) {
      int tile = 0;

      totalJpegSize = 0;
      start = getTime();
      for (row = 0, srcPtr = srcBuf; row < ntilesh;
           row++, srcPtr += pitch * tileh * sampleSize) {
        for (col = 0, srcPtr2 = srcPtr; col < ntilesw;
             col++, tile++, srcPtr2 += ps * tilew * sampleSize) {
          int width = min(tilew, w - col * tilew);
          int height = min(tileh, h - row * tileh);

          if (doYUV) {
            double startEncode = getTime();

            if (tj3EncodeYUV8(handle, srcPtr2, width, pitch, height, pf,
                              yuvBuf, yuvAlign) == -1)
              THROW_TJ();
            if (iter >= 0) elapsedEncode += getTime() - startEncode;
            if (tj3CompressFromYUV8(handle, yuvBuf, width, yuvAlign, height,
                                    &jpegBufs[tile], &jpegSizes[tile]) == -1)
              THROW_TJ();
          } else {
            if (precision == 8) {
              if (tj3Compress8(handle, srcPtr2, width, pitch, height, pf,
                               &jpegBufs[tile], &jpegSizes[tile]) == -1)
                THROW_TJ();
            } else if (precision == 12) {
              if (tj3Compress12(handle, (short *)srcPtr2, width, pitch, height,
                                pf, &jpegBufs[tile], &jpegSizes[tile]) == -1)
                THROW_TJ();
            } else {
              if (tj3Compress16(handle, (unsigned short *)srcPtr2, width,
                                pitch, height, pf, &jpegBufs[tile],
                                &jpegSizes[tile]) == -1)
                THROW_TJ();
            }
          }
          totalJpegSize += jpegSizes[tile];
        }
      }
      elapsed += getTime() - start;
      if (iter >= 0) {
        iter++;
        if (elapsed >= benchTime) break;
      } else if (elapsed >= warmup) {
        iter = 0;
        elapsed = elapsedEncode = 0.;
      }
    }
    if (doYUV) elapsed -= elapsedEncode;

    if (quiet == 1) printf("%-5d  %-5d   ", tilew, tileh);
    if (quiet) {
      if (doYUV)
        printf("%-6s%s",
               sigfig((double)(w * h) / 1000000. *
                      (double)iter / elapsedEncode, 4, tempStr, 1024),
               quiet == 2 ? "\n" : "  ");
      printf("%-6s%s",
             sigfig((double)(w * h) / 1000000. * (double)iter / elapsed, 4,
                    tempStr, 1024),
             quiet == 2 ? "\n" : "  ");
      printf("%-6s%s",
             sigfig((double)(w * h * ps) / (double)totalJpegSize, 4, tempStr2,
                    80),
             quiet == 2 ? "\n" : "  ");
    } else {
      printf("\n%s size: %d x %d\n", doTile ? "Tile" : "Image", tilew, tileh);
      if (doYUV) {
        printf("Encode YUV    --> Frame rate:         %f fps\n",
               (double)iter / elapsedEncode);
        printf("                  Output image size:  %lu bytes\n",
               (unsigned long)yuvSize);
        printf("                  Compression ratio:  %f:1\n",
               (double)(w * h * ps) / (double)yuvSize);
        printf("                  Throughput:         %f Megapixels/sec\n",
               (double)(w * h) / 1000000. * (double)iter / elapsedEncode);
        printf("                  Output bit stream:  %f Megabits/sec\n",
               (double)yuvSize * 8. / 1000000. * (double)iter / elapsedEncode);
      }
      printf("%s --> Frame rate:         %f fps\n",
             doYUV ? "Comp from YUV" : "Compress     ",
             (double)iter / elapsed);
      printf("                  Output image size:  %lu bytes\n",
             (unsigned long)totalJpegSize);
      printf("                  Compression ratio:  %f:1\n",
             (double)(w * h * ps) / (double)totalJpegSize);
      printf("                  Throughput:         %f Megapixels/sec\n",
             (double)(w * h) / 1000000. * (double)iter / elapsed);
      printf("                  Output bit stream:  %f Megabits/sec\n",
             (double)totalJpegSize * 8. / 1000000. * (double)iter / elapsed);
    }
    if (tilew == w && tileh == h && doWrite) {
     SNPRINTF(tempStr, 1024, "%s_%s_%s%d.jpg", fileName,
              lossless ? "LOSSLS" : subName[subsamp],
              lossless ? "PSV" : "Q", jpegQual);
      if ((file = fopen(tempStr, "wb")) == NULL)
        THROW_UNIX("opening reference image");
      if (fwrite(jpegBufs[0], jpegSizes[0], 1, file) != 1)
        THROW_UNIX("writing reference image");
      fclose(file);  file = NULL;
      if (!quiet) printf("Reference image written to %s\n", tempStr);
    }

    /* Decompression test */
    if (!compOnly) {
      if (decomp(jpegBufs, jpegSizes, tmpBuf, w, h, subsamp, jpegQual,
                 fileName, tilew, tileh) == -1)
        goto bailout;
    } else if (quiet == 1) printf("N/A\n");

    for (i = 0; i < ntilesw * ntilesh; i++) {
      tj3Free(jpegBufs[i]);
      jpegBufs[i] = NULL;
    }
    free(jpegBufs);  jpegBufs = NULL;
    free(jpegSizes);  jpegSizes = NULL;
    if (doYUV) {
      free(yuvBuf);  yuvBuf = NULL;
    }

    if (tilew == w && tileh == h) break;
  }

bailout:
  if (file) fclose(file);
  if (jpegBufs) {
    for (i = 0; i < ntilesw * ntilesh; i++)
      tj3Free(jpegBufs[i]);
  }
  free(jpegBufs);
  free(yuvBuf);
  free(jpegSizes);
  free(tmpBuf);
  return retval;
}


static int decompTest(char *fileName)
{
  FILE *file = NULL;
  tjhandle handle = NULL;
  unsigned char **jpegBufs = NULL, *srcBuf = NULL;
  size_t *jpegSizes = NULL, srcSize, totalJpegSize;
  tjtransform *t = NULL;
  double start, elapsed;
  int ps = tjPixelSize[pf], tile, row, col, i, iter, retval = 0, decompsrc = 0;
  char *temp = NULL, tempStr[80], tempStr2[80];
  /* Original image */
  int w = 0, h = 0, minTile = 16, tilew, tileh, ntilesw = 1, ntilesh = 1,
    subsamp = -1, cs = -1;
  /* Transformed image */
  int tw, th, ttilew, ttileh, tntilesw, tntilesh, tsubsamp;

  if ((file = fopen(fileName, "rb")) == NULL)
    THROW_UNIX("opening file");
  if (fseek(file, 0, SEEK_END) < 0 ||
      (srcSize = ftell(file)) == (size_t)-1)
    THROW_UNIX("determining file size");
  if ((srcBuf = (unsigned char *)malloc(srcSize)) == NULL)
    THROW_UNIX("allocating memory");
  if (fseek(file, 0, SEEK_SET) < 0)
    THROW_UNIX("setting file position");
  if (fread(srcBuf, srcSize, 1, file) < 1)
    THROW_UNIX("reading JPEG data");
  fclose(file);  file = NULL;

  temp = strrchr(fileName, '.');
  if (temp != NULL) *temp = '\0';

  if ((handle = tj3Init(TJINIT_TRANSFORM)) == NULL)
    THROW_TJG();
  if (tj3Set(handle, TJPARAM_STOPONWARNING, stopOnWarning) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_BOTTOMUP, bottomUp) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_NOREALLOC, noRealloc) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_FASTUPSAMPLE, fastUpsample) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_FASTDCT, fastDCT) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_SCANLIMIT, limitScans ? 500 : 0) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_MAXMEMORY, maxMemory) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_MAXPIXELS, maxPixels) == -1)
    THROW_TJ();

  if (tj3DecompressHeader(handle, srcBuf, srcSize) == -1)
    THROW_TJ();
  w = tj3Get(handle, TJPARAM_JPEGWIDTH);
  h = tj3Get(handle, TJPARAM_JPEGHEIGHT);
  subsamp = tj3Get(handle, TJPARAM_SUBSAMP);
  precision = tj3Get(handle, TJPARAM_PRECISION);
  if (tj3Get(handle, TJPARAM_PROGRESSIVE) == 1)
    printf("JPEG image uses progressive entropy coding\n\n");
  if (tj3Get(handle, TJPARAM_ARITHMETIC) == 1)
    printf("JPEG image uses arithmetic entropy coding\n\n");
  if (tj3Set(handle, TJPARAM_PROGRESSIVE, progressive) == -1)
    THROW_TJ();
  if (tj3Set(handle, TJPARAM_ARITHMETIC, arithmetic) == -1)
    THROW_TJ();

  lossless = tj3Get(handle, TJPARAM_LOSSLESS);
  sampleSize = (precision == 8 ? sizeof(unsigned char) : sizeof(short));
  cs = tj3Get(handle, TJPARAM_COLORSPACE);
  if (w < 1 || h < 1)
    THROW("reading JPEG header", "Invalid image dimensions");
  if (cs == TJCS_YCCK || cs == TJCS_CMYK) {
    pf = TJPF_CMYK;  ps = tjPixelSize[pf];
  }
  if (lossless) sf = TJUNSCALED;

  if (tj3SetScalingFactor(handle, sf) == -1)
    THROW_TJ();
  if (tj3SetCroppingRegion(handle, cr) == -1)
    THROW_TJ();

  if (quiet == 1) {
    printf("All performance values in Mpixels/sec\n\n");
    printf("Pixel     JPEG             %s  %s   Xform   Comp    Decomp  ",
           doTile ? "Tile " : "Image", doTile ? "Tile " : "Image");
    if (doYUV) printf("Decode");
    printf("\n");
    printf("Format    Format           Width  Height  Perf    Ratio   Perf    ");
    if (doYUV) printf("Perf");
    printf("\n\n");
  } else if (!quiet)
    printf(">>>>>  %d-bit JPEG (%s) --> %s (%s)  <<<<<\n", precision,
           formatName(subsamp, cs, tempStr), pixFormatStr[pf],
           bottomUp ? "Bottom-up" : "Top-down");

  if (doTile) {
    if (subsamp == TJSAMP_UNKNOWN)
      THROW("transforming",
            "Could not determine subsampling level of JPEG image");
    minTile = max(tjMCUWidth[subsamp], tjMCUHeight[subsamp]);
  }
  for (tilew = doTile ? minTile : w, tileh = doTile ? minTile : h; ;
       tilew *= 2, tileh *= 2) {
    if (tilew > w) tilew = w;
    if (tileh > h) tileh = h;
    ntilesw = (w + tilew - 1) / tilew;
    ntilesh = (h + tileh - 1) / tileh;

    if ((jpegBufs = (unsigned char **)malloc(sizeof(unsigned char *) *
                                             ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG tile array");
    memset(jpegBufs, 0, sizeof(unsigned char *) * ntilesw * ntilesh);
    if ((jpegSizes = (size_t *)malloc(sizeof(size_t) * ntilesw *
                                      ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG size array");
    memset(jpegSizes, 0, sizeof(size_t) * ntilesw * ntilesh);

    if (noRealloc &&
        (doTile || xformOp != TJXOP_NONE || xformOpt != 0 || customFilter)) {
      for (i = 0; i < ntilesw * ntilesh; i++) {
        size_t jpegBufSize;

        if (xformOp == TJXOP_TRANSPOSE || xformOp == TJXOP_TRANSVERSE ||
            xformOp == TJXOP_ROT90 || xformOp == TJXOP_ROT270)
          jpegBufSize = tj3JPEGBufSize(tileh, tilew, subsamp);
        else
          jpegBufSize = tj3JPEGBufSize(tilew, tileh, subsamp);
        if (jpegBufSize == 0)
          THROW_TJG();
        if ((jpegBufs[i] = tj3Alloc(jpegBufSize)) == NULL)
          THROW_UNIX("allocating JPEG tiles");
      }
    }

    tw = w;  th = h;  ttilew = tilew;  ttileh = tileh;
    if (!quiet) {
      printf("\n%s size: %d x %d", doTile ? "Tile" : "Image", ttilew, ttileh);
      if (sf.num != 1 || sf.denom != 1 || IS_CROPPED(cr))
        printf(" --> %d x %d", CROPPED_WIDTH(tw), CROPPED_HEIGHT(th));
      printf("\n");
    } else if (quiet == 1) {
      printf("%-4s(%s)  %-14s   ", pixFormatStr[pf],
             bottomUp ? "BU" : "TD", formatName(subsamp, cs, tempStr));
      printf("%-5d  %-5d   ", CROPPED_WIDTH(tilew), CROPPED_HEIGHT(tileh));
    }

    tsubsamp = subsamp;
    if (doTile || xformOp != TJXOP_NONE || xformOpt != 0 || customFilter) {
      if ((t = (tjtransform *)malloc(sizeof(tjtransform) * ntilesw *
                                     ntilesh)) == NULL)
        THROW_UNIX("allocating image transform array");

      if (xformOp == TJXOP_TRANSPOSE || xformOp == TJXOP_TRANSVERSE ||
          xformOp == TJXOP_ROT90 || xformOp == TJXOP_ROT270) {
        tw = h;  th = w;  ttilew = tileh;  ttileh = tilew;
      }

      if (xformOp != TJXOP_NONE && xformOp != TJXOP_TRANSPOSE &&
          subsamp == TJSAMP_UNKNOWN)
        THROW("transforming",
              "Could not determine subsampling level of JPEG image");
      if (xformOpt & TJXOPT_GRAY) tsubsamp = TJSAMP_GRAY;
      if (xformOp == TJXOP_HFLIP || xformOp == TJXOP_ROT180)
        tw = tw - (tw % tjMCUWidth[tsubsamp]);
      if (xformOp == TJXOP_VFLIP || xformOp == TJXOP_ROT180)
        th = th - (th % tjMCUHeight[tsubsamp]);
      if (xformOp == TJXOP_TRANSVERSE || xformOp == TJXOP_ROT90)
        tw = tw - (tw % tjMCUHeight[tsubsamp]);
      if (xformOp == TJXOP_TRANSVERSE || xformOp == TJXOP_ROT270)
        th = th - (th % tjMCUWidth[tsubsamp]);
      tntilesw = (tw + ttilew - 1) / ttilew;
      tntilesh = (th + ttileh - 1) / ttileh;

      if (xformOp == TJXOP_TRANSPOSE || xformOp == TJXOP_TRANSVERSE ||
          xformOp == TJXOP_ROT90 || xformOp == TJXOP_ROT270) {
        if (tsubsamp == TJSAMP_422) tsubsamp = TJSAMP_440;
        else if (tsubsamp == TJSAMP_440) tsubsamp = TJSAMP_422;
        else if (tsubsamp == TJSAMP_411) tsubsamp = TJSAMP_441;
        else if (tsubsamp == TJSAMP_441) tsubsamp = TJSAMP_411;
      }

      for (row = 0, tile = 0; row < tntilesh; row++) {
        for (col = 0; col < tntilesw; col++, tile++) {
          t[tile].r.w = min(ttilew, tw - col * ttilew);
          t[tile].r.h = min(ttileh, th - row * ttileh);
          t[tile].r.x = col * ttilew;
          t[tile].r.y = row * ttileh;
          t[tile].op = xformOp;
          t[tile].options = xformOpt | TJXOPT_TRIM;
          t[tile].customFilter = customFilter;
          if (t[tile].options & TJXOPT_NOOUTPUT && jpegBufs[tile]) {
            tj3Free(jpegBufs[tile]);  jpegBufs[tile] = NULL;
          }
        }
      }

      iter = -1;
      elapsed = 0.;
      while (1) {
        start = getTime();
        if (tj3Transform(handle, srcBuf, srcSize, tntilesw * tntilesh,
                         jpegBufs, jpegSizes, t) == -1)
          THROW_TJ();
        elapsed += getTime() - start;
        if (iter >= 0) {
          iter++;
          if (elapsed >= benchTime) break;
        } else if (elapsed >= warmup) {
          iter = 0;
          elapsed = 0.;
        }
      }

      free(t);  t = NULL;

      for (tile = 0, totalJpegSize = 0; tile < tntilesw * tntilesh; tile++)
        totalJpegSize += jpegSizes[tile];

      if (quiet) {
        printf("%-6s%s%-6s%s",
               sigfig((double)(w * h) / 1000000. / elapsed, 4, tempStr, 80),
               quiet == 2 ? "\n" : "  ",
               sigfig((double)(w * h * ps) / (double)totalJpegSize, 4,
                      tempStr2, 80),
               quiet == 2 ? "\n" : "  ");
      } else {
        printf("Transform     --> Frame rate:         %f fps\n",
               1.0 / elapsed);
        printf("                  Output image size:  %lu bytes\n",
               (unsigned long)totalJpegSize);
        printf("                  Compression ratio:  %f:1\n",
               (double)(w * h * ps) / (double)totalJpegSize);
        printf("                  Throughput:         %f Megapixels/sec\n",
               (double)(w * h) / 1000000. / elapsed);
        printf("                  Output bit stream:  %f Megabits/sec\n",
               (double)totalJpegSize * 8. / 1000000. / elapsed);
      }
    } else {
      if (quiet == 1) printf("N/A     N/A     ");
      tj3Free(jpegBufs[0]);
      jpegBufs[0] = NULL;
      decompsrc = 1;
    }

    if (w == tilew) ttilew = tw;
    if (h == tileh) ttileh = th;
    if (!(xformOpt & TJXOPT_NOOUTPUT)) {
      if (decomp(decompsrc ? &srcBuf : jpegBufs,
                 decompsrc ? &srcSize : jpegSizes, NULL, tw, th, tsubsamp, 0,
                 fileName, ttilew, ttileh) == -1)
        goto bailout;
    } else if (quiet == 1) printf("N/A\n");

    for (i = 0; i < ntilesw * ntilesh; i++) {
      tj3Free(jpegBufs[i]);
      jpegBufs[i] = NULL;
    }
    free(jpegBufs);  jpegBufs = NULL;
    free(jpegSizes);  jpegSizes = NULL;

    if (tilew == w && tileh == h) break;
  }

bailout:
  if (file) fclose(file);
  if (jpegBufs) {
    for (i = 0; i < ntilesw * ntilesh; i++)
      tj3Free(jpegBufs[i]);
  }
  free(jpegBufs);
  free(jpegSizes);
  free(srcBuf);
  free(t);
  tj3Destroy(handle);
  return retval;
}


static void usage(char *progName)
{
  int i;

  printf("USAGE: %s\n", progName);
  printf("       <Inputimage (BMP|PPM)> <Quality or PSV> [options]\n\n");
  printf("       %s\n", progName);
  printf("       <Inputimage (JPG)> [options]\n");

  printf("\nGENERAL OPTIONS\n");
  printf("---------------\n");
  printf("-alloc = Dynamically allocate JPEG buffers\n");
  printf("-benchtime T = Run each benchmark for at least T seconds [default = 5.0]\n");
  printf("-bmp = Use Windows Bitmap format for output images [default = PPM]\n");
  printf("     ** 8-bit data precision only **\n");
  printf("-bottomup = Use bottom-up row order for packed-pixel source/destination buffers\n");
  printf("-componly = Stop after running compression tests.  Do not test decompression.\n");
  printf("-lossless = Generate lossless JPEG images when compressing (implies\n");
  printf("     -subsamp 444).  PSV is the predictor selection value (1-7).\n");
  printf("-maxmemory = Memory limit (in megabytes) for intermediate buffers used with\n");
  printf("     progressive JPEG compression and decompression, optimized baseline entropy\n");
  printf("     coding, lossless JPEG compression, and lossless transformation\n");
  printf("     [default = no limit]\n");
  printf("-maxpixels = Input image size limit (in pixels) [default = no limit]\n");
  printf("-nowrite = Do not write reference or output images (improves consistency of\n");
  printf("     benchmark results)\n");
  printf("-rgb, -bgr, -rgbx, -bgrx, -xbgr, -xrgb =\n");
  printf("     Use the specified pixel format for packed-pixel source/destination buffers\n");
  printf("     [default = BGR]\n");
  printf("-cmyk = Indirectly test YCCK JPEG compression/decompression\n");
  printf("     (use the CMYK pixel format for packed-pixel source/destination buffers)\n");
  printf("-precision N = Use N-bit data precision when compressing [N is 8, 12, or 16;\n");
  printf("     default = 8; if N is 16, then -lossless must also be specified]\n");
  printf("     (-precision 12 implies -optimize unless -arithmetic is also specified)\n");
  printf("-quiet = Output results in tabular rather than verbose format\n");
  printf("-restart N = When compressing, add a restart marker every N MCU rows (lossy) or\n");
  printf("     N sample rows (lossless) [default = 0 (no restart markers)].  Append 'B'\n");
  printf("     to specify the restart marker interval in MCU blocks (lossy) or samples\n");
  printf("     (lossless).\n");
  printf("-stoponwarning = Immediately discontinue the current\n");
  printf("     compression/decompression/transform operation if a warning (non-fatal\n");
  printf("     error) occurs\n");
  printf("-tile = Compress/transform the input image into separate JPEG tiles of varying\n");
  printf("     sizes (useful for measuring JPEG overhead)\n");
  printf("-warmup T = Run each benchmark for T seconds [default = 1.0] prior to starting\n");
  printf("     the timer, in order to prime the caches and thus improve the consistency\n");
  printf("     of the benchmark results\n");

  printf("\nLOSSY JPEG OPTIONS\n");
  printf("------------------\n");
  printf("-arithmetic = Use arithmetic entropy coding in JPEG images generated by\n");
  printf("     compression and transform operations (can be combined with -progressive)\n");
  printf("-crop WxH+X+Y = Decompress only the specified region of the JPEG image, where W\n");
  printf("     and H are the width and height of the region (0 = maximum possible width\n");
  printf("     or height) and X and Y are the left and upper boundary of the region, all\n");
  printf("     specified relative to the scaled image dimensions.  X must be divible by\n");
  printf("     the scaled MCU width.\n");
  printf("-fastdct = Use the fastest DCT/IDCT algorithm available\n");
  printf("-fastupsample = Use the fastest chrominance upsampling algorithm available\n");
  printf("-optimize = Use optimized baseline entropy coding in JPEG images generated by\n");
  printf("     compession and transform operations\n");
  printf("-progressive = Use progressive entropy coding in JPEG images generated by\n");
  printf("     compression and transform operations (can be combined with -arithmetic;\n");
  printf("     implies -optimize unless -arithmetic is also specified)\n");
  printf("-limitscans = Refuse to decompress or transform progressive JPEG images that\n");
  printf("     have an unreasonably large number of scans\n");
  printf("-scale M/N = When decompressing, scale the width/height of the JPEG image by a\n");
  printf("     factor of M/N (M/N = ");
  for (i = 0; i < nsf; i++) {
    printf("%d/%d", scalingFactors[i].num, scalingFactors[i].denom);
    if (nsf == 2 && i != nsf - 1) printf(" or ");
    else if (nsf > 2) {
      if (i != nsf - 1) printf(", ");
      if (i == nsf - 2) printf("or ");
    }
    if (i % 8 == 0 && i != 0) printf("\n     ");
  }
  printf(")\n");
  printf("-subsamp S = When compressing, use the specified level of chrominance\n");
  printf("     subsampling (S = 444, 422, 440, 420, 411, 441, or GRAY) [default = test\n");
  printf("     Grayscale, 4:2:0, 4:2:2, and 4:4:4 in sequence]\n");
  printf("-hflip, -vflip, -transpose, -transverse, -rot90, -rot180, -rot270 =\n");
  printf("     Perform the specified lossless transform operation on the input image\n");
  printf("     prior to decompression (these operations are mutually exclusive)\n");
  printf("-grayscale = Transform the input image into a grayscale JPEG image prior to\n");
  printf("     decompression (can be combined with the other transform operations above)\n");
  printf("-copynone = Do not copy any extra markers (including EXIF and ICC profile data)\n");
  printf("     when transforming the input image\n");
  printf("-yuv = Compress from/decompress to intermediate planar YUV images\n");
  printf("     ** 8-bit data precision only **\n");
  printf("-yuvpad N = The number of bytes by which each row in each plane of an\n");
  printf("     intermediate YUV image is evenly divisible (N must be a power of 2)\n");
  printf("     [default = 1]\n");

  printf("\nNOTE:  If the quality/PSV is specified as a range (e.g. 90-100 or 1-4), a\n");
  printf("separate test will be performed for all values in the range.\n\n");
  exit(1);
}


int main(int argc, char *argv[])
{
  void *srcBuf = NULL;
  int w = 0, h = 0, i, j, minQual = -1, maxQual = -1;
  char *temp;
  int minArg = 2, retval = 0, subsamp = -1;
  tjhandle handle = NULL;

  if ((scalingFactors = tj3GetScalingFactors(&nsf)) == NULL || nsf == 0)
    THROW("executing tj3GetScalingFactors()", tj3GetErrorStr(NULL));

  if (argc < minArg) usage(argv[0]);

  temp = strrchr(argv[1], '.');
  if (temp != NULL) {
    if (!strcasecmp(temp, ".bmp")) ext = "bmp";
    if (!strcasecmp(temp, ".jpg") || !strcasecmp(temp, ".jpeg"))
      decompOnly = 1;
  }

  printf("\n");

  if (!decompOnly) {
    minArg = 3;
    if (argc < minArg) usage(argv[0]);
    minQual = atoi(argv[2]);
    if ((temp = strchr(argv[2], '-')) != NULL && strlen(temp) > 1 &&
        sscanf(&temp[1], "%d", &maxQual) == 1 && maxQual > minQual) {}
    else maxQual = minQual;
  }

  if (argc > minArg) {
    for (i = minArg; i < argc; i++) {
      if (!strcasecmp(argv[i], "-tile")) {
        doTile = 1;  xformOpt |= TJXOPT_CROP;
      } else if (!strcasecmp(argv[i], "-precision") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi != 8 && tempi != 12 && tempi != 16)
          usage(argv[0]);
        precision = tempi;
      } else if (!strcasecmp(argv[i], "-fastupsample")) {
        printf("Using fastest upsampling algorithm\n\n");
        fastUpsample = 1;
      } else if (!strcasecmp(argv[i], "-fastdct")) {
        printf("Using fastest DCT/IDCT algorithm\n\n");
        fastDCT = 1;
      } else if (!strcasecmp(argv[i], "-optimize")) {
        printf("Using optimized baseline entropy coding\n\n");
        optimize = 1;
        xformOpt |= TJXOPT_OPTIMIZE;
      } else if (!strcasecmp(argv[i], "-progressive")) {
        printf("Using progressive entropy coding\n\n");
        progressive = 1;
        xformOpt |= TJXOPT_PROGRESSIVE;
      } else if (!strcasecmp(argv[i], "-arithmetic")) {
        printf("Using arithmetic entropy coding\n\n");
        arithmetic = 1;
        xformOpt |= TJXOPT_ARITHMETIC;
      } else if (!strcasecmp(argv[i], "-lossless")) {
        lossless = 1;
        subsamp = TJSAMP_444;
      } else if (!strcasecmp(argv[i], "-rgb"))
        pf = TJPF_RGB;
      else if (!strcasecmp(argv[i], "-rgbx"))
        pf = TJPF_RGBX;
      else if (!strcasecmp(argv[i], "-bgr"))
        pf = TJPF_BGR;
      else if (!strcasecmp(argv[i], "-bgrx"))
        pf = TJPF_BGRX;
      else if (!strcasecmp(argv[i], "-xbgr"))
        pf = TJPF_XBGR;
      else if (!strcasecmp(argv[i], "-xrgb"))
        pf = TJPF_XRGB;
      else if (!strcasecmp(argv[i], "-cmyk"))
        pf = TJPF_CMYK;
      else if (!strcasecmp(argv[i], "-bottomup"))
        bottomUp = 1;
      else if (!strcasecmp(argv[i], "-quiet"))
        quiet = 1;
      else if (!strcasecmp(argv[i], "-qq"))
        quiet = 2;
      else if (!strcasecmp(argv[i], "-scale") && i < argc - 1) {
        int temp1 = 0, temp2 = 0, match = 0;

        if (sscanf(argv[++i], "%d/%d", &temp1, &temp2) == 2) {
          for (j = 0; j < nsf; j++) {
            if (temp1 == scalingFactors[j].num &&
                temp2 == scalingFactors[j].denom) {
              sf = scalingFactors[j];
              match = 1;  break;
            }
          }
          if (!match) usage(argv[0]);
        } else usage(argv[0]);
      } else if (!strcasecmp(argv[i], "-crop") && i < argc - 1) {
        int temp1 = -1, temp2 = -1, temp3 = -1, temp4 = -1;

        if (sscanf(argv[++i], "%dx%d+%d+%d", &temp1, &temp2, &temp3,
                   &temp4) == 4 && temp1 >= 0 && temp2 >= 0 && temp3 >= 0 &&
                   temp4 >= 0) {
          cr.w = temp1;  cr.h = temp2;  cr.x = temp3;  cr.y = temp4;
        } else usage(argv[0]);
      } else if (!strcasecmp(argv[i], "-hflip"))
        xformOp = TJXOP_HFLIP;
      else if (!strcasecmp(argv[i], "-vflip"))
        xformOp = TJXOP_VFLIP;
      else if (!strcasecmp(argv[i], "-transpose"))
        xformOp = TJXOP_TRANSPOSE;
      else if (!strcasecmp(argv[i], "-transverse"))
        xformOp = TJXOP_TRANSVERSE;
      else if (!strcasecmp(argv[i], "-rot90"))
        xformOp = TJXOP_ROT90;
      else if (!strcasecmp(argv[i], "-rot180"))
        xformOp = TJXOP_ROT180;
      else if (!strcasecmp(argv[i], "-rot270"))
        xformOp = TJXOP_ROT270;
      else if (!strcasecmp(argv[i], "-grayscale"))
        xformOpt |= TJXOPT_GRAY;
      else if (!strcasecmp(argv[i], "-custom"))
        customFilter = dummyDCTFilter;
      else if (!strcasecmp(argv[i], "-nooutput"))
        xformOpt |= TJXOPT_NOOUTPUT;
      else if (!strcasecmp(argv[i], "-copynone"))
        xformOpt |= TJXOPT_COPYNONE;
      else if (!strcasecmp(argv[i], "-benchtime") && i < argc - 1) {
        double tempd = atof(argv[++i]);

        if (tempd > 0.0) benchTime = tempd;
        else usage(argv[0]);
      } else if (!strcasecmp(argv[i], "-warmup") && i < argc - 1) {
        double tempd = atof(argv[++i]);

        if (tempd >= 0.0) warmup = tempd;
        else usage(argv[0]);
        printf("Warmup time = %.1f seconds\n\n", warmup);
      } else if (!strcasecmp(argv[i], "-alloc"))
        noRealloc = 0;
      else if (!strcasecmp(argv[i], "-bmp"))
        ext = "bmp";
      else if (!strcasecmp(argv[i], "-yuv")) {
        printf("Testing planar YUV encoding/decoding\n\n");
        doYUV = 1;
      } else if (!strcasecmp(argv[i], "-yuvpad") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi >= 1 && (tempi & (tempi - 1)) == 0) yuvAlign = tempi;
        else usage(argv[0]);
      } else if (!strcasecmp(argv[i], "-subsamp") && i < argc - 1) {
        i++;
        if (toupper(argv[i][0]) == 'G') subsamp = TJSAMP_GRAY;
        else {
          int tempi = atoi(argv[i]);

          switch (tempi) {
          case 444:  subsamp = TJSAMP_444;  break;
          case 422:  subsamp = TJSAMP_422;  break;
          case 440:  subsamp = TJSAMP_440;  break;
          case 420:  subsamp = TJSAMP_420;  break;
          case 411:  subsamp = TJSAMP_411;  break;
          case 441:  subsamp = TJSAMP_441;  break;
          default:  usage(argv[0]);
          }
        }
      } else if (!strcasecmp(argv[i], "-componly"))
        compOnly = 1;
      else if (!strcasecmp(argv[i], "-nowrite"))
        doWrite = 0;
      else if (!strcasecmp(argv[i], "-limitscans"))
        limitScans = 1;
      else if (!strcasecmp(argv[i], "-maxmemory") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi < 0) usage(argv[0]);
        maxMemory = tempi;
      } else if (!strcasecmp(argv[i], "-maxpixels") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi < 0) usage(argv[0]);
        maxPixels = tempi;
      } else if (!strcasecmp(argv[i], "-restart") && i < argc - 1) {
        int tempi = -1, nscan;  char tempc = 0;

        if ((nscan = sscanf(argv[++i], "%d%c", &tempi, &tempc)) < 1 ||
            tempi < 0 || tempi > 65535 ||
            (nscan == 2 && tempc != 'B' && tempc != 'b'))
          usage(argv[0]);

        if (tempc == 'B' || tempc == 'b')
          restartIntervalBlocks = tempi;
        else
          restartIntervalRows = tempi;
      } else if (!strcasecmp(argv[i], "-stoponwarning"))
        stopOnWarning = 1;
      else usage(argv[0]);
    }
  }

  if (precision == 16 && !lossless) {
    printf("ERROR: -lossless must be specified along with -precision 16\n");
    retval = -1;  goto bailout;
  }
  if (precision != 8 && doYUV) {
    printf("ERROR: -yuv requires 8-bit data precision\n");
    retval = -1;  goto bailout;
  }
  if (lossless && doYUV) {
    printf("ERROR: -lossless and -yuv are incompatible\n");
    retval = -1;  goto bailout;
  }
  sampleSize = (precision == 8 ? sizeof(unsigned char) : sizeof(short));

  if ((sf.num != 1 || sf.denom != 1) && doTile) {
    printf("Disabling tiled compression/decompression tests, because those tests do not\n");
    printf("work when scaled decompression is enabled.\n\n");
    doTile = 0;  xformOpt &= (~TJXOPT_CROP);
  }

  if (IS_CROPPED(cr)) {
    if (!decompOnly) {
      printf("ERROR: Partial image decompression can only be enabled for JPEG input images\n");
      retval = -1;  goto bailout;
    }
    if (doTile) {
      printf("Disabling tiled compression/decompression tests, because those tests do not\n");
      printf("work when partial image decompression is enabled.\n\n");
      doTile = 0;  xformOpt &= (~TJXOPT_CROP);
    }
    if (doYUV) {
      printf("ERROR: -crop and -yuv are incompatible\n");
      retval = -1;  goto bailout;
    }
  }

  if (!noRealloc && doTile) {
    printf("Disabling tiled compression/decompression tests, because those tests do not\n");
    printf("work when dynamic JPEG buffer allocation is enabled.\n\n");
    doTile = 0;  xformOpt &= (~TJXOPT_CROP);
  }

  if (!decompOnly) {
    if ((handle = tj3Init(TJINIT_COMPRESS)) == NULL)
      THROW_TJG();
    if (tj3Set(handle, TJPARAM_STOPONWARNING, stopOnWarning) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_BOTTOMUP, bottomUp) == -1)
      THROW_TJ();
    if (tj3Set(handle, TJPARAM_MAXPIXELS, maxPixels) == -1)
      THROW_TJ();

    if (precision == 8) {
      if ((srcBuf = tj3LoadImage8(handle, argv[1], &w, 1, &h, &pf)) == NULL)
        THROW_TJ();
    } else if (precision == 12) {
      if ((srcBuf = tj3LoadImage12(handle, argv[1], &w, 1, &h, &pf)) == NULL)
        THROW_TJ();
    } else {
      if ((srcBuf = tj3LoadImage16(handle, argv[1], &w, 1, &h, &pf)) == NULL)
        THROW_TJ();
    }
    temp = strrchr(argv[1], '.');
    if (temp != NULL) *temp = '\0';
  }

  if (quiet == 1 && !decompOnly) {
    printf("All performance values in Mpixels/sec\n\n");
    printf("Pixel     JPEG      JPEG  %s  %s   ",
           doTile ? "Tile " : "Image", doTile ? "Tile " : "Image");
    if (doYUV) printf("Encode  ");
    printf("Comp    Comp    Decomp  ");
    if (doYUV) printf("Decode");
    printf("\n");
    printf("Format    Format    %s  Width  Height  ",
           lossless ? "PSV " : "Qual");
    if (doYUV) printf("Perf    ");
    printf("Perf    Ratio   Perf    ");
    if (doYUV) printf("Perf");
    printf("\n\n");
  }

  if (decompOnly) {
    decompTest(argv[1]);
    printf("\n");
    goto bailout;
  }
  if (lossless) {
    if (minQual < 1 || minQual > 7 || maxQual < 1 || maxQual > 7) {
      puts("ERROR: PSV must be between 1 and 7.");
      exit(1);
    }
  } else {
    if (minQual < 1 || minQual > 100 || maxQual < 1 || maxQual > 100) {
      puts("ERROR: Quality must be between 1 and 100.");
      exit(1);
    }
  }
  if (subsamp >= 0 && subsamp < TJ_NUMSAMP) {
    for (i = maxQual; i >= minQual; i--)
      fullTest(handle, srcBuf, w, h, subsamp, i, argv[1]);
    printf("\n");
  } else {
    if (pf != TJPF_CMYK) {
      for (i = maxQual; i >= minQual; i--)
        fullTest(handle, srcBuf, w, h, TJSAMP_GRAY, i, argv[1]);
      printf("\n");
    }
    for (i = maxQual; i >= minQual; i--)
      fullTest(handle, srcBuf, w, h, TJSAMP_420, i, argv[1]);
    printf("\n");
    for (i = maxQual; i >= minQual; i--)
      fullTest(handle, srcBuf, w, h, TJSAMP_422, i, argv[1]);
    printf("\n");
    for (i = maxQual; i >= minQual; i--)
      fullTest(handle, srcBuf, w, h, TJSAMP_444, i, argv[1]);
    printf("\n");
  }

bailout:
  tj3Destroy(handle);
  tj3Free(srcBuf);
  return retval;
}
