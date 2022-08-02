/*
 * Copyright (C)2009-2019, 2021-2022 D. R. Commander.  All Rights Reserved.
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
#include <cdjpeg.h>
#include "./tjutil.h"
#include "./turbojpeg.h"


#define THROW(op, err) { \
  printf("ERROR in line %d while %s:\n%s\n", __LINE__, op, err); \
  retval = -1;  goto bailout; \
}
#define THROW_UNIX(m)  THROW(m, strerror(errno))

char tjErrorStr[JMSG_LENGTH_MAX] = "\0", tjErrorMsg[JMSG_LENGTH_MAX] = "\0";
int tjErrorLine = -1, tjErrorCode = -1;

#define THROW_TJG(m) { \
  printf("ERROR in line %d while %s:\n%s\n", __LINE__, m, \
         tjGetErrorStr2(NULL)); \
  retval = -1;  goto bailout; \
}

#define THROW_TJ(m) { \
  int _tjErrorCode = tjGetErrorCode(handle); \
  char *_tjErrorStr = tjGetErrorStr2(handle); \
  \
  if (!(flags & TJFLAG_STOPONWARNING) && _tjErrorCode == TJERR_WARNING) { \
    if (strncmp(tjErrorStr, _tjErrorStr, JMSG_LENGTH_MAX) || \
        strncmp(tjErrorMsg, m, JMSG_LENGTH_MAX) || \
        tjErrorCode != _tjErrorCode || tjErrorLine != __LINE__) { \
      strncpy(tjErrorStr, _tjErrorStr, JMSG_LENGTH_MAX); \
      tjErrorStr[JMSG_LENGTH_MAX - 1] = '\0'; \
      strncpy(tjErrorMsg, m, JMSG_LENGTH_MAX); \
      tjErrorMsg[JMSG_LENGTH_MAX - 1] = '\0'; \
      tjErrorCode = _tjErrorCode; \
      tjErrorLine = __LINE__; \
      printf("WARNING in line %d while %s:\n%s\n", __LINE__, m, _tjErrorStr); \
    } \
  } else { \
    printf("%s in line %d while %s:\n%s\n", \
           _tjErrorCode == TJERR_WARNING ? "WARNING" : "ERROR", __LINE__, m, \
           _tjErrorStr); \
    retval = -1;  goto bailout; \
  } \
}

int flags = TJFLAG_NOREALLOC, compOnly = 0, decompOnly = 0, doYUV = 0,
  quiet = 0, doTile = 0, pf = TJPF_BGR, yuvPad = 1, doWrite = 1;
char *ext = "ppm";
const char *pixFormatStr[TJ_NUMPF] = {
  "RGB", "BGR", "RGBX", "BGRX", "XBGR", "XRGB", "GRAY", "", "", "", "", "CMYK"
};
const char *subNameLong[TJ_NUMSAMP] = {
  "4:4:4", "4:2:2", "4:2:0", "GRAY", "4:4:0", "4:1:1"
};
const char *csName[TJ_NUMCS] = {
  "RGB", "YCbCr", "GRAY", "CMYK", "YCCK"
};
const char *subName[TJ_NUMSAMP] = {
  "444", "422", "420", "GRAY", "440", "411"
};
tjscalingfactor *scalingFactors = NULL, sf = { 1, 1 };
int nsf = 0, xformOp = TJXOP_NONE, xformOpt = 0;
int (*customFilter) (short *, tjregion, tjregion, int, int, tjtransform *);
double benchTime = 5.0, warmup = 1.0;


static char *formatName(int subsamp, int cs, char *buf)
{
  if (cs == TJCS_YCbCr)
    return (char *)subNameLong[subsamp];
  else if (cs == TJCS_YCCK || cs == TJCS_CMYK) {
    SNPRINTF(buf, 80, "%s %s", csName[cs], subNameLong[subsamp]);
    return buf;
  } else
    return (char *)csName[cs];
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
static int decomp(unsigned char *srcBuf, unsigned char **jpegBuf,
                  unsigned long *jpegSize, unsigned char *dstBuf, int w, int h,
                  int subsamp, int jpegQual, char *fileName, int tilew,
                  int tileh)
{
  char tempStr[1024], sizeStr[24] = "\0", qualStr[13] = "\0", *ptr;
  FILE *file = NULL;
  tjhandle handle = NULL;
  int row, col, iter = 0, dstBufAlloc = 0, retval = 0;
  double elapsed, elapsedDecode;
  int ps = tjPixelSize[pf];
  int scaledw = TJSCALED(w, sf);
  int scaledh = TJSCALED(h, sf);
  int pitch = scaledw * ps;
  int ntilesw = (w + tilew - 1) / tilew, ntilesh = (h + tileh - 1) / tileh;
  unsigned char *dstPtr, *dstPtr2, *yuvBuf = NULL;

  if (jpegQual > 0) {
    SNPRINTF(qualStr, 13, "_Q%d", jpegQual);
    qualStr[12] = 0;
  }

  if ((handle = tjInitDecompress()) == NULL)
    THROW_TJ("executing tjInitDecompress()");

  if (dstBuf == NULL) {
    if ((unsigned long long)pitch * (unsigned long long)scaledh >
        (unsigned long long)((size_t)-1))
      THROW("allocating destination buffer", "Image is too large");
    if ((dstBuf = (unsigned char *)malloc((size_t)pitch * scaledh)) == NULL)
      THROW_UNIX("allocating destination buffer");
    dstBufAlloc = 1;
  }
  /* Set the destination buffer to gray so we know whether the decompressor
     attempted to write to it */
  memset(dstBuf, 127, (size_t)pitch * scaledh);

  if (doYUV) {
    int width = doTile ? tilew : scaledw;
    int height = doTile ? tileh : scaledh;
    unsigned long yuvSize = tjBufSizeYUV2(width, yuvPad, height, subsamp);

    if (yuvSize == (unsigned long)-1)
      THROW_TJ("allocating YUV buffer");
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
         row++, dstPtr += (size_t)pitch * tileh) {
      for (col = 0, dstPtr2 = dstPtr; col < ntilesw;
           col++, tile++, dstPtr2 += ps * tilew) {
        int width = doTile ? min(tilew, w - col * tilew) : scaledw;
        int height = doTile ? min(tileh, h - row * tileh) : scaledh;

        if (doYUV) {
          double startDecode;

          if (tjDecompressToYUV2(handle, jpegBuf[tile], jpegSize[tile], yuvBuf,
                                 width, yuvPad, height, flags) == -1)
            THROW_TJ("executing tjDecompressToYUV2()");
          startDecode = getTime();
          if (tjDecodeYUV(handle, yuvBuf, yuvPad, subsamp, dstPtr2, width,
                          pitch, height, pf, flags) == -1)
            THROW_TJ("executing tjDecodeYUV()");
          if (iter >= 0) elapsedDecode += getTime() - startDecode;
        } else if (tjDecompress2(handle, jpegBuf[tile], jpegSize[tile],
                                 dstPtr2, width, pitch, height, pf,
                                 flags) == -1)
          THROW_TJ("executing tjDecompress2()");
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

  if (tjDestroy(handle) == -1) THROW_TJ("executing tjDestroy()");
  handle = NULL;

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
    SNPRINTF(tempStr, 1024, "%s_%s%s_%s.%s", fileName, subName[subsamp],
             qualStr, sizeStr, ext);

  if (tjSaveImage(tempStr, dstBuf, scaledw, 0, scaledh, pf, flags) == -1)
    THROW_TJG("saving bitmap");
  ptr = strrchr(tempStr, '.');
  SNPRINTF(ptr, 1024 - (ptr - tempStr), "-err.%s", ext);
  if (srcBuf && sf.num == 1 && sf.denom == 1) {
    if (!quiet) printf("Compression error written to %s.\n", tempStr);
    if (subsamp == TJ_GRAYSCALE) {
      unsigned long index, index2;

      for (row = 0, index = 0; row < h; row++, index += pitch) {
        for (col = 0, index2 = index; col < w; col++, index2 += ps) {
          unsigned long rindex = index2 + tjRedOffset[pf];
          unsigned long gindex = index2 + tjGreenOffset[pf];
          unsigned long bindex = index2 + tjBlueOffset[pf];
          int y = (int)((double)srcBuf[rindex] * 0.299 +
                        (double)srcBuf[gindex] * 0.587 +
                        (double)srcBuf[bindex] * 0.114 + 0.5);

          if (y > 255) y = 255;
          if (y < 0) y = 0;
          dstBuf[rindex] = (unsigned char)abs(dstBuf[rindex] - y);
          dstBuf[gindex] = (unsigned char)abs(dstBuf[gindex] - y);
          dstBuf[bindex] = (unsigned char)abs(dstBuf[bindex] - y);
        }
      }
    } else {
      for (row = 0; row < h; row++)
        for (col = 0; col < w * ps; col++)
          dstBuf[pitch * row + col] =
            (unsigned char)abs(dstBuf[pitch * row + col] -
                               srcBuf[pitch * row + col]);
    }
    if (tjSaveImage(tempStr, dstBuf, w, 0, h, pf, flags) == -1)
      THROW_TJG("saving bitmap");
  }

bailout:
  if (file) fclose(file);
  if (handle) tjDestroy(handle);
  if (dstBufAlloc) free(dstBuf);
  free(yuvBuf);
  return retval;
}


static int fullTest(unsigned char *srcBuf, int w, int h, int subsamp,
                    int jpegQual, char *fileName)
{
  char tempStr[1024], tempStr2[80];
  FILE *file = NULL;
  tjhandle handle = NULL;
  unsigned char **jpegBuf = NULL, *yuvBuf = NULL, *tmpBuf = NULL, *srcPtr,
    *srcPtr2;
  double start, elapsed, elapsedEncode;
  int totalJpegSize = 0, row, col, i, tilew = w, tileh = h, retval = 0;
  int iter;
  unsigned long *jpegSize = NULL, yuvSize = 0;
  int ps = tjPixelSize[pf];
  int ntilesw = 1, ntilesh = 1, pitch = w * ps;
  const char *pfStr = pixFormatStr[pf];

  if ((unsigned long long)pitch * (unsigned long long)h >
      (unsigned long long)((size_t)-1))
    THROW("allocating temporary image buffer", "Image is too large");
  if ((tmpBuf = (unsigned char *)malloc((size_t)pitch * h)) == NULL)
    THROW_UNIX("allocating temporary image buffer");

  if (!quiet)
    printf(">>>>>  %s (%s) <--> JPEG %s Q%d  <<<<<\n", pfStr,
           (flags & TJFLAG_BOTTOMUP) ? "Bottom-up" : "Top-down",
           subNameLong[subsamp], jpegQual);

  for (tilew = doTile ? 8 : w, tileh = doTile ? 8 : h; ;
       tilew *= 2, tileh *= 2) {
    if (tilew > w) tilew = w;
    if (tileh > h) tileh = h;
    ntilesw = (w + tilew - 1) / tilew;
    ntilesh = (h + tileh - 1) / tileh;

    if ((jpegBuf = (unsigned char **)malloc(sizeof(unsigned char *) *
                                            ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG tile array");
    memset(jpegBuf, 0, sizeof(unsigned char *) * ntilesw * ntilesh);
    if ((jpegSize = (unsigned long *)malloc(sizeof(unsigned long) *
                                            ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG size array");
    memset(jpegSize, 0, sizeof(unsigned long) * ntilesw * ntilesh);

    if ((flags & TJFLAG_NOREALLOC) != 0)
      for (i = 0; i < ntilesw * ntilesh; i++) {
        if (tjBufSize(tilew, tileh, subsamp) > (unsigned long)INT_MAX)
          THROW("getting buffer size", "Image is too large");
        if ((jpegBuf[i] = (unsigned char *)
                          tjAlloc(tjBufSize(tilew, tileh, subsamp))) == NULL)
          THROW_UNIX("allocating JPEG tiles");
      }

    /* Compression test */
    if (quiet == 1)
      printf("%-4s (%s)  %-5s    %-3d   ", pfStr,
             (flags & TJFLAG_BOTTOMUP) ? "BU" : "TD", subNameLong[subsamp],
             jpegQual);
    for (i = 0; i < h; i++)
      memcpy(&tmpBuf[pitch * i], &srcBuf[w * ps * i], w * ps);
    if ((handle = tjInitCompress()) == NULL)
      THROW_TJ("executing tjInitCompress()");

    if (doYUV) {
      yuvSize = tjBufSizeYUV2(tilew, yuvPad, tileh, subsamp);
      if (yuvSize == (unsigned long)-1)
        THROW_TJ("allocating YUV buffer");
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
           row++, srcPtr += pitch * tileh) {
        for (col = 0, srcPtr2 = srcPtr; col < ntilesw;
             col++, tile++, srcPtr2 += ps * tilew) {
          int width = min(tilew, w - col * tilew);
          int height = min(tileh, h - row * tileh);

          if (doYUV) {
            double startEncode = getTime();

            if (tjEncodeYUV3(handle, srcPtr2, width, pitch, height, pf, yuvBuf,
                             yuvPad, subsamp, flags) == -1)
              THROW_TJ("executing tjEncodeYUV3()");
            if (iter >= 0) elapsedEncode += getTime() - startEncode;
            if (tjCompressFromYUV(handle, yuvBuf, width, yuvPad, height,
                                  subsamp, &jpegBuf[tile], &jpegSize[tile],
                                  jpegQual, flags) == -1)
              THROW_TJ("executing tjCompressFromYUV()");
          } else {
            if (tjCompress2(handle, srcPtr2, width, pitch, height, pf,
                            &jpegBuf[tile], &jpegSize[tile], subsamp, jpegQual,
                            flags) == -1)
              THROW_TJ("executing tjCompress2()");
          }
          totalJpegSize += jpegSize[tile];
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

    if (tjDestroy(handle) == -1) THROW_TJ("executing tjDestroy()");
    handle = NULL;

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
        printf("                  Output image size:  %lu bytes\n", yuvSize);
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
      printf("                  Output image size:  %d bytes\n",
             totalJpegSize);
      printf("                  Compression ratio:  %f:1\n",
             (double)(w * h * ps) / (double)totalJpegSize);
      printf("                  Throughput:         %f Megapixels/sec\n",
             (double)(w * h) / 1000000. * (double)iter / elapsed);
      printf("                  Output bit stream:  %f Megabits/sec\n",
             (double)totalJpegSize * 8. / 1000000. * (double)iter / elapsed);
    }
    if (tilew == w && tileh == h && doWrite) {
      SNPRINTF(tempStr, 1024, "%s_%s_Q%d.jpg", fileName, subName[subsamp],
               jpegQual);
      if ((file = fopen(tempStr, "wb")) == NULL)
        THROW_UNIX("opening reference image");
      if (fwrite(jpegBuf[0], jpegSize[0], 1, file) != 1)
        THROW_UNIX("writing reference image");
      fclose(file);  file = NULL;
      if (!quiet) printf("Reference image written to %s\n", tempStr);
    }

    /* Decompression test */
    if (!compOnly) {
      if (decomp(srcBuf, jpegBuf, jpegSize, tmpBuf, w, h, subsamp, jpegQual,
                 fileName, tilew, tileh) == -1)
        goto bailout;
    } else if (quiet == 1) printf("N/A\n");

    for (i = 0; i < ntilesw * ntilesh; i++) {
      tjFree(jpegBuf[i]);
      jpegBuf[i] = NULL;
    }
    free(jpegBuf);  jpegBuf = NULL;
    free(jpegSize);  jpegSize = NULL;
    if (doYUV) {
      free(yuvBuf);  yuvBuf = NULL;
    }

    if (tilew == w && tileh == h) break;
  }

bailout:
  if (file) fclose(file);
  if (jpegBuf) {
    for (i = 0; i < ntilesw * ntilesh; i++)
      tjFree(jpegBuf[i]);
  }
  free(jpegBuf);
  free(yuvBuf);
  free(jpegSize);
  free(tmpBuf);
  if (handle) tjDestroy(handle);
  return retval;
}


static int decompTest(char *fileName)
{
  FILE *file = NULL;
  tjhandle handle = NULL;
  unsigned char **jpegBuf = NULL, *srcBuf = NULL;
  unsigned long *jpegSize = NULL, srcSize, totalJpegSize;
  tjtransform *t = NULL;
  double start, elapsed;
  int ps = tjPixelSize[pf], tile, row, col, i, iter, retval = 0, decompsrc = 0;
  char *temp = NULL, tempStr[80], tempStr2[80];
  /* Original image */
  int w = 0, h = 0, tilew, tileh, ntilesw = 1, ntilesh = 1, subsamp = -1,
    cs = -1;
  /* Transformed image */
  int tw, th, ttilew, ttileh, tntilesw, tntilesh, tsubsamp;

  if ((file = fopen(fileName, "rb")) == NULL)
    THROW_UNIX("opening file");
  if (fseek(file, 0, SEEK_END) < 0 ||
      (srcSize = ftell(file)) == (unsigned long)-1)
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

  if ((handle = tjInitTransform()) == NULL)
    THROW_TJ("executing tjInitTransform()");
  if (tjDecompressHeader3(handle, srcBuf, srcSize, &w, &h, &subsamp,
                          &cs) == -1)
    THROW_TJ("executing tjDecompressHeader3()");
  if (w < 1 || h < 1)
    THROW("reading JPEG header", "Invalid image dimensions");
  if (cs == TJCS_YCCK || cs == TJCS_CMYK) {
    pf = TJPF_CMYK;  ps = tjPixelSize[pf];
  }

  if (quiet == 1) {
    printf("All performance values in Mpixels/sec\n\n");
    printf("Bitmap     JPEG   JPEG     %s  %s   Xform   Comp    Decomp  ",
           doTile ? "Tile " : "Image", doTile ? "Tile " : "Image");
    if (doYUV) printf("Decode");
    printf("\n");
    printf("Format     CS     Subsamp  Width  Height  Perf    Ratio   Perf    ");
    if (doYUV) printf("Perf");
    printf("\n\n");
  } else if (!quiet)
    printf(">>>>>  JPEG %s --> %s (%s)  <<<<<\n",
           formatName(subsamp, cs, tempStr), pixFormatStr[pf],
           (flags & TJFLAG_BOTTOMUP) ? "Bottom-up" : "Top-down");

  for (tilew = doTile ? 16 : w, tileh = doTile ? 16 : h; ;
       tilew *= 2, tileh *= 2) {
    if (tilew > w) tilew = w;
    if (tileh > h) tileh = h;
    ntilesw = (w + tilew - 1) / tilew;
    ntilesh = (h + tileh - 1) / tileh;

    if ((jpegBuf = (unsigned char **)malloc(sizeof(unsigned char *) *
                                            ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG tile array");
    memset(jpegBuf, 0, sizeof(unsigned char *) * ntilesw * ntilesh);
    if ((jpegSize = (unsigned long *)malloc(sizeof(unsigned long) *
                                            ntilesw * ntilesh)) == NULL)
      THROW_UNIX("allocating JPEG size array");
    memset(jpegSize, 0, sizeof(unsigned long) * ntilesw * ntilesh);

    if ((flags & TJFLAG_NOREALLOC) != 0 &&
        (doTile || xformOp != TJXOP_NONE || xformOpt != 0 || customFilter))
      for (i = 0; i < ntilesw * ntilesh; i++) {
        if (tjBufSize(tilew, tileh, subsamp) > (unsigned long)INT_MAX)
          THROW("getting buffer size", "Image is too large");
        if ((jpegBuf[i] = (unsigned char *)
                          tjAlloc(tjBufSize(tilew, tileh, subsamp))) == NULL)
          THROW_UNIX("allocating JPEG tiles");
      }

    tw = w;  th = h;  ttilew = tilew;  ttileh = tileh;
    if (!quiet) {
      printf("\n%s size: %d x %d", doTile ? "Tile" : "Image", ttilew, ttileh);
      if (sf.num != 1 || sf.denom != 1)
        printf(" --> %d x %d", TJSCALED(tw, sf), TJSCALED(th, sf));
      printf("\n");
    } else if (quiet == 1) {
      printf("%-4s (%s)  %-5s  %-5s    ", pixFormatStr[pf],
             (flags & TJFLAG_BOTTOMUP) ? "BU" : "TD", csName[cs],
             subNameLong[subsamp]);
      printf("%-5d  %-5d   ", tilew, tileh);
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

      if (xformOpt & TJXOPT_GRAY) tsubsamp = TJ_GRAYSCALE;
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
          if (t[tile].options & TJXOPT_NOOUTPUT && jpegBuf[tile]) {
            tjFree(jpegBuf[tile]);  jpegBuf[tile] = NULL;
          }
        }
      }

      iter = -1;
      elapsed = 0.;
      while (1) {
        start = getTime();
        if (tjTransform(handle, srcBuf, srcSize, tntilesw * tntilesh, jpegBuf,
                        jpegSize, t, flags) == -1)
          THROW_TJ("executing tjTransform()");
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
        totalJpegSize += jpegSize[tile];

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
               totalJpegSize);
        printf("                  Compression ratio:  %f:1\n",
               (double)(w * h * ps) / (double)totalJpegSize);
        printf("                  Throughput:         %f Megapixels/sec\n",
               (double)(w * h) / 1000000. / elapsed);
        printf("                  Output bit stream:  %f Megabits/sec\n",
               (double)totalJpegSize * 8. / 1000000. / elapsed);
      }
    } else {
      if (quiet == 1) printf("N/A     N/A     ");
      tjFree(jpegBuf[0]);
      jpegBuf[0] = NULL;
      decompsrc = 1;
    }

    if (w == tilew) ttilew = tw;
    if (h == tileh) ttileh = th;
    if (!(xformOpt & TJXOPT_NOOUTPUT)) {
      if (decomp(NULL, decompsrc ? &srcBuf : jpegBuf,
                 decompsrc ? &srcSize : jpegSize, NULL, tw, th, tsubsamp, 0,
                 fileName, ttilew, ttileh) == -1)
        goto bailout;
    } else if (quiet == 1) printf("N/A\n");

    for (i = 0; i < ntilesw * ntilesh; i++) {
      tjFree(jpegBuf[i]);
      jpegBuf[i] = NULL;
    }
    free(jpegBuf);  jpegBuf = NULL;
    free(jpegSize);  jpegSize = NULL;

    if (tilew == w && tileh == h) break;
  }

bailout:
  if (file) fclose(file);
  if (jpegBuf) {
    for (i = 0; i < ntilesw * ntilesh; i++)
      tjFree(jpegBuf[i]);
  }
  free(jpegBuf);
  free(jpegSize);
  free(srcBuf);
  free(t);
  if (handle) { tjDestroy(handle);  handle = NULL; }
  return retval;
}


static void usage(char *progName)
{
  int i;

  printf("USAGE: %s\n", progName);
  printf("       <Inputfile (BMP|PPM)> <Quality> [options]\n\n");
  printf("       %s\n", progName);
  printf("       <Inputfile (JPG)> [options]\n\n");
  printf("Options:\n\n");
  printf("-alloc = Dynamically allocate JPEG image buffers\n");
  printf("-bmp = Generate output images in Windows Bitmap format (default = PPM)\n");
  printf("-bottomup = Test bottom-up compression/decompression\n");
  printf("-tile = Test performance of the codec when the image is encoded as separate\n");
  printf("     tiles of varying sizes.\n");
  printf("-rgb, -bgr, -rgbx, -bgrx, -xbgr, -xrgb =\n");
  printf("     Test the specified color conversion path in the codec (default = BGR)\n");
  printf("-cmyk = Indirectly test YCCK JPEG compression/decompression (the source\n");
  printf("     and destination bitmaps are still RGB.  The conversion is done\n");
  printf("     internally prior to compression or after decompression.)\n");
  printf("-fastupsample = Use the fastest chrominance upsampling algorithm available in\n");
  printf("     the underlying codec\n");
  printf("-fastdct = Use the fastest DCT/IDCT algorithms available in the underlying\n");
  printf("     codec\n");
  printf("-accuratedct = Use the most accurate DCT/IDCT algorithms available in the\n");
  printf("     underlying codec\n");
  printf("-progressive = Use progressive entropy coding in JPEG images generated by\n");
  printf("     compression and transform operations.\n");
  printf("-subsamp <s> = When testing JPEG compression, this option specifies the level\n");
  printf("     of chrominance subsampling to use (<s> = 444, 422, 440, 420, 411, or\n");
  printf("     GRAY).  The default is to test Grayscale, 4:2:0, 4:2:2, and 4:4:4 in\n");
  printf("     sequence.\n");
  printf("-quiet = Output results in tabular rather than verbose format\n");
  printf("-yuv = Test YUV encoding/decoding functions\n");
  printf("-yuvpad <p> = If testing YUV encoding/decoding, this specifies the number of\n");
  printf("     bytes to which each row of each plane in the intermediate YUV image is\n");
  printf("     padded (default = 1)\n");
  printf("-scale M/N = Scale down the width/height of the decompressed JPEG image by a\n");
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
  printf("-hflip, -vflip, -transpose, -transverse, -rot90, -rot180, -rot270 =\n");
  printf("     Perform the corresponding lossless transform prior to\n");
  printf("     decompression (these options are mutually exclusive)\n");
  printf("-grayscale = Perform lossless grayscale conversion prior to decompression\n");
  printf("     test (can be combined with the other transforms above)\n");
  printf("-copynone = Do not copy any extra markers (including EXIF and ICC profile data)\n");
  printf("     when transforming the image.\n");
  printf("-benchtime <t> = Run each benchmark for at least <t> seconds (default = 5.0)\n");
  printf("-warmup <t> = Run each benchmark for <t> seconds (default = 1.0) prior to\n");
  printf("     starting the timer, in order to prime the caches and thus improve the\n");
  printf("     consistency of the results.\n");
  printf("-componly = Stop after running compression tests.  Do not test decompression.\n");
  printf("-nowrite = Do not write reference or output images (improves consistency of\n");
  printf("     performance measurements.)\n");
  printf("-limitscans = Refuse to decompress or transform progressive JPEG images that\n");
  printf("     have an unreasonably large number of scans\n");
  printf("-stoponwarning = Immediately discontinue the current\n");
  printf("     compression/decompression/transform operation if the underlying codec\n");
  printf("     throws a warning (non-fatal error)\n\n");
  printf("NOTE:  If the quality is specified as a range (e.g. 90-100), a separate\n");
  printf("test will be performed for all quality values in the range.\n\n");
  exit(1);
}


int main(int argc, char *argv[])
{
  unsigned char *srcBuf = NULL;
  int w = 0, h = 0, i, j, minQual = -1, maxQual = -1;
  char *temp;
  int minArg = 2, retval = 0, subsamp = -1;

  if ((scalingFactors = tjGetScalingFactors(&nsf)) == NULL || nsf == 0)
    THROW("executing tjGetScalingFactors()", tjGetErrorStr());

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
    if ((minQual = atoi(argv[2])) < 1 || minQual > 100) {
      puts("ERROR: Quality must be between 1 and 100.");
      exit(1);
    }
    if ((temp = strchr(argv[2], '-')) != NULL && strlen(temp) > 1 &&
        sscanf(&temp[1], "%d", &maxQual) == 1 && maxQual > minQual &&
        maxQual >= 1 && maxQual <= 100) {}
    else maxQual = minQual;
  }

  if (argc > minArg) {
    for (i = minArg; i < argc; i++) {
      if (!strcasecmp(argv[i], "-tile")) {
        doTile = 1;  xformOpt |= TJXOPT_CROP;
      } else if (!strcasecmp(argv[i], "-fastupsample")) {
        printf("Using fast upsampling code\n\n");
        flags |= TJFLAG_FASTUPSAMPLE;
      } else if (!strcasecmp(argv[i], "-fastdct")) {
        printf("Using fastest DCT/IDCT algorithm\n\n");
        flags |= TJFLAG_FASTDCT;
      } else if (!strcasecmp(argv[i], "-accuratedct")) {
        printf("Using most accurate DCT/IDCT algorithm\n\n");
        flags |= TJFLAG_ACCURATEDCT;
      } else if (!strcasecmp(argv[i], "-progressive")) {
        printf("Using progressive entropy coding\n\n");
        flags |= TJFLAG_PROGRESSIVE;
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
        flags |= TJFLAG_BOTTOMUP;
      else if (!strcasecmp(argv[i], "-quiet"))
        quiet = 1;
      else if (!strcasecmp(argv[i], "-qq"))
        quiet = 2;
      else if (!strcasecmp(argv[i], "-scale") && i < argc - 1) {
        int temp1 = 0, temp2 = 0, match = 0;

        if (sscanf(argv[++i], "%d/%d", &temp1, &temp2) == 2) {
          for (j = 0; j < nsf; j++) {
            if ((double)temp1 / (double)temp2 ==
                (double)scalingFactors[j].num /
                (double)scalingFactors[j].denom) {
              sf = scalingFactors[j];
              match = 1;  break;
            }
          }
          if (!match) usage(argv[0]);
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
        flags &= (~TJFLAG_NOREALLOC);
      else if (!strcasecmp(argv[i], "-bmp"))
        ext = "bmp";
      else if (!strcasecmp(argv[i], "-yuv")) {
        printf("Testing YUV planar encoding/decoding\n\n");
        doYUV = 1;
      } else if (!strcasecmp(argv[i], "-yuvpad") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi >= 1) yuvPad = tempi;
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
          }
        }
      } else if (!strcasecmp(argv[i], "-componly"))
        compOnly = 1;
      else if (!strcasecmp(argv[i], "-nowrite"))
        doWrite = 0;
      else if (!strcasecmp(argv[i], "-limitscans"))
        flags |= TJFLAG_LIMITSCANS;
      else if (!strcasecmp(argv[i], "-stoponwarning"))
        flags |= TJFLAG_STOPONWARNING;
      else usage(argv[0]);
    }
  }

  if ((sf.num != 1 || sf.denom != 1) && doTile) {
    printf("Disabling tiled compression/decompression tests, because those tests do not\n");
    printf("work when scaled decompression is enabled.\n");
    doTile = 0;
  }

  if ((flags & TJFLAG_NOREALLOC) == 0 && doTile) {
    printf("Disabling tiled compression/decompression tests, because those tests do not\n");
    printf("work when dynamic JPEG buffer allocation is enabled.\n\n");
    doTile = 0;
  }

  if (!decompOnly) {
    if ((srcBuf = tjLoadImage(argv[1], &w, 1, &h, &pf, flags)) == NULL)
      THROW_TJG("loading bitmap");
    temp = strrchr(argv[1], '.');
    if (temp != NULL) *temp = '\0';
  }

  if (quiet == 1 && !decompOnly) {
    printf("All performance values in Mpixels/sec\n\n");
    printf("Bitmap     JPEG     JPEG  %s  %s   ",
           doTile ? "Tile " : "Image", doTile ? "Tile " : "Image");
    if (doYUV) printf("Encode  ");
    printf("Comp    Comp    Decomp  ");
    if (doYUV) printf("Decode");
    printf("\n");
    printf("Format     Subsamp  Qual  Width  Height  ");
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
  if (subsamp >= 0 && subsamp < TJ_NUMSAMP) {
    for (i = maxQual; i >= minQual; i--)
      fullTest(srcBuf, w, h, subsamp, i, argv[1]);
    printf("\n");
  } else {
    if (pf != TJPF_CMYK) {
      for (i = maxQual; i >= minQual; i--)
        fullTest(srcBuf, w, h, TJSAMP_GRAY, i, argv[1]);
      printf("\n");
    }
    for (i = maxQual; i >= minQual; i--)
      fullTest(srcBuf, w, h, TJSAMP_420, i, argv[1]);
    printf("\n");
    for (i = maxQual; i >= minQual; i--)
      fullTest(srcBuf, w, h, TJSAMP_422, i, argv[1]);
    printf("\n");
    for (i = maxQual; i >= minQual; i--)
      fullTest(srcBuf, w, h, TJSAMP_444, i, argv[1]);
    printf("\n");
  }

bailout:
  tjFree(srcBuf);
  return retval;
}
