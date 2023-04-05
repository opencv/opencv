/* -*- C++ -*-
 * File: aahd_demosaic.cpp
 * Copyright 2013 Anton Petrusevich
 * Created: Wed May  15, 2013
 *
 * This code is licensed under one of two licenses as you choose:
 *
 * 1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
 *    (See file LICENSE.LGPL provided in LibRaw distribution archive for
 * details).
 *
 * 2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
 *    (See file LICENSE.CDDL provided in LibRaw distribution archive for
 * details).
 *
 */

#include "../../internal/dmp_include.h"

typedef ushort ushort3[3];
typedef int int3[3];

#ifndef Pnw
#define Pnw (-1 - nr_width)
#define Pn (-nr_width)
#define Pne (+1 - nr_width)
#define Pe (+1)
#define Pse (+1 + nr_width)
#define Ps (+nr_width)
#define Psw (-1 + nr_width)
#define Pw (-1)
#endif

struct AAHD
{
  int nr_height, nr_width;
  static const int nr_margin = 4;
  static const int Thot = 4;
  static const int Tdead = 4;
  static const int OverFraction = 8;
  ushort3 *rgb_ahd[2];
  int3 *yuv[2];
  char *ndir, *homo[2];
  ushort channel_maximum[3], channels_max;
  ushort channel_minimum[3];
  static const float yuv_coeff[3][3];
  static float gammaLUT[0x10000];
  float yuv_cam[3][3];
  LibRaw &libraw;
  enum
  {
    HVSH = 1,
    HOR = 2,
    VER = 4,
    HORSH = HOR | HVSH,
    VERSH = VER | HVSH,
    HOT = 8
  };

  static inline float calc_dist(int c1, int c2) throw()
  {
    return c1 > c2 ? (float)c1 / c2 : (float)c2 / c1;
  }
  int inline Y(ushort3 &rgb) throw()
  {
    return yuv_cam[0][0] * rgb[0] + yuv_cam[0][1] * rgb[1] +
           yuv_cam[0][2] * rgb[2];
  }
  int inline U(ushort3 &rgb) throw()
  {
    return yuv_cam[1][0] * rgb[0] + yuv_cam[1][1] * rgb[1] +
           yuv_cam[1][2] * rgb[2];
  }
  int inline V(ushort3 &rgb) throw()
  {
    return yuv_cam[2][0] * rgb[0] + yuv_cam[2][1] * rgb[1] +
           yuv_cam[2][2] * rgb[2];
  }
  inline int nr_offset(int row, int col) throw()
  {
    return (row * nr_width + col);
  }
  ~AAHD();
  AAHD(LibRaw &_libraw);
  void make_ahd_greens();
  void make_ahd_gline(int i);
  void make_ahd_rb();
  void make_ahd_rb_hv(int i);
  void make_ahd_rb_last(int i);
  void evaluate_ahd();
  void combine_image();
  void hide_hots();
  void refine_hv_dirs();
  void refine_hv_dirs(int i, int js);
  void refine_ihv_dirs(int i);
  void illustrate_dirs();
  void illustrate_dline(int i);
};

const float AAHD::yuv_coeff[3][3] = {
    // YPbPr
    //	{
    //		0.299f,
    //		0.587f,
    //		0.114f },
    //	{
    //		-0.168736,
    //		-0.331264f,
    //		0.5f },
    //	{
    //		0.5f,
    //		-0.418688f,
    //		-0.081312f }
    //
    //	Rec. 2020
    //	Y'= 0,2627R' + 0,6780G' + 0,0593B'
    //	U = (B-Y)/1.8814 =  (-0,2627R' - 0,6780G' + 0.9407B) / 1.8814 =
    //-0.13963R - 0.36037G + 0.5B
    //	V = (R-Y)/1.4647 = (0.7373R - 0,6780G - 0,0593B) / 1.4647 = 0.5R -
    //0.4629G - 0.04049B
    {+0.2627f, +0.6780f, +0.0593f},
    {-0.13963f, -0.36037f, +0.5f},
    {+0.5034f, -0.4629f, -0.0405f}

};

float AAHD::gammaLUT[0x10000] = {-1.f};

AAHD::AAHD(LibRaw &_libraw) : libraw(_libraw)
{
  nr_height = libraw.imgdata.sizes.iheight + nr_margin * 2;
  nr_width = libraw.imgdata.sizes.iwidth + nr_margin * 2;
  rgb_ahd[0] = (ushort3 *)calloc(nr_height * nr_width,
                                 (sizeof(ushort3) * 2 + sizeof(int3) * 2 + 3));
  if (!rgb_ahd[0])
    throw LIBRAW_EXCEPTION_ALLOC;

  rgb_ahd[1] = rgb_ahd[0] + nr_height * nr_width;
  yuv[0] = (int3 *)(rgb_ahd[1] + nr_height * nr_width);
  yuv[1] = yuv[0] + nr_height * nr_width;
  ndir = (char *)(yuv[1] + nr_height * nr_width);
  homo[0] = ndir + nr_height * nr_width;
  homo[1] = homo[0] + nr_height * nr_width;
  channel_maximum[0] = channel_maximum[1] = channel_maximum[2] = 0;
  channel_minimum[0] = libraw.imgdata.image[0][0];
  channel_minimum[1] = libraw.imgdata.image[0][1];
  channel_minimum[2] = libraw.imgdata.image[0][2];
  int iwidth = libraw.imgdata.sizes.iwidth;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      yuv_cam[i][j] = 0;
      for (int k = 0; k < 3; ++k)
        yuv_cam[i][j] += yuv_coeff[i][k] * libraw.imgdata.color.rgb_cam[k][j];
    }
  if (gammaLUT[0] < -0.1f)
  {
    float r;
    for (int i = 0; i < 0x10000; i++)
    {
      r = (float)i / 0x10000;
      gammaLUT[i] =
          0x10000 * (r < 0.0181 ? 4.5f * r : 1.0993f * pow(r, 0.45f) - .0993f);
    }
  }
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    int col_cache[48];
    for (int j = 0; j < 48; ++j)
    {
      int c = libraw.COLOR(i, j);
      if (c == 3)
        c = 1;
      col_cache[j] = c;
    }
    int moff = nr_offset(i + nr_margin, nr_margin);
    for (int j = 0; j < iwidth; ++j, ++moff)
    {
      int c = col_cache[j % 48];
      unsigned short d = libraw.imgdata.image[i * iwidth + j][c];
      if (d != 0)
      {
        if (channel_maximum[c] < d)
          channel_maximum[c] = d;
        if (channel_minimum[c] > d)
          channel_minimum[c] = d;
        rgb_ahd[1][moff][c] = rgb_ahd[0][moff][c] = d;
      }
    }
  }
  channels_max =
      MAX(MAX(channel_maximum[0], channel_maximum[1]), channel_maximum[2]);
}

void AAHD::hide_hots()
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    int js = libraw.COLOR(i, 0) & 1;
    int kc = libraw.COLOR(i, js);
    /*
     * js -- начальная х-координата, которая попадает мимо известного зелёного
     * kc -- известный цвет в точке интерполирования
     */
    int moff = nr_offset(i + nr_margin, nr_margin + js);
    for (int j = js; j < iwidth; j += 2, moff += 2)
    {
      ushort3 *rgb = &rgb_ahd[0][moff];
      int c = rgb[0][kc];
      if ((c > rgb[2 * Pe][kc] && c > rgb[2 * Pw][kc] && c > rgb[2 * Pn][kc] &&
           c > rgb[2 * Ps][kc] && c > rgb[Pe][1] && c > rgb[Pw][1] &&
           c > rgb[Pn][1] && c > rgb[Ps][1]) ||
          (c < rgb[2 * Pe][kc] && c < rgb[2 * Pw][kc] && c < rgb[2 * Pn][kc] &&
           c < rgb[2 * Ps][kc] && c < rgb[Pe][1] && c < rgb[Pw][1] &&
           c < rgb[Pn][1] && c < rgb[Ps][1]))
      {
        int chot = c >> Thot;
        int cdead = c << Tdead;
        int avg = 0;
        for (int k = -2; k < 3; k += 2)
          for (int m = -2; m < 3; m += 2)
            if (m == 0 && k == 0)
              continue;
            else
              avg += rgb[nr_offset(k, m)][kc];
        avg /= 8;
        if (chot > avg || cdead < avg)
        {
          ndir[moff] |= HOT;
          int dh =
              ABS(rgb[2 * Pw][kc] - rgb[2 * Pe][kc]) +
              ABS(rgb[Pw][1] - rgb[Pe][1]) +
              ABS(rgb[Pw][1] - rgb[Pe][1] + rgb[2 * Pe][kc] - rgb[2 * Pw][kc]);
          int dv =
              ABS(rgb[2 * Pn][kc] - rgb[2 * Ps][kc]) +
              ABS(rgb[Pn][1] - rgb[Ps][1]) +
              ABS(rgb[Pn][1] - rgb[Ps][1] + rgb[2 * Ps][kc] - rgb[2 * Pn][kc]);
          int d;
          if (dv > dh)
            d = Pw;
          else
            d = Pn;
          rgb_ahd[1][moff][kc] = rgb[0][kc] =
              (rgb[+2 * d][kc] + rgb[-2 * d][kc]) / 2;
        }
      }
    }
    js ^= 1;
    moff = nr_offset(i + nr_margin, nr_margin + js);
    for (int j = js; j < iwidth; j += 2, moff += 2)
    {
      ushort3 *rgb = &rgb_ahd[0][moff];
      int c = rgb[0][1];
      if ((c > rgb[2 * Pe][1] && c > rgb[2 * Pw][1] && c > rgb[2 * Pn][1] &&
           c > rgb[2 * Ps][1] && c > rgb[Pe][kc] && c > rgb[Pw][kc] &&
           c > rgb[Pn][kc ^ 2] && c > rgb[Ps][kc ^ 2]) ||
          (c < rgb[2 * Pe][1] && c < rgb[2 * Pw][1] && c < rgb[2 * Pn][1] &&
           c < rgb[2 * Ps][1] && c < rgb[Pe][kc] && c < rgb[Pw][kc] &&
           c < rgb[Pn][kc ^ 2] && c < rgb[Ps][kc ^ 2]))
      {
        int chot = c >> Thot;
        int cdead = c << Tdead;
        int avg = 0;
        for (int k = -2; k < 3; k += 2)
          for (int m = -2; m < 3; m += 2)
            if (k == 0 && m == 0)
              continue;
            else
              avg += rgb[nr_offset(k, m)][1];
        avg /= 8;
        if (chot > avg || cdead < avg)
        {
          ndir[moff] |= HOT;
          int dh =
              ABS(rgb[2 * Pw][1] - rgb[2 * Pe][1]) +
              ABS(rgb[Pw][kc] - rgb[Pe][kc]) +
              ABS(rgb[Pw][kc] - rgb[Pe][kc] + rgb[2 * Pe][1] - rgb[2 * Pw][1]);
          int dv = ABS(rgb[2 * Pn][1] - rgb[2 * Ps][1]) +
                   ABS(rgb[Pn][kc ^ 2] - rgb[Ps][kc ^ 2]) +
                   ABS(rgb[Pn][kc ^ 2] - rgb[Ps][kc ^ 2] + rgb[2 * Ps][1] -
                       rgb[2 * Pn][1]);
          int d;
          if (dv > dh)
            d = Pw;
          else
            d = Pn;
          rgb_ahd[1][moff][1] = rgb[0][1] =
              (rgb[+2 * d][1] + rgb[-2 * d][1]) / 2;
        }
      }
    }
  }
}

void AAHD::evaluate_ahd()
{
  int hvdir[4] = {Pw, Pe, Pn, Ps};
  /*
   * YUV
   *
   */
  for (int d = 0; d < 2; ++d)
  {
    for (int i = 0; i < nr_width * nr_height; ++i)
    {
      ushort3 rgb;
      for (int c = 0; c < 3; ++c)
      {
        rgb[c] = gammaLUT[rgb_ahd[d][i][c]];
      }
      yuv[d][i][0] = Y(rgb);
      yuv[d][i][1] = U(rgb);
      yuv[d][i][2] = V(rgb);
    }
  }
  /* */
  /*
   * Lab
   *
   float r, cbrt[0x10000], xyz[3], xyz_cam[3][4];
   for (int i = 0; i < 0x10000; i++) {
   r = i / 65535.0;
   cbrt[i] = r > 0.008856 ? pow((double) r, (double) (1 / 3.0)) : 7.787 * r + 16
   / 116.0;
   }
   for (int i = 0; i < 3; i++)
   for (int j = 0; j < 3; j++) {
   xyz_cam[i][j] = 0;
   for (int k = 0; k < 3; k++)
   xyz_cam[i][j] += xyz_rgb[i][k] * libraw.imgdata.color.rgb_cam[k][j] /
   d65_white[i];
   }
   for (int d = 0; d < 2; ++d)
   for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i) {
   int moff = nr_offset(i + nr_margin, nr_margin);
   for (int j = 0; j < libraw.imgdata.sizes.iwidth; j++, ++moff) {
   xyz[0] = xyz[1] = xyz[2] = 0.5;
   for (int c = 0; c < 3; c++) {
   xyz[0] += xyz_cam[0][c] * rgb_ahd[d][moff][c];
   xyz[1] += xyz_cam[1][c] * rgb_ahd[d][moff][c];
   xyz[2] += xyz_cam[2][c] * rgb_ahd[d][moff][c];
   }
   xyz[0] = cbrt[CLIP((int) xyz[0])];
   xyz[1] = cbrt[CLIP((int) xyz[1])];
   xyz[2] = cbrt[CLIP((int) xyz[2])];
   yuv[d][moff][0] = 64 * (116 * xyz[1] - 16);
   yuv[d][moff][1] = 64 * 500 * (xyz[0] - xyz[1]);
   yuv[d][moff][2] = 64 * 200 * (xyz[1] - xyz[2]);
   }
   }
   * Lab */
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    int moff = nr_offset(i + nr_margin, nr_margin);
    for (int j = 0; j < libraw.imgdata.sizes.iwidth; j++, ++moff)
    {
      int3 *ynr;
      float ydiff[2][4];
      int uvdiff[2][4];
      for (int d = 0; d < 2; ++d)
      {
        ynr = &yuv[d][moff];
        for (int k = 0; k < 4; k++)
        {
          ydiff[d][k] = ABS(ynr[0][0] - ynr[hvdir[k]][0]);
          uvdiff[d][k] = SQR(ynr[0][1] - ynr[hvdir[k]][1]) +
                         SQR(ynr[0][2] - ynr[hvdir[k]][2]);
        }
      }
      float yeps =
          MIN(MAX(ydiff[0][0], ydiff[0][1]), MAX(ydiff[1][2], ydiff[1][3]));
      int uveps =
          MIN(MAX(uvdiff[0][0], uvdiff[0][1]), MAX(uvdiff[1][2], uvdiff[1][3]));
      for (int d = 0; d < 2; d++)
      {
        ynr = &yuv[d][moff];
        for (int k = 0; k < 4; k++)
          if (ydiff[d][k] <= yeps && uvdiff[d][k] <= uveps)
          {
            homo[d][moff + hvdir[k]]++;
            if (k / 2 == d)
            {
              // если в сонаправленном направлении интеполяции следующие точки
              // так же гомогенны, учтём их тоже
              for (int m = 2; m < 4; ++m)
              {
                int hvd = m * hvdir[k];
                if (ABS(ynr[0][0] - ynr[hvd][0]) < yeps &&
                    SQR(ynr[0][1] - ynr[hvd][1]) +
                            SQR(ynr[0][2] - ynr[hvd][2]) <
                        uveps)
                {
                  homo[d][moff + hvd]++;
                }
                else
                  break;
              }
            }
          }
      }
    }
  }
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    int moff = nr_offset(i + nr_margin, nr_margin);
    for (int j = 0; j < libraw.imgdata.sizes.iwidth; j++, ++moff)
    {
      char hm[2];
      for (int d = 0; d < 2; d++)
      {
        hm[d] = 0;
        char *hh = &homo[d][moff];
        for (int hx = -1; hx < 2; hx++)
          for (int hy = -1; hy < 2; hy++)
            hm[d] += hh[nr_offset(hy, hx)];
      }
      char d = 0;
      if (hm[0] != hm[1])
      {
        if (hm[1] > hm[0])
        {
          d = VERSH;
        }
        else
        {
          d = HORSH;
        }
      }
      else
      {
        int3 *ynr = &yuv[1][moff];
        int gv = SQR(2 * ynr[0][0] - ynr[Pn][0] - ynr[Ps][0]);
        gv += SQR(2 * ynr[0][1] - ynr[Pn][1] - ynr[Ps][1]) +
              SQR(2 * ynr[0][2] - ynr[Pn][2] - ynr[Ps][2]);
        ynr = &yuv[1][moff + Pn];
        gv += (SQR(2 * ynr[0][0] - ynr[Pn][0] - ynr[Ps][0]) +
               SQR(2 * ynr[0][1] - ynr[Pn][1] - ynr[Ps][1]) +
               SQR(2 * ynr[0][2] - ynr[Pn][2] - ynr[Ps][2])) /
              2;
        ynr = &yuv[1][moff + Ps];
        gv += (SQR(2 * ynr[0][0] - ynr[Pn][0] - ynr[Ps][0]) +
               SQR(2 * ynr[0][1] - ynr[Pn][1] - ynr[Ps][1]) +
               SQR(2 * ynr[0][2] - ynr[Pn][2] - ynr[Ps][2])) /
              2;
        ynr = &yuv[0][moff];
        int gh = SQR(2 * ynr[0][0] - ynr[Pw][0] - ynr[Pe][0]);
        gh += SQR(2 * ynr[0][1] - ynr[Pw][1] - ynr[Pe][1]) +
              SQR(2 * ynr[0][2] - ynr[Pw][2] - ynr[Pe][2]);
        ynr = &yuv[0][moff + Pw];
        gh += (SQR(2 * ynr[0][0] - ynr[Pw][0] - ynr[Pe][0]) +
               SQR(2 * ynr[0][1] - ynr[Pw][1] - ynr[Pe][1]) +
               SQR(2 * ynr[0][2] - ynr[Pw][2] - ynr[Pe][2])) /
              2;
        ynr = &yuv[0][moff + Pe];
        gh += (SQR(2 * ynr[0][0] - ynr[Pw][0] - ynr[Pe][0]) +
               SQR(2 * ynr[0][1] - ynr[Pw][1] - ynr[Pe][1]) +
               SQR(2 * ynr[0][2] - ynr[Pw][2] - ynr[Pe][2])) /
              2;
        if (gv > gh)
          d = HOR;
        else
          d = VER;
      }
      ndir[moff] |= d;
    }
  }
}

void AAHD::combine_image()
{
  for (int i = 0, i_out = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    int moff = nr_offset(i + nr_margin, nr_margin);
    for (int j = 0; j < libraw.imgdata.sizes.iwidth; j++, ++moff, ++i_out)
    {
      if (ndir[moff] & HOT)
      {
        int c = libraw.COLOR(i, j);
        rgb_ahd[1][moff][c] = rgb_ahd[0][moff][c] =
            libraw.imgdata.image[i_out][c];
      }
      if (ndir[moff] & VER)
      {
        libraw.imgdata.image[i_out][0] = rgb_ahd[1][moff][0];
        libraw.imgdata.image[i_out][3] = libraw.imgdata.image[i_out][1] =
            rgb_ahd[1][moff][1];
        libraw.imgdata.image[i_out][2] = rgb_ahd[1][moff][2];
      }
      else
      {
        libraw.imgdata.image[i_out][0] = rgb_ahd[0][moff][0];
        libraw.imgdata.image[i_out][3] = libraw.imgdata.image[i_out][1] =
            rgb_ahd[0][moff][1];
        libraw.imgdata.image[i_out][2] = rgb_ahd[0][moff][2];
      }
    }
  }
}

void AAHD::refine_hv_dirs()
{
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    refine_hv_dirs(i, i & 1);
  }
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    refine_hv_dirs(i, (i & 1) ^ 1);
  }
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    refine_ihv_dirs(i);
  }
}

void AAHD::refine_ihv_dirs(int i)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  int moff = nr_offset(i + nr_margin, nr_margin);
  for (int j = 0; j < iwidth; j++, ++moff)
  {
    if (ndir[moff] & HVSH)
      continue;
    int nv = (ndir[moff + Pn] & VER) + (ndir[moff + Ps] & VER) +
             (ndir[moff + Pw] & VER) + (ndir[moff + Pe] & VER);
    int nh = (ndir[moff + Pn] & HOR) + (ndir[moff + Ps] & HOR) +
             (ndir[moff + Pw] & HOR) + (ndir[moff + Pe] & HOR);
    nv /= VER;
    nh /= HOR;
    if ((ndir[moff] & VER) && nh > 3)
    {
      ndir[moff] &= ~VER;
      ndir[moff] |= HOR;
    }
    if ((ndir[moff] & HOR) && nv > 3)
    {
      ndir[moff] &= ~HOR;
      ndir[moff] |= VER;
    }
  }
}

void AAHD::refine_hv_dirs(int i, int js)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  int moff = nr_offset(i + nr_margin, nr_margin + js);
  for (int j = js; j < iwidth; j += 2, moff += 2)
  {
    int nv = (ndir[moff + Pn] & VER) + (ndir[moff + Ps] & VER) +
             (ndir[moff + Pw] & VER) + (ndir[moff + Pe] & VER);
    int nh = (ndir[moff + Pn] & HOR) + (ndir[moff + Ps] & HOR) +
             (ndir[moff + Pw] & HOR) + (ndir[moff + Pe] & HOR);
    bool codir = (ndir[moff] & VER)
                     ? ((ndir[moff + Pn] & VER) || (ndir[moff + Ps] & VER))
                     : ((ndir[moff + Pw] & HOR) || (ndir[moff + Pe] & HOR));
    nv /= VER;
    nh /= HOR;
    if ((ndir[moff] & VER) && (nh > 2 && !codir))
    {
      ndir[moff] &= ~VER;
      ndir[moff] |= HOR;
    }
    if ((ndir[moff] & HOR) && (nv > 2 && !codir))
    {
      ndir[moff] &= ~HOR;
      ndir[moff] |= VER;
    }
  }
}

/*
 * вычисление недостающих зелёных точек.
 */
void AAHD::make_ahd_greens()
{
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    make_ahd_gline(i);
  }
}

void AAHD::make_ahd_gline(int i)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  int js = libraw.COLOR(i, 0) & 1;
  int kc = libraw.COLOR(i, js);
  /*
   * js -- начальная х-координата, которая попадает мимо известного зелёного
   * kc -- известный цвет в точке интерполирования
   */
  int hvdir[2] = {Pe, Ps};
  for (int d = 0; d < 2; ++d)
  {
    int moff = nr_offset(i + nr_margin, nr_margin + js);
    for (int j = js; j < iwidth; j += 2, moff += 2)
    {
      ushort3 *cnr;
      cnr = &rgb_ahd[d][moff];
      int h1 = 2 * cnr[-hvdir[d]][1] - int(cnr[-2 * hvdir[d]][kc] + cnr[0][kc]);
      int h2 = 2 * cnr[+hvdir[d]][1] - int(cnr[+2 * hvdir[d]][kc] + cnr[0][kc]);
      int h0 = (h1 + h2) / 4;
      int eg = cnr[0][kc] + h0;
      int min = MIN(cnr[-hvdir[d]][1], cnr[+hvdir[d]][1]);
      int max = MAX(cnr[-hvdir[d]][1], cnr[+hvdir[d]][1]);
      min -= min / OverFraction;
      max += max / OverFraction;
      if (eg < min)
        eg = min - sqrt(float(min - eg));
      else if (eg > max)
        eg = max + sqrt(float(eg - max));
      if (eg > channel_maximum[1])
        eg = channel_maximum[1];
      else if (eg < channel_minimum[1])
        eg = channel_minimum[1];
      cnr[0][1] = eg;
    }
  }
}

/*
 * отладочная функция
 */

void AAHD::illustrate_dirs()
{
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    illustrate_dline(i);
  }
}

void AAHD::illustrate_dline(int i)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  for (int j = 0; j < iwidth; j++)
  {
    int x = j + nr_margin;
    int y = i + nr_margin;
    rgb_ahd[1][nr_offset(y, x)][0] = rgb_ahd[1][nr_offset(y, x)][1] =
        rgb_ahd[1][nr_offset(y, x)][2] = rgb_ahd[0][nr_offset(y, x)][0] =
            rgb_ahd[0][nr_offset(y, x)][1] = rgb_ahd[0][nr_offset(y, x)][2] = 0;
    int l = ndir[nr_offset(y, x)] & HVSH;
    l /= HVSH;
    if (ndir[nr_offset(y, x)] & VER)
      rgb_ahd[1][nr_offset(y, x)][0] =
          l * channel_maximum[0] / 4 + channel_maximum[0] / 4;
    else
      rgb_ahd[0][nr_offset(y, x)][2] =
          l * channel_maximum[2] / 4 + channel_maximum[2] / 4;
  }
}

void AAHD::make_ahd_rb_hv(int i)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  int js = libraw.COLOR(i, 0) & 1;
  int kc = libraw.COLOR(i, js);
  js ^= 1; // начальная координата зелёного
  int hvdir[2] = {Pe, Ps};
  // интерполяция вертикальных вертикально и горизонтальных горизонтально
  for (int j = js; j < iwidth; j += 2)
  {
    int x = j + nr_margin;
    int y = i + nr_margin;
    int moff = nr_offset(y, x);
    for (int d = 0; d < 2; ++d)
    {
      ushort3 *cnr;
      cnr = &rgb_ahd[d][moff];
      int c = kc ^ (d << 1); // цвет соответсвенного направления, для
                             // горизонтального c = kc, для вертикального c=kc^2
      int h1 = cnr[-hvdir[d]][c] - cnr[-hvdir[d]][1];
      int h2 = cnr[+hvdir[d]][c] - cnr[+hvdir[d]][1];
      int h0 = (h1 + h2) / 2;
      int eg = cnr[0][1] + h0;
      //			int min = MIN(cnr[-hvdir[d]][c], cnr[+hvdir[d]][c]);
      //			int max = MAX(cnr[-hvdir[d]][c], cnr[+hvdir[d]][c]);
      //			min -= min / OverFraction;
      //			max += max / OverFraction;
      //			if (eg < min)
      //				eg = min - sqrt(min - eg);
      //			else if (eg > max)
      //				eg = max + sqrt(eg - max);
      if (eg > channel_maximum[c])
        eg = channel_maximum[c];
      else if (eg < channel_minimum[c])
        eg = channel_minimum[c];
      cnr[0][c] = eg;
    }
  }
}

void AAHD::make_ahd_rb()
{
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    make_ahd_rb_hv(i);
  }
  for (int i = 0; i < libraw.imgdata.sizes.iheight; ++i)
  {
    make_ahd_rb_last(i);
  }
}

void AAHD::make_ahd_rb_last(int i)
{
  int iwidth = libraw.imgdata.sizes.iwidth;
  int js = libraw.COLOR(i, 0) & 1;
  int kc = libraw.COLOR(i, js);
  /*
   * js -- начальная х-координата, которая попадает мимо известного зелёного
   * kc -- известный цвет в точке интерполирования
   */
  int dirs[2][3] = {{Pnw, Pn, Pne}, {Pnw, Pw, Psw}};
  int moff = nr_offset(i + nr_margin, nr_margin);
  for (int j = 0; j < iwidth; j++)
  {
    for (int d = 0; d < 2; ++d)
    {
      ushort3 *cnr;
      cnr = &rgb_ahd[d][moff + j];
      int c = kc ^ 2;
      if ((j & 1) != js)
      {
        // точка зелёного, для вертикального направления нужен альтернативный
        // строчному цвет
        c ^= d << 1;
      }
      int bh = 0, bk = 0;
      int bgd = 0;
      for (int k = 0; k < 3; ++k)
        for (int h = 0; h < 3; ++h)
        {
          // градиент зелёного плюс градиент {r,b}
          int gd =
              ABS(2 * cnr[0][1] - (cnr[+dirs[d][k]][1] + cnr[-dirs[d][h]][1])) +
              ABS(cnr[+dirs[d][k]][c] - cnr[-dirs[d][h]][c]) / 4 +
              ABS(cnr[+dirs[d][k]][c] - cnr[+dirs[d][k]][1] +
                  cnr[-dirs[d][h]][1] - cnr[-dirs[d][h]][c]) /
                  4;
          if (bgd == 0 || gd < bgd)
          {
            bgd = gd;
            bh = h;
            bk = k;
          }
        }
      int h1 = cnr[+dirs[d][bk]][c] - cnr[+dirs[d][bk]][1];
      int h2 = cnr[-dirs[d][bh]][c] - cnr[-dirs[d][bh]][1];
      int eg = cnr[0][1] + (h1 + h2) / 2;
      //			int min = MIN(cnr[+dirs[d][bk]][c], cnr[-dirs[d][bh]][c]);
      //			int max = MAX(cnr[+dirs[d][bk]][c], cnr[-dirs[d][bh]][c]);
      //			min -= min / OverFraction;
      //			max += max / OverFraction;
      //			if (eg < min)
      //				eg = min - sqrt(min - eg);
      //			else if (eg > max)
      //				eg = max + sqrt(eg - max);
      if (eg > channel_maximum[c])
        eg = channel_maximum[c];
      else if (eg < channel_minimum[c])
        eg = channel_minimum[c];
      cnr[0][c] = eg;
    }
  }
}

AAHD::~AAHD() { free(rgb_ahd[0]); }

void LibRaw::aahd_interpolate()
{
  AAHD aahd(*this);
  aahd.hide_hots();
  aahd.make_ahd_greens();
  aahd.make_ahd_rb();
  aahd.evaluate_ahd();
  aahd.refine_hv_dirs();
  //	aahd.illustrate_dirs();
  aahd.combine_image();
}
