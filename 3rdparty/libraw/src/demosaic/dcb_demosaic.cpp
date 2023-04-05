/*
 *    Copyright (C) 2010,  Jacek Gozdz (cuniek@kft.umcs.lublin.pl)
 *
 *    This code is licensed under a (3-clause) BSD license as follows :
 *
 *    Redistribution and use in source and binary forms, with or without
 *    modification, are permitted provided that the following
 *	  conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *		disclaimer in the documentation and/or other materials provided
 * 	    with the distribution.
 *    * Neither the name of the author nor the names of its
 *      contributors may be used to endorse or promote products
 * 		derived from this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * 	  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * 	  THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * 	  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * 	  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * 	  SERVICES; LOSS OF USE,
 *    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * 	  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *	  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * 	  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 *    OF SUCH DAMAGE.
 */

// DCB demosaicing by Jacek Gozdz (cuniek@kft.umcs.lublin.pl)

// FBDD denoising by Jacek Gozdz (cuniek@kft.umcs.lublin.pl) and
// Luis Sanz Rodríguez (luis.sanz.rodriguez@gmail.com)

// last modification: 11.07.2010

#include "../../internal/dcraw_defs.h"

// interpolates green vertically and saves it to image3
void LibRaw::dcb_ver(float (*image3)[3])
{
  int row, col, u = width, indx;

  for (row = 2; row < height - 2; row++)
    for (col = 2 + (FC(row, 2) & 1), indx = row * width + col; col < u - 2;
         col += 2, indx += 2)
    {

      image3[indx][1] = CLIP((image[indx + u][1] + image[indx - u][1]) / 2.0);
    }
}

// interpolates green horizontally and saves it to image2
void LibRaw::dcb_hor(float (*image2)[3])
{
  int row, col, u = width, indx;

  for (row = 2; row < height - 2; row++)
    for (col = 2 + (FC(row, 2) & 1), indx = row * width + col; col < u - 2;
         col += 2, indx += 2)
    {

      image2[indx][1] = CLIP((image[indx + 1][1] + image[indx - 1][1]) / 2.0);
    }
}

// missing colors are interpolated
void LibRaw::dcb_color()
{
  int row, col, c, d, u = width, indx;

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), indx = row * width + col,
        c = 2 - FC(row, col);
         col < u - 1; col += 2, indx += 2)
    {

      image[indx][c] = CLIP((4 * image[indx][1] - image[indx + u + 1][1] -
                             image[indx + u - 1][1] - image[indx - u + 1][1] -
                             image[indx - u - 1][1] + image[indx + u + 1][c] +
                             image[indx + u - 1][c] + image[indx - u + 1][c] +
                             image[indx - u - 1][c]) /
                            4.0);
    }

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 2) & 1), indx = row * width + col,
        c = FC(row, col + 1), d = 2 - c;
         col < width - 1; col += 2, indx += 2)
    {

      image[indx][c] =
          CLIP((2 * image[indx][1] - image[indx + 1][1] - image[indx - 1][1] +
                image[indx + 1][c] + image[indx - 1][c]) /
               2.0);
      image[indx][d] =
          CLIP((2 * image[indx][1] - image[indx + u][1] - image[indx - u][1] +
                image[indx + u][d] + image[indx - u][d]) /
               2.0);
    }
}

// missing R and B are interpolated horizontally and saved in image2
void LibRaw::dcb_color2(float (*image2)[3])
{
  int row, col, c, d, u = width, indx;

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), indx = row * width + col,
        c = 2 - FC(row, col);
         col < u - 1; col += 2, indx += 2)
    {

      image2[indx][c] =
          CLIP((4 * image2[indx][1] - image2[indx + u + 1][1] -
                image2[indx + u - 1][1] - image2[indx - u + 1][1] -
                image2[indx - u - 1][1] + image[indx + u + 1][c] +
                image[indx + u - 1][c] + image[indx - u + 1][c] +
                image[indx - u - 1][c]) /
               4.0);
    }

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 2) & 1), indx = row * width + col,
        c = FC(row, col + 1), d = 2 - c;
         col < width - 1; col += 2, indx += 2)
    {

      image2[indx][c] = CLIP((image[indx + 1][c] + image[indx - 1][c]) / 2.0);
      image2[indx][d] =
          CLIP((2 * image2[indx][1] - image2[indx + u][1] -
                image2[indx - u][1] + image[indx + u][d] + image[indx - u][d]) /
               2.0);
    }
}

// missing R and B are interpolated vertically and saved in image3
void LibRaw::dcb_color3(float (*image3)[3])
{
  int row, col, c, d, u = width, indx;

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), indx = row * width + col,
        c = 2 - FC(row, col);
         col < u - 1; col += 2, indx += 2)
    {

      image3[indx][c] =
          CLIP((4 * image3[indx][1] - image3[indx + u + 1][1] -
                image3[indx + u - 1][1] - image3[indx - u + 1][1] -
                image3[indx - u - 1][1] + image[indx + u + 1][c] +
                image[indx + u - 1][c] + image[indx - u + 1][c] +
                image[indx - u - 1][c]) /
               4.0);
    }

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 2) & 1), indx = row * width + col,
        c = FC(row, col + 1), d = 2 - c;
         col < width - 1; col += 2, indx += 2)
    {

      image3[indx][c] =
          CLIP((2 * image3[indx][1] - image3[indx + 1][1] -
                image3[indx - 1][1] + image[indx + 1][c] + image[indx - 1][c]) /
               2.0);
      image3[indx][d] = CLIP((image[indx + u][d] + image[indx - u][d]) / 2.0);
    }
}

// decides the primary green interpolation direction
void LibRaw::dcb_decide(float (*image2)[3], float (*image3)[3])
{
  int row, col, c, d, u = width, v = 2 * u, indx;
  float current, current2, current3;

  for (row = 2; row < height - 2; row++)
    for (col = 2 + (FC(row, 2) & 1), indx = row * width + col, c = FC(row, col);
         col < u - 2; col += 2, indx += 2)
    {

      d = ABS(c - 2);

      current = MAX(image[indx + v][c],
                    MAX(image[indx - v][c],
                        MAX(image[indx - 2][c], image[indx + 2][c]))) -
                MIN(image[indx + v][c],
                    MIN(image[indx - v][c],
                        MIN(image[indx - 2][c], image[indx + 2][c]))) +
                MAX(image[indx + 1 + u][d],
                    MAX(image[indx + 1 - u][d],
                        MAX(image[indx - 1 + u][d], image[indx - 1 - u][d]))) -
                MIN(image[indx + 1 + u][d],
                    MIN(image[indx + 1 - u][d],
                        MIN(image[indx - 1 + u][d], image[indx - 1 - u][d])));

      current2 =
          MAX(image2[indx + v][d],
              MAX(image2[indx - v][d],
                  MAX(image2[indx - 2][d], image2[indx + 2][d]))) -
          MIN(image2[indx + v][d],
              MIN(image2[indx - v][d],
                  MIN(image2[indx - 2][d], image2[indx + 2][d]))) +
          MAX(image2[indx + 1 + u][c],
              MAX(image2[indx + 1 - u][c],
                  MAX(image2[indx - 1 + u][c], image2[indx - 1 - u][c]))) -
          MIN(image2[indx + 1 + u][c],
              MIN(image2[indx + 1 - u][c],
                  MIN(image2[indx - 1 + u][c], image2[indx - 1 - u][c])));

      current3 =
          MAX(image3[indx + v][d],
              MAX(image3[indx - v][d],
                  MAX(image3[indx - 2][d], image3[indx + 2][d]))) -
          MIN(image3[indx + v][d],
              MIN(image3[indx - v][d],
                  MIN(image3[indx - 2][d], image3[indx + 2][d]))) +
          MAX(image3[indx + 1 + u][c],
              MAX(image3[indx + 1 - u][c],
                  MAX(image3[indx - 1 + u][c], image3[indx - 1 - u][c]))) -
          MIN(image3[indx + 1 + u][c],
              MIN(image3[indx + 1 - u][c],
                  MIN(image3[indx - 1 + u][c], image3[indx - 1 - u][c])));

      if (ABS(current - current2) < ABS(current - current3))
        image[indx][1] = image2[indx][1];
      else
        image[indx][1] = image3[indx][1];
    }
}

// saves red and blue in image2
void LibRaw::dcb_copy_to_buffer(float (*image2)[3])
{
  int indx;

  for (indx = 0; indx < height * width; indx++)
  {
    image2[indx][0] = image[indx][0]; // R
    image2[indx][2] = image[indx][2]; // B
  }
}

// restores red and blue from image2
void LibRaw::dcb_restore_from_buffer(float (*image2)[3])
{
  int indx;

  for (indx = 0; indx < height * width; indx++)
  {
    image[indx][0] = image2[indx][0]; // R
    image[indx][2] = image2[indx][2]; // B
  }
}

// R and B smoothing using green contrast, all pixels except 2 pixel wide border
void LibRaw::dcb_pp()
{
  int g1, r1, b1, u = width, indx, row, col;

  for (row = 2; row < height - 2; row++)
    for (col = 2, indx = row * u + col; col < width - 2; col++, indx++)
    {

      r1 = (image[indx - 1][0] + image[indx + 1][0] + image[indx - u][0] +
            image[indx + u][0] + image[indx - u - 1][0] +
            image[indx + u + 1][0] + image[indx - u + 1][0] +
            image[indx + u - 1][0]) /
           8.0;
      g1 = (image[indx - 1][1] + image[indx + 1][1] + image[indx - u][1] +
            image[indx + u][1] + image[indx - u - 1][1] +
            image[indx + u + 1][1] + image[indx - u + 1][1] +
            image[indx + u - 1][1]) /
           8.0;
      b1 = (image[indx - 1][2] + image[indx + 1][2] + image[indx - u][2] +
            image[indx + u][2] + image[indx - u - 1][2] +
            image[indx + u + 1][2] + image[indx - u + 1][2] +
            image[indx + u - 1][2]) /
           8.0;

      image[indx][0] = CLIP(r1 + (image[indx][1] - g1));
      image[indx][2] = CLIP(b1 + (image[indx][1] - g1));
    }
}

// green blurring correction, helps to get the nyquist right
void LibRaw::dcb_nyquist()
{
  int row, col, c, u = width, v = 2 * u, indx;

  for (row = 2; row < height - 2; row++)
    for (col = 2 + (FC(row, 2) & 1), indx = row * width + col, c = FC(row, col);
         col < u - 2; col += 2, indx += 2)
    {

      image[indx][1] = CLIP((image[indx + v][1] + image[indx - v][1] +
                             image[indx - 2][1] + image[indx + 2][1]) /
                                4.0 +
                            image[indx][c] -
                            (image[indx + v][c] + image[indx - v][c] +
                             image[indx - 2][c] + image[indx + 2][c]) /
                                4.0);
    }
}

// missing colors are interpolated using high quality algorithm by Luis Sanz
// Rodríguez
void LibRaw::dcb_color_full()
{
  int row, col, c, d, u = width, w = 3 * u, indx, g1, g2;
  float f[4], g[4], (*chroma)[2];

  chroma = (float(*)[2])calloc(width * height, sizeof *chroma);

  for (row = 1; row < height - 1; row++)
    for (col = 1 + (FC(row, 1) & 1), indx = row * width + col, c = FC(row, col),
        d = c / 2;
         col < u - 1; col += 2, indx += 2)
      chroma[indx][d] = image[indx][c] - image[indx][1];

  for (row = 3; row < height - 3; row++)
    for (col = 3 + (FC(row, 1) & 1), indx = row * width + col,
        c = 1 - FC(row, col) / 2, d = 1 - c;
         col < u - 3; col += 2, indx += 2)
    {
      f[0] = 1.0 /
             (float)(1.0 +
                     fabs(chroma[indx - u - 1][c] - chroma[indx + u + 1][c]) +
                     fabs(chroma[indx - u - 1][c] - chroma[indx - w - 3][c]) +
                     fabs(chroma[indx + u + 1][c] - chroma[indx - w - 3][c]));
      f[1] = 1.0 /
             (float)(1.0 +
                     fabs(chroma[indx - u + 1][c] - chroma[indx + u - 1][c]) +
                     fabs(chroma[indx - u + 1][c] - chroma[indx - w + 3][c]) +
                     fabs(chroma[indx + u - 1][c] - chroma[indx - w + 3][c]));
      f[2] = 1.0 /
             (float)(1.0 +
                     fabs(chroma[indx + u - 1][c] - chroma[indx - u + 1][c]) +
                     fabs(chroma[indx + u - 1][c] - chroma[indx + w + 3][c]) +
                     fabs(chroma[indx - u + 1][c] - chroma[indx + w - 3][c]));
      f[3] = 1.0 /
             (float)(1.0 +
                     fabs(chroma[indx + u + 1][c] - chroma[indx - u - 1][c]) +
                     fabs(chroma[indx + u + 1][c] - chroma[indx + w - 3][c]) +
                     fabs(chroma[indx - u - 1][c] - chroma[indx + w + 3][c]));
      g[0] = 1.325 * chroma[indx - u - 1][c] - 0.175 * chroma[indx - w - 3][c] -
             0.075 * chroma[indx - w - 1][c] - 0.075 * chroma[indx - u - 3][c];
      g[1] = 1.325 * chroma[indx - u + 1][c] - 0.175 * chroma[indx - w + 3][c] -
             0.075 * chroma[indx - w + 1][c] - 0.075 * chroma[indx - u + 3][c];
      g[2] = 1.325 * chroma[indx + u - 1][c] - 0.175 * chroma[indx + w - 3][c] -
             0.075 * chroma[indx + w - 1][c] - 0.075 * chroma[indx + u - 3][c];
      g[3] = 1.325 * chroma[indx + u + 1][c] - 0.175 * chroma[indx + w + 3][c] -
             0.075 * chroma[indx + w + 1][c] - 0.075 * chroma[indx + u + 3][c];
      chroma[indx][c] =
          (f[0] * g[0] + f[1] * g[1] + f[2] * g[2] + f[3] * g[3]) /
          (f[0] + f[1] + f[2] + f[3]);
    }
  for (row = 3; row < height - 3; row++)
    for (col = 3 + (FC(row, 2) & 1), indx = row * width + col,
        c = FC(row, col + 1) / 2;
         col < u - 3; col += 2, indx += 2)
      for (d = 0; d <= 1; c = 1 - c, d++)
      {
        f[0] = 1.0 /
               (float)(1.0 + fabs(chroma[indx - u][c] - chroma[indx + u][c]) +
                       fabs(chroma[indx - u][c] - chroma[indx - w][c]) +
                       fabs(chroma[indx + u][c] - chroma[indx - w][c]));
        f[1] = 1.0 /
               (float)(1.0 + fabs(chroma[indx + 1][c] - chroma[indx - 1][c]) +
                       fabs(chroma[indx + 1][c] - chroma[indx + 3][c]) +
                       fabs(chroma[indx - 1][c] - chroma[indx + 3][c]));
        f[2] = 1.0 /
               (float)(1.0 + fabs(chroma[indx - 1][c] - chroma[indx + 1][c]) +
                       fabs(chroma[indx - 1][c] - chroma[indx - 3][c]) +
                       fabs(chroma[indx + 1][c] - chroma[indx - 3][c]));
        f[3] = 1.0 /
               (float)(1.0 + fabs(chroma[indx + u][c] - chroma[indx - u][c]) +
                       fabs(chroma[indx + u][c] - chroma[indx + w][c]) +
                       fabs(chroma[indx - u][c] - chroma[indx + w][c]));

        g[0] = 0.875 * chroma[indx - u][c] + 0.125 * chroma[indx - w][c];
        g[1] = 0.875 * chroma[indx + 1][c] + 0.125 * chroma[indx + 3][c];
        g[2] = 0.875 * chroma[indx - 1][c] + 0.125 * chroma[indx - 3][c];
        g[3] = 0.875 * chroma[indx + u][c] + 0.125 * chroma[indx + w][c];

        chroma[indx][c] =
            (f[0] * g[0] + f[1] * g[1] + f[2] * g[2] + f[3] * g[3]) /
            (f[0] + f[1] + f[2] + f[3]);
      }

  for (row = 6; row < height - 6; row++)
    for (col = 6, indx = row * width + col; col < width - 6; col++, indx++)
    {
      image[indx][0] = CLIP(chroma[indx][0] + image[indx][1]);
      image[indx][2] = CLIP(chroma[indx][1] + image[indx][1]);

      g1 = MIN(
          image[indx + 1 + u][0],
          MIN(image[indx + 1 - u][0],
              MIN(image[indx - 1 + u][0],
                  MIN(image[indx - 1 - u][0],
                      MIN(image[indx - 1][0],
                          MIN(image[indx + 1][0],
                              MIN(image[indx - u][0], image[indx + u][0])))))));

      g2 = MAX(
          image[indx + 1 + u][0],
          MAX(image[indx + 1 - u][0],
              MAX(image[indx - 1 + u][0],
                  MAX(image[indx - 1 - u][0],
                      MAX(image[indx - 1][0],
                          MAX(image[indx + 1][0],
                              MAX(image[indx - u][0], image[indx + u][0])))))));

      image[indx][0] = ULIM(image[indx][0], g2, g1);

      g1 = MIN(
          image[indx + 1 + u][2],
          MIN(image[indx + 1 - u][2],
              MIN(image[indx - 1 + u][2],
                  MIN(image[indx - 1 - u][2],
                      MIN(image[indx - 1][2],
                          MIN(image[indx + 1][2],
                              MIN(image[indx - u][2], image[indx + u][2])))))));

      g2 = MAX(
          image[indx + 1 + u][2],
          MAX(image[indx + 1 - u][2],
              MAX(image[indx - 1 + u][2],
                  MAX(image[indx - 1 - u][2],
                      MAX(image[indx - 1][2],
                          MAX(image[indx + 1][2],
                              MAX(image[indx - u][2], image[indx + u][2])))))));

      image[indx][2] = ULIM(image[indx][2], g2, g1);
    }

  free(chroma);
}

// green is used to create an interpolation direction map saved in image[][3]
// 1 = vertical
// 0 = horizontal
void LibRaw::dcb_map()
{
  int row, col, u = width, indx;

  for (row = 1; row < height - 1; row++)
  {
    for (col = 1, indx = row * width + col; col < width - 1; col++, indx++)
    {

      if (image[indx][1] > (image[indx - 1][1] + image[indx + 1][1] +
                            image[indx - u][1] + image[indx + u][1]) /
                               4.0)
        image[indx][3] = ((MIN(image[indx - 1][1], image[indx + 1][1]) +
                           image[indx - 1][1] + image[indx + 1][1]) <
                          (MIN(image[indx - u][1], image[indx + u][1]) +
                           image[indx - u][1] + image[indx + u][1]));
      else
        image[indx][3] = ((MAX(image[indx - 1][1], image[indx + 1][1]) +
                           image[indx - 1][1] + image[indx + 1][1]) >
                          (MAX(image[indx - u][1], image[indx + u][1]) +
                           image[indx - u][1] + image[indx + u][1]));
    }
  }
}

// interpolated green pixels are corrected using the map
void LibRaw::dcb_correction()
{
  int current, row, col, u = width, v = 2 * u, indx;

  for (row = 2; row < height - 2; row++)
    for (col = 2 + (FC(row, 2) & 1), indx = row * width + col; col < u - 2;
         col += 2, indx += 2)
    {

      current = 4 * image[indx][3] +
                2 * (image[indx + u][3] + image[indx - u][3] +
                     image[indx + 1][3] + image[indx - 1][3]) +
                image[indx + v][3] + image[indx - v][3] + image[indx + 2][3] +
                image[indx - 2][3];

      image[indx][1] =
          ((16 - current) * (image[indx - 1][1] + image[indx + 1][1]) / 2.0 +
           current * (image[indx - u][1] + image[indx + u][1]) / 2.0) /
          16.0;
    }
}

// interpolated green pixels are corrected using the map
// with contrast correction
void LibRaw::dcb_correction2()
{
  int current, row, col, c, u = width, v = 2 * u, indx;

  for (row = 4; row < height - 4; row++)
    for (col = 4 + (FC(row, 2) & 1), indx = row * width + col, c = FC(row, col);
         col < u - 4; col += 2, indx += 2)
    {

      current = 4 * image[indx][3] +
                2 * (image[indx + u][3] + image[indx - u][3] +
                     image[indx + 1][3] + image[indx - 1][3]) +
                image[indx + v][3] + image[indx - v][3] + image[indx + 2][3] +
                image[indx - 2][3];

      image[indx][1] = CLIP(
          ((16 - current) * ((image[indx - 1][1] + image[indx + 1][1]) / 2.0 +
                             image[indx][c] -
                             (image[indx + 2][c] + image[indx - 2][c]) / 2.0) +
           current * ((image[indx - u][1] + image[indx + u][1]) / 2.0 +
                      image[indx][c] -
                      (image[indx + v][c] + image[indx - v][c]) / 2.0)) /
          16.0);
    }
}

void LibRaw::dcb_refinement()
{
  int row, col, c, u = width, v = 2 * u, w = 3 * u, indx, current;
  float f[5], g1, g2;

  for (row = 4; row < height - 4; row++)
    for (col = 4 + (FC(row, 2) & 1), indx = row * width + col, c = FC(row, col);
         col < u - 4; col += 2, indx += 2)
    {

      current = 4 * image[indx][3] +
                2 * (image[indx + u][3] + image[indx - u][3] +
                     image[indx + 1][3] + image[indx - 1][3]) +
                image[indx + v][3] + image[indx - v][3] + image[indx - 2][3] +
                image[indx + 2][3];

      if (image[indx][c] > 1)
      {

        f[0] = (float)(image[indx - u][1] + image[indx + u][1]) /
               (2 * image[indx][c]);

        if (image[indx - v][c] > 0)
          f[1] = 2 * (float)image[indx - u][1] /
                 (image[indx - v][c] + image[indx][c]);
        else
          f[1] = f[0];

        if (image[indx - v][c] > 0)
          f[2] = (float)(image[indx - u][1] + image[indx - w][1]) /
                 (2 * image[indx - v][c]);
        else
          f[2] = f[0];

        if (image[indx + v][c] > 0)
          f[3] = 2 * (float)image[indx + u][1] /
                 (image[indx + v][c] + image[indx][c]);
        else
          f[3] = f[0];

        if (image[indx + v][c] > 0)
          f[4] = (float)(image[indx + u][1] + image[indx + w][1]) /
                 (2 * image[indx + v][c]);
        else
          f[4] = f[0];

        g1 = (5 * f[0] + 3 * f[1] + f[2] + 3 * f[3] + f[4]) / 13.0;

        f[0] = (float)(image[indx - 1][1] + image[indx + 1][1]) /
               (2 * image[indx][c]);

        if (image[indx - 2][c] > 0)
          f[1] = 2 * (float)image[indx - 1][1] /
                 (image[indx - 2][c] + image[indx][c]);
        else
          f[1] = f[0];

        if (image[indx - 2][c] > 0)
          f[2] = (float)(image[indx - 1][1] + image[indx - 3][1]) /
                 (2 * image[indx - 2][c]);
        else
          f[2] = f[0];

        if (image[indx + 2][c] > 0)
          f[3] = 2 * (float)image[indx + 1][1] /
                 (image[indx + 2][c] + image[indx][c]);
        else
          f[3] = f[0];

        if (image[indx + 2][c] > 0)
          f[4] = (float)(image[indx + 1][1] + image[indx + 3][1]) /
                 (2 * image[indx + 2][c]);
        else
          f[4] = f[0];

        g2 = (5 * f[0] + 3 * f[1] + f[2] + 3 * f[3] + f[4]) / 13.0;

        image[indx][1] = CLIP((image[indx][c]) *
                              (current * g1 + (16 - current) * g2) / 16.0);
      }
      else
        image[indx][1] = image[indx][c];

      // get rid of overshooted pixels

      g1 = MIN(
          image[indx + 1 + u][1],
          MIN(image[indx + 1 - u][1],
              MIN(image[indx - 1 + u][1],
                  MIN(image[indx - 1 - u][1],
                      MIN(image[indx - 1][1],
                          MIN(image[indx + 1][1],
                              MIN(image[indx - u][1], image[indx + u][1])))))));

      g2 = MAX(
          image[indx + 1 + u][1],
          MAX(image[indx + 1 - u][1],
              MAX(image[indx - 1 + u][1],
                  MAX(image[indx - 1 - u][1],
                      MAX(image[indx - 1][1],
                          MAX(image[indx + 1][1],
                              MAX(image[indx - u][1], image[indx + u][1])))))));

      image[indx][1] = ULIM(image[indx][1], g2, g1);
    }
}

// converts RGB to LCH colorspace and saves it to image3
void LibRaw::rgb_to_lch(double (*image2)[3])
{
  int indx;
  for (indx = 0; indx < height * width; indx++)
  {

    image2[indx][0] = image[indx][0] + image[indx][1] + image[indx][2]; // L
    image2[indx][1] = 1.732050808 * (image[indx][0] - image[indx][1]);  // C
    image2[indx][2] =
        2.0 * image[indx][2] - image[indx][0] - image[indx][1]; // H
  }
}

// converts LCH to RGB colorspace and saves it back to image
void LibRaw::lch_to_rgb(double (*image2)[3])
{
  int indx;
  for (indx = 0; indx < height * width; indx++)
  {

    image[indx][0] = CLIP(image2[indx][0] / 3.0 - image2[indx][2] / 6.0 +
                          image2[indx][1] / 3.464101615);
    image[indx][1] = CLIP(image2[indx][0] / 3.0 - image2[indx][2] / 6.0 -
                          image2[indx][1] / 3.464101615);
    image[indx][2] = CLIP(image2[indx][0] / 3.0 + image2[indx][2] / 3.0);
  }
}

// denoising using interpolated neighbours
void LibRaw::fbdd_correction()
{
  int row, col, c, u = width, indx;

  for (row = 2; row < height - 2; row++)
  {
    for (col = 2, indx = row * width + col; col < width - 2; col++, indx++)
    {

      c = fcol(row, col);

      image[indx][c] =
          ULIM(image[indx][c],
               MAX(image[indx - 1][c],
                   MAX(image[indx + 1][c],
                       MAX(image[indx - u][c], image[indx + u][c]))),
               MIN(image[indx - 1][c],
                   MIN(image[indx + 1][c],
                       MIN(image[indx - u][c], image[indx + u][c]))));
    }
  }
}

// corrects chroma noise
void LibRaw::fbdd_correction2(double (*image2)[3])
{
  int indx, v = 2 * width;
  int col, row;
  double Co, Ho, ratio;

  for (row = 6; row < height - 6; row++)
  {
    for (col = 6; col < width - 6; col++)
    {
      indx = row * width + col;

      if (image2[indx][1] * image2[indx][2] != 0)
      {
        Co = (image2[indx + v][1] + image2[indx - v][1] + image2[indx - 2][1] +
              image2[indx + 2][1] -
              MAX(image2[indx - 2][1],
                  MAX(image2[indx + 2][1],
                      MAX(image2[indx - v][1], image2[indx + v][1]))) -
              MIN(image2[indx - 2][1],
                  MIN(image2[indx + 2][1],
                      MIN(image2[indx - v][1], image2[indx + v][1])))) /
             2.0;
        Ho = (image2[indx + v][2] + image2[indx - v][2] + image2[indx - 2][2] +
              image2[indx + 2][2] -
              MAX(image2[indx - 2][2],
                  MAX(image2[indx + 2][2],
                      MAX(image2[indx - v][2], image2[indx + v][2]))) -
              MIN(image2[indx - 2][2],
                  MIN(image2[indx + 2][2],
                      MIN(image2[indx - v][2], image2[indx + v][2])))) /
             2.0;
        ratio = sqrt((Co * Co + Ho * Ho) / (image2[indx][1] * image2[indx][1] +
                                            image2[indx][2] * image2[indx][2]));

        if (ratio < 0.85)
        {
          image2[indx][0] =
              -(image2[indx][1] + image2[indx][2] - Co - Ho) + image2[indx][0];
          image2[indx][1] = Co;
          image2[indx][2] = Ho;
        }
      }
    }
  }
}

// Cubic Spline Interpolation by Li and Randhawa, modified by Jacek Gozdz and
// Luis Sanz Rodríguez
void LibRaw::fbdd_green()
{
  int row, col, c, u = width, v = 2 * u, w = 3 * u, x = 4 * u, y = 5 * u, indx,
                   min, max;
  float f[4], g[4];

  for (row = 5; row < height - 5; row++)
    for (col = 5 + (FC(row, 1) & 1), indx = row * width + col, c = FC(row, col);
         col < u - 5; col += 2, indx += 2)
    {

      f[0] = 1.0 / (1.0 + abs(image[indx - u][1] - image[indx - w][1]) +
                    abs(image[indx - w][1] - image[indx + y][1]));
      f[1] = 1.0 / (1.0 + abs(image[indx + 1][1] - image[indx + 3][1]) +
                    abs(image[indx + 3][1] - image[indx - 5][1]));
      f[2] = 1.0 / (1.0 + abs(image[indx - 1][1] - image[indx - 3][1]) +
                    abs(image[indx - 3][1] - image[indx + 5][1]));
      f[3] = 1.0 / (1.0 + abs(image[indx + u][1] - image[indx + w][1]) +
                    abs(image[indx + w][1] - image[indx - y][1]));

      g[0] = CLIP((23 * image[indx - u][1] + 23 * image[indx - w][1] +
                   2 * image[indx - y][1] +
                   8 * (image[indx - v][c] - image[indx - x][c]) +
                   40 * (image[indx][c] - image[indx - v][c])) /
                  48.0);
      g[1] = CLIP((23 * image[indx + 1][1] + 23 * image[indx + 3][1] +
                   2 * image[indx + 5][1] +
                   8 * (image[indx + 2][c] - image[indx + 4][c]) +
                   40 * (image[indx][c] - image[indx + 2][c])) /
                  48.0);
      g[2] = CLIP((23 * image[indx - 1][1] + 23 * image[indx - 3][1] +
                   2 * image[indx - 5][1] +
                   8 * (image[indx - 2][c] - image[indx - 4][c]) +
                   40 * (image[indx][c] - image[indx - 2][c])) /
                  48.0);
      g[3] = CLIP((23 * image[indx + u][1] + 23 * image[indx + w][1] +
                   2 * image[indx + y][1] +
                   8 * (image[indx + v][c] - image[indx + x][c]) +
                   40 * (image[indx][c] - image[indx + v][c])) /
                  48.0);

      image[indx][1] =
          CLIP((f[0] * g[0] + f[1] * g[1] + f[2] * g[2] + f[3] * g[3]) /
               (f[0] + f[1] + f[2] + f[3]));

      min = MIN(
          image[indx + 1 + u][1],
          MIN(image[indx + 1 - u][1],
              MIN(image[indx - 1 + u][1],
                  MIN(image[indx - 1 - u][1],
                      MIN(image[indx - 1][1],
                          MIN(image[indx + 1][1],
                              MIN(image[indx - u][1], image[indx + u][1])))))));

      max = MAX(
          image[indx + 1 + u][1],
          MAX(image[indx + 1 - u][1],
              MAX(image[indx - 1 + u][1],
                  MAX(image[indx - 1 - u][1],
                      MAX(image[indx - 1][1],
                          MAX(image[indx + 1][1],
                              MAX(image[indx - u][1], image[indx + u][1])))))));

      image[indx][1] = ULIM(image[indx][1], max, min);
    }
}

// FBDD (Fake Before Demosaicing Denoising)
void LibRaw::fbdd(int noiserd)
{
  double(*image2)[3];
  // safety net: disable for 4-color bayer or full-color images
  if (colors != 3 || !filters)
    return;
  image2 = (double(*)[3])calloc(width * height, sizeof *image2);

  border_interpolate(4);

  if (noiserd > 1)
  {
    fbdd_green();
    // dcb_color_full(image2);
    dcb_color_full();
    fbdd_correction();

    dcb_color();
    rgb_to_lch(image2);
    fbdd_correction2(image2);
    fbdd_correction2(image2);
    lch_to_rgb(image2);
  }
  else
  {
    fbdd_green();
    // dcb_color_full(image2);
    dcb_color_full();
    fbdd_correction();
  }

  free(image2);
}

// DCB demosaicing main routine
void LibRaw::dcb(int iterations, int dcb_enhance)
{

  int i = 1;

  float(*image2)[3];
  image2 = (float(*)[3])calloc(width * height, sizeof *image2);

  float(*image3)[3];
  image3 = (float(*)[3])calloc(width * height, sizeof *image3);

  border_interpolate(6);

  dcb_hor(image2);
  dcb_color2(image2);

  dcb_ver(image3);
  dcb_color3(image3);

  dcb_decide(image2, image3);

  free(image3);

  dcb_copy_to_buffer(image2);

  while (i <= iterations)
  {
    dcb_nyquist();
    dcb_nyquist();
    dcb_nyquist();
    dcb_map();
    dcb_correction();
    i++;
  }

  dcb_color();
  dcb_pp();

  dcb_map();
  dcb_correction2();

  dcb_map();
  dcb_correction();

  dcb_map();
  dcb_correction();

  dcb_map();
  dcb_correction();

  dcb_map();
  dcb_restore_from_buffer(image2);
  dcb_color();

  if (dcb_enhance)
  {
    dcb_refinement();
    // dcb_color_full(image2);
    dcb_color_full();
  }

  free(image2);
}
