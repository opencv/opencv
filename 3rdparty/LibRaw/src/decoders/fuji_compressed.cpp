/* -*- C++ -*-
 * File: libraw_fuji_compressed.cpp
 * Copyright (C) 2016-2019 Alexey Danilchenko
 *
 * Adopted to LibRaw by Alex Tutubalin, lexa@lexa.ru
 * LibRaw Fujifilm/compressed decoder

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/libraw_cxx_defs.h"

#ifdef _abs
#undef _abs
#undef _min
#undef _max
#endif
#define _abs(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define _min(a, b) ((a) < (b) ? (a) : (b))
#define _max(a, b) ((a) > (b) ? (a) : (b))

struct int_pair
{
  int value1;
  int value2;
};

enum _xt_lines
{
  _R0 = 0,
  _R1,
  _R2,
  _R3,
  _R4,
  _G0,
  _G1,
  _G2,
  _G3,
  _G4,
  _G5,
  _G6,
  _G7,
  _B0,
  _B1,
  _B2,
  _B3,
  _B4,
  _ltotal
};

// tables of gradients for single sample level
struct fuji_grads
{
  int_pair grads[41];
  int_pair lossy_grads[3][5];
};

struct fuji_compressed_block
{
  int cur_bit;            // current bit being read (from left to right)
  int cur_pos;            // current position in a buffer
  INT64 cur_buf_offset;   // offset of this buffer in a file
  unsigned max_read_size; // Amount of data to be read
  int cur_buf_size;       // buffer size
  uchar *cur_buf;         // currently read block
  int fillbytes;          // Counter to add extra byte for block size N*16
  LibRaw_abstract_datastream *input;
  fuji_grads even[3]; // tables of even gradients
  fuji_grads odd[3];  // tables of odd gradients
  ushort *linealloc;
  ushort *linebuf[_ltotal];
};

static inline int log2ceil(int val)
{
  int result = 0;
  if (val--)
    do
      ++result;
    while (val >>= 1);

  return result;
}

void setup_qlut(int8_t *qt, int *q_point)
{
  for (int curVal = -q_point[4]; curVal <= q_point[4]; ++qt, ++curVal)
  {
    if (curVal <= -q_point[3])
      *qt = -4;
    else if (curVal <= -q_point[2])
      *qt = -3;
    else if (curVal <= -q_point[1])
      *qt = -2;
    else if (curVal < -q_point[0])
      *qt = -1;
    else if (curVal <= q_point[0])
      *qt = 0;
    else if (curVal < q_point[1])
      *qt = 1;
    else if (curVal < q_point[2])
      *qt = 2;
    else if (curVal < q_point[3])
      *qt = 3;
    else
      *qt = 4;
  }
}

void init_main_qtable(fuji_compressed_params *params, uchar q_base)
{
  fuji_q_table *qt = params->qt;
  int qp[5];
  int maxVal = params->max_value + 1;
  qp[0] = q_base;
  qp[1] = 3 * q_base + 0x12;
  qp[2] = 5 * q_base + 0x43;
  qp[3] = 7 * q_base + 0x114;
  qp[4] = params->max_value;
  if (qp[1] >= maxVal || qp[1] < q_base + 1)
    qp[1] = q_base + 1;
  if (qp[2] < qp[1] || qp[2] >= maxVal)
    qp[2] = qp[1];
  if (qp[3] < qp[2] || qp[3] >= maxVal)
    qp[3] = qp[2];
  setup_qlut(qt->q_table, qp);
  qt->q_base = q_base;
  qt->max_grad = 0;
  qt->total_values = (qp[4] + 2 * q_base) / (2 * q_base + 1) + 1;
  qt->raw_bits = log2ceil(qt->total_values);
  qt->q_grad_mult = 9;
  params->max_bits = 4 * log2ceil(qp[4] + 1);
}

void LibRaw::init_fuji_compr(fuji_compressed_params *params)
{
  if ((libraw_internal_data.unpacker_data.fuji_block_width % 3 &&
       libraw_internal_data.unpacker_data.fuji_raw_type == 16) ||
      (libraw_internal_data.unpacker_data.fuji_block_width & 1 &&
       libraw_internal_data.unpacker_data.fuji_raw_type == 0))
    derror();

  size_t q_table_size = 2 << libraw_internal_data.unpacker_data.fuji_bits;
  if (libraw_internal_data.unpacker_data.fuji_lossless)
    params->buf = malloc(q_table_size);
  else
    params->buf = malloc(3 * q_table_size);

  if (libraw_internal_data.unpacker_data.fuji_raw_type == 16)
    params->line_width = (libraw_internal_data.unpacker_data.fuji_block_width * 2) / 3;
  else
    params->line_width = libraw_internal_data.unpacker_data.fuji_block_width >> 1;

  params->min_value = 0x40;
  params->max_value = (1 << libraw_internal_data.unpacker_data.fuji_bits) - 1;

  // setup qtables
  if (libraw_internal_data.unpacker_data.fuji_lossless)
  {
    // setup main qtable only, zero the rest
    memset(params->qt + 1, 0, 3 * sizeof(fuji_q_table));
    params->qt[0].q_table = (int8_t *)params->buf;
    params->qt[0].q_base = -1;
    init_main_qtable(params, 0);
  }
  else
  {
    // setup 3 extra qtables - main one will be set for each block
    memset(params->qt, 0, sizeof(fuji_q_table));
    int qp[5];

    qp[0] = 0;
    qp[4] = params->max_value;

    // table 0
    params->qt[1].q_table = (int8_t *)params->buf;
    params->qt[1].q_base = 0;
    params->qt[1].max_grad = 5;
    params->qt[1].q_grad_mult = 3;
    params->qt[1].total_values = qp[4] + 1;
    params->qt[1].raw_bits = log2ceil(params->qt[1].total_values);

    qp[1] = qp[4] >= 0x12 ? 0x12 : qp[0] + 1;
    qp[2] = qp[4] >= 0x43 ? 0x43 : qp[1];
    qp[3] = qp[4] >= 0x114 ? 0x114 : qp[2];
    setup_qlut(params->qt[1].q_table, qp);

    // table 1
    params->qt[2].q_table = params->qt[1].q_table + q_table_size;
    params->qt[2].q_base = 1;
    params->qt[2].max_grad = 6;
    params->qt[2].q_grad_mult = 3;
    params->qt[2].total_values = (qp[4] + 2) / 3 + 1;
    params->qt[2].raw_bits = log2ceil(params->qt[2].total_values);

    qp[0] = params->qt[2].q_base;
    qp[1] = qp[4] >= 0x15 ? 0x15 : qp[0] + 1;
    qp[2] = qp[4] >= 0x48 ? 0x48 : qp[1];
    qp[3] = qp[4] >= 0x11B ? 0x11B : qp[2];
    setup_qlut(params->qt[2].q_table, qp);

    // table 2
    params->qt[3].q_table = params->qt[2].q_table + q_table_size;
    params->qt[3].q_base = 2;
    params->qt[3].max_grad = 7;
    params->qt[3].q_grad_mult = 3;
    params->qt[3].total_values = (qp[4] + 4) / 5 + 1;
    params->qt[3].raw_bits = log2ceil(params->qt[3].total_values);

    qp[0] = params->qt[3].q_base;
    qp[1] = qp[4] >= 0x18 ? 0x18 : qp[0] + 1;
    qp[2] = qp[4] >= 0x4D ? 0x4D : qp[1];
    qp[3] = qp[4] >= 0x122 ? 0x122 : qp[2];
    setup_qlut(params->qt[3].q_table, qp);
  }
}

#define XTRANS_BUF_SIZE 0x10000

static inline void fuji_fill_buffer(fuji_compressed_block *info)
{
  if (info->cur_pos >= info->cur_buf_size)
  {
    info->cur_pos = 0;
    info->cur_buf_offset += info->cur_buf_size;
#ifdef LIBRAW_USE_OPENMP
#pragma omp critical
#endif
    {
#ifndef LIBRAW_USE_OPENMP
      info->input->lock();
#endif
      info->input->seek(info->cur_buf_offset, SEEK_SET);
      info->cur_buf_size = info->input->read(info->cur_buf, 1, _min(info->max_read_size, XTRANS_BUF_SIZE));
#ifndef LIBRAW_USE_OPENMP
      info->input->unlock();
#endif
      if (info->cur_buf_size < 1) // nothing read
      {
        if (info->fillbytes > 0)
        {
          int ls = _max(1, _min(info->fillbytes, XTRANS_BUF_SIZE));
          memset(info->cur_buf, 0, ls);
          info->fillbytes -= ls;
        }
        else
          throw LIBRAW_EXCEPTION_IO_EOF;
      }
      info->max_read_size -= info->cur_buf_size;
    }
  }
}

void init_main_grads(const fuji_compressed_params *params, fuji_compressed_block *info)
{
  int max_diff = _max(2, (params->qt->total_values + 0x20) >> 6);
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 41; i++)
    {
      info->even[j].grads[i].value1 = max_diff;
      info->even[j].grads[i].value2 = 1;
      info->odd[j].grads[i].value1 = max_diff;
      info->odd[j].grads[i].value2 = 1;
    }
}

void LibRaw::init_fuji_block(fuji_compressed_block *info, const fuji_compressed_params *params, INT64 raw_offset,
                             unsigned dsize)
{
  info->linealloc = (ushort *)calloc(sizeof(ushort), _ltotal * (params->line_width + 2));

  INT64 fsize = libraw_internal_data.internal_data.input->size();
  info->max_read_size = _min(unsigned(fsize - raw_offset), dsize); // Data size may be incorrect?
  info->fillbytes = 1;

  info->input = libraw_internal_data.internal_data.input;
  info->linebuf[_R0] = info->linealloc;
  for (int i = _R1; i <= _B4; i++)
    info->linebuf[i] = info->linebuf[i - 1] + params->line_width + 2;

  // init buffer
  info->cur_buf = (uchar *)malloc(XTRANS_BUF_SIZE);
  info->cur_bit = 0;
  info->cur_pos = 0;
  info->cur_buf_offset = raw_offset;
  info->cur_buf_size = 0;
  fuji_fill_buffer(info);

  // init grads for lossy and lossless
  if (libraw_internal_data.unpacker_data.fuji_lossless)
    init_main_grads(params, info);
  else
  {
    // init static grads for lossy only - main ones are done per line
    for (int k = 0; k < 3; ++k)
    {
      int max_diff = _max(2, ((params->qt[k + 1].total_values + 0x20) >> 6));
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 5; ++i)
        {
          info->even[j].lossy_grads[k][i].value1 = max_diff;
          info->even[j].lossy_grads[k][i].value2 = 1;
          info->odd[j].lossy_grads[k][i].value1 = max_diff;
          info->odd[j].lossy_grads[k][i].value2 = 1;
        }
    }
  }
}

void LibRaw::copy_line_to_xtrans(fuji_compressed_block *info, int cur_line, int cur_block, int cur_block_width)
{
  ushort *lineBufB[3];
  ushort *lineBufG[6];
  ushort *lineBufR[3];
  unsigned pixel_count;
  ushort *line_buf;
  int index;

  int offset = libraw_internal_data.unpacker_data.fuji_block_width * cur_block + 6 * imgdata.sizes.raw_width * cur_line;
  ushort *raw_block_data = imgdata.rawdata.raw_image + offset;
  int row_count = 0;

  for (int i = 0; i < 3; i++)
  {
    lineBufR[i] = info->linebuf[_R2 + i] + 1;
    lineBufB[i] = info->linebuf[_B2 + i] + 1;
  }
  for (int i = 0; i < 6; i++)
    lineBufG[i] = info->linebuf[_G2 + i] + 1;

  while (row_count < 6)
  {
    pixel_count = 0;
    while (pixel_count < (unsigned)cur_block_width)
    {
      switch (imgdata.idata.xtrans_abs[row_count][(pixel_count % 6)])
      {
      case 0: // red
        line_buf = lineBufR[row_count >> 1];
        break;
      case 1:  // green
      default: // to make static analyzer happy
        line_buf = lineBufG[row_count];
        break;
      case 2: // blue
        line_buf = lineBufB[row_count >> 1];
        break;
      }

      index = (((pixel_count * 2 / 3) & 0x7FFFFFFE) | ((pixel_count % 3) & 1)) + ((pixel_count % 3) >> 1);
      raw_block_data[pixel_count] = line_buf[index];

      ++pixel_count;
    }
    ++row_count;
    raw_block_data += imgdata.sizes.raw_width;
  }
}

void LibRaw::copy_line_to_bayer(fuji_compressed_block *info, int cur_line, int cur_block, int cur_block_width)
{
  ushort *lineBufB[3];
  ushort *lineBufG[6];
  ushort *lineBufR[3];
  unsigned pixel_count;
  ushort *line_buf;

  int fuji_bayer[2][2];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++)
      fuji_bayer[r][c] = FC(r, c); // We'll downgrade G2 to G below

  int offset = libraw_internal_data.unpacker_data.fuji_block_width * cur_block + 6 * imgdata.sizes.raw_width * cur_line;
  ushort *raw_block_data = imgdata.rawdata.raw_image + offset;
  int row_count = 0;

  for (int i = 0; i < 3; i++)
  {
    lineBufR[i] = info->linebuf[_R2 + i] + 1;
    lineBufB[i] = info->linebuf[_B2 + i] + 1;
  }
  for (int i = 0; i < 6; i++)
    lineBufG[i] = info->linebuf[_G2 + i] + 1;

  while (row_count < 6)
  {
    pixel_count = 0;
    while (pixel_count < (unsigned)cur_block_width)
    {
      switch (fuji_bayer[row_count & 1][pixel_count & 1])
      {
      case 0: // red
        line_buf = lineBufR[row_count >> 1];
        break;
      case 1:  // green
      case 3:  // second green
      default: // to make static analyzer happy
        line_buf = lineBufG[row_count];
        break;
      case 2: // blue
        line_buf = lineBufB[row_count >> 1];
        break;
      }

      raw_block_data[pixel_count] = line_buf[pixel_count >> 1];
      ++pixel_count;
    }
    ++row_count;
    raw_block_data += imgdata.sizes.raw_width;
  }
}

#define fuji_quant_gradient(max, q, v1, v2) (q->q_grad_mult * q->q_table[(max) + (v1)] + q->q_table[(max) + (v2)])

static inline void fuji_zerobits(fuji_compressed_block *info, int *count)
{
  uchar zero = 0;
  *count = 0;
  while (zero == 0)
  {
    zero = (info->cur_buf[info->cur_pos] >> (7 - info->cur_bit)) & 1;
    info->cur_bit++;
    info->cur_bit &= 7;
    if (!info->cur_bit)
    {
      ++info->cur_pos;
      fuji_fill_buffer(info);
    }
    if (zero)
      break;
    ++*count;
  }
}

static inline void fuji_read_code(fuji_compressed_block *info, int *data, int bits_to_read)
{
  uchar bits_left = bits_to_read;
  uchar bits_left_in_byte = 8 - (info->cur_bit & 7);
  *data = 0;
  if (!bits_to_read)
    return;
  if (bits_to_read >= bits_left_in_byte)
  {
    do
    {
      *data <<= bits_left_in_byte;
      bits_left -= bits_left_in_byte;
      *data |= info->cur_buf[info->cur_pos] & ((1 << bits_left_in_byte) - 1);
      ++info->cur_pos;
      fuji_fill_buffer(info);
      bits_left_in_byte = 8;
    } while (bits_left >= 8);
  }
  if (!bits_left)
  {
    info->cur_bit = (8 - (bits_left_in_byte & 7)) & 7;
    return;
  }
  *data <<= bits_left;
  bits_left_in_byte -= bits_left;
  *data |= ((1 << bits_left) - 1) & ((unsigned)info->cur_buf[info->cur_pos] >> bits_left_in_byte);
  info->cur_bit = (8 - (bits_left_in_byte & 7)) & 7;
}

static inline int bitDiff(int value1, int value2)
{
  int decBits = 0;
  if (value2 < value1)
    while (decBits <= 14 && (value2 << ++decBits) < value1)
      ;
  return decBits;
}

static inline int fuji_decode_sample_even(fuji_compressed_block *info, const fuji_compressed_params *params,
                                          ushort *line_buf, int pos, fuji_grads *grad_params)
{
  int interp_val = 0;
  // ushort decBits;
  int errcnt = 0;

  int sample = 0, code = 0;
  ushort *line_buf_cur = line_buf + pos;
  int Rb = line_buf_cur[-2 - params->line_width];
  int Rc = line_buf_cur[-3 - params->line_width];
  int Rd = line_buf_cur[-1 - params->line_width];
  int Rf = line_buf_cur[-4 - 2 * params->line_width];

  int grad, gradient, diffRcRb, diffRfRb, diffRdRb;

  diffRcRb = _abs(Rc - Rb);
  diffRfRb = _abs(Rf - Rb);
  diffRdRb = _abs(Rd - Rb);

  const fuji_q_table *qt = params->qt;
  int_pair *grads = grad_params->grads;
  for (int i = 1; params->qt[0].q_base >= i && i < 4; ++i)
    if (diffRfRb + diffRcRb <= params->qt[i].max_grad)
    {
      qt = params->qt + i;
      grads = grad_params->lossy_grads[i - 1];
      break;
    }

  grad = fuji_quant_gradient(params->max_value, qt, Rb - Rf, Rc - Rb);
  gradient = _abs(grad);

  if (diffRcRb > diffRfRb && diffRcRb > diffRdRb)
    interp_val = Rf + Rd + 2 * Rb;
  else if (diffRdRb > diffRcRb && diffRdRb > diffRfRb)
    interp_val = Rf + Rc + 2 * Rb;
  else
    interp_val = Rd + Rc + 2 * Rb;

  fuji_zerobits(info, &sample);

  if (sample < params->max_bits - qt->raw_bits - 1)
  {
    int decBits = bitDiff(grads[gradient].value1, grads[gradient].value2);
    fuji_read_code(info, &code, decBits);
    code += sample << decBits;
  }
  else
  {
    fuji_read_code(info, &code, qt->raw_bits);
    ++code;
  }

  if (code < 0 || code >= qt->total_values)
    ++errcnt;

  if (code & 1)
    code = -1 - code / 2;
  else
    code /= 2;

  grads[gradient].value1 += _abs(code);
  if (grads[gradient].value2 == params->min_value)
  {
    grads[gradient].value1 >>= 1;
    grads[gradient].value2 >>= 1;
  }
  ++grads[gradient].value2;
  if (grad < 0)
    interp_val = (interp_val >> 2) - code * (2 * qt->q_base + 1);
  else
    interp_val = (interp_val >> 2) + code * (2 * qt->q_base + 1);
  if (interp_val < -qt->q_base)
    interp_val += qt->total_values * (2 * qt->q_base + 1);
  else if (interp_val > qt->q_base + params->max_value)
    interp_val -= qt->total_values * (2 * qt->q_base + 1);

  if (interp_val >= 0)
    line_buf_cur[0] = _min(interp_val, params->max_value);
  else
    line_buf_cur[0] = 0;
  return errcnt;
}

static inline int fuji_decode_sample_odd(fuji_compressed_block *info, const fuji_compressed_params *params,
                                         ushort *line_buf, int pos, fuji_grads *grad_params)
{
  int interp_val = 0;
  int errcnt = 0;

  int sample = 0, code = 0;
  ushort *line_buf_cur = line_buf + pos;
  int Ra = line_buf_cur[-1];
  int Rb = line_buf_cur[-2 - params->line_width];
  int Rc = line_buf_cur[-3 - params->line_width];
  int Rd = line_buf_cur[-1 - params->line_width];
  int Rg = line_buf_cur[1];

  int grad, gradient;

  int diffRcRa = _abs(Rc - Ra);
  int diffRbRc = _abs(Rb - Rc);

  const fuji_q_table *qt = params->qt;
  int_pair *grads = grad_params->grads;
  for (int i = 1; params->qt[0].q_base >= i && i < 4; ++i)
    if (diffRbRc + diffRcRa <= params->qt[i].max_grad)
    {
      qt = params->qt + i;
      grads = grad_params->lossy_grads[i - 1];
      break;
    }

  grad = fuji_quant_gradient(params->max_value, qt, Rb - Rc, Rc - Ra);
  gradient = _abs(grad);

  if ((Rb > Rc && Rb > Rd) || (Rb < Rc && Rb < Rd))
    interp_val = (Rg + Ra + 2 * Rb) >> 2;
  else
    interp_val = (Ra + Rg) >> 1;

  fuji_zerobits(info, &sample);

  if (sample < params->max_bits - qt->raw_bits - 1)
  {
    int decBits = bitDiff(grads[gradient].value1, grads[gradient].value2);
    fuji_read_code(info, &code, decBits);
    code += sample << decBits;
  }
  else
  {
    fuji_read_code(info, &code, qt->raw_bits);
    ++code;
  }

  if (code < 0 || code >= qt->total_values)
    ++errcnt;

  if (code & 1)
    code = -1 - code / 2;
  else
    code /= 2;

  grads[gradient].value1 += _abs(code);
  if (grads[gradient].value2 == params->min_value)
  {
    grads[gradient].value1 >>= 1;
    grads[gradient].value2 >>= 1;
  }
  ++grads[gradient].value2;
  if (grad < 0)
    interp_val -= code * (2 * qt->q_base + 1);
  else
    interp_val += code * (2 * qt->q_base + 1);
  if (interp_val < -qt->q_base)
    interp_val += qt->total_values * (2 * qt->q_base + 1);
  else if (interp_val > qt->q_base + params->max_value)
    interp_val -= qt->total_values * (2 * qt->q_base + 1);

  if (interp_val >= 0)
    line_buf_cur[0] = _min(interp_val, params->max_value);
  else
    line_buf_cur[0] = 0;
  return errcnt;
}

static void fuji_decode_interpolation_even(int line_width, ushort *line_buf, int pos)
{
  ushort *line_buf_cur = line_buf + pos;
  int Rb = line_buf_cur[-2 - line_width];
  int Rc = line_buf_cur[-3 - line_width];
  int Rd = line_buf_cur[-1 - line_width];
  int Rf = line_buf_cur[-4 - 2 * line_width];
  int diffRcRb = _abs(Rc - Rb);
  int diffRfRb = _abs(Rf - Rb);
  int diffRdRb = _abs(Rd - Rb);
  if (diffRcRb > diffRfRb && diffRcRb > diffRdRb)
    *line_buf_cur = (Rf + Rd + 2 * Rb) >> 2;
  else if (diffRdRb > diffRcRb && diffRdRb > diffRfRb)
    *line_buf_cur = (Rf + Rc + 2 * Rb) >> 2;
  else
    *line_buf_cur = (Rd + Rc + 2 * Rb) >> 2;
}

static void fuji_extend_generic(ushort *linebuf[_ltotal], int line_width, int start, int end)
{
  for (int i = start; i <= end; i++)
  {
    linebuf[i][0] = linebuf[i - 1][1];
    linebuf[i][line_width + 1] = linebuf[i - 1][line_width];
  }
}

static void fuji_extend_red(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _R2, _R4);
}

static void fuji_extend_green(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _G2, _G7);
}

static void fuji_extend_blue(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _B2, _B4);
}

void LibRaw::xtrans_decode_block(fuji_compressed_block *info, const fuji_compressed_params *params, int /*cur_line*/)
{
  int r_even_pos = 0, r_odd_pos = 1;
  int g_even_pos = 0, g_odd_pos = 1;
  int b_even_pos = 0, b_odd_pos = 1;

  int errcnt = 0;

  const int line_width = params->line_width;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      fuji_decode_interpolation_even(line_width, info->linebuf[_R2] + 1, r_even_pos);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G2] + 1, g_even_pos, &info->even[0]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R2] + 1, r_odd_pos, &info->odd[0]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G2] + 1, g_odd_pos, &info->odd[0]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G3] + 1, g_even_pos, &info->even[1]);
      g_even_pos += 2;
      fuji_decode_interpolation_even(line_width, info->linebuf[_B2] + 1, b_even_pos);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G3] + 1, g_odd_pos, &info->odd[1]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B2] + 1, b_odd_pos, &info->odd[1]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      if (r_even_pos & 3)
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R3] + 1, r_even_pos, &info->even[2]);
      else
        fuji_decode_interpolation_even(line_width, info->linebuf[_R3] + 1, r_even_pos);
      r_even_pos += 2;
      fuji_decode_interpolation_even(line_width, info->linebuf[_G4] + 1, g_even_pos);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R3] + 1, r_odd_pos, &info->odd[2]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G4] + 1, g_odd_pos, &info->odd[2]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G5] + 1, g_even_pos, &info->even[0]);
      g_even_pos += 2;
      if ((b_even_pos & 3) == 2)
        fuji_decode_interpolation_even(line_width, info->linebuf[_B3] + 1, b_even_pos);
      else
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B3] + 1, b_even_pos, &info->even[0]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G5] + 1, g_odd_pos, &info->odd[0]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B3] + 1, b_odd_pos, &info->odd[0]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      if ((r_even_pos & 3) == 2)
        fuji_decode_interpolation_even(line_width, info->linebuf[_R4] + 1, r_even_pos);
      else
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R4] + 1, r_even_pos, &info->even[1]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G6] + 1, g_even_pos, &info->even[1]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R4] + 1, r_odd_pos, &info->odd[1]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G6] + 1, g_odd_pos, &info->odd[1]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      fuji_decode_interpolation_even(line_width, info->linebuf[_G7] + 1, g_even_pos);
      g_even_pos += 2;
      if (b_even_pos & 3)
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B4] + 1, b_even_pos, &info->even[2]);
      else
        fuji_decode_interpolation_even(line_width, info->linebuf[_B4] + 1, b_even_pos);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G7] + 1, g_odd_pos, &info->odd[2]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B4] + 1, b_odd_pos, &info->odd[2]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  if (errcnt)
    derror();
}

void LibRaw::fuji_bayer_decode_block(fuji_compressed_block *info, const fuji_compressed_params *params, int /*cur_line*/)
{
  int r_even_pos = 0, r_odd_pos = 1;
  int g_even_pos = 0, g_odd_pos = 1;
  int b_even_pos = 0, b_odd_pos = 1;

  int errcnt = 0;

  const int line_width = params->line_width;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R2] + 1, r_even_pos, &info->even[0]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G2] + 1, g_even_pos, &info->even[0]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R2] + 1, r_odd_pos, &info->odd[0]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G2] + 1, g_odd_pos, &info->odd[0]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G3] + 1, g_even_pos, &info->even[1]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B2] + 1, b_even_pos, &info->even[1]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G3] + 1, g_odd_pos, &info->odd[1]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B2] + 1, b_odd_pos, &info->odd[1]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R3] + 1, r_even_pos, &info->even[2]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G4] + 1, g_even_pos, &info->even[2]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R3] + 1, r_odd_pos, &info->odd[2]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G4] + 1, g_odd_pos, &info->odd[2]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G5] + 1, g_even_pos, &info->even[0]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B3] + 1, b_even_pos, &info->even[0]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G5] + 1, g_odd_pos, &info->odd[0]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B3] + 1, b_odd_pos, &info->odd[0]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R4] + 1, r_even_pos, &info->even[1]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G6] + 1, g_even_pos, &info->even[1]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R4] + 1, r_odd_pos, &info->odd[1]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G6] + 1, g_odd_pos, &info->odd[1]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G7] + 1, g_even_pos, &info->even[2]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B4] + 1, b_even_pos, &info->even[2]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G7] + 1, g_odd_pos, &info->odd[2]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B4] + 1, b_odd_pos, &info->odd[2]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  if (errcnt)
    derror();
}

void LibRaw::fuji_decode_strip(fuji_compressed_params *params, int cur_block, INT64 raw_offset, unsigned dsize,
                               uchar *q_bases)
{
  int cur_block_width, cur_line;
  unsigned line_size;
  fuji_compressed_block info;
  fuji_compressed_params *info_common = params;

  if (!libraw_internal_data.unpacker_data.fuji_lossless)
  {
    int buf_size = sizeof(fuji_compressed_params) + (2 << libraw_internal_data.unpacker_data.fuji_bits);

    info_common = (fuji_compressed_params *)malloc(buf_size);
    memcpy(info_common, params, sizeof(fuji_compressed_params));
    info_common->qt[0].q_table = (int8_t *)(info_common + 1);
    info_common->qt[0].q_base = -1;
  }
  init_fuji_block(&info, info_common, raw_offset, dsize);
  line_size = sizeof(ushort) * (info_common->line_width + 2);

  cur_block_width = libraw_internal_data.unpacker_data.fuji_block_width;
  if (cur_block + 1 == libraw_internal_data.unpacker_data.fuji_total_blocks)
  {
    cur_block_width = imgdata.sizes.raw_width - (libraw_internal_data.unpacker_data.fuji_block_width * cur_block);
    /* Old code, may get incorrect results on GFX50, but luckily large optical
    black cur_block_width = imgdata.sizes.raw_width %
    libraw_internal_data.unpacker_data.fuji_block_width;
    */
  }

  struct i_pair
  {
    int a, b;
  };
  const i_pair mtable[6] = {{_R0, _R3}, {_R1, _R4}, {_G0, _G6}, {_G1, _G7}, {_B0, _B3}, {_B1, _B4}},
               ztable[3] = {{_R2, 3}, {_G2, 6}, {_B2, 3}};
  for (cur_line = 0; cur_line < libraw_internal_data.unpacker_data.fuji_total_lines; cur_line++)
  {
    // init grads and main qtable
    if (!libraw_internal_data.unpacker_data.fuji_lossless)
    {
      int q_base = q_bases ? q_bases[cur_line] : 0;
      if (!cur_line || q_base != info_common->qt[0].q_base)
      {
        init_main_qtable(info_common, q_bases[cur_line]);
        init_main_grads(info_common, &info);
      }
    }

    if (libraw_internal_data.unpacker_data.fuji_raw_type == 16)
      xtrans_decode_block(&info, info_common, cur_line);
    else
      fuji_bayer_decode_block(&info, info_common, cur_line);

    // copy data from line buffers and advance
    for (int i = 0; i < 6; i++)
      memcpy(info.linebuf[mtable[i].a], info.linebuf[mtable[i].b], line_size);

    if (libraw_internal_data.unpacker_data.fuji_raw_type == 16)
      copy_line_to_xtrans(&info, cur_line, cur_block, cur_block_width);
    else
      copy_line_to_bayer(&info, cur_line, cur_block, cur_block_width);

    for (int i = 0; i < 3; i++)
    {
      memset(info.linebuf[ztable[i].a], 0, ztable[i].b * line_size);
      info.linebuf[ztable[i].a][0] = info.linebuf[ztable[i].a - 1][1];
      info.linebuf[ztable[i].a][info_common->line_width + 1] = info.linebuf[ztable[i].a - 1][info_common->line_width];
    }
  }

  // release data
  if (!libraw_internal_data.unpacker_data.fuji_lossless)
    free(info_common);
  free(info.linealloc);
  free(info.cur_buf);
}

void LibRaw::fuji_compressed_load_raw()
{
  fuji_compressed_params common_info;
  int cur_block;
  unsigned *block_sizes;
  uchar *q_bases = 0;
  INT64 raw_offset, *raw_block_offsets;

  init_fuji_compr(&common_info);

  // read block sizes
  block_sizes = (unsigned *)malloc(sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks);
  raw_block_offsets = (INT64 *)malloc(sizeof(INT64) * libraw_internal_data.unpacker_data.fuji_total_blocks);

  libraw_internal_data.internal_data.input->seek(libraw_internal_data.unpacker_data.data_offset, SEEK_SET);
  int sizesToRead = sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks;
  if (libraw_internal_data.internal_data.input->read(block_sizes, 1, sizesToRead) != sizesToRead)
  {
    free(block_sizes);
    free(raw_block_offsets);
    throw LIBRAW_EXCEPTION_IO_EOF;
  }

  raw_offset = ((sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks) + 0xF) & ~0xF;

  // read q bases for lossy
  if (!libraw_internal_data.unpacker_data.fuji_lossless)
  {
    int total_q_bases = libraw_internal_data.unpacker_data.fuji_total_blocks *
                        ((libraw_internal_data.unpacker_data.fuji_total_lines + 0xF) & ~0xF);
    q_bases = (uchar *)malloc(total_q_bases);
    libraw_internal_data.internal_data.input->seek(raw_offset + libraw_internal_data.unpacker_data.data_offset,
                                                   SEEK_SET);
    libraw_internal_data.internal_data.input->read(q_bases, 1, total_q_bases);
    raw_offset += total_q_bases;
  }

  raw_offset += libraw_internal_data.unpacker_data.data_offset;

  // calculating raw block offsets
  raw_block_offsets[0] = raw_offset;
  for (cur_block = 0; cur_block < libraw_internal_data.unpacker_data.fuji_total_blocks; cur_block++)
  {
    unsigned bsize = sgetn(4, (uchar *)(block_sizes + cur_block));
    block_sizes[cur_block] = bsize;
  }

  for (cur_block = 1; cur_block < libraw_internal_data.unpacker_data.fuji_total_blocks; cur_block++)
    raw_block_offsets[cur_block] = raw_block_offsets[cur_block - 1] + block_sizes[cur_block - 1];

  fuji_decode_loop(&common_info, libraw_internal_data.unpacker_data.fuji_total_blocks, raw_block_offsets, block_sizes,
                   q_bases);

  free(q_bases);
  free(block_sizes);
  free(raw_block_offsets);
  free(common_info.buf);
}

void LibRaw::fuji_decode_loop(fuji_compressed_params *common_info, int count, INT64 *raw_block_offsets,
                              unsigned *block_sizes, uchar *q_bases)
{
  int cur_block;
  const int lineStep = (libraw_internal_data.unpacker_data.fuji_total_lines + 0xF) & ~0xF;
#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for private(cur_block)
#endif
  for (cur_block = 0; cur_block < count; cur_block++)
  {
    fuji_decode_strip(common_info, cur_block, raw_block_offsets[cur_block], block_sizes[cur_block],
                      q_bases ? q_bases + cur_block * lineStep : 0);
  }
}

void LibRaw::parse_fuji_compressed_header()
{
  unsigned signature, lossless, h_raw_type, h_raw_bits, h_raw_height, h_raw_rounded_width, h_raw_width, h_block_size,
      h_blocks_in_row, h_total_lines;

  uchar header[16];

  libraw_internal_data.internal_data.input->seek(libraw_internal_data.unpacker_data.data_offset, SEEK_SET);
  if (libraw_internal_data.internal_data.input->read(header, 1, sizeof(header)) != sizeof(header))
    return;

  // read all header
  signature = sgetn(2, header);
  lossless = header[2];
  h_raw_type = header[3];
  h_raw_bits = header[4];
  h_raw_height = sgetn(2, header + 5);
  h_raw_rounded_width = sgetn(2, header + 7);
  h_raw_width = sgetn(2, header + 9);
  h_block_size = sgetn(2, header + 11);
  h_blocks_in_row = header[13];
  h_total_lines = sgetn(2, header + 14);

  // general validation
  if (signature != 0x4953 || lossless > 1 || h_raw_height > 0x4002 || h_raw_height < 6 || h_raw_height % 6 ||
      h_block_size < 1 || h_raw_width > 0x4200 || h_raw_width < 0x300 || h_raw_width % 24 ||
      h_raw_rounded_width > 0x4200 || h_raw_rounded_width < h_block_size || h_raw_rounded_width % h_block_size ||
      h_raw_rounded_width - h_raw_width >= h_block_size || h_block_size != 0x300 || h_blocks_in_row > 0x10 ||
      h_blocks_in_row == 0 || h_blocks_in_row != h_raw_rounded_width / h_block_size || h_total_lines > 0xAAB ||
      h_total_lines == 0 || h_total_lines != h_raw_height / 6 ||
      (h_raw_bits != 12 && h_raw_bits != 14 && h_raw_bits != 16) || (h_raw_type != 16 && h_raw_type != 0))
    return;

  // modify data
  libraw_internal_data.unpacker_data.fuji_total_lines = h_total_lines;
  libraw_internal_data.unpacker_data.fuji_total_blocks = h_blocks_in_row;
  libraw_internal_data.unpacker_data.fuji_block_width = h_block_size;
  libraw_internal_data.unpacker_data.fuji_bits = h_raw_bits;
  libraw_internal_data.unpacker_data.fuji_raw_type = h_raw_type;
  libraw_internal_data.unpacker_data.fuji_lossless = lossless;
  imgdata.sizes.raw_width = h_raw_width;
  imgdata.sizes.raw_height = h_raw_height;
  libraw_internal_data.unpacker_data.data_offset += 16;
  load_raw = &LibRaw::fuji_compressed_load_raw;
}

#undef _abs
#undef _min
