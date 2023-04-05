#ifdef USE_X3FTOOLS

/* Library for accessing X3F Files
----------------------------------------------------------------
BSD-style License
----------------------------------------------------------------

* Copyright (c) 2010, Roland Karlsson (roland@proxel.se)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY ROLAND KARLSSON ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL ROLAND KARLSSON BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "../../internal/libraw_cxx_defs.h"

#if defined __sun && defined DS
#undef DS
#endif
#ifdef ID
#undef ID /* used in x3f utils */
#endif

#include "../../internal/x3f_tools.h"

/* extern */ int legacy_offset = 0;
/* extern */ bool_t auto_legacy_offset = 1;

/* --------------------------------------------------------------------- */
/* Reading and writing - assuming little endian in the file              */
/* --------------------------------------------------------------------- */

static int x3f_get1(LibRaw_abstract_datastream *f)
{
  /* Little endian file */
  return f->get_char();
}

static int x3f_sget2(uchar *s) { return s[0] | s[1] << 8; }

static int x3f_get2(LibRaw_abstract_datastream *f)
{
  uchar str[2] = {0xff, 0xff};
  f->read(str, 1, 2);
  return x3f_sget2(str);
}

unsigned x3f_sget4(uchar *s)
{
  return s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
}

unsigned x3f_get4(LibRaw_abstract_datastream *f)
{
  uchar str[4] = {0xff, 0xff, 0xff, 0xff};
  f->read(str, 1, 4);
  return x3f_sget4(str);
}

#define FREE(P)                                                                \
  do                                                                           \
  {                                                                            \
    free(P);                                                                   \
    (P) = NULL;                                                                \
  } while (0)

#define PUT_GET_N(_buffer, _size, _file, _func)                                \
  do                                                                           \
  {                                                                            \
    int _left = _size;                                                         \
    while (_left != 0)                                                         \
    {                                                                          \
      int _cur = _file->_func(_buffer, 1, _left);                              \
      if (_cur == 0)                                                           \
      {                                                                        \
        throw LIBRAW_EXCEPTION_IO_CORRUPT;                                     \
      }                                                                        \
      _left -= _cur;                                                           \
    }                                                                          \
  } while (0)

#define GET1(_v)                                                               \
  do                                                                           \
  {                                                                            \
    (_v) = x3f_get1(I->input.file);                                            \
  } while (0)
#define GET2(_v)                                                               \
  do                                                                           \
  {                                                                            \
    (_v) = x3f_get2(I->input.file);                                            \
  } while (0)
#define GET4(_v)                                                               \
  do                                                                           \
  {                                                                            \
    (_v) = x3f_get4(I->input.file);                                            \
  } while (0)

#define GET4F(_v)                                                              \
  do                                                                           \
  {                                                                            \
    union {                                                                    \
      int32_t i;                                                               \
      float f;                                                                 \
    } _tmp;                                                                    \
    _tmp.i = x3f_get4(I->input.file);                                          \
    (_v) = _tmp.f;                                                             \
  } while (0)

#define GETN(_v, _s) PUT_GET_N(_v, _s, I->input.file, read)

#define GET_TABLE(_T, _GETX, _NUM, _TYPE)                                      \
  do                                                                           \
  {                                                                            \
    int _i;                                                                    \
    (_T).size = (_NUM);                                                        \
    (_T).element =                                                             \
        (_TYPE *)realloc((_T).element, (_NUM) * sizeof((_T).element[0]));      \
    for (_i = 0; _i < (int)(_T).size; _i++)                                         \
      _GETX((_T).element[_i]);                                                 \
  } while (0)

#define GET_PROPERTY_TABLE(_T, _NUM)                                           \
  do                                                                           \
  {                                                                            \
    int _i;                                                                    \
    (_T).size = (_NUM);                                                        \
    (_T).element = (x3f_property_t *)realloc(                                  \
        (_T).element, (_NUM) * sizeof((_T).element[0]));                       \
    for (_i = 0; _i < (int)(_T).size; _i++)                                         \
    {                                                                          \
      GET4((_T).element[_i].name_offset);                                      \
      GET4((_T).element[_i].value_offset);                                     \
    }                                                                          \
  } while (0)

#define GET_TRUE_HUFF_TABLE(_T)                                                \
  do                                                                           \
  {                                                                            \
    int _i;                                                                    \
    (_T).element = NULL;                                                       \
    for (_i = 0;; _i++)                                                        \
    {                                                                          \
      (_T).size = _i + 1;                                                      \
      (_T).element = (x3f_true_huffman_element_t *)realloc(                    \
          (_T).element, (_i + 1) * sizeof((_T).element[0]));                   \
      GET1((_T).element[_i].code_size);                                        \
      GET1((_T).element[_i].code);                                             \
      if ((_T).element[_i].code_size == 0)                                     \
        break;                                                                 \
    }                                                                          \
  } while (0)

/* --------------------------------------------------------------------- */
/* Allocating Huffman tree help data                                   */
/* --------------------------------------------------------------------- */

static void cleanup_huffman_tree(x3f_hufftree_t *HTP) { free(HTP->nodes); }

static void new_huffman_tree(x3f_hufftree_t *HTP, int bits)
{
  int leaves = 1 << bits;

  HTP->free_node_index = 0;
  HTP->total_node_index = HUF_TREE_MAX_NODES(leaves);
  HTP->nodes = (x3f_huffnode_t *)calloc(1, HUF_TREE_MAX_NODES(leaves) *
                                               sizeof(x3f_huffnode_t));
}

/* --------------------------------------------------------------------- */
/* Allocating TRUE engine RAW help data                                  */
/* --------------------------------------------------------------------- */

static void cleanup_true(x3f_true_t **TRUP)
{
  x3f_true_t *TRU = *TRUP;

  if (TRU == NULL)
    return;

  FREE(TRU->table.element);
  FREE(TRU->plane_size.element);
  cleanup_huffman_tree(&TRU->tree);
  FREE(TRU->x3rgb16.buf);

  FREE(TRU);

  *TRUP = NULL;
}

static x3f_true_t *new_true(x3f_true_t **TRUP)
{
  x3f_true_t *TRU = (x3f_true_t *)calloc(1, sizeof(x3f_true_t));

  cleanup_true(TRUP);

  TRU->table.size = 0;
  TRU->table.element = NULL;
  TRU->plane_size.size = 0;
  TRU->plane_size.element = NULL;
  TRU->tree.nodes = NULL;
  TRU->x3rgb16.data = NULL;
  TRU->x3rgb16.buf = NULL;

  *TRUP = TRU;

  return TRU;
}

static void cleanup_quattro(x3f_quattro_t **QP)
{
  x3f_quattro_t *Q = *QP;

  if (Q == NULL)
    return;

  FREE(Q->top16.buf);
  FREE(Q);

  *QP = NULL;
}

static x3f_quattro_t *new_quattro(x3f_quattro_t **QP)
{
  x3f_quattro_t *Q = (x3f_quattro_t *)calloc(1, sizeof(x3f_quattro_t));
  int i;

  cleanup_quattro(QP);

  for (i = 0; i < TRUE_PLANES; i++)
  {
    Q->plane[i].columns = 0;
    Q->plane[i].rows = 0;
  }

  Q->unknown = 0;

  Q->top16.data = NULL;
  Q->top16.buf = NULL;

  *QP = Q;

  return Q;
}

/* --------------------------------------------------------------------- */
/* Allocating Huffman engine help data                                   */
/* --------------------------------------------------------------------- */

static void cleanup_huffman(x3f_huffman_t **HUFP)
{
  x3f_huffman_t *HUF = *HUFP;

  if (HUF == NULL)
    return;

  FREE(HUF->mapping.element);
  FREE(HUF->table.element);
  cleanup_huffman_tree(&HUF->tree);
  FREE(HUF->row_offsets.element);
  FREE(HUF->rgb8.buf);
  FREE(HUF->x3rgb16.buf);
  FREE(HUF);

  *HUFP = NULL;
}

static x3f_huffman_t *new_huffman(x3f_huffman_t **HUFP)
{
  x3f_huffman_t *HUF = (x3f_huffman_t *)calloc(1, sizeof(x3f_huffman_t));

  cleanup_huffman(HUFP);

  /* Set all not read data block pointers to NULL */
  HUF->mapping.size = 0;
  HUF->mapping.element = NULL;
  HUF->table.size = 0;
  HUF->table.element = NULL;
  HUF->tree.nodes = NULL;
  HUF->row_offsets.size = 0;
  HUF->row_offsets.element = NULL;
  HUF->rgb8.data = NULL;
  HUF->rgb8.buf = NULL;
  HUF->x3rgb16.data = NULL;
  HUF->x3rgb16.buf = NULL;

  *HUFP = HUF;

  return HUF;
}

/* --------------------------------------------------------------------- */
/* Creating a new x3f structure from file                                */
/* --------------------------------------------------------------------- */

/* extern */ x3f_t *x3f_new_from_file(LibRaw_abstract_datastream *infile)
{
  if (!infile)
    return NULL;
  INT64 fsize = infile->size();
  x3f_t *x3f = (x3f_t *)calloc(1, sizeof(x3f_t));
  if (!x3f)
    throw LIBRAW_EXCEPTION_ALLOC;
  try
  {
    x3f_info_t *I = NULL;
    x3f_header_t *H = NULL;
    x3f_directory_section_t *DS = NULL;
    int i, d;

    I = &x3f->info;
    I->error = NULL;
    I->input.file = infile;
    I->output.file = NULL;

    /* Read file header */
    H = &x3f->header;
    infile->seek(0, SEEK_SET);
    GET4(H->identifier);

    if (H->identifier != X3F_FOVb)
    {
      free(x3f);
      return NULL;
    }

    GET4(H->version);
    GETN(H->unique_identifier, SIZE_UNIQUE_IDENTIFIER);
    /* TODO: the meaning of the rest of the header for version >= 4.0 (Quattro)
     * is unknown */
    if (H->version < X3F_VERSION_4_0)
    {
      GET4(H->mark_bits);
      GET4(H->columns);
      GET4(H->rows);
      GET4(H->rotation);
      if (H->version >= X3F_VERSION_2_1)
      {
        int num_ext_data =
            H->version >= X3F_VERSION_3_0 ? NUM_EXT_DATA_3_0 : NUM_EXT_DATA_2_1;

        GETN(H->white_balance, SIZE_WHITE_BALANCE);
        if (H->version >= X3F_VERSION_2_3)
          GETN(H->color_mode, SIZE_COLOR_MODE);
        GETN(H->extended_types, num_ext_data);
        for (i = 0; i < num_ext_data; i++)
          GET4F(H->extended_data[i]);
      }
    }

    /* Go to the beginning of the directory */
    infile->seek(-4, SEEK_END);
    infile->seek(x3f_get4(infile), SEEK_SET);

    /* Read the directory header */
    DS = &x3f->directory_section;
    GET4(DS->identifier);
    GET4(DS->version);
    GET4(DS->num_directory_entries);

    if (DS->num_directory_entries > 50)
      goto _err; // too much direntries, most likely broken file

    if (DS->num_directory_entries > 0)
    {
      size_t size = DS->num_directory_entries * sizeof(x3f_directory_entry_t);
      DS->directory_entry = (x3f_directory_entry_t *)calloc(1, size);
    }

    /* Traverse the directory */
    for (d = 0; d < (int)DS->num_directory_entries; d++)
    {
      x3f_directory_entry_t *DE = &DS->directory_entry[d];
      x3f_directory_entry_header_t *DEH = &DE->header;
      uint32_t save_dir_pos;

      /* Read the directory entry info */
      GET4(DE->input.offset);
      GET4(DE->input.size);
      if (DE->input.offset + DE->input.size > fsize * 2)
        goto _err;

      DE->output.offset = 0;
      DE->output.size = 0;

      GET4(DE->type);

      /* Save current pos and go to the entry */
      save_dir_pos = infile->tell();
      infile->seek(DE->input.offset, SEEK_SET);

      /* Read the type independent part of the entry header */
      DEH = &DE->header;
      GET4(DEH->identifier);
      GET4(DEH->version);

      /* NOTE - the tests below could be made on DE->type instead */

      if (DEH->identifier == X3F_SECp)
      {
        x3f_property_list_t *PL = &DEH->data_subsection.property_list;
        if (!PL)
          goto _err;
        /* Read the property part of the header */
        GET4(PL->num_properties);
        GET4(PL->character_format);
        GET4(PL->reserved);
        GET4(PL->total_length);

        /* Set all not read data block pointers to NULL */
        PL->data = NULL;
        PL->data_size = 0;
      }

      if (DEH->identifier == X3F_SECi)
      {
        x3f_image_data_t *ID = &DEH->data_subsection.image_data;
        if (!ID)
          goto _err;
        /* Read the image part of the header */
        GET4(ID->type);
        GET4(ID->format);
        ID->type_format = (ID->type << 16) + (ID->format);
        GET4(ID->columns);
        GET4(ID->rows);
        GET4(ID->row_stride);

        /* Set all not read data block pointers to NULL */
        ID->huffman = NULL;

        ID->data = NULL;
        ID->data_size = 0;
      }

      if (DEH->identifier == X3F_SECc)
      {
        x3f_camf_t *CAMF = &DEH->data_subsection.camf;
        if (!CAMF)
          goto _err;
        /* Read the CAMF part of the header */
        GET4(CAMF->type);
        GET4(CAMF->tN.val0);
        GET4(CAMF->tN.val1);
        GET4(CAMF->tN.val2);
        GET4(CAMF->tN.val3);

        /* Set all not read data block pointers to NULL */
        CAMF->data = NULL;
        CAMF->data_size = 0;

        /* Set all not allocated help pointers to NULL */
        CAMF->table.element = NULL;
        CAMF->table.size = 0;
        CAMF->tree.nodes = NULL;
        CAMF->decoded_data = NULL;
        CAMF->decoded_data_size = 0;
        CAMF->entry_table.element = NULL;
        CAMF->entry_table.size = 0;
      }

      /* Reset the file pointer back to the directory */
      infile->seek(save_dir_pos, SEEK_SET);
    }

    return x3f;
  _err:
    if (x3f)
    {
      DS = &x3f->directory_section;
      if (DS && DS->directory_entry)
        free(DS->directory_entry);
      free(x3f);
    }
    return NULL;
  }
  catch (...)
  {
    x3f_directory_section_t *DS = &x3f->directory_section;
    if (DS && DS->directory_entry)
      free(DS->directory_entry);
    free(x3f);
    return NULL;
  }
}

/* --------------------------------------------------------------------- */
/* Clean up an x3f structure                                             */
/* --------------------------------------------------------------------- */

static void free_camf_entry(camf_entry_t *entry)
{
  FREE(entry->property_name);
  FREE(entry->property_value);
  FREE(entry->matrix_decoded);
  FREE(entry->matrix_dim_entry);
}

/* extern */ x3f_return_t x3f_delete(x3f_t *x3f)
{
  x3f_directory_section_t *DS;
  int d;

  if (x3f == NULL)
    return X3F_ARGUMENT_ERROR;

  DS = &x3f->directory_section;
  if (DS->num_directory_entries > 50)
    return X3F_ARGUMENT_ERROR;

  for (d = 0; d < (int)DS->num_directory_entries; d++)
  {
    x3f_directory_entry_t *DE = &DS->directory_entry[d];
    x3f_directory_entry_header_t *DEH = &DE->header;
    if (DEH->identifier == X3F_SECp)
    {
      x3f_property_list_t *PL = &DEH->data_subsection.property_list;
      FREE(PL->property_table.element);
      FREE(PL->data);
    }

    if (DEH->identifier == X3F_SECi)
    {
      x3f_image_data_t *ID = &DEH->data_subsection.image_data;

      if (ID)
      {
        cleanup_huffman(&ID->huffman);
        cleanup_true(&ID->tru);
        cleanup_quattro(&ID->quattro);
        FREE(ID->data);
      }
    }

    if (DEH->identifier == X3F_SECc)
    {
      x3f_camf_t *CAMF = &DEH->data_subsection.camf;
      int i;
      if (CAMF)
      {
        FREE(CAMF->data);
        FREE(CAMF->table.element);
        cleanup_huffman_tree(&CAMF->tree);
        FREE(CAMF->decoded_data);
        for (i = 0; i < (int)CAMF->entry_table.size; i++)
        {
          free_camf_entry(&CAMF->entry_table.element[i]);
        }
      }
      FREE(CAMF->entry_table.element);
    }
  }

  FREE(DS->directory_entry);
  FREE(x3f);

  return X3F_OK;
}

/* --------------------------------------------------------------------- */
/* Getting a reference to a directory entry                              */
/* --------------------------------------------------------------------- */

/* TODO: all those only get the first instance */

static x3f_directory_entry_t *x3f_get(x3f_t *x3f, uint32_t type,
                                      uint32_t image_type)
{
  x3f_directory_section_t *DS;
  int d;

  if (x3f == NULL)
    return NULL;

  DS = &x3f->directory_section;

  for (d = 0; d < (int)DS->num_directory_entries; d++)
  {
    x3f_directory_entry_t *DE = &DS->directory_entry[d];
    x3f_directory_entry_header_t *DEH = &DE->header;

    if (DEH->identifier == type)
    {
      switch (DEH->identifier)
      {
      case X3F_SECi:
      {
        x3f_image_data_t *ID = &DEH->data_subsection.image_data;

        if (ID->type_format == image_type)
          return DE;
      }
      break;
      default:
        return DE;
      }
    }
  }

  return NULL;
}

/* extern */ x3f_directory_entry_t *x3f_get_raw(x3f_t *x3f)
{
  x3f_directory_entry_t *DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_HUFFMAN_X530)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_HUFFMAN_10BIT)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_TRUE)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_MERRILL)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_QUATTRO)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_SDQ)) != NULL)
    return DE;

  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_SDQH)) != NULL)
    return DE;
  if ((DE = x3f_get(x3f, X3F_SECi, X3F_IMAGE_RAW_SDQH2)) != NULL)
    return DE;

  return NULL;
}

/* extern */ x3f_directory_entry_t *x3f_get_thumb_plain(x3f_t *x3f)
{
  return x3f_get(x3f, X3F_SECi, X3F_IMAGE_THUMB_PLAIN);
}

/* extern */ x3f_directory_entry_t *x3f_get_thumb_huffman(x3f_t *x3f)
{
  return x3f_get(x3f, X3F_SECi, X3F_IMAGE_THUMB_HUFFMAN);
}

/* extern */ x3f_directory_entry_t *x3f_get_thumb_jpeg(x3f_t *x3f)
{
  return x3f_get(x3f, X3F_SECi, X3F_IMAGE_THUMB_JPEG);
}

/* extern */ x3f_directory_entry_t *x3f_get_camf(x3f_t *x3f)
{
  return x3f_get(x3f, X3F_SECc, 0);
}

/* extern */ x3f_directory_entry_t *x3f_get_prop(x3f_t *x3f)
{
  return x3f_get(x3f, X3F_SECp, 0);
}

/* For some obscure reason, the bit numbering is weird. It is
   generally some kind of "big endian" style - e.g. the bit 7 is the
   first in a byte and bit 31 first in a 4 byte int. For patterns in
   the huffman pattern table, bit 27 is the first bit and bit 26 the
   next one. */

#define PATTERN_BIT_POS(_len, _bit) ((_len) - (_bit)-1)
#define MEMORY_BIT_POS(_bit) PATTERN_BIT_POS(8, _bit)

/* --------------------------------------------------------------------- */
/* Huffman Decode                                                        */
/* --------------------------------------------------------------------- */

/* Make the huffman tree */

#ifdef DBG_PRNT
static char *display_code(int length, uint32_t code, char *buffer)
{
  int i;

  for (i = 0; i < length; i++)
  {
    int pos = PATTERN_BIT_POS(length, i);
    buffer[i] = ((code >> pos) & 1) == 0 ? '0' : '1';
  }

  buffer[i] = 0;

  return buffer;
}
#endif

static x3f_huffnode_t *new_node(x3f_hufftree_t *tree)
{
	if (tree->free_node_index >= tree->total_node_index)
		throw LIBRAW_EXCEPTION_IO_CORRUPT;
  x3f_huffnode_t *t = &tree->nodes[tree->free_node_index];

  t->branch[0] = NULL;
  t->branch[1] = NULL;
  t->leaf = UNDEFINED_LEAF;

  tree->free_node_index++;

  return t;
}

static void add_code_to_tree(x3f_hufftree_t *tree, int length, uint32_t code,
                             uint32_t value)
{
  int i;

  x3f_huffnode_t *t = tree->nodes;

  for (i = 0; i < length; i++)
  {
    int pos = PATTERN_BIT_POS(length, i);
    int bit = (code >> pos) & 1;
    x3f_huffnode_t *t_next = t->branch[bit];

    if (t_next == NULL)
      t_next = t->branch[bit] = new_node(tree);

    t = t_next;
  }

  t->leaf = value;
}

static void populate_true_huffman_tree(x3f_hufftree_t *tree,
                                       x3f_true_huffman_t *table)
{
  int i;

  new_node(tree);

  for (i = 0; i < (int)table->size; i++)
  {
    x3f_true_huffman_element_t *element = &table->element[i];
    uint32_t length = element->code_size;

    if (length != 0)
    {
      /* add_code_to_tree wants the code right adjusted */
      uint32_t code = ((element->code) >> (8 - length)) & 0xff;
      uint32_t value = i;

      add_code_to_tree(tree, length, code, value);

#ifdef DBG_PRNT
      {
        char buffer[100];

        x3f_printf(DEBUG, "H %5d : %5x : %5d : %02x %08x (%08x) (%s)\n", i, i,
                   value, length, code, value,
                   display_code(length, code, buffer));
      }
#endif
    }
  }
}

static void populate_huffman_tree(x3f_hufftree_t *tree, x3f_table32_t *table,
                                  x3f_table16_t *mapping)
{
  int i;

  new_node(tree);

  for (i = 0; i < (int)table->size; i++)
  {
    uint32_t element = table->element[i];

    if (element != 0)
    {
      uint32_t length = HUF_TREE_GET_LENGTH(element);
      uint32_t code = HUF_TREE_GET_CODE(element);
      uint32_t value;

      /* If we have a valid mapping table - then the value from the
         mapping table shall be used. Otherwise we use the current
         index in the table as value. */
      if (table->size == mapping->size)
        value = mapping->element[i];
      else
        value = i;

      add_code_to_tree(tree, length, code, value);

#ifdef DBG_PRNT
      {
        char buffer[100];

        x3f_printf(DEBUG, "H %5d : %5x : %5d : %02x %08x (%08x) (%s)\n", i, i,
                   value, length, code, element,
                   display_code(length, code, buffer));
      }
#endif
    }
  }
}

#ifdef DBG_PRNT
static void print_huffman_tree(x3f_huffnode_t *t, int length, uint32_t code)
{
  char buf1[100];
  char buf2[100];

  x3f_printf(DEBUG, "%*s (%s,%s) %s (%s)\n", length,
             length < 1 ? "-" : (code & 1) ? "1" : "0",
             t->branch[0] == NULL ? "-" : "0", t->branch[1] == NULL ? "-" : "1",
             t->leaf == UNDEFINED_LEAF ? "-"
                                       : (sprintf(buf1, "%x", t->leaf), buf1),
             display_code(length, code, buf2));

  code = code << 1;
  if (t->branch[0])
    print_huffman_tree(t->branch[0], length + 1, code + 0);
  if (t->branch[1])
    print_huffman_tree(t->branch[1], length + 1, code + 1);
}
#endif

/* Help machinery for reading bits in a memory */

typedef struct bit_state_s
{
  uint8_t *next_address;
  uint8_t bit_offset;
  uint8_t bits[8];
} bit_state_t;

static void set_bit_state(bit_state_t *BS, uint8_t *address)
{
  BS->next_address = address;
  BS->bit_offset = 8;
}

static uint8_t get_bit(bit_state_t *BS)
{
  if (BS->bit_offset == 8)
  {
    uint8_t byte = *BS->next_address;
    int i;

    for (i = 7; i >= 0; i--)
    {
      BS->bits[i] = byte & 1;
      byte = byte >> 1;
    }
    BS->next_address++;
    BS->bit_offset = 0;
  }

  return BS->bits[BS->bit_offset++];
}

/* Decode use the TRUE algorithm */

static int32_t get_true_diff(bit_state_t *BS, x3f_hufftree_t *HTP)
{
  int32_t diff;
  x3f_huffnode_t *node = &HTP->nodes[0];
  uint8_t bits;

  while (node->branch[0] != NULL || node->branch[1] != NULL)
  {
    uint8_t bit = get_bit(BS);
    x3f_huffnode_t *new_node = node->branch[bit];

    node = new_node;
    if (node == NULL)
    {
      /* TODO: Shouldn't this be treated as a fatal error? */
      return 0;
    }
  }

  bits = node->leaf;

  if (bits == 0)
    diff = 0;
  else
  {
    uint8_t first_bit = get_bit(BS);
    int i;

    diff = first_bit;

    for (i = 1; i < bits; i++)
      diff = (diff << 1) + get_bit(BS);

    if (first_bit == 0)
      diff -= (1 << bits) - 1;
  }

  return diff;
}

/* This code (that decodes one of the X3F color planes, really is a
   decoding of a compression algorithm suited for Bayer CFA data. In
   Bayer CFA the data is divided into 2x2 squares that represents
   (R,G1,G2,B) data. Those four positions are (in this compression)
   treated as one data stream each, where you store the differences to
   previous data in the stream. The reason for this is, of course,
   that the date is more often than not near to the next data in a
   stream that represents the same color. */

/* TODO: write more about the compression */

static void true_decode_one_color(x3f_image_data_t *ID, int color)
{
  x3f_true_t *TRU = ID->tru;
  x3f_quattro_t *Q = ID->quattro;
  uint32_t seed = TRU->seed[color]; /* TODO : Is this correct ? */
  int row;

  x3f_hufftree_t *tree = &TRU->tree;
  bit_state_t BS;

  int32_t row_start_acc[2][2];
  uint32_t rows = ID->rows;
  uint32_t cols = ID->columns;
  x3f_area16_t *area = &TRU->x3rgb16;
  uint16_t *dst = area->data + color;

  set_bit_state(&BS, TRU->plane_address[color]);

  row_start_acc[0][0] = seed;
  row_start_acc[0][1] = seed;
  row_start_acc[1][0] = seed;
  row_start_acc[1][1] = seed;

  if (ID->type_format == X3F_IMAGE_RAW_QUATTRO ||
      ID->type_format == X3F_IMAGE_RAW_SDQ ||
      ID->type_format == X3F_IMAGE_RAW_SDQH ||
      ID->type_format == X3F_IMAGE_RAW_SDQH2)
  {
    rows = Q->plane[color].rows;
    cols = Q->plane[color].columns;

    if (Q->quattro_layout && color == 2)
    {
      area = &Q->top16;
      dst = area->data;
    }
  }
  else
  {
  }

  if (rows != area->rows || cols < area->columns)
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  for (row = 0; row < (int)rows; row++)
  {
    int col;
    bool_t odd_row = row & 1;
    int32_t acc[2];

    for (col = 0; col < (int)cols; col++)
    {
      bool_t odd_col = col & 1;
      int32_t diff = get_true_diff(&BS, tree);
      int32_t prev = col < 2 ? row_start_acc[odd_row][odd_col] : acc[odd_col];
      int32_t value = prev + diff;

      acc[odd_col] = value;
      if (col < 2)
        row_start_acc[odd_row][odd_col] = value;

      /* Discard additional data at the right for binned Quattro plane 2 */
      if (col >= (int)area->columns)
        continue;

      *dst = value;
      dst += area->channels;
    }
  }
}

static void true_decode(x3f_info_t * /*I*/, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  int color;

  for (color = 0; color < 3; color++)
  {
    true_decode_one_color(ID, color);
  }
}

/* Decode use the huffman tree */

static int32_t get_huffman_diff(bit_state_t *BS, x3f_hufftree_t *HTP)
{
  int32_t diff;
  x3f_huffnode_t *node = &HTP->nodes[0];

  while (node->branch[0] != NULL || node->branch[1] != NULL)
  {
    uint8_t bit = get_bit(BS);
    x3f_huffnode_t *new_node = node->branch[bit];

    node = new_node;
    if (node == NULL)
    {
      /* TODO: Shouldn't this be treated as a fatal error? */
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
      return 0; /* unreachable code */
    }
  }

  diff = node->leaf;

  return diff;
}

static void huffman_decode_row(x3f_info_t * /*I*/, x3f_directory_entry_t *DE,
                               int /*bits*/, int row, int offset, int *minimum)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  x3f_huffman_t *HUF = ID->huffman;

  int16_t c[3] = {(int16_t)offset, (int16_t)offset, (int16_t)offset};
  int col;
  bit_state_t BS;

  if (HUF->row_offsets.element[row] > ID->data_size - 1)
	  throw LIBRAW_EXCEPTION_IO_CORRUPT;
  set_bit_state(&BS, (uint8_t *)ID->data + HUF->row_offsets.element[row]);

  for (col = 0; col < (int)ID->columns; col++)
  {
    int color;

    for (color = 0; color < 3; color++)
    {
      uint16_t c_fix;

      c[color] += get_huffman_diff(&BS, &HUF->tree);
      if (c[color] < 0)
      {
        c_fix = 0;
        if (c[color] < *minimum)
          *minimum = c[color];
      }
      else
      {
        c_fix = c[color];
      }

      switch (ID->type_format)
      {
      case X3F_IMAGE_RAW_HUFFMAN_X530:
      case X3F_IMAGE_RAW_HUFFMAN_10BIT:
        HUF->x3rgb16.data[3 * (row * ID->columns + col) + color] =
            (uint16_t)c_fix;
        break;
      case X3F_IMAGE_THUMB_HUFFMAN:
        HUF->rgb8.data[3 * (row * ID->columns + col) + color] = (uint8_t)c_fix;
        break;
      default:
        /* TODO: Shouldn't this be treated as a fatal error? */
        throw LIBRAW_EXCEPTION_IO_CORRUPT;
      }
    }
  }
}

static void huffman_decode(x3f_info_t *I, x3f_directory_entry_t *DE, int bits)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;

  int row;
  int minimum = 0;
  int offset = legacy_offset;

  for (row = 0; row < (int)ID->rows; row++)
    huffman_decode_row(I, DE, bits, row, offset, &minimum);

  if (auto_legacy_offset && minimum < 0)
  {
    offset = -minimum;
    for (row = 0; row < (int)ID->rows; row++)
      huffman_decode_row(I, DE, bits, row, offset, &minimum);
  }
}

static int32_t get_simple_diff(x3f_huffman_t *HUF, uint16_t index)
{
  if (HUF->mapping.size == 0)
    return index;
  else
    return HUF->mapping.element[index];
}

static void simple_decode_row(x3f_info_t * /*I*/, x3f_directory_entry_t *DE,
                              int bits, int row, int row_stride)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  x3f_huffman_t *HUF = ID->huffman;

  if (row*row_stride > (int)(ID->data_size - (ID->columns*sizeof(uint32_t))))
	  throw LIBRAW_EXCEPTION_IO_CORRUPT;
  uint32_t *data = (uint32_t *)((unsigned char *)ID->data + row * row_stride);

  uint16_t c[3] = {0, 0, 0};
  int col;

  uint32_t mask = 0;

  switch (bits)
  {
  case 8:
    mask = 0x0ff;
    break;
  case 9:
    mask = 0x1ff;
    break;
  case 10:
    mask = 0x3ff;
    break;
  case 11:
    mask = 0x7ff;
    break;
  case 12:
    mask = 0xfff;
    break;
  default:
    mask = 0;
    /* TODO: Shouldn't this be treated as a fatal error? */
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
    break;
  }

  for (col = 0; col < (int)ID->columns; col++)
  {
    int color;
    uint32_t val = data[col];

    for (color = 0; color < 3; color++)
    {
      uint16_t c_fix;
      c[color] += get_simple_diff(HUF, (val >> (color * bits)) & mask);

      switch (ID->type_format)
      {
      case X3F_IMAGE_RAW_HUFFMAN_X530:
      case X3F_IMAGE_RAW_HUFFMAN_10BIT:
        c_fix = (int16_t)c[color] > 0 ? c[color] : 0;

        HUF->x3rgb16.data[3 * (row * ID->columns + col) + color] = c_fix;
        break;
      case X3F_IMAGE_THUMB_HUFFMAN:
        c_fix = (int8_t)c[color] > 0 ? c[color] : 0;

        HUF->rgb8.data[3 * (row * ID->columns + col) + color] = c_fix;
        break;
      default:
        /* TODO: Shouldn't this be treated as a fatal error? */
        throw LIBRAW_EXCEPTION_IO_CORRUPT;
      }
    }
  }
}

static void simple_decode(x3f_info_t *I, x3f_directory_entry_t *DE, int bits,
                          int row_stride)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;

  int row;

  for (row = 0; row < (int)ID->rows; row++)
    simple_decode_row(I, DE, bits, row, row_stride);
}

/* --------------------------------------------------------------------- */
/* Loading the data in a directory entry                                 */
/* --------------------------------------------------------------------- */

/* First you set the offset to where to start reading the data ... */

static void read_data_set_offset(x3f_info_t *I, x3f_directory_entry_t *DE,
                                 uint32_t header_size)
{
  uint32_t i_off = DE->input.offset + header_size;

  I->input.file->seek(i_off, SEEK_SET);
}

/* ... then you read the data, block for block */

static uint32_t read_data_block(void **data, x3f_info_t *I,
                                x3f_directory_entry_t *DE, uint32_t footer)
{
  INT64 fpos = I->input.file->tell();
  uint32_t size = DE->input.size + DE->input.offset - fpos - footer;

  if (fpos + size > I->input.file->size())
    throw LIBRAW_EXCEPTION_IO_CORRUPT;

  *data = (void *)malloc(size);

  GETN(*data, size);

  return size;
}

static uint32_t data_block_size(void ** /*data*/, x3f_info_t *I,
                                x3f_directory_entry_t *DE, uint32_t footer)
{
  uint32_t size =
      DE->input.size + DE->input.offset - I->input.file->tell() - footer;
  return size;
}

static void x3f_load_image_verbatim(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  if (!ID->data_size)
    ID->data_size = read_data_block(&ID->data, I, DE, 0);
}

static int32_t x3f_load_image_verbatim_size(x3f_info_t *I,
                                            x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  return data_block_size(&ID->data, I, DE, 0);
}

static void x3f_load_property_list(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_property_list_t *PL = &DEH->data_subsection.property_list;
  int i;

  read_data_set_offset(I, DE, X3F_PROPERTY_LIST_HEADER_SIZE);

  GET_PROPERTY_TABLE(PL->property_table, PL->num_properties);

  if (!PL->data_size)
    PL->data_size = read_data_block(&PL->data, I, DE, 0);
  uint32_t maxoffset = PL->data_size / sizeof(utf16_t) -
                       2; // at least 2 chars, value + terminating 0x0000

  for (i = 0; i < (int)PL->num_properties; i++)
  {
    x3f_property_t *P = &PL->property_table.element[i];
    if (P->name_offset > maxoffset || P->value_offset > maxoffset)
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    P->name = ((utf16_t *)PL->data + P->name_offset);
    P->value = ((utf16_t *)PL->data + P->value_offset);
  }
}

static void x3f_load_true(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  x3f_true_t *TRU = new_true(&ID->tru);
  x3f_quattro_t *Q = NULL;
  int i;

  if (ID->type_format == X3F_IMAGE_RAW_QUATTRO ||
      ID->type_format == X3F_IMAGE_RAW_SDQ ||
      ID->type_format == X3F_IMAGE_RAW_SDQH ||
      ID->type_format == X3F_IMAGE_RAW_SDQH2)
  {
    Q = new_quattro(&ID->quattro);

    for (i = 0; i < TRUE_PLANES; i++)
    {
      GET2(Q->plane[i].columns);
      GET2(Q->plane[i].rows);
    }

    if (Q->plane[0].rows == ID->rows / 2)
    {
      Q->quattro_layout = 1;
    }
    else if (Q->plane[0].rows == ID->rows)
    {
      Q->quattro_layout = 0;
    }
    else
    {
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    }
  }

  /* Read TRUE header data */
  GET2(TRU->seed[0]);
  GET2(TRU->seed[1]);
  GET2(TRU->seed[2]);
  GET2(TRU->unknown);
  GET_TRUE_HUFF_TABLE(TRU->table);

  if (ID->type_format == X3F_IMAGE_RAW_QUATTRO ||
      ID->type_format == X3F_IMAGE_RAW_SDQ ||
      ID->type_format == X3F_IMAGE_RAW_SDQH ||
      ID->type_format == X3F_IMAGE_RAW_SDQH2)
  {
    GET4(Q->unknown);
  }

  GET_TABLE(TRU->plane_size, GET4, TRUE_PLANES, uint32_t);

  /* Read image data */
  if (!ID->data_size)
    ID->data_size = read_data_block(&ID->data, I, DE, 0);

  /* TODO: can it be fewer than 8 bits? Maybe taken from TRU->table? */
  new_huffman_tree(&TRU->tree, 8);

  populate_true_huffman_tree(&TRU->tree, &TRU->table);

#ifdef DBG_PRNT
  print_huffman_tree(TRU->tree.nodes, 0, 0);
#endif

  TRU->plane_address[0] = (uint8_t *)ID->data;
  for (i = 1; i < TRUE_PLANES; i++)
    TRU->plane_address[i] = TRU->plane_address[i - 1] +
                            (((TRU->plane_size.element[i - 1] + 15) / 16) * 16);

  if ((ID->type_format == X3F_IMAGE_RAW_QUATTRO ||
       ID->type_format == X3F_IMAGE_RAW_SDQ ||
       ID->type_format == X3F_IMAGE_RAW_SDQH ||
       ID->type_format == X3F_IMAGE_RAW_SDQH2) &&
      Q->quattro_layout)
  {
    uint32_t columns = Q->plane[0].columns;
    uint32_t rows = Q->plane[0].rows;
    uint32_t channels = 3;
    uint32_t size = columns * rows * channels;

    TRU->x3rgb16.columns = columns;
    TRU->x3rgb16.rows = rows;
    TRU->x3rgb16.channels = channels;
    TRU->x3rgb16.row_stride = columns * channels;
    TRU->x3rgb16.buf = malloc(sizeof(uint16_t) * size);
    TRU->x3rgb16.data = (uint16_t *)TRU->x3rgb16.buf;

    columns = Q->plane[2].columns;
    rows = Q->plane[2].rows;
    channels = 1;
    size = columns * rows * channels;

    Q->top16.columns = columns;
    Q->top16.rows = rows;
    Q->top16.channels = channels;
    Q->top16.row_stride = columns * channels;
    Q->top16.buf = malloc(sizeof(uint16_t) * size);
    Q->top16.data = (uint16_t *)Q->top16.buf;
  }
  else
  {
    uint32_t size = ID->columns * ID->rows * 3;

    TRU->x3rgb16.columns = ID->columns;
    TRU->x3rgb16.rows = ID->rows;
    TRU->x3rgb16.channels = 3;
    TRU->x3rgb16.row_stride = ID->columns * 3;
    TRU->x3rgb16.buf = malloc(sizeof(uint16_t) * size);
    TRU->x3rgb16.data = (uint16_t *)TRU->x3rgb16.buf;
  }

  true_decode(I, DE);
}

static void x3f_load_huffman_compressed(x3f_info_t *I,
                                        x3f_directory_entry_t *DE, int bits,
                                        int /*use_map_table*/)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  x3f_huffman_t *HUF = ID->huffman;
  int table_size = 1 << bits;
  int row_offsets_size = ID->rows * sizeof(HUF->row_offsets.element[0]);

  GET_TABLE(HUF->table, GET4, table_size, uint32_t);

  if (!ID->data_size)
    ID->data_size = read_data_block(&ID->data, I, DE, row_offsets_size);

  GET_TABLE(HUF->row_offsets, GET4, ID->rows, uint32_t);

  new_huffman_tree(&HUF->tree, bits);
  populate_huffman_tree(&HUF->tree, &HUF->table, &HUF->mapping);

  huffman_decode(I, DE, bits);
}

static void x3f_load_huffman_not_compressed(x3f_info_t *I,
                                            x3f_directory_entry_t *DE, int bits,
                                            int /*use_map_table*/, int row_stride)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;

  if (!ID->data_size)
    ID->data_size = read_data_block(&ID->data, I, DE, 0);

  simple_decode(I, DE, bits, row_stride);
}

static void x3f_load_huffman(x3f_info_t *I, x3f_directory_entry_t *DE, int bits,
                             int use_map_table, int row_stride)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;
  x3f_huffman_t *HUF = new_huffman(&ID->huffman);
  uint32_t size;

  if (use_map_table)
  {
    int table_size = 1 << bits;

    GET_TABLE(HUF->mapping, GET2, table_size, uint16_t);
  }

  switch (ID->type_format)
  {
  case X3F_IMAGE_RAW_HUFFMAN_X530:
  case X3F_IMAGE_RAW_HUFFMAN_10BIT:
    size = ID->columns * ID->rows * 3;
    HUF->x3rgb16.columns = ID->columns;
    HUF->x3rgb16.rows = ID->rows;
    HUF->x3rgb16.channels = 3;
    HUF->x3rgb16.row_stride = ID->columns * 3;
    HUF->x3rgb16.buf = malloc(sizeof(uint16_t) * size);
    HUF->x3rgb16.data = (uint16_t *)HUF->x3rgb16.buf;
    break;
  case X3F_IMAGE_THUMB_HUFFMAN:
    size = ID->columns * ID->rows * 3;
    HUF->rgb8.columns = ID->columns;
    HUF->rgb8.rows = ID->rows;
    HUF->rgb8.channels = 3;
    HUF->rgb8.row_stride = ID->columns * 3;
    HUF->rgb8.buf = malloc(sizeof(uint8_t) * size);
    HUF->rgb8.data = (uint8_t *)HUF->rgb8.buf;
    break;
  default:
    /* TODO: Shouldn't this be treated as a fatal error? */
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  }

  if (row_stride == 0)
    return x3f_load_huffman_compressed(I, DE, bits, use_map_table);
  else
    return x3f_load_huffman_not_compressed(I, DE, bits, use_map_table,
                                           row_stride);
}

static void x3f_load_pixmap(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_load_image_verbatim(I, DE);
}

static uint32_t x3f_load_pixmap_size(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  return x3f_load_image_verbatim_size(I, DE);
}

static void x3f_load_jpeg(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_load_image_verbatim(I, DE);
}

static uint32_t x3f_load_jpeg_size(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  return x3f_load_image_verbatim_size(I, DE);
}

static void x3f_load_image(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;

  if (ID->rows > 65535 || ID->columns > 65535)
	  throw LIBRAW_EXCEPTION_IO_CORRUPT;

  read_data_set_offset(I, DE, X3F_IMAGE_HEADER_SIZE);

  switch (ID->type_format)
  {
  case X3F_IMAGE_RAW_TRUE:
  case X3F_IMAGE_RAW_MERRILL:
  case X3F_IMAGE_RAW_QUATTRO:
  case X3F_IMAGE_RAW_SDQ:
  case X3F_IMAGE_RAW_SDQH:
  case X3F_IMAGE_RAW_SDQH2:
    x3f_load_true(I, DE);
    break;
  case X3F_IMAGE_RAW_HUFFMAN_X530:
  case X3F_IMAGE_RAW_HUFFMAN_10BIT:
    x3f_load_huffman(I, DE, 10, 1, ID->row_stride);
    break;
  case X3F_IMAGE_THUMB_PLAIN:
    x3f_load_pixmap(I, DE);
    break;
  case X3F_IMAGE_THUMB_HUFFMAN:
    x3f_load_huffman(I, DE, 8, 0, ID->row_stride);
    break;
  case X3F_IMAGE_THUMB_JPEG:
    x3f_load_jpeg(I, DE);
    break;
  default:
    /* TODO: Shouldn't this be treated as a fatal error? */
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  }
}

// Used only for thumbnail size estimation
static uint32_t x3f_load_image_size(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_image_data_t *ID = &DEH->data_subsection.image_data;

  read_data_set_offset(I, DE, X3F_IMAGE_HEADER_SIZE);

  switch (ID->type_format)
  {
  case X3F_IMAGE_THUMB_PLAIN:
    return x3f_load_pixmap_size(I, DE);
  case X3F_IMAGE_THUMB_JPEG:
    return x3f_load_jpeg_size(I, DE);
    break;
  default:
    return 0;
  }
}

static void x3f_load_camf_decode_type2(x3f_camf_t *CAMF)
{
  uint32_t key = CAMF->t2.crypt_key;
  int i;

  CAMF->decoded_data_size = CAMF->data_size;
  CAMF->decoded_data = malloc(CAMF->decoded_data_size);

  for (i = 0; i < (int)CAMF->data_size; i++)
  {
    uint8_t old, _new;
    uint32_t tmp;

    old = ((uint8_t *)CAMF->data)[i];
    key = (key * 1597 + 51749) % 244944;
    tmp = (uint32_t)(key * ((int64_t)301593171) >> 24);
    _new = (uint8_t)(old ^ (uint8_t)(((((key << 8) - tmp) >> 1) + tmp) >> 17));
    ((uint8_t *)CAMF->decoded_data)[i] = _new;
  }
}

/* NOTE: the unpacking in this code is in big respects identical to
   true_decode_one_color(). The difference is in the output you
   build. It might be possible to make some parts shared. NOTE ALSO:
   This means that the meta data is obfuscated using an image
   compression algorithm. */

static void camf_decode_type4(x3f_camf_t *CAMF)
{
  uint32_t seed = CAMF->t4.decode_bias;
  int row;

  uint8_t *dst;
  uint32_t dst_size = CAMF->t4.decoded_data_size;
  uint8_t *dst_end;

  bool_t odd_dst = 0;

  x3f_hufftree_t *tree = &CAMF->tree;
  bit_state_t BS;

  int32_t row_start_acc[2][2];
  uint32_t rows = CAMF->t4.block_count;
  uint32_t cols = CAMF->t4.block_size;

  CAMF->decoded_data_size = dst_size;

  CAMF->decoded_data = malloc(CAMF->decoded_data_size);
  memset(CAMF->decoded_data, 0, CAMF->decoded_data_size);

  dst = (uint8_t *)CAMF->decoded_data;
  dst_end = dst + dst_size;

  set_bit_state(&BS, CAMF->decoding_start);

  row_start_acc[0][0] = seed;
  row_start_acc[0][1] = seed;
  row_start_acc[1][0] = seed;
  row_start_acc[1][1] = seed;

  for (row = 0; row < (int)rows; row++)
  {
    int col;
    bool_t odd_row = row & 1;
    int32_t acc[2];

    /* We loop through all the columns and the rows. But the actual
       data is smaller than that, so we break the loop when reaching
       the end. */
    for (col = 0; col < (int)cols; col++)
    {
      bool_t odd_col = col & 1;
      int32_t diff = get_true_diff(&BS, tree);
      int32_t prev = col < 2 ? row_start_acc[odd_row][odd_col] : acc[odd_col];
      int32_t value = prev + diff;

      acc[odd_col] = value;
      if (col < 2)
        row_start_acc[odd_row][odd_col] = value;

      switch (odd_dst)
      {
      case 0:
        *dst++ = (uint8_t)((value >> 4) & 0xff);

        if (dst >= dst_end)
        {
          goto ready;
        }

        *dst = (uint8_t)((value << 4) & 0xf0);
        break;
      case 1:
        *dst++ |= (uint8_t)((value >> 8) & 0x0f);

        if (dst >= dst_end)
        {
          goto ready;
        }

        *dst++ = (uint8_t)((value << 0) & 0xff);

        if (dst >= dst_end)
        {
          goto ready;
        }

        break;
      }

      odd_dst = !odd_dst;
    } /* end col */
  }   /* end row */

ready:;
}

static void x3f_load_camf_decode_type4(x3f_camf_t *CAMF)
{
  int i;
  uint8_t *p;
  x3f_true_huffman_element_t *element = NULL;

  for (i = 0, p = (uint8_t *)CAMF->data; *p != 0; i++)
  {
    /* TODO: Is this too expensive ??*/
    element = (x3f_true_huffman_element_t *)realloc(element,
                                                    (i + 1) * sizeof(*element));

    element[i].code_size = *p++;
    element[i].code = *p++;
  }

  CAMF->table.size = i;
  CAMF->table.element = element;

  /* TODO: where does the values 28 and 32 come from? */
#define CAMF_T4_DATA_SIZE_OFFSET 28
#define CAMF_T4_DATA_OFFSET 32
  CAMF->decoding_size =
      *(uint32_t *)((unsigned char *)CAMF->data + CAMF_T4_DATA_SIZE_OFFSET);
  CAMF->decoding_start = (uint8_t *)CAMF->data + CAMF_T4_DATA_OFFSET;

  /* TODO: can it be fewer than 8 bits? Maybe taken from TRU->table? */
  new_huffman_tree(&CAMF->tree, 8);

  populate_true_huffman_tree(&CAMF->tree, &CAMF->table);

#ifdef DBG_PRNT
  print_huffman_tree(CAMF->tree.nodes, 0, 0);
#endif

  camf_decode_type4(CAMF);
}

static void camf_decode_type5(x3f_camf_t *CAMF)
{
  int32_t acc = CAMF->t5.decode_bias;

  uint8_t *dst;

  x3f_hufftree_t *tree = &CAMF->tree;
  bit_state_t BS;

  int32_t i;

  CAMF->decoded_data_size = CAMF->t5.decoded_data_size;
  CAMF->decoded_data = malloc(CAMF->decoded_data_size);

  dst = (uint8_t *)CAMF->decoded_data;

  set_bit_state(&BS, CAMF->decoding_start);

  for (i = 0; i < (int)CAMF->decoded_data_size; i++)
  {
    int32_t diff = get_true_diff(&BS, tree);

    acc = acc + diff;
    *dst++ = (uint8_t)(acc & 0xff);
  }
}

static void x3f_load_camf_decode_type5(x3f_camf_t *CAMF)
{
  int i;
  uint8_t *p;
  x3f_true_huffman_element_t *element = NULL;

  for (i = 0, p = (uint8_t *)CAMF->data; *p != 0; i++)
  {
    /* TODO: Is this too expensive ??*/
    element = (x3f_true_huffman_element_t *)realloc(element,
                                                    (i + 1) * sizeof(*element));

    element[i].code_size = *p++;
    element[i].code = *p++;
  }

  CAMF->table.size = i;
  CAMF->table.element = element;

  /* TODO: where does the values 28 and 32 come from? */
#define CAMF_T5_DATA_SIZE_OFFSET 28
#define CAMF_T5_DATA_OFFSET 32
  CAMF->decoding_size =
      *(uint32_t *)((uint8_t *)CAMF->data + CAMF_T5_DATA_SIZE_OFFSET);
  CAMF->decoding_start = (uint8_t *)CAMF->data + CAMF_T5_DATA_OFFSET;

  /* TODO: can it be fewer than 8 bits? Maybe taken from TRU->table? */
  new_huffman_tree(&CAMF->tree, 8);

  populate_true_huffman_tree(&CAMF->tree, &CAMF->table);

#ifdef DBG_PRNT
  print_huffman_tree(CAMF->tree.nodes, 0, 0);
#endif

  camf_decode_type5(CAMF);
}

static void x3f_setup_camf_text_entry(camf_entry_t *entry)
{
  entry->text_size = *(uint32_t *)entry->value_address;
  entry->text = (char *)entry->value_address + 4;
}

static void x3f_setup_camf_property_entry(camf_entry_t *entry)
{
  int i;
  uint8_t *e = (uint8_t *)entry->entry;
  uint8_t *v = (uint8_t *)entry->value_address;
  uint32_t num = entry->property_num = *(uint32_t *)v;
  uint32_t off = *(uint32_t *)(v + 4);

  entry->property_name = (char **)malloc(num * sizeof(uint8_t *));
  entry->property_value = (uint8_t **)malloc(num * sizeof(uint8_t *));

  for (i = 0; i < (int)num; i++)
  {
    uint32_t name_off = off + *(uint32_t *)(v + 8 + 8 * i);
    uint32_t value_off = off + *(uint32_t *)(v + 8 + 8 * i + 4);

    entry->property_name[i] = (char *)(e + name_off);
    entry->property_value[i] = e + value_off;
  }
}

static void set_matrix_element_info(uint32_t type, uint32_t *size,
                                    matrix_type_t *decoded_type)
{
  switch (type)
  {
  case 0:
    *size = 2;
    *decoded_type = M_INT; /* known to be true */
    break;
  case 1:
    *size = 4;
    *decoded_type = M_UINT; /* TODO: unknown ???? */
    break;
  case 2:
    *size = 4;
    *decoded_type = M_UINT; /* TODO: unknown ???? */
    break;
  case 3:
    *size = 4;
    *decoded_type = M_FLOAT; /* known to be true */
    break;
  case 5:
    *size = 1;
    *decoded_type = M_UINT; /* TODO: unknown ???? */
    break;
  case 6:
    *size = 2;
    *decoded_type = M_UINT; /* TODO: unknown ???? */
    break;
  default:
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  }
}

static void get_matrix_copy(camf_entry_t *entry)
{
  uint32_t element_size = entry->matrix_element_size;
  uint32_t elements = entry->matrix_elements;
  int i, size = (entry->matrix_decoded_type == M_FLOAT ? sizeof(double)
                                                       : sizeof(uint32_t)) *
                elements;

  entry->matrix_decoded = malloc(size);

  switch (element_size)
  {
  case 4:
    switch (entry->matrix_decoded_type)
    {
    case M_INT:
    case M_UINT:
      memcpy(entry->matrix_decoded, entry->matrix_data, size);
      break;
    case M_FLOAT:
      for (i = 0; i < (int)elements; i++)
        ((double *)entry->matrix_decoded)[i] =
            (double)((float *)entry->matrix_data)[i];
      break;
    default:
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    }
    break;
  case 2:
    switch (entry->matrix_decoded_type)
    {
    case M_INT:
      for (i = 0; i < (int)elements; i++)
        ((int32_t *)entry->matrix_decoded)[i] =
            (int32_t)((int16_t *)entry->matrix_data)[i];
      break;
    case M_UINT:
      for (i = 0; i < (int)elements; i++)
        ((uint32_t *)entry->matrix_decoded)[i] =
            (uint32_t)((uint16_t *)entry->matrix_data)[i];
      break;
    default:
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    }
    break;
  case 1:
    switch (entry->matrix_decoded_type)
    {
    case M_INT:
      for (i = 0; i < (int)elements; i++)
        ((int32_t *)entry->matrix_decoded)[i] =
            (int32_t)((int8_t *)entry->matrix_data)[i];
      break;
    case M_UINT:
      for (i = 0; i < (int)elements; i++)
        ((uint32_t *)entry->matrix_decoded)[i] =
            (uint32_t)((uint8_t *)entry->matrix_data)[i];
      break;
    default:
      throw LIBRAW_EXCEPTION_IO_CORRUPT;
    }
    break;
  default:
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  }
}

static void x3f_setup_camf_matrix_entry(camf_entry_t *entry)
{
  int i;
  int totalsize = 1;

  uint8_t *e = (uint8_t *)entry->entry;
  uint8_t *v = (uint8_t *)entry->value_address;
  uint32_t type = entry->matrix_type = *(uint32_t *)(v + 0);
  uint32_t dim = entry->matrix_dim = *(uint32_t *)(v + 4);
  uint32_t off = entry->matrix_data_off = *(uint32_t *)(v + 8);
  camf_dim_entry_t *dentry = entry->matrix_dim_entry =
      (camf_dim_entry_t *)malloc(dim * sizeof(camf_dim_entry_t));

  for (i = 0; i < (int)dim; i++)
  {
    uint32_t size = dentry[i].size = *(uint32_t *)(v + 12 + 12 * i + 0);
    dentry[i].name_offset = *(uint32_t *)(v + 12 + 12 * i + 4);
    dentry[i].n = *(uint32_t *)(v + 12 + 12 * i + 8);
    dentry[i].name = (char *)(e + dentry[i].name_offset);

    if ((int)dentry[i].n != i)
    {
    }

    totalsize *= size;
  }

  set_matrix_element_info(type, &entry->matrix_element_size,
                          &entry->matrix_decoded_type);
  entry->matrix_data = (void *)(e + off);

  entry->matrix_elements = totalsize;
  entry->matrix_used_space = entry->entry_size - off;

  /* This estimate only works for matrices above a certain size */
  entry->matrix_estimated_element_size = entry->matrix_used_space / totalsize;

  get_matrix_copy(entry);
}

static void x3f_setup_camf_entries(x3f_camf_t *CAMF)
{
  uint8_t *p = (uint8_t *)CAMF->decoded_data;
  uint8_t *end = p + CAMF->decoded_data_size;
  camf_entry_t *entry = NULL;
  int i;

  for (i = 0; p < end; i++)
  {
    uint32_t *p4 = (uint32_t *)p;

    switch (*p4)
    {
    case X3F_CMbP:
    case X3F_CMbT:
    case X3F_CMbM:
      break;
    default:
      goto stop;
    }

    /* TODO: lots of realloc - may be inefficient */
    entry = (camf_entry_t *)realloc(entry, (i + 1) * sizeof(camf_entry_t));

    /* Pointer */
    entry[i].entry = p;

    /* Header */
    entry[i].id = *p4++;
    entry[i].version = *p4++;
    entry[i].entry_size = *p4++;
    entry[i].name_offset = *p4++;
    entry[i].value_offset = *p4++;

    /* Compute addresses and sizes */
    entry[i].name_address = (char *)(p + entry[i].name_offset);
    entry[i].value_address = p + entry[i].value_offset;
    entry[i].name_size = entry[i].value_offset - entry[i].name_offset;
    entry[i].value_size = entry[i].entry_size - entry[i].value_offset;

    entry[i].text_size = 0;
    entry[i].text = NULL;
    entry[i].property_num = 0;
    entry[i].property_name = NULL;
    entry[i].property_value = NULL;
    entry[i].matrix_type = 0;
    entry[i].matrix_dim = 0;
    entry[i].matrix_data_off = 0;
    entry[i].matrix_data = NULL;
    entry[i].matrix_dim_entry = NULL;

    entry[i].matrix_decoded = NULL;

    switch (entry[i].id)
    {
    case X3F_CMbP:
      x3f_setup_camf_property_entry(&entry[i]);
      break;
    case X3F_CMbT:
      x3f_setup_camf_text_entry(&entry[i]);
      break;
    case X3F_CMbM:
      x3f_setup_camf_matrix_entry(&entry[i]);
      break;
    }

    p += entry[i].entry_size;
  }

stop:

  CAMF->entry_table.size = i;
  CAMF->entry_table.element = entry;
}

static void x3f_load_camf(x3f_info_t *I, x3f_directory_entry_t *DE)
{
  x3f_directory_entry_header_t *DEH = &DE->header;
  x3f_camf_t *CAMF = &DEH->data_subsection.camf;

  read_data_set_offset(I, DE, X3F_CAMF_HEADER_SIZE);

  if (!CAMF->data_size)
    CAMF->data_size = read_data_block(&CAMF->data, I, DE, 0);

  switch (CAMF->type)
  {
  case 2: /* Older SD9-SD14 */
    x3f_load_camf_decode_type2(CAMF);
    break;
  case 4: /* TRUE ... Merrill */
    x3f_load_camf_decode_type4(CAMF);
    break;
  case 5: /* Quattro ... */
    x3f_load_camf_decode_type5(CAMF);
    break;
  default:
    /* TODO: Shouldn't this be treated as a fatal error? */
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
  }

  if (CAMF->decoded_data != NULL)
    x3f_setup_camf_entries(CAMF);
  else
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
}

/* extern */ x3f_return_t x3f_load_data(x3f_t *x3f, x3f_directory_entry_t *DE)
{
  x3f_info_t *I = &x3f->info;

  if (DE == NULL)
    return X3F_ARGUMENT_ERROR;

  switch (DE->header.identifier)
  {
  case X3F_SECp:
    x3f_load_property_list(I, DE);
    break;
  case X3F_SECi:
    x3f_load_image(I, DE);
    break;
  case X3F_SECc:
    x3f_load_camf(I, DE);
    break;
  default:
    return X3F_INTERNAL_ERROR;
  }
  return X3F_OK;
}

/* extern */ int64_t x3f_load_data_size(x3f_t *x3f, x3f_directory_entry_t *DE)
{
  x3f_info_t *I = &x3f->info;

  if (DE == NULL)
    return -1;

  switch (DE->header.identifier)
  {
  case X3F_SECi:
    return x3f_load_image_size(I, DE);
  default:
    return 0;
  }
}

/* extern */ x3f_return_t x3f_load_image_block(x3f_t *x3f,
                                               x3f_directory_entry_t *DE)
{
  x3f_info_t *I = &x3f->info;

  if (DE == NULL)
    return X3F_ARGUMENT_ERROR;

  switch (DE->header.identifier)
  {
  case X3F_SECi:
    read_data_set_offset(I, DE, X3F_IMAGE_HEADER_SIZE);
    x3f_load_image_verbatim(I, DE);
    break;
  default:
    throw LIBRAW_EXCEPTION_IO_CORRUPT;
    return X3F_INTERNAL_ERROR; /* unreachable code*/
  }

  return X3F_OK;
}

/* --------------------------------------------------------------------- */
/* The End                                                               */
/* --------------------------------------------------------------------- */

#endif
