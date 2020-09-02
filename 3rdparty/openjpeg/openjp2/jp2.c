/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2010-2011, Kaori Hagihara
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "opj_includes.h"

/** @defgroup JP2 JP2 - JPEG-2000 file format reader/writer */
/*@{*/

#define OPJ_BOX_SIZE    1024

#define OPJ_UNUSED(x) (void)x

/** @name Local static functions */
/*@{*/

/*static void jp2_write_url(opj_cio_t *cio, char *Idx_file);*/

/**
 * Reads a IHDR box - Image Header box
 *
 * @param   p_image_header_data         pointer to actual data (already read from file)
 * @param   jp2                         the jpeg2000 file codec.
 * @param   p_image_header_size         the size of the image header
 * @param   p_manager                   the user event manager.
 *
 * @return  true if the image header is valid, false else.
 */
static OPJ_BOOL opj_jp2_read_ihdr(opj_jp2_t *jp2,
                                  OPJ_BYTE *p_image_header_data,
                                  OPJ_UINT32 p_image_header_size,
                                  opj_event_mgr_t * p_manager);

/**
 * Writes the Image Header box - Image Header box.
 *
 * @param jp2                   jpeg2000 file codec.
 * @param p_nb_bytes_written    pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static OPJ_BYTE * opj_jp2_write_ihdr(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written);

/**
 * Writes the Bit per Component box.
 *
 * @param   jp2                     jpeg2000 file codec.
 * @param   p_nb_bytes_written      pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static OPJ_BYTE * opj_jp2_write_bpcc(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written);

/**
 * Reads a Bit per Component box.
 *
 * @param   p_bpc_header_data           pointer to actual data (already read from file)
 * @param   jp2                         the jpeg2000 file codec.
 * @param   p_bpc_header_size           the size of the bpc header
 * @param   p_manager                   the user event manager.
 *
 * @return  true if the bpc header is valid, false else.
 */
static OPJ_BOOL opj_jp2_read_bpcc(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_bpc_header_data,
                                  OPJ_UINT32 p_bpc_header_size,
                                  opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_jp2_read_cdef(opj_jp2_t * jp2,
                                  OPJ_BYTE * p_cdef_header_data,
                                  OPJ_UINT32 p_cdef_header_size,
                                  opj_event_mgr_t * p_manager);

static void opj_jp2_apply_cdef(opj_image_t *image, opj_jp2_color_t *color,
                               opj_event_mgr_t *);

/**
 * Writes the Channel Definition box.
 *
 * @param jp2                   jpeg2000 file codec.
 * @param p_nb_bytes_written    pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
 */
static OPJ_BYTE * opj_jp2_write_cdef(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written);

/**
 * Writes the Colour Specification box.
 *
 * @param jp2                   jpeg2000 file codec.
 * @param p_nb_bytes_written    pointer to store the nb of bytes written by the function.
 *
 * @return  the data being copied.
*/
static OPJ_BYTE * opj_jp2_write_colr(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written);

/**
 * Writes a FTYP box - File type box
 *
 * @param   cio         the stream to write data to.
 * @param   jp2         the jpeg2000 file codec.
 * @param   p_manager   the user event manager.
 *
 * @return  true if writing was successful.
 */
static OPJ_BOOL opj_jp2_write_ftyp(opj_jp2_t *jp2,
                                   opj_stream_private_t *cio,
                                   opj_event_mgr_t * p_manager);

/**
 * Reads a a FTYP box - File type box
 *
 * @param   p_header_data   the data contained in the FTYP box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the FTYP box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the FTYP box is valid.
 */
static OPJ_BOOL opj_jp2_read_ftyp(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_header_data,
                                  OPJ_UINT32 p_header_size,
                                  opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_jp2_skip_jp2c(opj_jp2_t *jp2,
                                  opj_stream_private_t *stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param   p_header_data   the data contained in the file header box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the file header box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the JP2 Header box was successfully recognized.
*/
static OPJ_BOOL opj_jp2_read_jp2h(opj_jp2_t *jp2,
                                  OPJ_BYTE *p_header_data,
                                  OPJ_UINT32 p_header_size,
                                  opj_event_mgr_t * p_manager);

/**
 * Writes the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param  jp2      the jpeg2000 file codec.
 * @param  stream      the stream to write data to.
 * @param  p_manager  user event manager.
 *
 * @return true if writing was successful.
 */
static OPJ_BOOL opj_jp2_write_jp2h(opj_jp2_t *jp2,
                                   opj_stream_private_t *stream,
                                   opj_event_mgr_t * p_manager);

/**
 * Writes the Jpeg2000 codestream Header box - JP2C Header box. This function must be called AFTER the coding has been done.
 *
 * @param   cio         the stream to write data to.
 * @param   jp2         the jpeg2000 file codec.
 * @param   p_manager   user event manager.
 *
 * @return true if writing was successful.
*/
static OPJ_BOOL opj_jp2_write_jp2c(opj_jp2_t *jp2,
                                   opj_stream_private_t *cio,
                                   opj_event_mgr_t * p_manager);

#ifdef USE_JPIP
/**
 * Write index Finder box
 * @param cio     the stream to write to.
 * @param   jp2         the jpeg2000 file codec.
 * @param   p_manager   user event manager.
*/
static OPJ_BOOL opj_jpip_write_iptr(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager);

/**
 * Write index Finder box
 * @param cio     the stream to write to.
 * @param   jp2         the jpeg2000 file codec.
 * @param   p_manager   user event manager.
 */
static OPJ_BOOL opj_jpip_write_cidx(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager);

/**
 * Write file Index (superbox)
 * @param cio     the stream to write to.
 * @param   jp2         the jpeg2000 file codec.
 * @param   p_manager   user event manager.
 */
static OPJ_BOOL opj_jpip_write_fidx(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager);
#endif /* USE_JPIP */

/**
 * Reads a jpeg2000 file signature box.
 *
 * @param   p_header_data   the data contained in the signature box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the signature box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the file signature box is valid.
 */
static OPJ_BOOL opj_jp2_read_jp(opj_jp2_t *jp2,
                                OPJ_BYTE * p_header_data,
                                OPJ_UINT32 p_header_size,
                                opj_event_mgr_t * p_manager);

/**
 * Writes a jpeg2000 file signature box.
 *
 * @param cio the stream to write data to.
 * @param   jp2         the jpeg2000 file codec.
 * @param p_manager the user event manager.
 *
 * @return true if writing was successful.
 */
static OPJ_BOOL opj_jp2_write_jp(opj_jp2_t *jp2,
                                 opj_stream_private_t *cio,
                                 opj_event_mgr_t * p_manager);

/**
Apply collected palette data
@param image Image.
@param color Collector for profile, cdef and pclr data.
@param p_manager the user event manager.
@return true in case of success
*/
static OPJ_BOOL opj_jp2_apply_pclr(opj_image_t *image,
                                   opj_jp2_color_t *color,
                                   opj_event_mgr_t * p_manager);

static void opj_jp2_free_pclr(opj_jp2_color_t *color);

/**
 * Collect palette data
 *
 * @param jp2 JP2 handle
 * @param p_pclr_header_data    FIXME DOC
 * @param p_pclr_header_size    FIXME DOC
 * @param p_manager
 *
 * @return Returns true if successful, returns false otherwise
*/
static OPJ_BOOL opj_jp2_read_pclr(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_pclr_header_data,
                                  OPJ_UINT32 p_pclr_header_size,
                                  opj_event_mgr_t * p_manager);

/**
 * Collect component mapping data
 *
 * @param jp2                 JP2 handle
 * @param p_cmap_header_data  FIXME DOC
 * @param p_cmap_header_size  FIXME DOC
 * @param p_manager           FIXME DOC
 *
 * @return Returns true if successful, returns false otherwise
*/

static OPJ_BOOL opj_jp2_read_cmap(opj_jp2_t * jp2,
                                  OPJ_BYTE * p_cmap_header_data,
                                  OPJ_UINT32 p_cmap_header_size,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads the Color Specification box.
 *
 * @param   p_colr_header_data          pointer to actual data (already read from file)
 * @param   jp2                         the jpeg2000 file codec.
 * @param   p_colr_header_size          the size of the color header
 * @param   p_manager                   the user event manager.
 *
 * @return  true if the bpc header is valid, false else.
*/
static OPJ_BOOL opj_jp2_read_colr(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_colr_header_data,
                                  OPJ_UINT32 p_colr_header_size,
                                  opj_event_mgr_t * p_manager);

/*@}*/

/*@}*/

/**
 * Sets up the procedures to do on writing header after the codestream.
 * Developpers wanting to extend the library can add their own writing procedures.
 */
static OPJ_BOOL opj_jp2_setup_end_header_writing(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the procedures to do on reading header after the codestream.
 * Developpers wanting to extend the library can add their own writing procedures.
 */
static OPJ_BOOL opj_jp2_setup_end_header_reading(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

/**
 * Reads a jpeg2000 file header structure.
 *
 * @param jp2 the jpeg2000 file header structure.
 * @param stream the stream to read data from.
 * @param p_manager the user event manager.
 *
 * @return true if the box is valid.
 */
static OPJ_BOOL opj_jp2_read_header_procedure(opj_jp2_t *jp2,
        opj_stream_private_t *stream,
        opj_event_mgr_t * p_manager);

/**
 * Executes the given procedures on the given codec.
 *
 * @param   p_procedure_list    the list of procedures to execute
 * @param   jp2                 the jpeg2000 file codec to execute the procedures on.
 * @param   stream                  the stream to execute the procedures on.
 * @param   p_manager           the user manager.
 *
 * @return  true                if all the procedures were successfully executed.
 */
static OPJ_BOOL opj_jp2_exec(opj_jp2_t * jp2,
                             opj_procedure_list_t * p_procedure_list,
                             opj_stream_private_t *stream,
                             opj_event_mgr_t * p_manager);

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure.
 *
 * @param   cio                     the input stream to read data from.
 * @param   box                     the box structure to fill.
 * @param   p_number_bytes_read     pointer to an int that will store the number of bytes read from the stream (shoul usually be 2).
 * @param   p_manager               user event manager.
 *
 * @return  true if the box is recognized, false otherwise
*/
static OPJ_BOOL opj_jp2_read_boxhdr(opj_jp2_box_t *box,
                                    OPJ_UINT32 * p_number_bytes_read,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager);

/**
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static OPJ_BOOL opj_jp2_setup_encoding_validation(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the procedures to do on writing header. Developpers wanting to extend the library can add their own writing procedures.
 */
static OPJ_BOOL opj_jp2_setup_header_writing(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_jp2_default_validation(opj_jp2_t * jp2,
        opj_stream_private_t *cio,
        opj_event_mgr_t * p_manager);

/**
 * Finds the image execution function related to the given box id.
 *
 * @param   p_id    the id of the handler to fetch.
 *
 * @return  the given handler or NULL if it could not be found.
 */
static const opj_jp2_header_handler_t * opj_jp2_img_find_handler(
    OPJ_UINT32 p_id);

/**
 * Finds the execution function related to the given box id.
 *
 * @param   p_id    the id of the handler to fetch.
 *
 * @return  the given handler or NULL if it could not be found.
 */
static const opj_jp2_header_handler_t * opj_jp2_find_handler(OPJ_UINT32 p_id);

static const opj_jp2_header_handler_t jp2_header [] = {
    {JP2_JP, opj_jp2_read_jp},
    {JP2_FTYP, opj_jp2_read_ftyp},
    {JP2_JP2H, opj_jp2_read_jp2h}
};

static const opj_jp2_header_handler_t jp2_img_header [] = {
    {JP2_IHDR, opj_jp2_read_ihdr},
    {JP2_COLR, opj_jp2_read_colr},
    {JP2_BPCC, opj_jp2_read_bpcc},
    {JP2_PCLR, opj_jp2_read_pclr},
    {JP2_CMAP, opj_jp2_read_cmap},
    {JP2_CDEF, opj_jp2_read_cdef}

};

/**
 * Reads a box header. The box is the way data is packed inside a jpeg2000 file structure. Data is read from a character string
 *
 * @param   box                     the box structure to fill.
 * @param   p_data                  the character string to read data from.
 * @param   p_number_bytes_read     pointer to an int that will store the number of bytes read from the stream (shoul usually be 2).
 * @param   p_box_max_size          the maximum number of bytes in the box.
 * @param   p_manager         FIXME DOC
 *
 * @return  true if the box is recognized, false otherwise
*/
static OPJ_BOOL opj_jp2_read_boxhdr_char(opj_jp2_box_t *box,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_number_bytes_read,
        OPJ_UINT32 p_box_max_size,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static OPJ_BOOL opj_jp2_setup_decoding_validation(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the procedures to do on reading header.
 * Developpers wanting to extend the library can add their own writing procedures.
 */
static OPJ_BOOL opj_jp2_setup_header_reading(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager);

/* ----------------------------------------------------------------------- */
static OPJ_BOOL opj_jp2_read_boxhdr(opj_jp2_box_t *box,
                                    OPJ_UINT32 * p_number_bytes_read,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager)
{
    /* read header from file */
    OPJ_BYTE l_data_header [8];

    /* preconditions */
    assert(cio != 00);
    assert(box != 00);
    assert(p_number_bytes_read != 00);
    assert(p_manager != 00);

    *p_number_bytes_read = (OPJ_UINT32)opj_stream_read_data(cio, l_data_header, 8,
                           p_manager);
    if (*p_number_bytes_read != 8) {
        return OPJ_FALSE;
    }

    /* process read data */
    opj_read_bytes(l_data_header, &(box->length), 4);
    opj_read_bytes(l_data_header + 4, &(box->type), 4);

    if (box->length == 0) { /* last box */
        const OPJ_OFF_T bleft = opj_stream_get_number_byte_left(cio);
        if (bleft > (OPJ_OFF_T)(0xFFFFFFFFU - 8U)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Cannot handle box sizes higher than 2^32\n");
            return OPJ_FALSE;
        }
        box->length = (OPJ_UINT32)bleft + 8U;
        assert((OPJ_OFF_T)box->length == bleft + 8);
        return OPJ_TRUE;
    }

    /* do we have a "special very large box ?" */
    /* read then the XLBox */
    if (box->length == 1) {
        OPJ_UINT32 l_xl_part_size;

        OPJ_UINT32 l_nb_bytes_read = (OPJ_UINT32)opj_stream_read_data(cio,
                                     l_data_header, 8, p_manager);
        if (l_nb_bytes_read != 8) {
            if (l_nb_bytes_read > 0) {
                *p_number_bytes_read += l_nb_bytes_read;
            }

            return OPJ_FALSE;
        }

        *p_number_bytes_read = 16;
        opj_read_bytes(l_data_header, &l_xl_part_size, 4);
        if (l_xl_part_size != 0) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Cannot handle box sizes higher than 2^32\n");
            return OPJ_FALSE;
        }
        opj_read_bytes(l_data_header + 4, &(box->length), 4);
    }
    return OPJ_TRUE;
}

#if 0
static void jp2_write_url(opj_cio_t *cio, char *Idx_file)
{
    OPJ_UINT32 i;
    opj_jp2_box_t box;

    box.init_pos = cio_tell(cio);
    cio_skip(cio, 4);
    cio_write(cio, JP2_URL, 4); /* DBTL */
    cio_write(cio, 0, 1);       /* VERS */
    cio_write(cio, 0, 3);       /* FLAG */

    if (Idx_file) {
        for (i = 0; i < strlen(Idx_file); i++) {
            cio_write(cio, Idx_file[i], 1);
        }
    }

    box.length = cio_tell(cio) - box.init_pos;
    cio_seek(cio, box.init_pos);
    cio_write(cio, box.length, 4);  /* L */
    cio_seek(cio, box.init_pos + box.length);
}
#endif

static OPJ_BOOL opj_jp2_read_ihdr(opj_jp2_t *jp2,
                                  OPJ_BYTE *p_image_header_data,
                                  OPJ_UINT32 p_image_header_size,
                                  opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_image_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (jp2->comps != NULL) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Ignoring ihdr box. First ihdr box already read\n");
        return OPJ_TRUE;
    }

    if (p_image_header_size != 14) {
        opj_event_msg(p_manager, EVT_ERROR, "Bad image header box (bad size)\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_image_header_data, &(jp2->h), 4);          /* HEIGHT */
    p_image_header_data += 4;
    opj_read_bytes(p_image_header_data, &(jp2->w), 4);          /* WIDTH */
    p_image_header_data += 4;
    opj_read_bytes(p_image_header_data, &(jp2->numcomps), 2);   /* NC */
    p_image_header_data += 2;

    if ((jp2->numcomps - 1U) >=
            16384U) { /* unsigned underflow is well defined: 1U <= jp2->numcomps <= 16384U */
        opj_event_msg(p_manager, EVT_ERROR, "Invalid number of components (ihdr)\n");
        return OPJ_FALSE;
    }

    /* allocate memory for components */
    jp2->comps = (opj_jp2_comps_t*) opj_calloc(jp2->numcomps,
                 sizeof(opj_jp2_comps_t));
    if (jp2->comps == 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to handle image header (ihdr)\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_image_header_data, &(jp2->bpc), 1);        /* BPC */
    ++ p_image_header_data;

    opj_read_bytes(p_image_header_data, &(jp2->C), 1);          /* C */
    ++ p_image_header_data;

    /* Should be equal to 7 cf. chapter about image header box of the norm */
    if (jp2->C != 7) {
        opj_event_msg(p_manager, EVT_INFO,
                      "JP2 IHDR box: compression type indicate that the file is not a conforming JP2 file (%d) \n",
                      jp2->C);
    }

    opj_read_bytes(p_image_header_data, &(jp2->UnkC), 1);       /* UnkC */
    ++ p_image_header_data;
    opj_read_bytes(p_image_header_data, &(jp2->IPR), 1);        /* IPR */
    ++ p_image_header_data;

    jp2->j2k->m_cp.allow_different_bit_depth_sign = (jp2->bpc == 255);
    jp2->j2k->ihdr_w = jp2->w;
    jp2->j2k->ihdr_h = jp2->h;
    jp2->has_ihdr = 1;

    return OPJ_TRUE;
}

static OPJ_BYTE * opj_jp2_write_ihdr(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written
                                    )
{
    OPJ_BYTE * l_ihdr_data, * l_current_ihdr_ptr;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_nb_bytes_written != 00);

    /* default image header is 22 bytes wide */
    l_ihdr_data = (OPJ_BYTE *) opj_calloc(1, 22);
    if (l_ihdr_data == 00) {
        return 00;
    }

    l_current_ihdr_ptr = l_ihdr_data;

    opj_write_bytes(l_current_ihdr_ptr, 22, 4);             /* write box size */
    l_current_ihdr_ptr += 4;

    opj_write_bytes(l_current_ihdr_ptr, JP2_IHDR, 4);       /* IHDR */
    l_current_ihdr_ptr += 4;

    opj_write_bytes(l_current_ihdr_ptr, jp2->h, 4);     /* HEIGHT */
    l_current_ihdr_ptr += 4;

    opj_write_bytes(l_current_ihdr_ptr, jp2->w, 4);     /* WIDTH */
    l_current_ihdr_ptr += 4;

    opj_write_bytes(l_current_ihdr_ptr, jp2->numcomps, 2);      /* NC */
    l_current_ihdr_ptr += 2;

    opj_write_bytes(l_current_ihdr_ptr, jp2->bpc, 1);       /* BPC */
    ++l_current_ihdr_ptr;

    opj_write_bytes(l_current_ihdr_ptr, jp2->C, 1);     /* C : Always 7 */
    ++l_current_ihdr_ptr;

    opj_write_bytes(l_current_ihdr_ptr, jp2->UnkC,
                    1);      /* UnkC, colorspace unknown */
    ++l_current_ihdr_ptr;

    opj_write_bytes(l_current_ihdr_ptr, jp2->IPR,
                    1);       /* IPR, no intellectual property */
    ++l_current_ihdr_ptr;

    *p_nb_bytes_written = 22;

    return l_ihdr_data;
}

static OPJ_BYTE * opj_jp2_write_bpcc(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written
                                    )
{
    OPJ_UINT32 i;
    /* room for 8 bytes for box and 1 byte for each component */
    OPJ_UINT32 l_bpcc_size;
    OPJ_BYTE * l_bpcc_data, * l_current_bpcc_ptr;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_nb_bytes_written != 00);
    l_bpcc_size = 8 + jp2->numcomps;

    l_bpcc_data = (OPJ_BYTE *) opj_calloc(1, l_bpcc_size);
    if (l_bpcc_data == 00) {
        return 00;
    }

    l_current_bpcc_ptr = l_bpcc_data;

    opj_write_bytes(l_current_bpcc_ptr, l_bpcc_size,
                    4);            /* write box size */
    l_current_bpcc_ptr += 4;

    opj_write_bytes(l_current_bpcc_ptr, JP2_BPCC, 4);               /* BPCC */
    l_current_bpcc_ptr += 4;

    for (i = 0; i < jp2->numcomps; ++i)  {
        opj_write_bytes(l_current_bpcc_ptr, jp2->comps[i].bpcc,
                        1); /* write each component information */
        ++l_current_bpcc_ptr;
    }

    *p_nb_bytes_written = l_bpcc_size;

    return l_bpcc_data;
}

static OPJ_BOOL opj_jp2_read_bpcc(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_bpc_header_data,
                                  OPJ_UINT32 p_bpc_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 i;

    /* preconditions */
    assert(p_bpc_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);


    if (jp2->bpc != 255) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "A BPCC header box is available although BPC given by the IHDR box (%d) indicate components bit depth is constant\n",
                      jp2->bpc);
    }

    /* and length is relevant */
    if (p_bpc_header_size != jp2->numcomps) {
        opj_event_msg(p_manager, EVT_ERROR, "Bad BPCC header box (bad size)\n");
        return OPJ_FALSE;
    }

    /* read info for each component */
    for (i = 0; i < jp2->numcomps; ++i) {
        opj_read_bytes(p_bpc_header_data, &jp2->comps[i].bpcc,
                       1);  /* read each BPCC component */
        ++p_bpc_header_data;
    }

    return OPJ_TRUE;
}
static OPJ_BYTE * opj_jp2_write_cdef(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written)
{
    /* room for 8 bytes for box, 2 for n */
    OPJ_UINT32 l_cdef_size = 10;
    OPJ_BYTE * l_cdef_data, * l_current_cdef_ptr;
    OPJ_UINT32 l_value;
    OPJ_UINT16 i;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_nb_bytes_written != 00);
    assert(jp2->color.jp2_cdef != 00);
    assert(jp2->color.jp2_cdef->info != 00);
    assert(jp2->color.jp2_cdef->n > 0U);

    l_cdef_size += 6U * jp2->color.jp2_cdef->n;

    l_cdef_data = (OPJ_BYTE *) opj_malloc(l_cdef_size);
    if (l_cdef_data == 00) {
        return 00;
    }

    l_current_cdef_ptr = l_cdef_data;

    opj_write_bytes(l_current_cdef_ptr, l_cdef_size, 4);        /* write box size */
    l_current_cdef_ptr += 4;

    opj_write_bytes(l_current_cdef_ptr, JP2_CDEF, 4);               /* BPCC */
    l_current_cdef_ptr += 4;

    l_value = jp2->color.jp2_cdef->n;
    opj_write_bytes(l_current_cdef_ptr, l_value, 2);                /* N */
    l_current_cdef_ptr += 2;

    for (i = 0U; i < jp2->color.jp2_cdef->n; ++i) {
        l_value = jp2->color.jp2_cdef->info[i].cn;
        opj_write_bytes(l_current_cdef_ptr, l_value, 2);                /* Cni */
        l_current_cdef_ptr += 2;
        l_value = jp2->color.jp2_cdef->info[i].typ;
        opj_write_bytes(l_current_cdef_ptr, l_value, 2);                /* Typi */
        l_current_cdef_ptr += 2;
        l_value = jp2->color.jp2_cdef->info[i].asoc;
        opj_write_bytes(l_current_cdef_ptr, l_value, 2);                /* Asoci */
        l_current_cdef_ptr += 2;
    }
    *p_nb_bytes_written = l_cdef_size;

    return l_cdef_data;
}

static OPJ_BYTE * opj_jp2_write_colr(opj_jp2_t *jp2,
                                     OPJ_UINT32 * p_nb_bytes_written
                                    )
{
    /* room for 8 bytes for box 3 for common data and variable upon profile*/
    OPJ_UINT32 l_colr_size = 11;
    OPJ_BYTE * l_colr_data, * l_current_colr_ptr;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_nb_bytes_written != 00);
    assert(jp2->meth == 1 || jp2->meth == 2);

    switch (jp2->meth) {
    case 1 :
        l_colr_size += 4; /* EnumCS */
        break;
    case 2 :
        assert(jp2->color.icc_profile_len); /* ICC profile */
        l_colr_size += jp2->color.icc_profile_len;
        break;
    default :
        return 00;
    }

    l_colr_data = (OPJ_BYTE *) opj_calloc(1, l_colr_size);
    if (l_colr_data == 00) {
        return 00;
    }

    l_current_colr_ptr = l_colr_data;

    opj_write_bytes(l_current_colr_ptr, l_colr_size,
                    4);            /* write box size */
    l_current_colr_ptr += 4;

    opj_write_bytes(l_current_colr_ptr, JP2_COLR, 4);               /* BPCC */
    l_current_colr_ptr += 4;

    opj_write_bytes(l_current_colr_ptr, jp2->meth, 1);              /* METH */
    ++l_current_colr_ptr;

    opj_write_bytes(l_current_colr_ptr, jp2->precedence, 1);        /* PRECEDENCE */
    ++l_current_colr_ptr;

    opj_write_bytes(l_current_colr_ptr, jp2->approx, 1);            /* APPROX */
    ++l_current_colr_ptr;

    if (jp2->meth ==
            1) { /* Meth value is restricted to 1 or 2 (Table I.9 of part 1) */
        opj_write_bytes(l_current_colr_ptr, jp2->enumcs, 4);
    }       /* EnumCS */
    else {
        if (jp2->meth == 2) {                                      /* ICC profile */
            OPJ_UINT32 i;
            for (i = 0; i < jp2->color.icc_profile_len; ++i) {
                opj_write_bytes(l_current_colr_ptr, jp2->color.icc_profile_buf[i], 1);
                ++l_current_colr_ptr;
            }
        }
    }

    *p_nb_bytes_written = l_colr_size;

    return l_colr_data;
}

static void opj_jp2_free_pclr(opj_jp2_color_t *color)
{
    opj_free(color->jp2_pclr->channel_sign);
    opj_free(color->jp2_pclr->channel_size);
    opj_free(color->jp2_pclr->entries);

    if (color->jp2_pclr->cmap) {
        opj_free(color->jp2_pclr->cmap);
    }

    opj_free(color->jp2_pclr);
    color->jp2_pclr = NULL;
}

static OPJ_BOOL opj_jp2_check_color(opj_image_t *image, opj_jp2_color_t *color,
                                    opj_event_mgr_t *p_manager)
{
    OPJ_UINT16 i;

    /* testcase 4149.pdf.SIGSEGV.cf7.3501 */
    if (color->jp2_cdef) {
        opj_jp2_cdef_info_t *info = color->jp2_cdef->info;
        OPJ_UINT16 n = color->jp2_cdef->n;
        OPJ_UINT32 nr_channels =
            image->numcomps; /* FIXME image->numcomps == jp2->numcomps before color is applied ??? */

        /* cdef applies to cmap channels if any */
        if (color->jp2_pclr && color->jp2_pclr->cmap) {
            nr_channels = (OPJ_UINT32)color->jp2_pclr->nr_channels;
        }

        for (i = 0; i < n; i++) {
            if (info[i].cn >= nr_channels) {
                opj_event_msg(p_manager, EVT_ERROR, "Invalid component index %d (>= %d).\n",
                              info[i].cn, nr_channels);
                return OPJ_FALSE;
            }
            if (info[i].asoc == 65535U) {
                continue;
            }

            if (info[i].asoc > 0 && (OPJ_UINT32)(info[i].asoc - 1) >= nr_channels) {
                opj_event_msg(p_manager, EVT_ERROR, "Invalid component index %d (>= %d).\n",
                              info[i].asoc - 1, nr_channels);
                return OPJ_FALSE;
            }
        }

        /* issue 397 */
        /* ISO 15444-1 states that if cdef is present, it shall contain a complete list of channel definitions. */
        while (nr_channels > 0) {
            for (i = 0; i < n; ++i) {
                if ((OPJ_UINT32)info[i].cn == (nr_channels - 1U)) {
                    break;
                }
            }
            if (i == n) {
                opj_event_msg(p_manager, EVT_ERROR, "Incomplete channel definitions.\n");
                return OPJ_FALSE;
            }
            --nr_channels;
        }
    }

    /* testcases 451.pdf.SIGSEGV.f4c.3723, 451.pdf.SIGSEGV.5b5.3723 and
       66ea31acbb0f23a2bbc91f64d69a03f5_signal_sigsegv_13937c0_7030_5725.pdf */
    if (color->jp2_pclr && color->jp2_pclr->cmap) {
        OPJ_UINT16 nr_channels = color->jp2_pclr->nr_channels;
        opj_jp2_cmap_comp_t *cmap = color->jp2_pclr->cmap;
        OPJ_BOOL *pcol_usage, is_sane = OPJ_TRUE;

        /* verify that all original components match an existing one */
        for (i = 0; i < nr_channels; i++) {
            if (cmap[i].cmp >= image->numcomps) {
                opj_event_msg(p_manager, EVT_ERROR, "Invalid component index %d (>= %d).\n",
                              cmap[i].cmp, image->numcomps);
                is_sane = OPJ_FALSE;
            }
        }

        pcol_usage = (OPJ_BOOL *) opj_calloc(nr_channels, sizeof(OPJ_BOOL));
        if (!pcol_usage) {
            opj_event_msg(p_manager, EVT_ERROR, "Unexpected OOM.\n");
            return OPJ_FALSE;
        }
        /* verify that no component is targeted more than once */
        for (i = 0; i < nr_channels; i++) {
            OPJ_BYTE mtyp = cmap[i].mtyp;
            OPJ_BYTE pcol = cmap[i].pcol;
            /* See ISO 15444-1 Table I.14 – MTYPi field values */
            if (mtyp != 0 && mtyp != 1) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid value for cmap[%d].mtyp = %d.\n", i,
                              mtyp);
                is_sane = OPJ_FALSE;
            } else if (pcol >= nr_channels) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid component/palette index for direct mapping %d.\n", pcol);
                is_sane = OPJ_FALSE;
            } else if (pcol_usage[pcol] && mtyp == 1) {
                opj_event_msg(p_manager, EVT_ERROR, "Component %d is mapped twice.\n", pcol);
                is_sane = OPJ_FALSE;
            } else if (mtyp == 0 && pcol != 0) {
                /* I.5.3.5 PCOL: If the value of the MTYP field for this channel is 0, then
                 * the value of this field shall be 0. */
                opj_event_msg(p_manager, EVT_ERROR, "Direct use at #%d however pcol=%d.\n", i,
                              pcol);
                is_sane = OPJ_FALSE;
            } else if (mtyp == 1 && pcol != i) {
                /* OpenJPEG implementation limitation. See assert(i == pcol); */
                /* in opj_jp2_apply_pclr() */
                opj_event_msg(p_manager, EVT_ERROR,
                              "Implementation limitation: for palette mapping, "
                              "pcol[%d] should be equal to %d, but is equal "
                              "to %d.\n", i, i, pcol);
                is_sane = OPJ_FALSE;
            } else {
                pcol_usage[pcol] = OPJ_TRUE;
            }
        }
        /* verify that all components are targeted at least once */
        for (i = 0; i < nr_channels; i++) {
            if (!pcol_usage[i] && cmap[i].mtyp != 0) {
                opj_event_msg(p_manager, EVT_ERROR, "Component %d doesn't have a mapping.\n",
                              i);
                is_sane = OPJ_FALSE;
            }
        }
        /* Issue 235/447 weird cmap */
        if (1 && is_sane && (image->numcomps == 1U)) {
            for (i = 0; i < nr_channels; i++) {
                if (!pcol_usage[i]) {
                    is_sane = 0U;
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "Component mapping seems wrong. Trying to correct.\n");
                    break;
                }
            }
            if (!is_sane) {
                is_sane = OPJ_TRUE;
                for (i = 0; i < nr_channels; i++) {
                    cmap[i].mtyp = 1U;
                    cmap[i].pcol = (OPJ_BYTE) i;
                }
            }
        }
        opj_free(pcol_usage);
        if (!is_sane) {
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}

/* file9.jp2 */
static OPJ_BOOL opj_jp2_apply_pclr(opj_image_t *image,
                                   opj_jp2_color_t *color,
                                   opj_event_mgr_t * p_manager)
{
    opj_image_comp_t *old_comps, *new_comps;
    OPJ_BYTE *channel_size, *channel_sign;
    OPJ_UINT32 *entries;
    opj_jp2_cmap_comp_t *cmap;
    OPJ_INT32 *src, *dst;
    OPJ_UINT32 j, max;
    OPJ_UINT16 i, nr_channels, cmp, pcol;
    OPJ_INT32 k, top_k;

    channel_size = color->jp2_pclr->channel_size;
    channel_sign = color->jp2_pclr->channel_sign;
    entries = color->jp2_pclr->entries;
    cmap = color->jp2_pclr->cmap;
    nr_channels = color->jp2_pclr->nr_channels;

    for (i = 0; i < nr_channels; ++i) {
        /* Palette mapping: */
        cmp = cmap[i].cmp;
        if (image->comps[cmp].data == NULL) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "image->comps[%d].data == NULL in opj_jp2_apply_pclr().\n", i);
            return OPJ_FALSE;
        }
    }

    old_comps = image->comps;
    new_comps = (opj_image_comp_t*)
                opj_malloc(nr_channels * sizeof(opj_image_comp_t));
    if (!new_comps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Memory allocation failure in opj_jp2_apply_pclr().\n");
        return OPJ_FALSE;
    }
    for (i = 0; i < nr_channels; ++i) {
        pcol = cmap[i].pcol;
        cmp = cmap[i].cmp;

        /* Direct use */
        if (cmap[i].mtyp == 0) {
            assert(pcol == 0);
            new_comps[i] = old_comps[cmp];
        } else {
            assert(i == pcol);
            new_comps[pcol] = old_comps[cmp];
        }

        /* Palette mapping: */
        new_comps[i].data = (OPJ_INT32*)
                            opj_image_data_alloc(sizeof(OPJ_INT32) * old_comps[cmp].w * old_comps[cmp].h);
        if (!new_comps[i].data) {
            while (i > 0) {
                -- i;
                opj_image_data_free(new_comps[i].data);
            }
            opj_free(new_comps);
            opj_event_msg(p_manager, EVT_ERROR,
                          "Memory allocation failure in opj_jp2_apply_pclr().\n");
            return OPJ_FALSE;
        }
        new_comps[i].prec = channel_size[i];
        new_comps[i].sgnd = channel_sign[i];
    }

    top_k = color->jp2_pclr->nr_entries - 1;

    for (i = 0; i < nr_channels; ++i) {
        /* Palette mapping: */
        cmp = cmap[i].cmp;
        pcol = cmap[i].pcol;
        src = old_comps[cmp].data;
        assert(src); /* verified above */
        max = new_comps[pcol].w * new_comps[pcol].h;

        /* Direct use: */
        if (cmap[i].mtyp == 0) {
            dst = new_comps[i].data;
            assert(dst);
            for (j = 0; j < max; ++j) {
                dst[j] = src[j];
            }
        } else {
            assert(i == pcol);
            dst = new_comps[pcol].data;
            assert(dst);
            for (j = 0; j < max; ++j) {
                /* The index */
                if ((k = src[j]) < 0) {
                    k = 0;
                } else if (k > top_k) {
                    k = top_k;
                }

                /* The colour */
                dst[j] = (OPJ_INT32)entries[k * nr_channels + pcol];
            }
        }
    }

    max = image->numcomps;
    for (i = 0; i < max; ++i) {
        if (old_comps[i].data) {
            opj_image_data_free(old_comps[i].data);
        }
    }

    opj_free(old_comps);
    image->comps = new_comps;
    image->numcomps = nr_channels;

    return OPJ_TRUE;
}/* apply_pclr() */

static OPJ_BOOL opj_jp2_read_pclr(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_pclr_header_data,
                                  OPJ_UINT32 p_pclr_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    opj_jp2_pclr_t *jp2_pclr;
    OPJ_BYTE *channel_size, *channel_sign;
    OPJ_UINT32 *entries;
    OPJ_UINT16 nr_entries, nr_channels;
    OPJ_UINT16 i, j;
    OPJ_UINT32 l_value;
    OPJ_BYTE *orig_header_data = p_pclr_header_data;

    /* preconditions */
    assert(p_pclr_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);
    (void)p_pclr_header_size;

    if (jp2->color.jp2_pclr) {
        return OPJ_FALSE;
    }

    if (p_pclr_header_size < 3) {
        return OPJ_FALSE;
    }

    opj_read_bytes(p_pclr_header_data, &l_value, 2);    /* NE */
    p_pclr_header_data += 2;
    nr_entries = (OPJ_UINT16) l_value;
    if ((nr_entries == 0U) || (nr_entries > 1024U)) {
        opj_event_msg(p_manager, EVT_ERROR, "Invalid PCLR box. Reports %d entries\n",
                      (int)nr_entries);
        return OPJ_FALSE;
    }

    opj_read_bytes(p_pclr_header_data, &l_value, 1);    /* NPC */
    ++p_pclr_header_data;
    nr_channels = (OPJ_UINT16) l_value;
    if (nr_channels == 0U) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid PCLR box. Reports 0 palette columns\n");
        return OPJ_FALSE;
    }

    if (p_pclr_header_size < 3 + (OPJ_UINT32)nr_channels) {
        return OPJ_FALSE;
    }

    entries = (OPJ_UINT32*) opj_malloc(sizeof(OPJ_UINT32) * nr_channels *
                                       nr_entries);
    if (!entries) {
        return OPJ_FALSE;
    }
    channel_size = (OPJ_BYTE*) opj_malloc(nr_channels);
    if (!channel_size) {
        opj_free(entries);
        return OPJ_FALSE;
    }
    channel_sign = (OPJ_BYTE*) opj_malloc(nr_channels);
    if (!channel_sign) {
        opj_free(entries);
        opj_free(channel_size);
        return OPJ_FALSE;
    }

    jp2_pclr = (opj_jp2_pclr_t*)opj_malloc(sizeof(opj_jp2_pclr_t));
    if (!jp2_pclr) {
        opj_free(entries);
        opj_free(channel_size);
        opj_free(channel_sign);
        return OPJ_FALSE;
    }

    jp2_pclr->channel_sign = channel_sign;
    jp2_pclr->channel_size = channel_size;
    jp2_pclr->entries = entries;
    jp2_pclr->nr_entries = nr_entries;
    jp2_pclr->nr_channels = (OPJ_BYTE) l_value;
    jp2_pclr->cmap = NULL;

    jp2->color.jp2_pclr = jp2_pclr;

    for (i = 0; i < nr_channels; ++i) {
        opj_read_bytes(p_pclr_header_data, &l_value, 1);    /* Bi */
        ++p_pclr_header_data;

        channel_size[i] = (OPJ_BYTE)((l_value & 0x7f) + 1);
        channel_sign[i] = (l_value & 0x80) ? 1 : 0;
    }

    for (j = 0; j < nr_entries; ++j) {
        for (i = 0; i < nr_channels; ++i) {
            OPJ_UINT32 bytes_to_read = (OPJ_UINT32)((channel_size[i] + 7) >> 3);

            if (bytes_to_read > sizeof(OPJ_UINT32)) {
                bytes_to_read = sizeof(OPJ_UINT32);
            }
            if ((ptrdiff_t)p_pclr_header_size < (ptrdiff_t)(p_pclr_header_data -
                    orig_header_data) + (ptrdiff_t)bytes_to_read) {
                return OPJ_FALSE;
            }

            opj_read_bytes(p_pclr_header_data, &l_value, bytes_to_read);    /* Cji */
            p_pclr_header_data += bytes_to_read;
            *entries = (OPJ_UINT32) l_value;
            entries++;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_read_cmap(opj_jp2_t * jp2,
                                  OPJ_BYTE * p_cmap_header_data,
                                  OPJ_UINT32 p_cmap_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    opj_jp2_cmap_comp_t *cmap;
    OPJ_BYTE i, nr_channels;
    OPJ_UINT32 l_value;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_cmap_header_data != 00);
    assert(p_manager != 00);
    (void)p_cmap_header_size;

    /* Need nr_channels: */
    if (jp2->color.jp2_pclr == NULL) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Need to read a PCLR box before the CMAP box.\n");
        return OPJ_FALSE;
    }

    /* Part 1, I.5.3.5: 'There shall be at most one Component Mapping box
     * inside a JP2 Header box' :
    */
    if (jp2->color.jp2_pclr->cmap) {
        opj_event_msg(p_manager, EVT_ERROR, "Only one CMAP box is allowed.\n");
        return OPJ_FALSE;
    }

    nr_channels = jp2->color.jp2_pclr->nr_channels;
    if (p_cmap_header_size < (OPJ_UINT32)nr_channels * 4) {
        opj_event_msg(p_manager, EVT_ERROR, "Insufficient data for CMAP box.\n");
        return OPJ_FALSE;
    }

    cmap = (opj_jp2_cmap_comp_t*) opj_malloc(nr_channels * sizeof(
                opj_jp2_cmap_comp_t));
    if (!cmap) {
        return OPJ_FALSE;
    }


    for (i = 0; i < nr_channels; ++i) {
        opj_read_bytes(p_cmap_header_data, &l_value, 2);            /* CMP^i */
        p_cmap_header_data += 2;
        cmap[i].cmp = (OPJ_UINT16) l_value;

        opj_read_bytes(p_cmap_header_data, &l_value, 1);            /* MTYP^i */
        ++p_cmap_header_data;
        cmap[i].mtyp = (OPJ_BYTE) l_value;

        opj_read_bytes(p_cmap_header_data, &l_value, 1);            /* PCOL^i */
        ++p_cmap_header_data;
        cmap[i].pcol = (OPJ_BYTE) l_value;
    }

    jp2->color.jp2_pclr->cmap = cmap;

    return OPJ_TRUE;
}

static void opj_jp2_apply_cdef(opj_image_t *image, opj_jp2_color_t *color,
                               opj_event_mgr_t *manager)
{
    opj_jp2_cdef_info_t *info;
    OPJ_UINT16 i, n, cn, asoc, acn;

    info = color->jp2_cdef->info;
    n = color->jp2_cdef->n;

    for (i = 0; i < n; ++i) {
        /* WATCH: acn = asoc - 1 ! */
        asoc = info[i].asoc;
        cn = info[i].cn;

        if (cn >= image->numcomps) {
            opj_event_msg(manager, EVT_WARNING, "opj_jp2_apply_cdef: cn=%d, numcomps=%d\n",
                          cn, image->numcomps);
            continue;
        }
        if (asoc == 0 || asoc == 65535) {
            image->comps[cn].alpha = info[i].typ;
            continue;
        }

        acn = (OPJ_UINT16)(asoc - 1);
        if (acn >= image->numcomps) {
            opj_event_msg(manager, EVT_WARNING, "opj_jp2_apply_cdef: acn=%d, numcomps=%d\n",
                          acn, image->numcomps);
            continue;
        }

        /* Swap only if color channel */
        if ((cn != acn) && (info[i].typ == 0)) {
            opj_image_comp_t saved;
            OPJ_UINT16 j;

            memcpy(&saved, &image->comps[cn], sizeof(opj_image_comp_t));
            memcpy(&image->comps[cn], &image->comps[acn], sizeof(opj_image_comp_t));
            memcpy(&image->comps[acn], &saved, sizeof(opj_image_comp_t));

            /* Swap channels in following channel definitions, don't bother with j <= i that are already processed */
            for (j = (OPJ_UINT16)(i + 1U); j < n ; ++j) {
                if (info[j].cn == cn) {
                    info[j].cn = acn;
                } else if (info[j].cn == acn) {
                    info[j].cn = cn;
                }
                /* asoc is related to color index. Do not update. */
            }
        }

        image->comps[cn].alpha = info[i].typ;
    }

    if (color->jp2_cdef->info) {
        opj_free(color->jp2_cdef->info);
    }

    opj_free(color->jp2_cdef);
    color->jp2_cdef = NULL;

}/* jp2_apply_cdef() */

static OPJ_BOOL opj_jp2_read_cdef(opj_jp2_t * jp2,
                                  OPJ_BYTE * p_cdef_header_data,
                                  OPJ_UINT32 p_cdef_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    opj_jp2_cdef_info_t *cdef_info;
    OPJ_UINT16 i;
    OPJ_UINT32 l_value;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_cdef_header_data != 00);
    assert(p_manager != 00);
    (void)p_cdef_header_size;

    /* Part 1, I.5.3.6: 'The shall be at most one Channel Definition box
     * inside a JP2 Header box.'*/
    if (jp2->color.jp2_cdef) {
        return OPJ_FALSE;
    }

    if (p_cdef_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Insufficient data for CDEF box.\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_cdef_header_data, &l_value, 2);        /* N */
    p_cdef_header_data += 2;

    if ((OPJ_UINT16)l_value == 0) { /* szukw000: FIXME */
        opj_event_msg(p_manager, EVT_ERROR,
                      "Number of channel description is equal to zero in CDEF box.\n");
        return OPJ_FALSE;
    }

    if (p_cdef_header_size < 2 + (OPJ_UINT32)(OPJ_UINT16)l_value * 6) {
        opj_event_msg(p_manager, EVT_ERROR, "Insufficient data for CDEF box.\n");
        return OPJ_FALSE;
    }

    cdef_info = (opj_jp2_cdef_info_t*) opj_malloc(l_value * sizeof(
                    opj_jp2_cdef_info_t));
    if (!cdef_info) {
        return OPJ_FALSE;
    }

    jp2->color.jp2_cdef = (opj_jp2_cdef_t*)opj_malloc(sizeof(opj_jp2_cdef_t));
    if (!jp2->color.jp2_cdef) {
        opj_free(cdef_info);
        return OPJ_FALSE;
    }
    jp2->color.jp2_cdef->info = cdef_info;
    jp2->color.jp2_cdef->n = (OPJ_UINT16) l_value;

    for (i = 0; i < jp2->color.jp2_cdef->n; ++i) {
        opj_read_bytes(p_cdef_header_data, &l_value, 2);            /* Cn^i */
        p_cdef_header_data += 2;
        cdef_info[i].cn = (OPJ_UINT16) l_value;

        opj_read_bytes(p_cdef_header_data, &l_value, 2);            /* Typ^i */
        p_cdef_header_data += 2;
        cdef_info[i].typ = (OPJ_UINT16) l_value;

        opj_read_bytes(p_cdef_header_data, &l_value, 2);            /* Asoc^i */
        p_cdef_header_data += 2;
        cdef_info[i].asoc = (OPJ_UINT16) l_value;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_read_colr(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_colr_header_data,
                                  OPJ_UINT32 p_colr_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_value;

    /* preconditions */
    assert(jp2 != 00);
    assert(p_colr_header_data != 00);
    assert(p_manager != 00);

    if (p_colr_header_size < 3) {
        opj_event_msg(p_manager, EVT_ERROR, "Bad COLR header box (bad size)\n");
        return OPJ_FALSE;
    }

    /* Part 1, I.5.3.3 : 'A conforming JP2 reader shall ignore all Colour
     * Specification boxes after the first.'
    */
    if (jp2->color.jp2_has_colr) {
        opj_event_msg(p_manager, EVT_INFO,
                      "A conforming JP2 reader shall ignore all Colour Specification boxes after the first, so we ignore this one.\n");
        p_colr_header_data += p_colr_header_size;
        return OPJ_TRUE;
    }

    opj_read_bytes(p_colr_header_data, &jp2->meth, 1);          /* METH */
    ++p_colr_header_data;

    opj_read_bytes(p_colr_header_data, &jp2->precedence, 1);    /* PRECEDENCE */
    ++p_colr_header_data;

    opj_read_bytes(p_colr_header_data, &jp2->approx, 1);        /* APPROX */
    ++p_colr_header_data;

    if (jp2->meth == 1) {
        if (p_colr_header_size < 7) {
            opj_event_msg(p_manager, EVT_ERROR, "Bad COLR header box (bad size: %d)\n",
                          p_colr_header_size);
            return OPJ_FALSE;
        }
        if ((p_colr_header_size > 7) &&
                (jp2->enumcs != 14)) { /* handled below for CIELab) */
            /* testcase Altona_Technical_v20_x4.pdf */
            opj_event_msg(p_manager, EVT_WARNING, "Bad COLR header box (bad size: %d)\n",
                          p_colr_header_size);
        }

        opj_read_bytes(p_colr_header_data, &jp2->enumcs, 4);        /* EnumCS */

        p_colr_header_data += 4;

        if (jp2->enumcs == 14) { /* CIELab */
            OPJ_UINT32 *cielab;
            OPJ_UINT32 rl, ol, ra, oa, rb, ob, il;

            cielab = (OPJ_UINT32*)opj_malloc(9 * sizeof(OPJ_UINT32));
            if (cielab == NULL) {
                opj_event_msg(p_manager, EVT_ERROR, "Not enough memory for cielab\n");
                return OPJ_FALSE;
            }
            cielab[0] = 14; /* enumcs */

            /* default values */
            rl = ra = rb = ol = oa = ob = 0;
            il = 0x00443530; /* D50 */
            cielab[1] = 0x44454600;/* DEF */

            if (p_colr_header_size == 35) {
                opj_read_bytes(p_colr_header_data, &rl, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &ol, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &ra, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &oa, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &rb, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &ob, 4);
                p_colr_header_data += 4;
                opj_read_bytes(p_colr_header_data, &il, 4);
                p_colr_header_data += 4;

                cielab[1] = 0;
            } else if (p_colr_header_size != 7) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Bad COLR header box (CIELab, bad size: %d)\n", p_colr_header_size);
            }
            cielab[2] = rl;
            cielab[4] = ra;
            cielab[6] = rb;
            cielab[3] = ol;
            cielab[5] = oa;
            cielab[7] = ob;
            cielab[8] = il;

            jp2->color.icc_profile_buf = (OPJ_BYTE*)cielab;
            jp2->color.icc_profile_len = 0;
        }
        jp2->color.jp2_has_colr = 1;
    } else if (jp2->meth == 2) {
        /* ICC profile */
        OPJ_INT32 it_icc_value = 0;
        OPJ_INT32 icc_len = (OPJ_INT32)p_colr_header_size - 3;

        jp2->color.icc_profile_len = (OPJ_UINT32)icc_len;
        jp2->color.icc_profile_buf = (OPJ_BYTE*) opj_calloc(1, (size_t)icc_len);
        if (!jp2->color.icc_profile_buf) {
            jp2->color.icc_profile_len = 0;
            return OPJ_FALSE;
        }

        for (it_icc_value = 0; it_icc_value < icc_len; ++it_icc_value) {
            opj_read_bytes(p_colr_header_data, &l_value, 1);    /* icc values */
            ++p_colr_header_data;
            jp2->color.icc_profile_buf[it_icc_value] = (OPJ_BYTE) l_value;
        }

        jp2->color.jp2_has_colr = 1;
    } else if (jp2->meth > 2) {
        /*  ISO/IEC 15444-1:2004 (E), Table I.9 Legal METH values:
        conforming JP2 reader shall ignore the entire Colour Specification box.*/
        opj_event_msg(p_manager, EVT_INFO,
                      "COLR BOX meth value is not a regular value (%d), "
                      "so we will ignore the entire Colour Specification box. \n", jp2->meth);
    }
    if (jp2->color.jp2_has_colr) {
        jp2->j2k->enumcs = jp2->enumcs;
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_jp2_decode(opj_jp2_t *jp2,
                        opj_stream_private_t *p_stream,
                        opj_image_t* p_image,
                        opj_event_mgr_t * p_manager)
{
    if (!p_image) {
        return OPJ_FALSE;
    }

    /* J2K decoding */
    if (! opj_j2k_decode(jp2->j2k, p_stream, p_image, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Failed to decode the codestream in the JP2 file\n");
        return OPJ_FALSE;
    }

    if (jp2->j2k->m_specific_param.m_decoder.m_numcomps_to_decode) {
        /* Bypass all JP2 component transforms */
        return OPJ_TRUE;
    }

    if (!jp2->ignore_pclr_cmap_cdef) {
        if (!opj_jp2_check_color(p_image, &(jp2->color), p_manager)) {
            return OPJ_FALSE;
        }

        /* Set Image Color Space */
        if (jp2->enumcs == 16) {
            p_image->color_space = OPJ_CLRSPC_SRGB;
        } else if (jp2->enumcs == 17) {
            p_image->color_space = OPJ_CLRSPC_GRAY;
        } else if (jp2->enumcs == 18) {
            p_image->color_space = OPJ_CLRSPC_SYCC;
        } else if (jp2->enumcs == 24) {
            p_image->color_space = OPJ_CLRSPC_EYCC;
        } else if (jp2->enumcs == 12) {
            p_image->color_space = OPJ_CLRSPC_CMYK;
        } else {
            p_image->color_space = OPJ_CLRSPC_UNKNOWN;
        }

        if (jp2->color.jp2_pclr) {
            /* Part 1, I.5.3.4: Either both or none : */
            if (!jp2->color.jp2_pclr->cmap) {
                opj_jp2_free_pclr(&(jp2->color));
            } else {
                if (!opj_jp2_apply_pclr(p_image, &(jp2->color), p_manager)) {
                    return OPJ_FALSE;
                }
            }
        }

        /* Apply the color space if needed */
        if (jp2->color.jp2_cdef) {
            opj_jp2_apply_cdef(p_image, &(jp2->color), p_manager);
        }

        if (jp2->color.icc_profile_buf) {
            p_image->icc_profile_buf = jp2->color.icc_profile_buf;
            p_image->icc_profile_len = jp2->color.icc_profile_len;
            jp2->color.icc_profile_buf = NULL;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_write_jp2h(opj_jp2_t *jp2,
                                   opj_stream_private_t *stream,
                                   opj_event_mgr_t * p_manager
                                  )
{
    opj_jp2_img_header_writer_handler_t l_writers [4];
    opj_jp2_img_header_writer_handler_t * l_current_writer;

    OPJ_INT32 i, l_nb_pass;
    /* size of data for super box*/
    OPJ_UINT32 l_jp2h_size = 8;
    OPJ_BOOL l_result = OPJ_TRUE;

    /* to store the data of the super box */
    OPJ_BYTE l_jp2h_data [8];

    /* preconditions */
    assert(stream != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    memset(l_writers, 0, sizeof(l_writers));

    if (jp2->bpc == 255) {
        l_nb_pass = 3;
        l_writers[0].handler = opj_jp2_write_ihdr;
        l_writers[1].handler = opj_jp2_write_bpcc;
        l_writers[2].handler = opj_jp2_write_colr;
    } else {
        l_nb_pass = 2;
        l_writers[0].handler = opj_jp2_write_ihdr;
        l_writers[1].handler = opj_jp2_write_colr;
    }

    if (jp2->color.jp2_cdef != NULL) {
        l_writers[l_nb_pass].handler = opj_jp2_write_cdef;
        l_nb_pass++;
    }

    /* write box header */
    /* write JP2H type */
    opj_write_bytes(l_jp2h_data + 4, JP2_JP2H, 4);

    l_current_writer = l_writers;
    for (i = 0; i < l_nb_pass; ++i) {
        l_current_writer->m_data = l_current_writer->handler(jp2,
                                   &(l_current_writer->m_size));
        if (l_current_writer->m_data == 00) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to hold JP2 Header data\n");
            l_result = OPJ_FALSE;
            break;
        }

        l_jp2h_size += l_current_writer->m_size;
        ++l_current_writer;
    }

    if (! l_result) {
        l_current_writer = l_writers;
        for (i = 0; i < l_nb_pass; ++i) {
            if (l_current_writer->m_data != 00) {
                opj_free(l_current_writer->m_data);
            }
            ++l_current_writer;
        }

        return OPJ_FALSE;
    }

    /* write super box size */
    opj_write_bytes(l_jp2h_data, l_jp2h_size, 4);

    /* write super box data on stream */
    if (opj_stream_write_data(stream, l_jp2h_data, 8, p_manager) != 8) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Stream error while writing JP2 Header box\n");
        l_result = OPJ_FALSE;
    }

    if (l_result) {
        l_current_writer = l_writers;
        for (i = 0; i < l_nb_pass; ++i) {
            if (opj_stream_write_data(stream, l_current_writer->m_data,
                                      l_current_writer->m_size, p_manager) != l_current_writer->m_size) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Stream error while writing JP2 Header box\n");
                l_result = OPJ_FALSE;
                break;
            }
            ++l_current_writer;
        }
    }

    l_current_writer = l_writers;

    /* cleanup */
    for (i = 0; i < l_nb_pass; ++i) {
        if (l_current_writer->m_data != 00) {
            opj_free(l_current_writer->m_data);
        }
        ++l_current_writer;
    }

    return l_result;
}

static OPJ_BOOL opj_jp2_write_ftyp(opj_jp2_t *jp2,
                                   opj_stream_private_t *cio,
                                   opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_ftyp_size;
    OPJ_BYTE * l_ftyp_data, * l_current_data_ptr;
    OPJ_BOOL l_result;

    /* preconditions */
    assert(cio != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);
    l_ftyp_size = 16 + 4 * jp2->numcl;

    l_ftyp_data = (OPJ_BYTE *) opj_calloc(1, l_ftyp_size);

    if (l_ftyp_data == 00) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to handle ftyp data\n");
        return OPJ_FALSE;
    }

    l_current_data_ptr = l_ftyp_data;

    opj_write_bytes(l_current_data_ptr, l_ftyp_size, 4); /* box size */
    l_current_data_ptr += 4;

    opj_write_bytes(l_current_data_ptr, JP2_FTYP, 4); /* FTYP */
    l_current_data_ptr += 4;

    opj_write_bytes(l_current_data_ptr, jp2->brand, 4); /* BR */
    l_current_data_ptr += 4;

    opj_write_bytes(l_current_data_ptr, jp2->minversion, 4); /* MinV */
    l_current_data_ptr += 4;

    for (i = 0; i < jp2->numcl; i++)  {
        opj_write_bytes(l_current_data_ptr, jp2->cl[i], 4); /* CL */
    }

    l_result = (opj_stream_write_data(cio, l_ftyp_data, l_ftyp_size,
                                      p_manager) == l_ftyp_size);
    if (! l_result) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error while writing ftyp data to stream\n");
    }

    opj_free(l_ftyp_data);

    return l_result;
}

static OPJ_BOOL opj_jp2_write_jp2c(opj_jp2_t *jp2,
                                   opj_stream_private_t *cio,
                                   opj_event_mgr_t * p_manager)
{
    OPJ_OFF_T j2k_codestream_exit;
    OPJ_BYTE l_data_header [8];

    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);
    assert(opj_stream_has_seek(cio));

    j2k_codestream_exit = opj_stream_tell(cio);
    opj_write_bytes(l_data_header,
                    (OPJ_UINT32)(j2k_codestream_exit - jp2->j2k_codestream_offset),
                    4); /* size of codestream */
    opj_write_bytes(l_data_header + 4, JP2_JP2C,
                    4);                                     /* JP2C */

    if (! opj_stream_seek(cio, jp2->j2k_codestream_offset, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(cio, l_data_header, 8, p_manager) != 8) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    if (! opj_stream_seek(cio, j2k_codestream_exit, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_write_jp(opj_jp2_t *jp2,
                                 opj_stream_private_t *cio,
                                 opj_event_mgr_t * p_manager)
{
    /* 12 bytes will be read */
    OPJ_BYTE l_signature_data [12];

    /* preconditions */
    assert(cio != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(jp2);

    /* write box length */
    opj_write_bytes(l_signature_data, 12, 4);
    /* writes box type */
    opj_write_bytes(l_signature_data + 4, JP2_JP, 4);
    /* writes magic number*/
    opj_write_bytes(l_signature_data + 8, 0x0d0a870a, 4);

    if (opj_stream_write_data(cio, l_signature_data, 12, p_manager) != 12) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/* ----------------------------------------------------------------------- */
/* JP2 decoder interface                                             */
/* ----------------------------------------------------------------------- */

void opj_jp2_setup_decoder(opj_jp2_t *jp2, opj_dparameters_t *parameters)
{
    /* setup the J2K codec */
    opj_j2k_setup_decoder(jp2->j2k, parameters);

    /* further JP2 initializations go here */
    jp2->color.jp2_has_colr = 0;
    jp2->ignore_pclr_cmap_cdef = parameters->flags &
                                 OPJ_DPARAMETERS_IGNORE_PCLR_CMAP_CDEF_FLAG;
}

OPJ_BOOL opj_jp2_set_threads(opj_jp2_t *jp2, OPJ_UINT32 num_threads)
{
    return opj_j2k_set_threads(jp2->j2k, num_threads);
}

/* ----------------------------------------------------------------------- */
/* JP2 encoder interface                                             */
/* ----------------------------------------------------------------------- */

OPJ_BOOL opj_jp2_setup_encoder(opj_jp2_t *jp2,
                               opj_cparameters_t *parameters,
                               opj_image_t *image,
                               opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i;
    OPJ_UINT32 depth_0;
    OPJ_UINT32 sign;
    OPJ_UINT32 alpha_count;
    OPJ_UINT32 color_channels = 0U;
    OPJ_UINT32 alpha_channel = 0U;


    if (!jp2 || !parameters || !image) {
        return OPJ_FALSE;
    }

    /* setup the J2K codec */
    /* ------------------- */

    /* Check if number of components respects standard */
    if (image->numcomps < 1 || image->numcomps > 16384) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid number of components specified while setting up JP2 encoder\n");
        return OPJ_FALSE;
    }

    if (opj_j2k_setup_encoder(jp2->j2k, parameters, image,
                              p_manager) == OPJ_FALSE) {
        return OPJ_FALSE;
    }

    /* setup the JP2 codec */
    /* ------------------- */

    /* Profile box */

    jp2->brand = JP2_JP2;   /* BR */
    jp2->minversion = 0;    /* MinV */
    jp2->numcl = 1;
    jp2->cl = (OPJ_UINT32*) opj_malloc(jp2->numcl * sizeof(OPJ_UINT32));
    if (!jp2->cl) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory when setup the JP2 encoder\n");
        return OPJ_FALSE;
    }
    jp2->cl[0] = JP2_JP2;   /* CL0 : JP2 */

    /* Image Header box */

    jp2->numcomps = image->numcomps;    /* NC */
    jp2->comps = (opj_jp2_comps_t*) opj_malloc(jp2->numcomps * sizeof(
                     opj_jp2_comps_t));
    if (!jp2->comps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory when setup the JP2 encoder\n");
        /* Memory of jp2->cl will be freed by opj_jp2_destroy */
        return OPJ_FALSE;
    }

    jp2->h = image->y1 - image->y0;     /* HEIGHT */
    jp2->w = image->x1 - image->x0;     /* WIDTH */
    /* BPC */
    depth_0 = image->comps[0].prec - 1;
    sign = image->comps[0].sgnd;
    jp2->bpc = depth_0 + (sign << 7);
    for (i = 1; i < image->numcomps; i++) {
        OPJ_UINT32 depth = image->comps[i].prec - 1;
        sign = image->comps[i].sgnd;
        if (depth_0 != depth) {
            jp2->bpc = 255;
        }
    }
    jp2->C = 7;         /* C : Always 7 */
    jp2->UnkC = 0;      /* UnkC, colorspace specified in colr box */
    jp2->IPR = 0;       /* IPR, no intellectual property */

    /* BitsPerComponent box */
    for (i = 0; i < image->numcomps; i++) {
        jp2->comps[i].bpcc = image->comps[i].prec - 1 + (image->comps[i].sgnd << 7);
    }

    /* Colour Specification box */
    if (image->icc_profile_len) {
        jp2->meth = 2;
        jp2->enumcs = 0;
    } else {
        jp2->meth = 1;
        if (image->color_space == 1) {
            jp2->enumcs = 16;    /* sRGB as defined by IEC 61966-2-1 */
        } else if (image->color_space == 2) {
            jp2->enumcs = 17;    /* greyscale */
        } else if (image->color_space == 3) {
            jp2->enumcs = 18;    /* YUV */
        }
    }

    /* Channel Definition box */
    /* FIXME not provided by parameters */
    /* We try to do what we can... */
    alpha_count = 0U;
    for (i = 0; i < image->numcomps; i++) {
        if (image->comps[i].alpha != 0) {
            alpha_count++;
            alpha_channel = i;
        }
    }
    if (alpha_count == 1U) { /* no way to deal with more than 1 alpha channel */
        switch (jp2->enumcs) {
        case 16:
        case 18:
            color_channels = 3;
            break;
        case 17:
            color_channels = 1;
            break;
        default:
            alpha_count = 0U;
            break;
        }
        if (alpha_count == 0U) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Alpha channel specified but unknown enumcs. No cdef box will be created.\n");
        } else if (image->numcomps < (color_channels + 1)) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Alpha channel specified but not enough image components for an automatic cdef box creation.\n");
            alpha_count = 0U;
        } else if ((OPJ_UINT32)alpha_channel < color_channels) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Alpha channel position conflicts with color channel. No cdef box will be created.\n");
            alpha_count = 0U;
        }
    } else if (alpha_count > 1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Multiple alpha channels specified. No cdef box will be created.\n");
    }
    if (alpha_count == 1U) { /* if here, we know what we can do */
        jp2->color.jp2_cdef = (opj_jp2_cdef_t*)opj_malloc(sizeof(opj_jp2_cdef_t));
        if (!jp2->color.jp2_cdef) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to setup the JP2 encoder\n");
            return OPJ_FALSE;
        }
        /* no memset needed, all values will be overwritten except if jp2->color.jp2_cdef->info allocation fails, */
        /* in which case jp2->color.jp2_cdef->info will be NULL => valid for destruction */
        jp2->color.jp2_cdef->info = (opj_jp2_cdef_info_t*) opj_malloc(
                                        image->numcomps * sizeof(opj_jp2_cdef_info_t));
        if (!jp2->color.jp2_cdef->info) {
            /* memory will be freed by opj_jp2_destroy */
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to setup the JP2 encoder\n");
            return OPJ_FALSE;
        }
        jp2->color.jp2_cdef->n = (OPJ_UINT16)
                                 image->numcomps; /* cast is valid : image->numcomps [1,16384] */
        for (i = 0U; i < color_channels; i++) {
            jp2->color.jp2_cdef->info[i].cn = (OPJ_UINT16)
                                              i; /* cast is valid : image->numcomps [1,16384] */
            jp2->color.jp2_cdef->info[i].typ = 0U;
            jp2->color.jp2_cdef->info[i].asoc = (OPJ_UINT16)(i +
                                                1U); /* No overflow + cast is valid : image->numcomps [1,16384] */
        }
        for (; i < image->numcomps; i++) {
            if (image->comps[i].alpha != 0) { /* we'll be here exactly once */
                jp2->color.jp2_cdef->info[i].cn = (OPJ_UINT16)
                                                  i; /* cast is valid : image->numcomps [1,16384] */
                jp2->color.jp2_cdef->info[i].typ = 1U; /* Opacity channel */
                jp2->color.jp2_cdef->info[i].asoc =
                    0U; /* Apply alpha channel to the whole image */
            } else {
                /* Unknown channel */
                jp2->color.jp2_cdef->info[i].cn = (OPJ_UINT16)
                                                  i; /* cast is valid : image->numcomps [1,16384] */
                jp2->color.jp2_cdef->info[i].typ = 65535U;
                jp2->color.jp2_cdef->info[i].asoc = 65535U;
            }
        }
    }

    jp2->precedence = 0;    /* PRECEDENCE */
    jp2->approx = 0;        /* APPROX */

    jp2->jpip_on = parameters->jpip_on;

    return OPJ_TRUE;
}

OPJ_BOOL opj_jp2_encode(opj_jp2_t *jp2,
                        opj_stream_private_t *stream,
                        opj_event_mgr_t * p_manager)
{
    return opj_j2k_encode(jp2->j2k, stream, p_manager);
}

OPJ_BOOL opj_jp2_end_decompress(opj_jp2_t *jp2,
                                opj_stream_private_t *cio,
                                opj_event_mgr_t * p_manager
                               )
{
    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);

    /* customization of the end encoding */
    if (! opj_jp2_setup_end_header_reading(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    /* write header */
    if (! opj_jp2_exec(jp2, jp2->m_procedure_list, cio, p_manager)) {
        return OPJ_FALSE;
    }

    return opj_j2k_end_decompress(jp2->j2k, cio, p_manager);
}

OPJ_BOOL opj_jp2_end_compress(opj_jp2_t *jp2,
                              opj_stream_private_t *cio,
                              opj_event_mgr_t * p_manager
                             )
{
    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);

    /* customization of the end encoding */
    if (! opj_jp2_setup_end_header_writing(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    if (! opj_j2k_end_compress(jp2->j2k, cio, p_manager)) {
        return OPJ_FALSE;
    }

    /* write header */
    return opj_jp2_exec(jp2, jp2->m_procedure_list, cio, p_manager);
}

static OPJ_BOOL opj_jp2_setup_end_header_writing(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

#ifdef USE_JPIP
    if (jp2->jpip_on) {
        if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                               (opj_procedure)opj_jpip_write_iptr, p_manager)) {
            return OPJ_FALSE;
        }
    }
#endif
    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_write_jp2c, p_manager)) {
        return OPJ_FALSE;
    }
    /* DEVELOPER CORNER, add your custom procedures */
#ifdef USE_JPIP
    if (jp2->jpip_on) {
        if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                               (opj_procedure)opj_jpip_write_cidx, p_manager)) {
            return OPJ_FALSE;
        }
        if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                               (opj_procedure)opj_jpip_write_fidx, p_manager)) {
            return OPJ_FALSE;
        }
    }
#endif
    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_setup_end_header_reading(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_read_header_procedure, p_manager)) {
        return OPJ_FALSE;
    }
    /* DEVELOPER CORNER, add your custom procedures */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_default_validation(opj_jp2_t * jp2,
        opj_stream_private_t *cio,
        opj_event_mgr_t * p_manager
                                          )
{
    OPJ_BOOL l_is_valid = OPJ_TRUE;
    OPJ_UINT32 i;

    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_manager);

    /* JPEG2000 codec validation */

    /* STATE checking */
    /* make sure the state is at 0 */
    l_is_valid &= (jp2->jp2_state == JP2_STATE_NONE);

    /* make sure not reading a jp2h ???? WEIRD */
    l_is_valid &= (jp2->jp2_img_state == JP2_IMG_STATE_NONE);

    /* POINTER validation */
    /* make sure a j2k codec is present */
    l_is_valid &= (jp2->j2k != 00);

    /* make sure a procedure list is present */
    l_is_valid &= (jp2->m_procedure_list != 00);

    /* make sure a validation list is present */
    l_is_valid &= (jp2->m_validation_list != 00);

    /* PARAMETER VALIDATION */
    /* number of components */
    l_is_valid &= (jp2->numcl > 0);
    /* width */
    l_is_valid &= (jp2->h > 0);
    /* height */
    l_is_valid &= (jp2->w > 0);
    /* precision */
    for (i = 0; i < jp2->numcomps; ++i) {
        l_is_valid &= ((jp2->comps[i].bpcc & 0x7FU) <
                       38U); /* 0 is valid, ignore sign for check */
    }

    /* METH */
    l_is_valid &= ((jp2->meth > 0) && (jp2->meth < 3));

    /* stream validation */
    /* back and forth is needed */
    l_is_valid &= opj_stream_has_seek(cio);

    return l_is_valid;
}

static OPJ_BOOL opj_jp2_read_header_procedure(opj_jp2_t *jp2,
        opj_stream_private_t *stream,
        opj_event_mgr_t * p_manager
                                             )
{
    opj_jp2_box_t box;
    OPJ_UINT32 l_nb_bytes_read;
    const opj_jp2_header_handler_t * l_current_handler;
    const opj_jp2_header_handler_t * l_current_handler_misplaced;
    OPJ_UINT32 l_last_data_size = OPJ_BOX_SIZE;
    OPJ_UINT32 l_current_data_size;
    OPJ_BYTE * l_current_data = 00;

    /* preconditions */
    assert(stream != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    l_current_data = (OPJ_BYTE*)opj_calloc(1, l_last_data_size);

    if (l_current_data == 00) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to handle jpeg2000 file header\n");
        return OPJ_FALSE;
    }

    while (opj_jp2_read_boxhdr(&box, &l_nb_bytes_read, stream, p_manager)) {
        /* is it the codestream box ? */
        if (box.type == JP2_JP2C) {
            if (jp2->jp2_state & JP2_STATE_HEADER) {
                jp2->jp2_state |= JP2_STATE_CODESTREAM;
                opj_free(l_current_data);
                return OPJ_TRUE;
            } else {
                opj_event_msg(p_manager, EVT_ERROR, "bad placed jpeg codestream\n");
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
        } else if (box.length == 0) {
            opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
            opj_free(l_current_data);
            return OPJ_FALSE;
        }
        /* testcase 1851.pdf.SIGSEGV.ce9.948 */
        else if (box.length < l_nb_bytes_read) {
            opj_event_msg(p_manager, EVT_ERROR, "invalid box size %d (%x)\n", box.length,
                          box.type);
            opj_free(l_current_data);
            return OPJ_FALSE;
        }

        l_current_handler = opj_jp2_find_handler(box.type);
        l_current_handler_misplaced = opj_jp2_img_find_handler(box.type);
        l_current_data_size = box.length - l_nb_bytes_read;

        if ((l_current_handler != 00) || (l_current_handler_misplaced != 00)) {
            if (l_current_handler == 00) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Found a misplaced '%c%c%c%c' box outside jp2h box\n",
                              (OPJ_BYTE)(box.type >> 24), (OPJ_BYTE)(box.type >> 16),
                              (OPJ_BYTE)(box.type >> 8), (OPJ_BYTE)(box.type >> 0));
                if (jp2->jp2_state & JP2_STATE_HEADER) {
                    /* read anyway, we already have jp2h */
                    l_current_handler = l_current_handler_misplaced;
                } else {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "JPEG2000 Header box not read yet, '%c%c%c%c' box will be ignored\n",
                                  (OPJ_BYTE)(box.type >> 24), (OPJ_BYTE)(box.type >> 16),
                                  (OPJ_BYTE)(box.type >> 8), (OPJ_BYTE)(box.type >> 0));
                    jp2->jp2_state |= JP2_STATE_UNKNOWN;
                    if (opj_stream_skip(stream, l_current_data_size,
                                        p_manager) != l_current_data_size) {
                        opj_event_msg(p_manager, EVT_ERROR,
                                      "Problem with skipping JPEG2000 box, stream error\n");
                        opj_free(l_current_data);
                        return OPJ_FALSE;
                    }
                    continue;
                }
            }
            if ((OPJ_OFF_T)l_current_data_size > opj_stream_get_number_byte_left(stream)) {
                /* do not even try to malloc if we can't read */
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid box size %d for box '%c%c%c%c'. Need %d bytes, %d bytes remaining \n",
                              box.length, (OPJ_BYTE)(box.type >> 24), (OPJ_BYTE)(box.type >> 16),
                              (OPJ_BYTE)(box.type >> 8), (OPJ_BYTE)(box.type >> 0), l_current_data_size,
                              (OPJ_UINT32)opj_stream_get_number_byte_left(stream));
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
            if (l_current_data_size > l_last_data_size) {
                OPJ_BYTE* new_current_data = (OPJ_BYTE*)opj_realloc(l_current_data,
                                             l_current_data_size);
                if (!new_current_data) {
                    opj_free(l_current_data);
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Not enough memory to handle jpeg2000 box\n");
                    return OPJ_FALSE;
                }
                l_current_data = new_current_data;
                l_last_data_size = l_current_data_size;
            }

            l_nb_bytes_read = (OPJ_UINT32)opj_stream_read_data(stream, l_current_data,
                              l_current_data_size, p_manager);
            if (l_nb_bytes_read != l_current_data_size) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Problem with reading JPEG2000 box, stream error\n");
                opj_free(l_current_data);
                return OPJ_FALSE;
            }

            if (! l_current_handler->handler(jp2, l_current_data, l_current_data_size,
                                             p_manager)) {
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
        } else {
            if (!(jp2->jp2_state & JP2_STATE_SIGNATURE)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Malformed JP2 file format: first box must be JPEG 2000 signature box\n");
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
            if (!(jp2->jp2_state & JP2_STATE_FILE_TYPE)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Malformed JP2 file format: second box must be file type box\n");
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
            jp2->jp2_state |= JP2_STATE_UNKNOWN;
            if (opj_stream_skip(stream, l_current_data_size,
                                p_manager) != l_current_data_size) {
                if (jp2->jp2_state & JP2_STATE_CODESTREAM) {
                    /* If we already read the codestream, do not error out */
                    /* Needed for data/input/nonregression/issue254.jp2 */
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "Problem with skipping JPEG2000 box, stream error\n");
                    opj_free(l_current_data);
                    return OPJ_TRUE;
                } else {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Problem with skipping JPEG2000 box, stream error\n");
                    opj_free(l_current_data);
                    return OPJ_FALSE;
                }
            }
        }
    }

    opj_free(l_current_data);

    return OPJ_TRUE;
}

/**
 * Executes the given procedures on the given codec.
 *
 * @param   p_procedure_list    the list of procedures to execute
 * @param   jp2                 the jpeg2000 file codec to execute the procedures on.
 * @param   stream                  the stream to execute the procedures on.
 * @param   p_manager           the user manager.
 *
 * @return  true                if all the procedures were successfully executed.
 */
static OPJ_BOOL opj_jp2_exec(opj_jp2_t * jp2,
                             opj_procedure_list_t * p_procedure_list,
                             opj_stream_private_t *stream,
                             opj_event_mgr_t * p_manager
                            )

{
    OPJ_BOOL(** l_procedure)(opj_jp2_t * jp2, opj_stream_private_t *,
                             opj_event_mgr_t *) = 00;
    OPJ_BOOL l_result = OPJ_TRUE;
    OPJ_UINT32 l_nb_proc, i;

    /* preconditions */
    assert(p_procedure_list != 00);
    assert(jp2 != 00);
    assert(stream != 00);
    assert(p_manager != 00);

    l_nb_proc = opj_procedure_list_get_nb_procedures(p_procedure_list);
    l_procedure = (OPJ_BOOL(**)(opj_jp2_t * jp2, opj_stream_private_t *,
                                opj_event_mgr_t *)) opj_procedure_list_get_first_procedure(p_procedure_list);

    for (i = 0; i < l_nb_proc; ++i) {
        l_result = l_result && (*l_procedure)(jp2, stream, p_manager);
        ++l_procedure;
    }

    /* and clear the procedure list at the end. */
    opj_procedure_list_clear(p_procedure_list);
    return l_result;
}

OPJ_BOOL opj_jp2_start_compress(opj_jp2_t *jp2,
                                opj_stream_private_t *stream,
                                opj_image_t * p_image,
                                opj_event_mgr_t * p_manager
                               )
{
    /* preconditions */
    assert(jp2 != 00);
    assert(stream != 00);
    assert(p_manager != 00);

    /* customization of the validation */
    if (! opj_jp2_setup_encoding_validation(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    /* validation of the parameters codec */
    if (! opj_jp2_exec(jp2, jp2->m_validation_list, stream, p_manager)) {
        return OPJ_FALSE;
    }

    /* customization of the encoding */
    if (! opj_jp2_setup_header_writing(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    /* write header */
    if (! opj_jp2_exec(jp2, jp2->m_procedure_list, stream, p_manager)) {
        return OPJ_FALSE;
    }

    return opj_j2k_start_compress(jp2->j2k, stream, p_image, p_manager);
}

static const opj_jp2_header_handler_t * opj_jp2_find_handler(OPJ_UINT32 p_id)
{
    OPJ_UINT32 i, l_handler_size = sizeof(jp2_header) / sizeof(
                                       opj_jp2_header_handler_t);

    for (i = 0; i < l_handler_size; ++i) {
        if (jp2_header[i].id == p_id) {
            return &jp2_header[i];
        }
    }
    return NULL;
}

/**
 * Finds the image execution function related to the given box id.
 *
 * @param   p_id    the id of the handler to fetch.
 *
 * @return  the given handler or 00 if it could not be found.
 */
static const opj_jp2_header_handler_t * opj_jp2_img_find_handler(
    OPJ_UINT32 p_id)
{
    OPJ_UINT32 i, l_handler_size = sizeof(jp2_img_header) / sizeof(
                                       opj_jp2_header_handler_t);
    for (i = 0; i < l_handler_size; ++i) {
        if (jp2_img_header[i].id == p_id) {
            return &jp2_img_header[i];
        }
    }

    return NULL;
}

/**
 * Reads a jpeg2000 file signature box.
 *
 * @param   p_header_data   the data contained in the signature box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the signature box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the file signature box is valid.
 */
static OPJ_BOOL opj_jp2_read_jp(opj_jp2_t *jp2,
                                OPJ_BYTE * p_header_data,
                                OPJ_UINT32 p_header_size,
                                opj_event_mgr_t * p_manager
                               )

{
    OPJ_UINT32 l_magic_number;

    /* preconditions */
    assert(p_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (jp2->jp2_state != JP2_STATE_NONE) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "The signature box must be the first box in the file.\n");
        return OPJ_FALSE;
    }

    /* assure length of data is correct (4 -> magic number) */
    if (p_header_size != 4) {
        opj_event_msg(p_manager, EVT_ERROR, "Error with JP signature Box size\n");
        return OPJ_FALSE;
    }

    /* rearrange data */
    opj_read_bytes(p_header_data, &l_magic_number, 4);
    if (l_magic_number != 0x0d0a870a) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with JP Signature : bad magic number\n");
        return OPJ_FALSE;
    }

    jp2->jp2_state |= JP2_STATE_SIGNATURE;

    return OPJ_TRUE;
}

/**
 * Reads a a FTYP box - File type box
 *
 * @param   p_header_data   the data contained in the FTYP box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the FTYP box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the FTYP box is valid.
 */
static OPJ_BOOL opj_jp2_read_ftyp(opj_jp2_t *jp2,
                                  OPJ_BYTE * p_header_data,
                                  OPJ_UINT32 p_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 i, l_remaining_bytes;

    /* preconditions */
    assert(p_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (jp2->jp2_state != JP2_STATE_SIGNATURE) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "The ftyp box must be the second box in the file.\n");
        return OPJ_FALSE;
    }

    /* assure length of data is correct */
    if (p_header_size < 8) {
        opj_event_msg(p_manager, EVT_ERROR, "Error with FTYP signature Box size\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &jp2->brand, 4);      /* BR */
    p_header_data += 4;

    opj_read_bytes(p_header_data, &jp2->minversion, 4);     /* MinV */
    p_header_data += 4;

    l_remaining_bytes = p_header_size - 8;

    /* the number of remaining bytes should be a multiple of 4 */
    if ((l_remaining_bytes & 0x3) != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error with FTYP signature Box size\n");
        return OPJ_FALSE;
    }

    /* div by 4 */
    jp2->numcl = l_remaining_bytes >> 2;
    if (jp2->numcl) {
        jp2->cl = (OPJ_UINT32 *) opj_calloc(jp2->numcl, sizeof(OPJ_UINT32));
        if (jp2->cl == 00) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory with FTYP Box\n");
            return OPJ_FALSE;
        }
    }

    for (i = 0; i < jp2->numcl; ++i) {
        opj_read_bytes(p_header_data, &jp2->cl[i], 4);      /* CLi */
        p_header_data += 4;
    }

    jp2->jp2_state |= JP2_STATE_FILE_TYPE;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_skip_jp2c(opj_jp2_t *jp2,
                                  opj_stream_private_t *stream,
                                  opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(stream != 00);
    assert(p_manager != 00);

    jp2->j2k_codestream_offset = opj_stream_tell(stream);

    if (opj_stream_skip(stream, 8, p_manager) != 8) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jpip_skip_iptr(opj_jp2_t *jp2,
                                   opj_stream_private_t *stream,
                                   opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(stream != 00);
    assert(p_manager != 00);

    jp2->jpip_iptr_offset = opj_stream_tell(stream);

    if (opj_stream_skip(stream, 24, p_manager) != 24) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads the Jpeg2000 file Header box - JP2 Header box (warning, this is a super box).
 *
 * @param   p_header_data   the data contained in the file header box.
 * @param   jp2             the jpeg2000 file codec.
 * @param   p_header_size   the size of the data contained in the file header box.
 * @param   p_manager       the user event manager.
 *
 * @return true if the JP2 Header box was successfully recognized.
*/
static OPJ_BOOL opj_jp2_read_jp2h(opj_jp2_t *jp2,
                                  OPJ_BYTE *p_header_data,
                                  OPJ_UINT32 p_header_size,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_box_size = 0, l_current_data_size = 0;
    opj_jp2_box_t box;
    const opj_jp2_header_handler_t * l_current_handler;
    OPJ_BOOL l_has_ihdr = 0;

    /* preconditions */
    assert(p_header_data != 00);
    assert(jp2 != 00);
    assert(p_manager != 00);

    /* make sure the box is well placed */
    if ((jp2->jp2_state & JP2_STATE_FILE_TYPE) != JP2_STATE_FILE_TYPE) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "The  box must be the first box in the file.\n");
        return OPJ_FALSE;
    }

    jp2->jp2_img_state = JP2_IMG_STATE_NONE;

    /* iterate while remaining data */
    while (p_header_size > 0) {

        if (! opj_jp2_read_boxhdr_char(&box, p_header_data, &l_box_size, p_header_size,
                                       p_manager)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Stream error while reading JP2 Header box\n");
            return OPJ_FALSE;
        }

        if (box.length > p_header_size) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Stream error while reading JP2 Header box: box length is inconsistent.\n");
            return OPJ_FALSE;
        }

        l_current_handler = opj_jp2_img_find_handler(box.type);
        l_current_data_size = box.length - l_box_size;
        p_header_data += l_box_size;

        if (l_current_handler != 00) {
            if (! l_current_handler->handler(jp2, p_header_data, l_current_data_size,
                                             p_manager)) {
                return OPJ_FALSE;
            }
        } else {
            jp2->jp2_img_state |= JP2_IMG_STATE_UNKNOWN;
        }

        if (box.type == JP2_IHDR) {
            l_has_ihdr = 1;
        }

        p_header_data += l_current_data_size;
        p_header_size -= box.length;
    }

    if (l_has_ihdr == 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Stream error while reading JP2 Header box: no 'ihdr' box.\n");
        return OPJ_FALSE;
    }

    jp2->jp2_state |= JP2_STATE_HEADER;
    jp2->has_jp2h = 1;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_read_boxhdr_char(opj_jp2_box_t *box,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_number_bytes_read,
        OPJ_UINT32 p_box_max_size,
        opj_event_mgr_t * p_manager
                                        )
{
    OPJ_UINT32 l_value;

    /* preconditions */
    assert(p_data != 00);
    assert(box != 00);
    assert(p_number_bytes_read != 00);
    assert(p_manager != 00);

    if (p_box_max_size < 8) {
        opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of less than 8 bytes\n");
        return OPJ_FALSE;
    }

    /* process read data */
    opj_read_bytes(p_data, &l_value, 4);
    p_data += 4;
    box->length = (OPJ_UINT32)(l_value);

    opj_read_bytes(p_data, &l_value, 4);
    p_data += 4;
    box->type = (OPJ_UINT32)(l_value);

    *p_number_bytes_read = 8;

    /* do we have a "special very large box ?" */
    /* read then the XLBox */
    if (box->length == 1) {
        OPJ_UINT32 l_xl_part_size;

        if (p_box_max_size < 16) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Cannot handle XL box of less than 16 bytes\n");
            return OPJ_FALSE;
        }

        opj_read_bytes(p_data, &l_xl_part_size, 4);
        p_data += 4;
        *p_number_bytes_read += 4;

        if (l_xl_part_size != 0) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Cannot handle box sizes higher than 2^32\n");
            return OPJ_FALSE;
        }

        opj_read_bytes(p_data, &l_value, 4);
        *p_number_bytes_read += 4;
        box->length = (OPJ_UINT32)(l_value);

        if (box->length == 0) {
            opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
            return OPJ_FALSE;
        }
    } else if (box->length == 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Cannot handle box of undefined sizes\n");
        return OPJ_FALSE;
    }
    if (box->length < *p_number_bytes_read) {
        opj_event_msg(p_manager, EVT_ERROR, "Box length is inconsistent.\n");
        return OPJ_FALSE;
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_jp2_read_header(opj_stream_private_t *p_stream,
                             opj_jp2_t *jp2,
                             opj_image_t ** p_image,
                             opj_event_mgr_t * p_manager
                            )
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    /* customization of the validation */
    if (! opj_jp2_setup_decoding_validation(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    /* customization of the encoding */
    if (! opj_jp2_setup_header_reading(jp2, p_manager)) {
        return OPJ_FALSE;
    }

    /* validation of the parameters codec */
    if (! opj_jp2_exec(jp2, jp2->m_validation_list, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    /* read header */
    if (! opj_jp2_exec(jp2, jp2->m_procedure_list, p_stream, p_manager)) {
        return OPJ_FALSE;
    }
    if (jp2->has_jp2h == 0) {
        opj_event_msg(p_manager, EVT_ERROR, "JP2H box missing. Required.\n");
        return OPJ_FALSE;
    }
    if (jp2->has_ihdr == 0) {
        opj_event_msg(p_manager, EVT_ERROR, "IHDR box_missing. Required.\n");
        return OPJ_FALSE;
    }

    return opj_j2k_read_header(p_stream,
                               jp2->j2k,
                               p_image,
                               p_manager);
}

static OPJ_BOOL opj_jp2_setup_encoding_validation(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(jp2->m_validation_list,
                                           (opj_procedure)opj_jp2_default_validation, p_manager)) {
        return OPJ_FALSE;
    }
    /* DEVELOPER CORNER, add your custom validation procedure */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_setup_decoding_validation(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(jp2);
    OPJ_UNUSED(p_manager);

    /* DEVELOPER CORNER, add your custom validation procedure */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_setup_header_writing(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_write_jp, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_write_ftyp, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_write_jp2h, p_manager)) {
        return OPJ_FALSE;
    }
    if (jp2->jpip_on) {
        if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                               (opj_procedure)opj_jpip_skip_iptr, p_manager)) {
            return OPJ_FALSE;
        }
    }
    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_skip_jp2c, p_manager)) {
        return OPJ_FALSE;
    }

    /* DEVELOPER CORNER, insert your custom procedures */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jp2_setup_header_reading(opj_jp2_t *jp2,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(jp2 != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(jp2->m_procedure_list,
                                           (opj_procedure)opj_jp2_read_header_procedure, p_manager)) {
        return OPJ_FALSE;
    }

    /* DEVELOPER CORNER, add your custom procedures */

    return OPJ_TRUE;
}

OPJ_BOOL opj_jp2_read_tile_header(opj_jp2_t * p_jp2,
                                  OPJ_UINT32 * p_tile_index,
                                  OPJ_UINT32 * p_data_size,
                                  OPJ_INT32 * p_tile_x0,
                                  OPJ_INT32 * p_tile_y0,
                                  OPJ_INT32 * p_tile_x1,
                                  OPJ_INT32 * p_tile_y1,
                                  OPJ_UINT32 * p_nb_comps,
                                  OPJ_BOOL * p_go_on,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    return opj_j2k_read_tile_header(p_jp2->j2k,
                                    p_tile_index,
                                    p_data_size,
                                    p_tile_x0, p_tile_y0,
                                    p_tile_x1, p_tile_y1,
                                    p_nb_comps,
                                    p_go_on,
                                    p_stream,
                                    p_manager);
}

OPJ_BOOL opj_jp2_write_tile(opj_jp2_t *p_jp2,
                            OPJ_UINT32 p_tile_index,
                            OPJ_BYTE * p_data,
                            OPJ_UINT32 p_data_size,
                            opj_stream_private_t *p_stream,
                            opj_event_mgr_t * p_manager
                           )

{
    return opj_j2k_write_tile(p_jp2->j2k, p_tile_index, p_data, p_data_size,
                              p_stream, p_manager);
}

OPJ_BOOL opj_jp2_decode_tile(opj_jp2_t * p_jp2,
                             OPJ_UINT32 p_tile_index,
                             OPJ_BYTE * p_data,
                             OPJ_UINT32 p_data_size,
                             opj_stream_private_t *p_stream,
                             opj_event_mgr_t * p_manager
                            )
{
    return opj_j2k_decode_tile(p_jp2->j2k, p_tile_index, p_data, p_data_size,
                               p_stream, p_manager);
}

void opj_jp2_destroy(opj_jp2_t *jp2)
{
    if (jp2) {
        /* destroy the J2K codec */
        opj_j2k_destroy(jp2->j2k);
        jp2->j2k = 00;

        if (jp2->comps) {
            opj_free(jp2->comps);
            jp2->comps = 00;
        }

        if (jp2->cl) {
            opj_free(jp2->cl);
            jp2->cl = 00;
        }

        if (jp2->color.icc_profile_buf) {
            opj_free(jp2->color.icc_profile_buf);
            jp2->color.icc_profile_buf = 00;
        }

        if (jp2->color.jp2_cdef) {
            if (jp2->color.jp2_cdef->info) {
                opj_free(jp2->color.jp2_cdef->info);
                jp2->color.jp2_cdef->info = NULL;
            }

            opj_free(jp2->color.jp2_cdef);
            jp2->color.jp2_cdef = 00;
        }

        if (jp2->color.jp2_pclr) {
            if (jp2->color.jp2_pclr->cmap) {
                opj_free(jp2->color.jp2_pclr->cmap);
                jp2->color.jp2_pclr->cmap = NULL;
            }
            if (jp2->color.jp2_pclr->channel_sign) {
                opj_free(jp2->color.jp2_pclr->channel_sign);
                jp2->color.jp2_pclr->channel_sign = NULL;
            }
            if (jp2->color.jp2_pclr->channel_size) {
                opj_free(jp2->color.jp2_pclr->channel_size);
                jp2->color.jp2_pclr->channel_size = NULL;
            }
            if (jp2->color.jp2_pclr->entries) {
                opj_free(jp2->color.jp2_pclr->entries);
                jp2->color.jp2_pclr->entries = NULL;
            }

            opj_free(jp2->color.jp2_pclr);
            jp2->color.jp2_pclr = 00;
        }

        if (jp2->m_validation_list) {
            opj_procedure_list_destroy(jp2->m_validation_list);
            jp2->m_validation_list = 00;
        }

        if (jp2->m_procedure_list) {
            opj_procedure_list_destroy(jp2->m_procedure_list);
            jp2->m_procedure_list = 00;
        }

        opj_free(jp2);
    }
}

OPJ_BOOL opj_jp2_set_decoded_components(opj_jp2_t *p_jp2,
                                        OPJ_UINT32 numcomps,
                                        const OPJ_UINT32* comps_indices,
                                        opj_event_mgr_t * p_manager)
{
    return opj_j2k_set_decoded_components(p_jp2->j2k,
                                          numcomps, comps_indices,
                                          p_manager);
}

OPJ_BOOL opj_jp2_set_decode_area(opj_jp2_t *p_jp2,
                                 opj_image_t* p_image,
                                 OPJ_INT32 p_start_x, OPJ_INT32 p_start_y,
                                 OPJ_INT32 p_end_x, OPJ_INT32 p_end_y,
                                 opj_event_mgr_t * p_manager
                                )
{
    return opj_j2k_set_decode_area(p_jp2->j2k, p_image, p_start_x, p_start_y,
                                   p_end_x, p_end_y, p_manager);
}

OPJ_BOOL opj_jp2_get_tile(opj_jp2_t *p_jp2,
                          opj_stream_private_t *p_stream,
                          opj_image_t* p_image,
                          opj_event_mgr_t * p_manager,
                          OPJ_UINT32 tile_index
                         )
{
    if (!p_image) {
        return OPJ_FALSE;
    }

    opj_event_msg(p_manager, EVT_WARNING,
                  "JP2 box which are after the codestream will not be read by this function.\n");

    if (! opj_j2k_get_tile(p_jp2->j2k, p_stream, p_image, p_manager, tile_index)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Failed to decode the codestream in the JP2 file\n");
        return OPJ_FALSE;
    }

    if (p_jp2->j2k->m_specific_param.m_decoder.m_numcomps_to_decode) {
        /* Bypass all JP2 component transforms */
        return OPJ_TRUE;
    }

    if (!opj_jp2_check_color(p_image, &(p_jp2->color), p_manager)) {
        return OPJ_FALSE;
    }

    /* Set Image Color Space */
    if (p_jp2->enumcs == 16) {
        p_image->color_space = OPJ_CLRSPC_SRGB;
    } else if (p_jp2->enumcs == 17) {
        p_image->color_space = OPJ_CLRSPC_GRAY;
    } else if (p_jp2->enumcs == 18) {
        p_image->color_space = OPJ_CLRSPC_SYCC;
    } else if (p_jp2->enumcs == 24) {
        p_image->color_space = OPJ_CLRSPC_EYCC;
    } else if (p_jp2->enumcs == 12) {
        p_image->color_space = OPJ_CLRSPC_CMYK;
    } else {
        p_image->color_space = OPJ_CLRSPC_UNKNOWN;
    }

    if (p_jp2->color.jp2_pclr) {
        /* Part 1, I.5.3.4: Either both or none : */
        if (!p_jp2->color.jp2_pclr->cmap) {
            opj_jp2_free_pclr(&(p_jp2->color));
        } else {
            if (!opj_jp2_apply_pclr(p_image, &(p_jp2->color), p_manager)) {
                return OPJ_FALSE;
            }
        }
    }

    /* Apply the color space if needed */
    if (p_jp2->color.jp2_cdef) {
        opj_jp2_apply_cdef(p_image, &(p_jp2->color), p_manager);
    }

    if (p_jp2->color.icc_profile_buf) {
        p_image->icc_profile_buf = p_jp2->color.icc_profile_buf;
        p_image->icc_profile_len = p_jp2->color.icc_profile_len;
        p_jp2->color.icc_profile_buf = NULL;
    }

    return OPJ_TRUE;
}

/* ----------------------------------------------------------------------- */
/* JP2 encoder interface                                             */
/* ----------------------------------------------------------------------- */

opj_jp2_t* opj_jp2_create(OPJ_BOOL p_is_decoder)
{
    opj_jp2_t *jp2 = (opj_jp2_t*)opj_calloc(1, sizeof(opj_jp2_t));
    if (jp2) {

        /* create the J2K codec */
        if (! p_is_decoder) {
            jp2->j2k = opj_j2k_create_compress();
        } else {
            jp2->j2k = opj_j2k_create_decompress();
        }

        if (jp2->j2k == 00) {
            opj_jp2_destroy(jp2);
            return 00;
        }

        /* Color structure */
        jp2->color.icc_profile_buf = NULL;
        jp2->color.icc_profile_len = 0;
        jp2->color.jp2_cdef = NULL;
        jp2->color.jp2_pclr = NULL;
        jp2->color.jp2_has_colr = 0;

        /* validation list creation */
        jp2->m_validation_list = opj_procedure_list_create();
        if (! jp2->m_validation_list) {
            opj_jp2_destroy(jp2);
            return 00;
        }

        /* execution list creation */
        jp2->m_procedure_list = opj_procedure_list_create();
        if (! jp2->m_procedure_list) {
            opj_jp2_destroy(jp2);
            return 00;
        }
    }

    return jp2;
}

void jp2_dump(opj_jp2_t* p_jp2, OPJ_INT32 flag, FILE* out_stream)
{
    /* preconditions */
    assert(p_jp2 != 00);

    j2k_dump(p_jp2->j2k,
             flag,
             out_stream);
}

opj_codestream_index_t* jp2_get_cstr_index(opj_jp2_t* p_jp2)
{
    return j2k_get_cstr_index(p_jp2->j2k);
}

opj_codestream_info_v2_t* jp2_get_cstr_info(opj_jp2_t* p_jp2)
{
    return j2k_get_cstr_info(p_jp2->j2k);
}

OPJ_BOOL opj_jp2_set_decoded_resolution_factor(opj_jp2_t *p_jp2,
        OPJ_UINT32 res_factor,
        opj_event_mgr_t * p_manager)
{
    return opj_j2k_set_decoded_resolution_factor(p_jp2->j2k, res_factor, p_manager);
}

/* JPIP specific */

#ifdef USE_JPIP
static OPJ_BOOL opj_jpip_write_iptr(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager)
{
    OPJ_OFF_T j2k_codestream_exit;
    OPJ_BYTE l_data_header [24];

    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);
    assert(opj_stream_has_seek(cio));

    j2k_codestream_exit = opj_stream_tell(cio);
    opj_write_bytes(l_data_header, 24, 4); /* size of iptr */
    opj_write_bytes(l_data_header + 4, JPIP_IPTR,
                    4);                                      /* IPTR */
#if 0
    opj_write_bytes(l_data_header + 4 + 4, 0, 8); /* offset */
    opj_write_bytes(l_data_header + 8 + 8, 0, 8); /* length */
#else
    opj_write_double(l_data_header + 4 + 4, 0); /* offset */
    opj_write_double(l_data_header + 8 + 8, 0); /* length */
#endif

    if (! opj_stream_seek(cio, jp2->jpip_iptr_offset, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(cio, l_data_header, 24, p_manager) != 24) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    if (! opj_stream_seek(cio, j2k_codestream_exit, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jpip_write_fidx(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager)
{
    OPJ_OFF_T j2k_codestream_exit;
    OPJ_BYTE l_data_header [24];

    OPJ_UNUSED(jp2);

    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);
    assert(opj_stream_has_seek(cio));

    opj_write_bytes(l_data_header, 24, 4); /* size of iptr */
    opj_write_bytes(l_data_header + 4, JPIP_FIDX,
                    4);                                      /* IPTR */
    opj_write_double(l_data_header + 4 + 4, 0); /* offset */
    opj_write_double(l_data_header + 8 + 8, 0); /* length */

    if (opj_stream_write_data(cio, l_data_header, 24, p_manager) != 24) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    j2k_codestream_exit = opj_stream_tell(cio);
    if (! opj_stream_seek(cio, j2k_codestream_exit, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_jpip_write_cidx(opj_jp2_t *jp2,
                                    opj_stream_private_t *cio,
                                    opj_event_mgr_t * p_manager)
{
    OPJ_OFF_T j2k_codestream_exit;
    OPJ_BYTE l_data_header [24];

    OPJ_UNUSED(jp2);

    /* preconditions */
    assert(jp2 != 00);
    assert(cio != 00);
    assert(p_manager != 00);
    assert(opj_stream_has_seek(cio));

    j2k_codestream_exit = opj_stream_tell(cio);
    opj_write_bytes(l_data_header, 24, 4); /* size of iptr */
    opj_write_bytes(l_data_header + 4, JPIP_CIDX,
                    4);                                      /* IPTR */
#if 0
    opj_write_bytes(l_data_header + 4 + 4, 0, 8); /* offset */
    opj_write_bytes(l_data_header + 8 + 8, 0, 8); /* length */
#else
    opj_write_double(l_data_header + 4 + 4, 0); /* offset */
    opj_write_double(l_data_header + 8 + 8, 0); /* length */
#endif

    if (! opj_stream_seek(cio, j2k_codestream_exit, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(cio, l_data_header, 24, p_manager) != 24) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    j2k_codestream_exit = opj_stream_tell(cio);
    if (! opj_stream_seek(cio, j2k_codestream_exit, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to seek in the stream.\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

#if 0
static void write_prxy(int offset_jp2c, int length_jp2c, int offset_idx,
                       int length_idx, opj_stream_private_t *cio,
                       opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [8];
    OPJ_OFF_T len, lenp;

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4, p_manager);         /* L [at the end] */
    opj_write_bytes(l_data_header, JPIP_PRXY, 4); /* IPTR           */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);

    opj_write_bytes(l_data_header, offset_jp2c, 8);  /* OOFF           */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);
    opj_write_bytes(l_data_header, length_jp2c, 4);  /* OBH part 1     */
    opj_write_bytes(l_data_header + 4, JP2_JP2C, 4); /* OBH part 2     */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);

    opj_write_bytes(l_data_header, 1, 1); /* NI             */
    opj_stream_write_data(cio, l_data_header, 1, p_manager);

    opj_write_bytes(l_data_header, offset_idx, 8);   /* IOFF           */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);
    opj_write_bytes(l_data_header, length_idx, 4);   /* IBH part 1     */
    opj_write_bytes(l_data_header + 4, JPIP_CIDX, 4);  /* IBH part 2     */
    opj_stream_write_data(cio, l_data_header, 8, p_manager);

    len = opj_stream_tell(cio) - lenp;
    opj_stream_skip(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4); /* L              */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);
}
#endif


#if 0
static int write_fidx(int offset_jp2c, int length_jp2c, int offset_idx,
                      int length_idx, opj_stream_private_t *cio,
                      opj_event_mgr_t * p_manager)
{
    OPJ_BYTE l_data_header [4];
    OPJ_OFF_T len, lenp;

    lenp = opj_stream_tell(cio);
    opj_stream_skip(cio, 4, p_manager);
    opj_write_bytes(l_data_header, JPIP_FIDX, 4); /* FIDX */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);

    write_prxy(offset_jp2c, length_jp2c, offset_idx, length_idx, cio, p_manager);

    len = opj_stream_tell(cio) - lenp;
    opj_stream_skip(cio, lenp, p_manager);
    opj_write_bytes(l_data_header, len, 4); /* L              */
    opj_stream_write_data(cio, l_data_header, 4, p_manager);
    opj_stream_seek(cio, lenp + len, p_manager);

    return len;
}
#endif
#endif /* USE_JPIP */
