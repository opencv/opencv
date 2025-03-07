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
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
 * Copyright (c) 2006-2007, Parvatha Elangovan
 * Copyright (c) 2010-2011, Kaori Hagihara
 * Copyright (c) 2011-2012, Centre National d'Etudes Spatiales (CNES), France
 * Copyright (c) 2012, CS Systemes d'Information, France
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
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

/** @defgroup J2K J2K - JPEG-2000 codestream reader/writer */
/*@{*/

/** @name Local static functions */
/*@{*/

/**
 * Sets up the procedures to do on reading header. Developers wanting to extend the library can add their own reading procedures.
 */
static OPJ_BOOL opj_j2k_setup_header_reading(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager);

/**
 * The read header procedure.
 */
static OPJ_BOOL opj_j2k_read_header_procedure(opj_j2k_t *p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * The default encoding validation procedure without any extension.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
static OPJ_BOOL opj_j2k_encoding_validation(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * The default decoding validation procedure without any extension.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
static OPJ_BOOL opj_j2k_decoding_validation(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
static OPJ_BOOL opj_j2k_setup_encoding_validation(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
static OPJ_BOOL opj_j2k_setup_decoding_validation(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager);

/**
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
static OPJ_BOOL opj_j2k_setup_end_compress(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager);

/**
 * The mct encoding validation procedure.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
static OPJ_BOOL opj_j2k_mct_validation(opj_j2k_t * p_j2k,
                                       opj_stream_private_t *p_stream,
                                       opj_event_mgr_t * p_manager);

/**
 * Builds the tcd decoder to use to decode tile.
 */
static OPJ_BOOL opj_j2k_build_decoder(opj_j2k_t * p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager);
/**
 * Builds the tcd encoder to use to encode tile.
 */
static OPJ_BOOL opj_j2k_build_encoder(opj_j2k_t * p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager);

/**
 * Creates a tile-coder encoder.
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager                   the user event manager.
*/
static OPJ_BOOL opj_j2k_create_tcd(opj_j2k_t *p_j2k,
                                   opj_stream_private_t *p_stream,
                                   opj_event_mgr_t * p_manager);

/**
 * Executes the given procedures on the given codec.
 *
 * @param       p_procedure_list        the list of procedures to execute
 * @param       p_j2k                           the jpeg2000 codec to execute the procedures on.
 * @param       p_stream                        the stream to execute the procedures on.
 * @param       p_manager                       the user manager.
 *
 * @return      true                            if all the procedures were successfully executed.
 */
static OPJ_BOOL opj_j2k_exec(opj_j2k_t * p_j2k,
                             opj_procedure_list_t * p_procedure_list,
                             opj_stream_private_t *p_stream,
                             opj_event_mgr_t * p_manager);

/**
 * Updates the rates of the tcp.
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_update_rates(opj_j2k_t *p_j2k,
                                     opj_stream_private_t *p_stream,
                                     opj_event_mgr_t * p_manager);

/**
 * Copies the decoding tile parameters onto all the tile parameters.
 * Creates also the tile decoder.
 */
static OPJ_BOOL opj_j2k_copy_default_tcp_and_create_tcd(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Destroys the memory associated with the decoding of headers.
 */
static OPJ_BOOL opj_j2k_destroy_header_memory(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Reads the lookup table containing all the marker, status and action, and returns the handler associated
 * with the marker value.
 * @param       p_id            Marker value to look up
 *
 * @return      the handler associated with the id.
*/
static const struct opj_dec_memory_marker_handler * opj_j2k_get_marker_handler(
    OPJ_UINT32 p_id);

/**
 * Destroys a tile coding parameter structure.
 *
 * @param       p_tcp           the tile coding parameter to destroy.
 */
static void opj_j2k_tcp_destroy(opj_tcp_t *p_tcp);

/**
 * Destroys the data inside a tile coding parameter structure.
 *
 * @param       p_tcp           the tile coding parameter which contain data to destroy.
 */
static void opj_j2k_tcp_data_destroy(opj_tcp_t *p_tcp);

/**
 * Destroys a coding parameter structure.
 *
 * @param       p_cp            the coding parameter to destroy.
 */
static void opj_j2k_cp_destroy(opj_cp_t *p_cp);

/**
 * Compare 2 a SPCod/ SPCoc elements, i.e. the coding style of a given component of a tile.
 *
 * @param       p_j2k            J2K codec.
 * @param       p_tile_no        Tile number
 * @param       p_first_comp_no  The 1st component number to compare.
 * @param       p_second_comp_no The 1st component number to compare.
 *
 * @return OPJ_TRUE if SPCdod are equals.
 */
static OPJ_BOOL opj_j2k_compare_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no, OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no);

/**
 * Writes a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 *
 * @param       p_j2k           J2K codec.
 * @param       p_tile_no       FIXME DOC
 * @param       p_comp_no       the component number to output.
 * @param       p_data          FIXME DOC
 * @param       p_header_size   FIXME DOC
 * @param       p_manager       the user event manager.
 *
 * @return FIXME DOC
*/
static OPJ_BOOL opj_j2k_write_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_header_size,
        opj_event_mgr_t * p_manager);

/**
 * Gets the size taken by writing a SPCod or SPCoc for the given tile and component.
 *
 * @param       p_j2k                   the J2K codec.
 * @param       p_tile_no               the tile index.
 * @param       p_comp_no               the component being outputted.
 *
 * @return      the number of bytes taken by the SPCod element.
 */
static OPJ_UINT32 opj_j2k_get_SPCod_SPCoc_size(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no);

/**
 * Reads a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 * @param       p_j2k           the jpeg2000 codec.
 * @param       compno          FIXME DOC
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 compno,
        OPJ_BYTE * p_header_data,
        OPJ_UINT32 * p_header_size,
        opj_event_mgr_t * p_manager);

/**
 * Gets the size taken by writing SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_tile_no               the tile index.
 * @param       p_comp_no               the component being outputted.
 * @param       p_j2k                   the J2K codec.
 *
 * @return      the number of bytes taken by the SPCod element.
 */
static OPJ_UINT32 opj_j2k_get_SQcd_SQcc_size(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no);

/**
 * Compares 2 SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_tile_no               the tile to output.
 * @param       p_first_comp_no         the first component number to compare.
 * @param       p_second_comp_no        the second component number to compare.
 *
 * @return OPJ_TRUE if equals.
 */
static OPJ_BOOL opj_j2k_compare_SQcd_SQcc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no, OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no);


/**
 * Writes a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_tile_no               the tile to output.
 * @param       p_comp_no               the component number to output.
 * @param       p_data                  the data buffer.
 * @param       p_header_size   pointer to the size of the data buffer, it is changed by the function.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
 *
*/
static OPJ_BOOL opj_j2k_write_SQcd_SQcc(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_tile_no,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_header_size,
                                        opj_event_mgr_t * p_manager);

/**
 * Updates the Tile Length Marker.
 */
static void opj_j2k_update_tlm(opj_j2k_t * p_j2k, OPJ_UINT32 p_tile_part_size);

/**
 * Reads a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_j2k           J2K codec.
 * @param       compno          the component number to output.
 * @param       p_header_data   the data buffer.
 * @param       p_header_size   pointer to the size of the data buffer, it is changed by the function.
 * @param       p_manager       the user event manager.
 *
*/
static OPJ_BOOL opj_j2k_read_SQcd_SQcc(opj_j2k_t *p_j2k,
                                       OPJ_UINT32 compno,
                                       OPJ_BYTE * p_header_data,
                                       OPJ_UINT32 * p_header_size,
                                       opj_event_mgr_t * p_manager);

/**
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param               p_j2k           the J2k codec.
 */
static void opj_j2k_copy_tile_component_parameters(opj_j2k_t *p_j2k);

/**
 * Copies the tile quantization parameters of all the component from the first tile component.
 *
 * @param               p_j2k           the J2k codec.
 */
static void opj_j2k_copy_tile_quantization_parameters(opj_j2k_t *p_j2k);

/**
 * Reads the tiles.
 */
static OPJ_BOOL opj_j2k_decode_tiles(opj_j2k_t *p_j2k,
                                     opj_stream_private_t *p_stream,
                                     opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_j2k_pre_write_tile(opj_j2k_t * p_j2k,
                                       OPJ_UINT32 p_tile_index,
                                       opj_stream_private_t *p_stream,
                                       opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_j2k_update_image_data(opj_tcd_t * p_tcd,
        opj_image_t* p_output_image);

static void opj_get_tile_dimensions(opj_image_t * l_image,
                                    opj_tcd_tilecomp_t * l_tilec,
                                    opj_image_comp_t * l_img_comp,
                                    OPJ_UINT32* l_size_comp,
                                    OPJ_UINT32* l_width,
                                    OPJ_UINT32* l_height,
                                    OPJ_UINT32* l_offset_x,
                                    OPJ_UINT32* l_offset_y,
                                    OPJ_UINT32* l_image_width,
                                    OPJ_UINT32* l_stride,
                                    OPJ_UINT32* l_tile_offset);

static void opj_j2k_get_tile_data(opj_tcd_t * p_tcd, OPJ_BYTE * p_data);

static OPJ_BOOL opj_j2k_post_write_tile(opj_j2k_t * p_j2k,
                                        opj_stream_private_t *p_stream,
                                        opj_event_mgr_t * p_manager);

/**
 * Sets up the procedures to do on writing header.
 * Developers wanting to extend the library can add their own writing procedures.
 */
static OPJ_BOOL opj_j2k_setup_header_writing(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_j2k_write_first_tile_part(opj_j2k_t *p_j2k,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_data_written,
        OPJ_UINT32 total_data_size,
        opj_stream_private_t *p_stream,
        struct opj_event_mgr * p_manager);

static OPJ_BOOL opj_j2k_write_all_tile_parts(opj_j2k_t *p_j2k,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_data_written,
        OPJ_UINT32 total_data_size,
        opj_stream_private_t *p_stream,
        struct opj_event_mgr * p_manager);

/**
 * Gets the offset of the header.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_get_end_header(opj_j2k_t *p_j2k,
                                       opj_stream_private_t *p_stream,
                                       opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_j2k_allocate_tile_element_cstr_index(opj_j2k_t *p_j2k);

/*
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 */

/**
 * Writes the SOC marker (Start Of Codestream)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_soc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a SOC marker (Start of Codestream)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_stream        XXX needs data
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_soc(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the SIZ marker (image and tile size)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_stream        the stream to write data to.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_siz(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a SIZ marker (image and tile size)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the SIZ box.
 * @param       p_header_size   the size of the data contained in the SIZ marker.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_siz(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the COM marker (comment)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_com(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a COM marker (comments)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_com(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);
/**
 * Writes the COD marker (Coding style default)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_cod(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a COD marker (Coding style defaults)
 * @param       p_header_data   the data contained in the COD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cod(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Compares 2 COC markers (Coding style component)
 *
 * @param       p_j2k            J2K codec.
 * @param       p_first_comp_no  the index of the first component to compare.
 * @param       p_second_comp_no the index of the second component to compare.
 *
 * @return      OPJ_TRUE if equals
 */
static OPJ_BOOL opj_j2k_compare_coc(opj_j2k_t *p_j2k,
                                    OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no);

/**
 * Writes the COC marker (Coding style component)
 *
 * @param       p_j2k       J2K codec.
 * @param       p_comp_no   the index of the component to output.
 * @param       p_stream    the stream to write data to.
 * @param       p_manager   the user event manager.
*/
static OPJ_BOOL opj_j2k_write_coc(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_comp_no,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Writes the COC marker (Coding style component)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_comp_no               the index of the component to output.
 * @param       p_data          FIXME DOC
 * @param       p_data_written  FIXME DOC
 * @param       p_manager               the user event manager.
*/
static void opj_j2k_write_coc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager);

/**
 * Gets the maximum size taken by a coc.
 *
 * @param       p_j2k   the jpeg2000 codec to use.
 */
static OPJ_UINT32 opj_j2k_get_max_coc_size(opj_j2k_t *p_j2k);

/**
 * Reads a COC marker (Coding Style Component)
 * @param       p_header_data   the data contained in the COC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_coc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the QCD marker (quantization default)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_stream                the stream to write data to.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_qcd(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a QCD marker (Quantization defaults)
 * @param       p_header_data   the data contained in the QCD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_qcd(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Compare QCC markers (quantization component)
 *
 * @param       p_j2k                 J2K codec.
 * @param       p_first_comp_no       the index of the first component to compare.
 * @param       p_second_comp_no      the index of the second component to compare.
 *
 * @return OPJ_TRUE if equals.
 */
static OPJ_BOOL opj_j2k_compare_qcc(opj_j2k_t *p_j2k,
                                    OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no);

/**
 * Writes the QCC marker (quantization component)
 *
 * @param       p_comp_no       the index of the component to output.
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_qcc(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_comp_no,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Writes the QCC marker (quantization component)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_comp_no       the index of the component to output.
 * @param       p_data          FIXME DOC
 * @param       p_data_written  the stream to write data to.
 * @param       p_manager       the user event manager.
*/
static void opj_j2k_write_qcc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager);

/**
 * Gets the maximum size taken by a qcc.
 */
static OPJ_UINT32 opj_j2k_get_max_qcc_size(opj_j2k_t *p_j2k);

/**
 * Reads a QCC marker (Quantization component)
 * @param       p_header_data   the data contained in the QCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_qcc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);
/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_poc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);
/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param       p_j2k          J2K codec.
 * @param       p_data         FIXME DOC
 * @param       p_data_written the stream to write data to.
 * @param       p_manager      the user event manager.
 */
static void opj_j2k_write_poc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager);
/**
 * Gets the maximum size taken by the writing of a POC.
 */
static OPJ_UINT32 opj_j2k_get_max_poc_size(opj_j2k_t *p_j2k);

/**
 * Reads a POC marker (Progression Order Change)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_poc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Gets the maximum size taken by the toc headers of all the tile parts of any given tile.
 */
static OPJ_UINT32 opj_j2k_get_max_toc_size(opj_j2k_t *p_j2k);

/**
 * Gets the maximum size taken by the headers of the SOT.
 *
 * @param       p_j2k   the jpeg2000 codec to use.
 */
static OPJ_UINT32 opj_j2k_get_specific_header_sizes(opj_j2k_t *p_j2k);

/**
 * Reads a CRG marker (Component registration)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_crg(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);
/**
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_tlm(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the updated tlm.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_updated_tlm(opj_j2k_t *p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_plm(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);
/**
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param       p_header_data   the data contained in the PLT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PLT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_plt(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Reads a PPM marker (Packed headers, main header)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
 */

static OPJ_BOOL opj_j2k_read_ppm(
    opj_j2k_t *p_j2k,
    OPJ_BYTE * p_header_data,
    OPJ_UINT32 p_header_size,
    opj_event_mgr_t * p_manager);

/**
 * Merges all PPM markers read (Packed headers, main header)
 *
 * @param       p_cp      main coding parameters.
 * @param       p_manager the user event manager.
 */
static OPJ_BOOL opj_j2k_merge_ppm(opj_cp_t *p_cp, opj_event_mgr_t * p_manager);

/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param       p_header_data   the data contained in the PPT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_ppt(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Merges all PPT markers read (Packed headers, tile-part header)
 *
 * @param       p_tcp   the tile.
 * @param       p_manager               the user event manager.
 */
static OPJ_BOOL opj_j2k_merge_ppt(opj_tcp_t *p_tcp,
                                  opj_event_mgr_t * p_manager);


/**
 * Writes the TLM marker (Tile Length Marker)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_tlm(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Writes the SOT marker (Start of tile-part)
 *
 * @param       p_j2k            J2K codec.
 * @param       p_data           Output buffer
 * @param       total_data_size  Output buffer size
 * @param       p_data_written   Number of bytes written into stream
 * @param       p_stream         the stream to write data to.
 * @param       p_manager        the user event manager.
*/
static OPJ_BOOL opj_j2k_write_sot(opj_j2k_t *p_j2k,
                                  OPJ_BYTE * p_data,
                                  OPJ_UINT32 total_data_size,
                                  OPJ_UINT32 * p_data_written,
                                  const opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads values from a SOT marker (Start of tile-part)
 *
 * the j2k decoder state is not affected. No side effects, no checks except for p_header_size.
 *
 * @param       p_header_data   the data contained in the SOT marker.
 * @param       p_header_size   the size of the data contained in the SOT marker.
 * @param       p_tile_no       Isot.
 * @param       p_tot_len       Psot.
 * @param       p_current_part  TPsot.
 * @param       p_num_parts     TNsot.
 * @param       p_manager       the user event manager.
 */
static OPJ_BOOL opj_j2k_get_sot_values(OPJ_BYTE *  p_header_data,
                                       OPJ_UINT32  p_header_size,
                                       OPJ_UINT32* p_tile_no,
                                       OPJ_UINT32* p_tot_len,
                                       OPJ_UINT32* p_current_part,
                                       OPJ_UINT32* p_num_parts,
                                       opj_event_mgr_t * p_manager);
/**
 * Reads a SOT marker (Start of tile-part)
 *
 * @param       p_header_data   the data contained in the SOT marker.
 * @param       p_j2k           the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_sot(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);
/**
 * Writes the SOD marker (Start of data)
 *
 * This also writes optional PLT markers (before SOD)
 *
 * @param       p_j2k               J2K codec.
 * @param       p_tile_coder        FIXME DOC
 * @param       p_data              FIXME DOC
 * @param       p_data_written      FIXME DOC
 * @param       total_data_size   FIXME DOC
 * @param       p_stream            the stream to write data to.
 * @param       p_manager           the user event manager.
*/
static OPJ_BOOL opj_j2k_write_sod(opj_j2k_t *p_j2k,
                                  opj_tcd_t * p_tile_coder,
                                  OPJ_BYTE * p_data,
                                  OPJ_UINT32 * p_data_written,
                                  OPJ_UINT32 total_data_size,
                                  const opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a SOD marker (Start Of Data)
 *
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_stream                FIXME DOC
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_sod(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager);

static void opj_j2k_update_tlm(opj_j2k_t * p_j2k, OPJ_UINT32 p_tile_part_size)
{
    if (p_j2k->m_specific_param.m_encoder.m_Ttlmi_is_byte) {
        opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current,
                        p_j2k->m_current_tile_number, 1);
        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current += 1;
    } else {
        opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current,
                        p_j2k->m_current_tile_number, 2);
        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current += 2;
    }

    opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current,
                    p_tile_part_size, 4);                                       /* PSOT */
    p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current += 4;
}

/**
 * Writes the RGN marker (Region Of Interest)
 *
 * @param       p_tile_no               the tile to output
 * @param       p_comp_no               the component to output
 * @param       nb_comps                the number of components
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_rgn(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_tile_no,
                                  OPJ_UINT32 p_comp_no,
                                  OPJ_UINT32 nb_comps,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a RGN marker (Region Of Interest)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_rgn(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the EOC marker (End of Codestream)
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_eoc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

#if 0
/**
 * Reads a EOC marker (End Of Codestream)
 *
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_stream                FIXME DOC
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_eoc(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager);
#endif

/**
 * Writes the CBD-MCT-MCC-MCO markers (Multi components transform)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_mct_data_group(opj_j2k_t *p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Inits the Info
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_init_info(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
Add main header marker information
@param cstr_index    Codestream information structure
@param type         marker type
@param pos          byte offset of marker segment
@param len          length of marker segment
 */
static OPJ_BOOL opj_j2k_add_mhmarker(opj_codestream_index_t *cstr_index,
                                     OPJ_UINT32 type, OPJ_OFF_T pos, OPJ_UINT32 len) ;
/**
Add tile header marker information
@param tileno       tile index number
@param cstr_index   Codestream information structure
@param type         marker type
@param pos          byte offset of marker segment
@param len          length of marker segment
 */
static OPJ_BOOL opj_j2k_add_tlmarker(OPJ_UINT32 tileno,
                                     opj_codestream_index_t *cstr_index, OPJ_UINT32 type, OPJ_OFF_T pos,
                                     OPJ_UINT32 len);

/**
 * Reads an unknown marker
 *
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_stream                the stream object to read from.
 * @param       output_marker           FIXME DOC
 * @param       p_manager               the user event manager.
 *
 * @return      true                    if the marker could be deduced.
*/
static OPJ_BOOL opj_j2k_read_unk(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 OPJ_UINT32 *output_marker,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the MCT marker (Multiple Component Transform)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_mct_record    FIXME DOC
 * @param       p_stream        the stream to write data to.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_write_mct_record(opj_j2k_t *p_j2k,
        opj_mct_data_t * p_mct_record,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param       p_header_data   the data contained in the MCT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_mct(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the MCC marker (Multiple Component Collection)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_mcc_record            FIXME DOC
 * @param       p_stream                the stream to write data to.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_mcc_record(opj_j2k_t *p_j2k,
        opj_simple_mcc_decorrelation_data_t * p_mcc_record,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager);

/**
 * Reads a MCC marker (Multiple Component Collection)
 *
 * @param       p_header_data   the data contained in the MCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_mcc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Writes the MCO marker (Multiple component transformation ordering)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_mco(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param       p_header_data   the data contained in the MCO box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCO marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_mco(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

static OPJ_BOOL opj_j2k_add_mct(opj_tcp_t * p_tcp, opj_image_t * p_image,
                                OPJ_UINT32 p_index);

static void  opj_j2k_read_int16_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_int32_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_float32_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_float64_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);

static void  opj_j2k_read_int16_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_int32_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_float32_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_read_float64_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);

static void  opj_j2k_write_float_to_int16(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_write_float_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_write_float_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  opj_j2k_write_float_to_float64(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem);

/**
 * Ends the encoding, i.e. frees memory.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_end_encoding(opj_j2k_t *p_j2k,
                                     opj_stream_private_t *p_stream,
                                     opj_event_mgr_t * p_manager);

/**
 * Writes the CBD marker (Component bit depth definition)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_cbd(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Reads a CBD marker (Component bit depth definition)
 * @param       p_header_data   the data contained in the CBD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CBD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cbd(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Reads a CAP marker (extended capabilities definition). Empty implementation.
 * Found in HTJ2K files
 *
 * @param       p_header_data   the data contained in the CAP box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CAP marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cap(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);

/**
 * Reads a CPF marker (corresponding profile). Empty implementation. Found in HTJ2K files
 * @param       p_header_data   the data contained in the CPF box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CPF marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cpf(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager);


/**
 * Writes COC marker for each component.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_all_coc(opj_j2k_t *p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager);

/**
 * Writes QCC marker for each component.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_all_qcc(opj_j2k_t *p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager);

/**
 * Writes regions of interests.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_regions(opj_j2k_t *p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager);

/**
 * Writes EPC ????
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_write_epc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager);

/**
 * Checks the progression order changes values. Tells of the poc given as input are valid.
 * A nice message is outputted at errors.
 *
 * @param       p_pocs                  the progression order changes.
 * @param       tileno                  the tile number of interest
 * @param       p_nb_pocs               the number of progression order changes.
 * @param       p_nb_resolutions        the number of resolutions.
 * @param       numcomps                the number of components
 * @param       numlayers               the number of layers.
 * @param       p_manager               the user event manager.
 *
 * @return      true if the pocs are valid.
 */
static OPJ_BOOL opj_j2k_check_poc_val(const opj_poc_t *p_pocs,
                                      OPJ_UINT32 tileno,
                                      OPJ_UINT32 p_nb_pocs,
                                      OPJ_UINT32 p_nb_resolutions,
                                      OPJ_UINT32 numcomps,
                                      OPJ_UINT32 numlayers,
                                      opj_event_mgr_t * p_manager);

/**
 * Gets the number of tile parts used for the given change of progression (if any) and the given tile.
 *
 * @param               cp                      the coding parameters.
 * @param               pino            the offset of the given poc (i.e. its position in the coding parameter).
 * @param               tileno          the given tile.
 *
 * @return              the number of tile parts.
 */
static OPJ_UINT32 opj_j2k_get_num_tp(opj_cp_t *cp, OPJ_UINT32 pino,
                                     OPJ_UINT32 tileno);

/**
 * Calculates the total number of tile parts needed by the encoder to
 * encode such an image. If not enough memory is available, then the function return false.
 *
 * @param       p_nb_tiles      pointer that will hold the number of tile parts.
 * @param       cp                      the coding parameters for the image.
 * @param       image           the image to encode.
 * @param       p_j2k                   the p_j2k encoder.
 * @param       p_manager       the user event manager.
 *
 * @return true if the function was successful, false else.
 */
static OPJ_BOOL opj_j2k_calculate_tp(opj_j2k_t *p_j2k,
                                     opj_cp_t *cp,
                                     OPJ_UINT32 * p_nb_tiles,
                                     opj_image_t *image,
                                     opj_event_mgr_t * p_manager);

static void opj_j2k_dump_MH_info(opj_j2k_t* p_j2k, FILE* out_stream);

static void opj_j2k_dump_MH_index(opj_j2k_t* p_j2k, FILE* out_stream);

static opj_codestream_index_t* opj_j2k_create_cstr_index(void);

static OPJ_FLOAT32 opj_j2k_get_tp_stride(opj_tcp_t * p_tcp);

static OPJ_FLOAT32 opj_j2k_get_default_stride(opj_tcp_t * p_tcp);

static int opj_j2k_initialise_4K_poc(opj_poc_t *POC, int numres);

static void opj_j2k_set_cinema_parameters(opj_cparameters_t *parameters,
        opj_image_t *image, opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_j2k_is_cinema_compliant(opj_image_t *image, OPJ_UINT16 rsiz,
        opj_event_mgr_t *p_manager);

static void opj_j2k_set_imf_parameters(opj_cparameters_t *parameters,
                                       opj_image_t *image, opj_event_mgr_t *p_manager);

static OPJ_BOOL opj_j2k_is_imf_compliant(opj_cparameters_t *parameters,
        opj_image_t *image,
        opj_event_mgr_t *p_manager);

/**
 * Checks for invalid number of tile-parts in SOT marker (TPsot==TNsot). See issue 254.
 *
 * @param       p_stream            the stream to read data from.
 * @param       tile_no             tile number we're looking for.
 * @param       p_correction_needed output value. if true, non conformant codestream needs TNsot correction.
 * @param       p_manager       the user event manager.
 *
 * @return true if the function was successful, false else.
 */
static OPJ_BOOL opj_j2k_need_nb_tile_parts_correction(opj_stream_private_t
        *p_stream, OPJ_UINT32 tile_no, OPJ_BOOL* p_correction_needed,
        opj_event_mgr_t * p_manager);

/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */
typedef struct j2k_prog_order {
    OPJ_PROG_ORDER enum_prog;
    char str_prog[5];
} j2k_prog_order_t;

static const j2k_prog_order_t j2k_prog_order_list[] = {
    {OPJ_CPRL, "CPRL"},
    {OPJ_LRCP, "LRCP"},
    {OPJ_PCRL, "PCRL"},
    {OPJ_RLCP, "RLCP"},
    {OPJ_RPCL, "RPCL"},
    {(OPJ_PROG_ORDER) - 1, ""}
};

/**
 * FIXME DOC
 */
static const OPJ_UINT32 MCT_ELEMENT_SIZE [] = {
    2,
    4,
    4,
    8
};

typedef void (* opj_j2k_mct_function)(const void * p_src_data,
                                      void * p_dest_data, OPJ_UINT32 p_nb_elem);

static const opj_j2k_mct_function j2k_mct_read_functions_to_float [] = {
    opj_j2k_read_int16_to_float,
    opj_j2k_read_int32_to_float,
    opj_j2k_read_float32_to_float,
    opj_j2k_read_float64_to_float
};

static const opj_j2k_mct_function j2k_mct_read_functions_to_int32 [] = {
    opj_j2k_read_int16_to_int32,
    opj_j2k_read_int32_to_int32,
    opj_j2k_read_float32_to_int32,
    opj_j2k_read_float64_to_int32
};

static const opj_j2k_mct_function j2k_mct_write_functions_from_float [] = {
    opj_j2k_write_float_to_int16,
    opj_j2k_write_float_to_int32,
    opj_j2k_write_float_to_float,
    opj_j2k_write_float_to_float64
};

typedef struct opj_dec_memory_marker_handler {
    /** marker value */
    OPJ_UINT32 id;
    /** value of the state when the marker can appear */
    OPJ_UINT32 states;
    /** action linked to the marker */
    OPJ_BOOL(*handler)(opj_j2k_t *p_j2k,
                       OPJ_BYTE * p_header_data,
                       OPJ_UINT32 p_header_size,
                       opj_event_mgr_t * p_manager);
}
opj_dec_memory_marker_handler_t;

static const opj_dec_memory_marker_handler_t j2k_memory_marker_handler_tab [] =
{
    {J2K_MS_SOT, J2K_STATE_MH | J2K_STATE_TPHSOT, opj_j2k_read_sot},
    {J2K_MS_COD, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_cod},
    {J2K_MS_COC, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_coc},
    {J2K_MS_RGN, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_rgn},
    {J2K_MS_QCD, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_qcd},
    {J2K_MS_QCC, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_qcc},
    {J2K_MS_POC, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_poc},
    {J2K_MS_SIZ, J2K_STATE_MHSIZ, opj_j2k_read_siz},
    {J2K_MS_TLM, J2K_STATE_MH, opj_j2k_read_tlm},
    {J2K_MS_PLM, J2K_STATE_MH, opj_j2k_read_plm},
    {J2K_MS_PLT, J2K_STATE_TPH, opj_j2k_read_plt},
    {J2K_MS_PPM, J2K_STATE_MH, opj_j2k_read_ppm},
    {J2K_MS_PPT, J2K_STATE_TPH, opj_j2k_read_ppt},
    {J2K_MS_SOP, 0, 0},
    {J2K_MS_CRG, J2K_STATE_MH, opj_j2k_read_crg},
    {J2K_MS_COM, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_com},
    {J2K_MS_MCT, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_mct},
    {J2K_MS_CBD, J2K_STATE_MH, opj_j2k_read_cbd},
    {J2K_MS_CAP, J2K_STATE_MH, opj_j2k_read_cap},
    {J2K_MS_CPF, J2K_STATE_MH, opj_j2k_read_cpf},
    {J2K_MS_MCC, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_mcc},
    {J2K_MS_MCO, J2K_STATE_MH | J2K_STATE_TPH, opj_j2k_read_mco},
#ifdef USE_JPWL
#ifdef TODO_MS /* remove these functions which are not compatible with the v2 API */
    {J2K_MS_EPC, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_epc},
    {J2K_MS_EPB, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_epb},
    {J2K_MS_ESD, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_esd},
    {J2K_MS_RED, J2K_STATE_MH | J2K_STATE_TPH, j2k_read_red},
#endif
#endif /* USE_JPWL */
#ifdef USE_JPSEC
    {J2K_MS_SEC, J2K_DEC_STATE_MH, j2k_read_sec},
    {J2K_MS_INSEC, 0, j2k_read_insec}
#endif /* USE_JPSEC */
    {J2K_MS_UNK, J2K_STATE_MH | J2K_STATE_TPH, 0}/*opj_j2k_read_unk is directly used*/
};

static void  opj_j2k_read_int16_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_bytes(l_src_data, &l_temp, 2);

        l_src_data += sizeof(OPJ_INT16);

        *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
    }
}

static void  opj_j2k_read_int32_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_bytes(l_src_data, &l_temp, 4);

        l_src_data += sizeof(OPJ_INT32);

        *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
    }
}

static void  opj_j2k_read_float32_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_FLOAT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_float(l_src_data, &l_temp);

        l_src_data += sizeof(OPJ_FLOAT32);

        *(l_dest_data++) = l_temp;
    }
}

static void  opj_j2k_read_float64_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_FLOAT64 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_double(l_src_data, &l_temp);

        l_src_data += sizeof(OPJ_FLOAT64);

        *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
    }
}

static void  opj_j2k_read_int16_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_bytes(l_src_data, &l_temp, 2);

        l_src_data += sizeof(OPJ_INT16);

        *(l_dest_data++) = (OPJ_INT32) l_temp;
    }
}

static void  opj_j2k_read_int32_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_bytes(l_src_data, &l_temp, 4);

        l_src_data += sizeof(OPJ_INT32);

        *(l_dest_data++) = (OPJ_INT32) l_temp;
    }
}

static void  opj_j2k_read_float32_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_FLOAT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_float(l_src_data, &l_temp);

        l_src_data += sizeof(OPJ_FLOAT32);

        *(l_dest_data++) = (OPJ_INT32) l_temp;
    }
}

static void  opj_j2k_read_float64_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
    OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
    OPJ_UINT32 i;
    OPJ_FLOAT64 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        opj_read_double(l_src_data, &l_temp);

        l_src_data += sizeof(OPJ_FLOAT64);

        *(l_dest_data++) = (OPJ_INT32) l_temp;
    }
}

static void  opj_j2k_write_float_to_int16(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
    OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        l_temp = (OPJ_UINT32) * (l_src_data++);

        opj_write_bytes(l_dest_data, l_temp, sizeof(OPJ_INT16));

        l_dest_data += sizeof(OPJ_INT16);
    }
}

static void opj_j2k_write_float_to_int32(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
    OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
    OPJ_UINT32 i;
    OPJ_UINT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        l_temp = (OPJ_UINT32) * (l_src_data++);

        opj_write_bytes(l_dest_data, l_temp, sizeof(OPJ_INT32));

        l_dest_data += sizeof(OPJ_INT32);
    }
}

static void  opj_j2k_write_float_to_float(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
    OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
    OPJ_UINT32 i;
    OPJ_FLOAT32 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        l_temp = (OPJ_FLOAT32) * (l_src_data++);

        opj_write_float(l_dest_data, l_temp);

        l_dest_data += sizeof(OPJ_FLOAT32);
    }
}

static void  opj_j2k_write_float_to_float64(const void * p_src_data,
        void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
    OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
    OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
    OPJ_UINT32 i;
    OPJ_FLOAT64 l_temp;

    for (i = 0; i < p_nb_elem; ++i) {
        l_temp = (OPJ_FLOAT64) * (l_src_data++);

        opj_write_double(l_dest_data, l_temp);

        l_dest_data += sizeof(OPJ_FLOAT64);
    }
}

const char *opj_j2k_convert_progression_order(OPJ_PROG_ORDER prg_order)
{
    const j2k_prog_order_t *po;
    for (po = j2k_prog_order_list; po->enum_prog != -1; po++) {
        if (po->enum_prog == prg_order) {
            return po->str_prog;
        }
    }
    return po->str_prog;
}

static OPJ_BOOL opj_j2k_check_poc_val(const opj_poc_t *p_pocs,
                                      OPJ_UINT32 tileno,
                                      OPJ_UINT32 p_nb_pocs,
                                      OPJ_UINT32 p_nb_resolutions,
                                      OPJ_UINT32 p_num_comps,
                                      OPJ_UINT32 p_num_layers,
                                      opj_event_mgr_t * p_manager)
{
    OPJ_UINT32* packet_array;
    OPJ_UINT32 index, resno, compno, layno;
    OPJ_UINT32 i;
    OPJ_UINT32 step_c = 1;
    OPJ_UINT32 step_r = p_num_comps * step_c;
    OPJ_UINT32 step_l = p_nb_resolutions * step_r;
    OPJ_BOOL loss = OPJ_FALSE;

    assert(p_nb_pocs > 0);

    packet_array = (OPJ_UINT32*) opj_calloc((size_t)step_l * p_num_layers,
                                            sizeof(OPJ_UINT32));
    if (packet_array == 00) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory for checking the poc values.\n");
        return OPJ_FALSE;
    }

    /* iterate through all the pocs that match our tile of interest. */
    for (i = 0; i < p_nb_pocs; ++i) {
        const opj_poc_t *poc = &p_pocs[i];
        if (tileno + 1 == poc->tile) {
            index = step_r * poc->resno0;

            /* take each resolution for each poc */
            for (resno = poc->resno0 ;
                    resno < opj_uint_min(poc->resno1, p_nb_resolutions); ++resno) {
                OPJ_UINT32 res_index = index + poc->compno0 * step_c;

                /* take each comp of each resolution for each poc */
                for (compno = poc->compno0 ;
                        compno < opj_uint_min(poc->compno1, p_num_comps); ++compno) {
                    /* The layer index always starts at zero for every progression. */
                    const OPJ_UINT32 layno0 = 0;
                    OPJ_UINT32 comp_index = res_index + layno0 * step_l;

                    /* and finally take each layer of each res of ... */
                    for (layno = layno0; layno < opj_uint_min(poc->layno1, p_num_layers);
                            ++layno) {
                        packet_array[comp_index] = 1;
                        comp_index += step_l;
                    }

                    res_index += step_c;
                }

                index += step_r;
            }
        }
    }

    index = 0;
    for (layno = 0; layno < p_num_layers ; ++layno) {
        for (resno = 0; resno < p_nb_resolutions; ++resno) {
            for (compno = 0; compno < p_num_comps; ++compno) {
                loss |= (packet_array[index] != 1);
#ifdef DEBUG_VERBOSE
                if (packet_array[index] != 1) {
                    fprintf(stderr,
                            "Missing packet in POC: layno=%d resno=%d compno=%d\n",
                            layno, resno, compno);
                }
#endif
                index += step_c;
            }
        }
    }

    if (loss) {
        opj_event_msg(p_manager, EVT_ERROR, "Missing packets possible loss of data\n");
    }

    opj_free(packet_array);

    return !loss;
}

/* ----------------------------------------------------------------------- */

static OPJ_UINT32 opj_j2k_get_num_tp(opj_cp_t *cp, OPJ_UINT32 pino,
                                     OPJ_UINT32 tileno)
{
    const OPJ_CHAR *prog = 00;
    OPJ_INT32 i;
    OPJ_UINT32 tpnum = 1;
    opj_tcp_t *tcp = 00;
    opj_poc_t * l_current_poc = 00;

    /*  preconditions */
    assert(tileno < (cp->tw * cp->th));
    assert(pino < (cp->tcps[tileno].numpocs + 1));

    /* get the given tile coding parameter */
    tcp = &cp->tcps[tileno];
    assert(tcp != 00);

    l_current_poc = &(tcp->pocs[pino]);
    assert(l_current_poc != 0);

    /* get the progression order as a character string */
    prog = opj_j2k_convert_progression_order(tcp->prg);
    assert(strlen(prog) > 0);

    if (cp->m_specific_param.m_enc.m_tp_on == 1) {
        for (i = 0; i < 4; ++i) {
            switch (prog[i]) {
            /* component wise */
            case 'C':
                tpnum *= l_current_poc->compE;
                break;
            /* resolution wise */
            case 'R':
                tpnum *= l_current_poc->resE;
                break;
            /* precinct wise */
            case 'P':
                tpnum *= l_current_poc->prcE;
                break;
            /* layer wise */
            case 'L':
                tpnum *= l_current_poc->layE;
                break;
            }
            /* would we split here ? */
            if (cp->m_specific_param.m_enc.m_tp_flag == prog[i]) {
                cp->m_specific_param.m_enc.m_tp_pos = i;
                break;
            }
        }
    } else {
        tpnum = 1;
    }

    return tpnum;
}

static OPJ_BOOL opj_j2k_calculate_tp(opj_j2k_t *p_j2k,
                                     opj_cp_t *cp,
                                     OPJ_UINT32 * p_nb_tiles,
                                     opj_image_t *image,
                                     opj_event_mgr_t * p_manager
                                    )
{
    OPJ_UINT32 pino, tileno;
    OPJ_UINT32 l_nb_tiles;
    opj_tcp_t *tcp;

    /* preconditions */
    assert(p_nb_tiles != 00);
    assert(cp != 00);
    assert(image != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_j2k);
    OPJ_UNUSED(p_manager);

    l_nb_tiles = cp->tw * cp->th;
    * p_nb_tiles = 0;
    tcp = cp->tcps;

    /* INDEX >> */
    /* TODO mergeV2: check this part which use cstr_info */
    /*if (p_j2k->cstr_info) {
            opj_tile_info_t * l_info_tile_ptr = p_j2k->cstr_info->tile;

            for (tileno = 0; tileno < l_nb_tiles; ++tileno) {
                    OPJ_UINT32 cur_totnum_tp = 0;

                    opj_pi_update_encoding_parameters(image,cp,tileno);

                    for (pino = 0; pino <= tcp->numpocs; ++pino)
                    {
                            OPJ_UINT32 tp_num = opj_j2k_get_num_tp(cp,pino,tileno);

                            *p_nb_tiles = *p_nb_tiles + tp_num;

                            cur_totnum_tp += tp_num;
                    }

                    tcp->m_nb_tile_parts = cur_totnum_tp;

                    l_info_tile_ptr->tp = (opj_tp_info_t *) opj_malloc(cur_totnum_tp * sizeof(opj_tp_info_t));
                    if (l_info_tile_ptr->tp == 00) {
                            return OPJ_FALSE;
                    }

                    memset(l_info_tile_ptr->tp,0,cur_totnum_tp * sizeof(opj_tp_info_t));

                    l_info_tile_ptr->num_tps = cur_totnum_tp;

                    ++l_info_tile_ptr;
                    ++tcp;
            }
    }
    else */{
        for (tileno = 0; tileno < l_nb_tiles; ++tileno) {
            OPJ_UINT32 cur_totnum_tp = 0;

            opj_pi_update_encoding_parameters(image, cp, tileno);

            for (pino = 0; pino <= tcp->numpocs; ++pino) {
                OPJ_UINT32 tp_num = opj_j2k_get_num_tp(cp, pino, tileno);

                *p_nb_tiles = *p_nb_tiles + tp_num;

                cur_totnum_tp += tp_num;
            }
            tcp->m_nb_tile_parts = cur_totnum_tp;

            ++tcp;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_soc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager)
{
    /* 2 bytes will be written */
    OPJ_BYTE * l_start_stream = 00;

    /* preconditions */
    assert(p_stream != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_start_stream = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    /* write SOC identifier */
    opj_write_bytes(l_start_stream, J2K_MS_SOC, 2);

    if (opj_stream_write_data(p_stream, l_start_stream, 2, p_manager) != 2) {
        return OPJ_FALSE;
    }

    /* UniPG>> */
#ifdef USE_JPWL
    /* update markers struct */
    /*
            OPJ_BOOL res = j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOC, p_stream_tell(p_stream) - 2, 2);
    */
    assert(0 && "TODO");
#endif /* USE_JPWL */
    /* <<UniPG */

    return OPJ_TRUE;
}

/**
 * Reads a SOC marker (Start of Codestream)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_stream        FIXME DOC
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_soc(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_BYTE l_data [2];
    OPJ_UINT32 l_marker;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    if (opj_stream_read_data(p_stream, l_data, 2, p_manager) != 2) {
        return OPJ_FALSE;
    }

    opj_read_bytes(l_data, &l_marker, 2);
    if (l_marker != J2K_MS_SOC) {
        return OPJ_FALSE;
    }

    /* Next marker should be a SIZ marker in the main header */
    p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_MHSIZ;

    /* FIXME move it in a index structure included in p_j2k*/
    p_j2k->cstr_index->main_head_start = opj_stream_tell(p_stream) - 2;

    opj_event_msg(p_manager, EVT_INFO,
                  "Start to read j2k main header (%" PRId64 ").\n",
                  p_j2k->cstr_index->main_head_start);

    /* Add the marker to the codestream index*/
    if (OPJ_FALSE == opj_j2k_add_mhmarker(p_j2k->cstr_index, J2K_MS_SOC,
                                          p_j2k->cstr_index->main_head_start, 2)) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n");
        return OPJ_FALSE;
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_siz(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_size_len;
    OPJ_BYTE * l_current_ptr;
    opj_image_t * l_image = 00;
    opj_cp_t *cp = 00;
    opj_image_comp_t * l_img_comp = 00;

    /* preconditions */
    assert(p_stream != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_image = p_j2k->m_private_image;
    cp = &(p_j2k->m_cp);
    l_size_len = 40 + 3 * l_image->numcomps;
    l_img_comp = l_image->comps;

    if (l_size_len > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {

        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_size_len);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory for the SIZ marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_size_len;
    }

    l_current_ptr = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    /* write SOC identifier */
    opj_write_bytes(l_current_ptr, J2K_MS_SIZ, 2);  /* SIZ */
    l_current_ptr += 2;

    opj_write_bytes(l_current_ptr, l_size_len - 2, 2); /* L_SIZ */
    l_current_ptr += 2;

    opj_write_bytes(l_current_ptr, cp->rsiz, 2);    /* Rsiz (capabilities) */
    l_current_ptr += 2;

    opj_write_bytes(l_current_ptr, l_image->x1, 4); /* Xsiz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, l_image->y1, 4); /* Ysiz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, l_image->x0, 4); /* X0siz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, l_image->y0, 4); /* Y0siz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, cp->tdx, 4);             /* XTsiz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, cp->tdy, 4);             /* YTsiz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, cp->tx0, 4);             /* XT0siz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, cp->ty0, 4);             /* YT0siz */
    l_current_ptr += 4;

    opj_write_bytes(l_current_ptr, l_image->numcomps, 2);   /* Csiz */
    l_current_ptr += 2;

    for (i = 0; i < l_image->numcomps; ++i) {
        /* TODO here with MCT ? */
        opj_write_bytes(l_current_ptr, l_img_comp->prec - 1 + (l_img_comp->sgnd << 7),
                        1);      /* Ssiz_i */
        ++l_current_ptr;

        opj_write_bytes(l_current_ptr, l_img_comp->dx, 1);      /* XRsiz_i */
        ++l_current_ptr;

        opj_write_bytes(l_current_ptr, l_img_comp->dy, 1);      /* YRsiz_i */
        ++l_current_ptr;

        ++l_img_comp;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_size_len,
                              p_manager) != l_size_len) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a SIZ marker (image and tile size)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the SIZ box.
 * @param       p_header_size   the size of the data contained in the SIZ marker.
 * @param       p_manager       the user event manager.
*/
static OPJ_BOOL opj_j2k_read_siz(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_nb_comp;
    OPJ_UINT32 l_nb_comp_remain;
    OPJ_UINT32 l_remaining_size;
    OPJ_UINT32 l_nb_tiles;
    OPJ_UINT32 l_tmp, l_tx1, l_ty1;
    OPJ_UINT32 l_prec0, l_sgnd0;
    opj_image_t *l_image = 00;
    opj_cp_t *l_cp = 00;
    opj_image_comp_t * l_img_comp = 00;
    opj_tcp_t * l_current_tile_param = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_header_data != 00);

    l_image = p_j2k->m_private_image;
    l_cp = &(p_j2k->m_cp);

    /* minimum size == 39 - 3 (= minimum component parameter) */
    if (p_header_size < 36) {
        opj_event_msg(p_manager, EVT_ERROR, "Error with SIZ marker size\n");
        return OPJ_FALSE;
    }

    l_remaining_size = p_header_size - 36;
    l_nb_comp = l_remaining_size / 3;
    l_nb_comp_remain = l_remaining_size % 3;
    if (l_nb_comp_remain != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error with SIZ marker size\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_tmp,
                   2);                                                /* Rsiz (capabilities) */
    p_header_data += 2;
    l_cp->rsiz = (OPJ_UINT16) l_tmp;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_image->x1, 4);   /* Xsiz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_image->y1, 4);   /* Ysiz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_image->x0, 4);   /* X0siz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_image->y0, 4);   /* Y0siz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_cp->tdx,
                   4);             /* XTsiz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_cp->tdy,
                   4);             /* YTsiz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_cp->tx0,
                   4);             /* XT0siz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_cp->ty0,
                   4);             /* YT0siz */
    p_header_data += 4;
    opj_read_bytes(p_header_data, (OPJ_UINT32*) &l_tmp,
                   2);                 /* Csiz */
    p_header_data += 2;
    if (l_tmp < 16385) {
        l_image->numcomps = (OPJ_UINT16) l_tmp;
    } else {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with SIZ marker: number of component is illegal -> %d\n", l_tmp);
        return OPJ_FALSE;
    }

    if (l_image->numcomps != l_nb_comp) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with SIZ marker: number of component is not compatible with the remaining number of parameters ( %d vs %d)\n",
                      l_image->numcomps, l_nb_comp);
        return OPJ_FALSE;
    }

    /* testcase 4035.pdf.SIGSEGV.d8b.3375 */
    /* testcase issue427-null-image-size.jp2 */
    if ((l_image->x0 >= l_image->x1) || (l_image->y0 >= l_image->y1)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with SIZ marker: negative or zero image size (%" PRId64 " x %" PRId64
                      ")\n", (OPJ_INT64)l_image->x1 - l_image->x0,
                      (OPJ_INT64)l_image->y1 - l_image->y0);
        return OPJ_FALSE;
    }
    /* testcase 2539.pdf.SIGFPE.706.1712 (also 3622.pdf.SIGFPE.706.2916 and 4008.pdf.SIGFPE.706.3345 and maybe more) */
    if ((l_cp->tdx == 0U) || (l_cp->tdy == 0U)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with SIZ marker: invalid tile size (tdx: %d, tdy: %d)\n", l_cp->tdx,
                      l_cp->tdy);
        return OPJ_FALSE;
    }

    /* testcase issue427-illegal-tile-offset.jp2 */
    l_tx1 = opj_uint_adds(l_cp->tx0, l_cp->tdx); /* manage overflow */
    l_ty1 = opj_uint_adds(l_cp->ty0, l_cp->tdy); /* manage overflow */
    if ((l_cp->tx0 > l_image->x0) || (l_cp->ty0 > l_image->y0) ||
            (l_tx1 <= l_image->x0) || (l_ty1 <= l_image->y0)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error with SIZ marker: illegal tile offset\n");
        return OPJ_FALSE;
    }
    if (!p_j2k->dump_state) {
        OPJ_UINT32 siz_w, siz_h;

        siz_w = l_image->x1 - l_image->x0;
        siz_h = l_image->y1 - l_image->y0;

        if (p_j2k->ihdr_w > 0 && p_j2k->ihdr_h > 0
                && (p_j2k->ihdr_w != siz_w || p_j2k->ihdr_h != siz_h)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Error with SIZ marker: IHDR w(%u) h(%u) vs. SIZ w(%u) h(%u)\n", p_j2k->ihdr_w,
                          p_j2k->ihdr_h, siz_w, siz_h);
            return OPJ_FALSE;
        }
    }
#ifdef USE_JPWL
    if (l_cp->correct) {
        /* if JPWL is on, we check whether TX errors have damaged
          too much the SIZ parameters */
        if (!(l_image->x1 * l_image->y1)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "JPWL: bad image size (%d x %d)\n",
                          l_image->x1, l_image->y1);
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
        }

        /* FIXME check previously in the function so why keep this piece of code ? Need by the norm ?
                if (l_image->numcomps != ((len - 38) / 3)) {
                        opj_event_msg(p_manager, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                                "JPWL: Csiz is %d => space in SIZ only for %d comps.!!!\n",
                                l_image->numcomps, ((len - 38) / 3));
                        if (!JPWL_ASSUME) {
                                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                                return OPJ_FALSE;
                        }
        */              /* we try to correct */
        /*              opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust this\n");
                        if (l_image->numcomps < ((len - 38) / 3)) {
                                len = 38 + 3 * l_image->numcomps;
                                opj_event_msg(p_manager, EVT_WARNING, "- setting Lsiz to %d => HYPOTHESIS!!!\n",
                                        len);
                        } else {
                                l_image->numcomps = ((len - 38) / 3);
                                opj_event_msg(p_manager, EVT_WARNING, "- setting Csiz to %d => HYPOTHESIS!!!\n",
                                        l_image->numcomps);
                        }
                }
        */

        /* update components number in the jpwl_exp_comps filed */
        l_cp->exp_comps = l_image->numcomps;
    }
#endif /* USE_JPWL */

    /* Allocate the resulting image components */
    l_image->comps = (opj_image_comp_t*) opj_calloc(l_image->numcomps,
                     sizeof(opj_image_comp_t));
    if (l_image->comps == 00) {
        l_image->numcomps = 0;
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to take in charge SIZ marker\n");
        return OPJ_FALSE;
    }

    l_img_comp = l_image->comps;

    l_prec0 = 0;
    l_sgnd0 = 0;
    /* Read the component information */
    for (i = 0; i < l_image->numcomps; ++i) {
        OPJ_UINT32 tmp;
        opj_read_bytes(p_header_data, &tmp, 1); /* Ssiz_i */
        ++p_header_data;
        l_img_comp->prec = (tmp & 0x7f) + 1;
        l_img_comp->sgnd = tmp >> 7;

        if (p_j2k->dump_state == 0) {
            if (i == 0) {
                l_prec0 = l_img_comp->prec;
                l_sgnd0 = l_img_comp->sgnd;
            } else if (!l_cp->allow_different_bit_depth_sign
                       && (l_img_comp->prec != l_prec0 || l_img_comp->sgnd != l_sgnd0)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Despite JP2 BPC!=255, precision and/or sgnd values for comp[%d] is different than comp[0]:\n"
                              "        [0] prec(%d) sgnd(%d) [%d] prec(%d) sgnd(%d)\n", i, l_prec0, l_sgnd0,
                              i, l_img_comp->prec, l_img_comp->sgnd);
            }
            /* TODO: we should perhaps also check against JP2 BPCC values */
        }
        opj_read_bytes(p_header_data, &tmp, 1); /* XRsiz_i */
        ++p_header_data;
        l_img_comp->dx = (OPJ_UINT32)tmp; /* should be between 1 and 255 */
        opj_read_bytes(p_header_data, &tmp, 1); /* YRsiz_i */
        ++p_header_data;
        l_img_comp->dy = (OPJ_UINT32)tmp; /* should be between 1 and 255 */
        if (l_img_comp->dx < 1 || l_img_comp->dx > 255 ||
                l_img_comp->dy < 1 || l_img_comp->dy > 255) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : dx=%u dy=%u (should be between 1 and 255 according to the JPEG2000 norm)\n",
                          i, l_img_comp->dx, l_img_comp->dy);
            return OPJ_FALSE;
        }
        /* Avoids later undefined shift in computation of */
        /* p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps[i].m_dc_level_shift = 1
                    << (l_image->comps[i].prec - 1); */
        if (l_img_comp->prec > 31) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : prec=%u (should be between 1 and 38 according to the JPEG2000 norm. OpenJpeg only supports up to 31)\n",
                          i, l_img_comp->prec);
            return OPJ_FALSE;
        }
#ifdef USE_JPWL
        if (l_cp->correct) {
            /* if JPWL is on, we check whether TX errors have damaged
                    too much the SIZ parameters, again */
            if (!(l_image->comps[i].dx * l_image->comps[i].dy)) {
                opj_event_msg(p_manager, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                              "JPWL: bad XRsiz_%d/YRsiz_%d (%d x %d)\n",
                              i, i, l_image->comps[i].dx, l_image->comps[i].dy);
                if (!JPWL_ASSUME) {
                    opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                    return OPJ_FALSE;
                }
                /* we try to correct */
                opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust them\n");
                if (!l_image->comps[i].dx) {
                    l_image->comps[i].dx = 1;
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "- setting XRsiz_%d to %d => HYPOTHESIS!!!\n",
                                  i, l_image->comps[i].dx);
                }
                if (!l_image->comps[i].dy) {
                    l_image->comps[i].dy = 1;
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "- setting YRsiz_%d to %d => HYPOTHESIS!!!\n",
                                  i, l_image->comps[i].dy);
                }
            }
        }
#endif /* USE_JPWL */
        l_img_comp->resno_decoded =
            0;                                                          /* number of resolution decoded */
        l_img_comp->factor =
            l_cp->m_specific_param.m_dec.m_reduce; /* reducing factor per component */
        ++l_img_comp;
    }

    if (l_cp->tdx == 0 || l_cp->tdy == 0) {
        return OPJ_FALSE;
    }

    /* Compute the number of tiles */
    l_cp->tw = opj_uint_ceildiv(l_image->x1 - l_cp->tx0, l_cp->tdx);
    l_cp->th = opj_uint_ceildiv(l_image->y1 - l_cp->ty0, l_cp->tdy);

    /* Check that the number of tiles is valid */
    if (l_cp->tw == 0 || l_cp->th == 0 || l_cp->tw > 65535 / l_cp->th) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid number of tiles : %u x %u (maximum fixed by jpeg2000 norm is 65535 tiles)\n",
                      l_cp->tw, l_cp->th);
        return OPJ_FALSE;
    }
    l_nb_tiles = l_cp->tw * l_cp->th;

    /* Define the tiles which will be decoded */
    if (p_j2k->m_specific_param.m_decoder.m_discard_tiles) {
        p_j2k->m_specific_param.m_decoder.m_start_tile_x =
            (p_j2k->m_specific_param.m_decoder.m_start_tile_x - l_cp->tx0) / l_cp->tdx;
        p_j2k->m_specific_param.m_decoder.m_start_tile_y =
            (p_j2k->m_specific_param.m_decoder.m_start_tile_y - l_cp->ty0) / l_cp->tdy;
        p_j2k->m_specific_param.m_decoder.m_end_tile_x = opj_uint_ceildiv(
                    p_j2k->m_specific_param.m_decoder.m_end_tile_x - l_cp->tx0,
                    l_cp->tdx);
        p_j2k->m_specific_param.m_decoder.m_end_tile_y = opj_uint_ceildiv(
                    p_j2k->m_specific_param.m_decoder.m_end_tile_y - l_cp->ty0,
                    l_cp->tdy);
    } else {
        p_j2k->m_specific_param.m_decoder.m_start_tile_x = 0;
        p_j2k->m_specific_param.m_decoder.m_start_tile_y = 0;
        p_j2k->m_specific_param.m_decoder.m_end_tile_x = l_cp->tw;
        p_j2k->m_specific_param.m_decoder.m_end_tile_y = l_cp->th;
    }

#ifdef USE_JPWL
    if (l_cp->correct) {
        /* if JPWL is on, we check whether TX errors have damaged
          too much the SIZ parameters */
        if ((l_cp->tw < 1) || (l_cp->th < 1) || (l_cp->tw > l_cp->max_tiles) ||
                (l_cp->th > l_cp->max_tiles)) {
            opj_event_msg(p_manager, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                          "JPWL: bad number of tiles (%d x %d)\n",
                          l_cp->tw, l_cp->th);
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
            /* we try to correct */
            opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust them\n");
            if (l_cp->tw < 1) {
                l_cp->tw = 1;
                opj_event_msg(p_manager, EVT_WARNING,
                              "- setting %d tiles in x => HYPOTHESIS!!!\n",
                              l_cp->tw);
            }
            if (l_cp->tw > l_cp->max_tiles) {
                l_cp->tw = 1;
                opj_event_msg(p_manager, EVT_WARNING,
                              "- too large x, increase expectance of %d\n"
                              "- setting %d tiles in x => HYPOTHESIS!!!\n",
                              l_cp->max_tiles, l_cp->tw);
            }
            if (l_cp->th < 1) {
                l_cp->th = 1;
                opj_event_msg(p_manager, EVT_WARNING,
                              "- setting %d tiles in y => HYPOTHESIS!!!\n",
                              l_cp->th);
            }
            if (l_cp->th > l_cp->max_tiles) {
                l_cp->th = 1;
                opj_event_msg(p_manager, EVT_WARNING,
                              "- too large y, increase expectance of %d to continue\n",
                              "- setting %d tiles in y => HYPOTHESIS!!!\n",
                              l_cp->max_tiles, l_cp->th);
            }
        }
    }
#endif /* USE_JPWL */

    /* memory allocations */
    l_cp->tcps = (opj_tcp_t*) opj_calloc(l_nb_tiles, sizeof(opj_tcp_t));
    if (l_cp->tcps == 00) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to take in charge SIZ marker\n");
        return OPJ_FALSE;
    }

#ifdef USE_JPWL
    if (l_cp->correct) {
        if (!l_cp->tcps) {
            opj_event_msg(p_manager, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                          "JPWL: could not alloc tcps field of cp\n");
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
        }
    }
#endif /* USE_JPWL */

    p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps =
        (opj_tccp_t*) opj_calloc(l_image->numcomps, sizeof(opj_tccp_t));
    if (p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps  == 00) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to take in charge SIZ marker\n");
        return OPJ_FALSE;
    }

    p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mct_records =
        (opj_mct_data_t*)opj_calloc(OPJ_J2K_MCT_DEFAULT_NB_RECORDS,
                                    sizeof(opj_mct_data_t));

    if (! p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mct_records) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to take in charge SIZ marker\n");
        return OPJ_FALSE;
    }
    p_j2k->m_specific_param.m_decoder.m_default_tcp->m_nb_max_mct_records =
        OPJ_J2K_MCT_DEFAULT_NB_RECORDS;

    p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mcc_records =
        (opj_simple_mcc_decorrelation_data_t*)
        opj_calloc(OPJ_J2K_MCC_DEFAULT_NB_RECORDS,
                   sizeof(opj_simple_mcc_decorrelation_data_t));

    if (! p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mcc_records) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to take in charge SIZ marker\n");
        return OPJ_FALSE;
    }
    p_j2k->m_specific_param.m_decoder.m_default_tcp->m_nb_max_mcc_records =
        OPJ_J2K_MCC_DEFAULT_NB_RECORDS;

    /* set up default dc level shift */
    for (i = 0; i < l_image->numcomps; ++i) {
        if (! l_image->comps[i].sgnd) {
            p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps[i].m_dc_level_shift = 1
                    << (l_image->comps[i].prec - 1);
        }
    }

    l_current_tile_param = l_cp->tcps;
    for (i = 0; i < l_nb_tiles; ++i) {
        l_current_tile_param->tccps = (opj_tccp_t*) opj_calloc(l_image->numcomps,
                                      sizeof(opj_tccp_t));
        if (l_current_tile_param->tccps == 00) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to take in charge SIZ marker\n");
            return OPJ_FALSE;
        }

        ++l_current_tile_param;
    }

    /*Allocate and initialize some elements of codestrem index*/
    if (!opj_j2k_allocate_tile_element_cstr_index(p_j2k)) {
        return OPJ_FALSE;
    }

    p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_MH;
    opj_image_comp_header_update(l_image, l_cp);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_com(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_comment_size;
    OPJ_UINT32 l_total_com_size;
    const OPJ_CHAR *l_comment;
    OPJ_BYTE * l_current_ptr = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    l_comment = p_j2k->m_cp.comment;
    l_comment_size = (OPJ_UINT32)strlen(l_comment);
    l_total_com_size = l_comment_size + 6;

    if (l_total_com_size >
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_total_com_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to write the COM marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_total_com_size;
    }

    l_current_ptr = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_ptr, J2K_MS_COM, 2);  /* COM */
    l_current_ptr += 2;

    opj_write_bytes(l_current_ptr, l_total_com_size - 2, 2);        /* L_COM */
    l_current_ptr += 2;

    opj_write_bytes(l_current_ptr, 1,
                    2);   /* General use (IS 8859-15:1999 (Latin) values) */
    l_current_ptr += 2;

    memcpy(l_current_ptr, l_comment, l_comment_size);

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_total_com_size,
                              p_manager) != l_total_com_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a COM marker (comments)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_com(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_header_data != 00);

    OPJ_UNUSED(p_j2k);
    OPJ_UNUSED(p_header_data);
    OPJ_UNUSED(p_header_size);
    OPJ_UNUSED(p_manager);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_cod(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager)
{
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_code_size, l_remaining_size;
    OPJ_BYTE * l_current_data = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
    l_code_size = 9 + opj_j2k_get_SPCod_SPCoc_size(p_j2k,
                  p_j2k->m_current_tile_number, 0);
    l_remaining_size = l_code_size;

    if (l_code_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_code_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write COD marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_code_size;
    }

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_COD, 2);           /* COD */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_code_size - 2, 2);      /* L_COD */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_tcp->csty, 1);          /* Scod */
    ++l_current_data;

    opj_write_bytes(l_current_data, (OPJ_UINT32)l_tcp->prg, 1); /* SGcod (A) */
    ++l_current_data;

    opj_write_bytes(l_current_data, l_tcp->numlayers, 2);     /* SGcod (B) */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_tcp->mct, 1);           /* SGcod (C) */
    ++l_current_data;

    l_remaining_size -= 9;

    if (! opj_j2k_write_SPCod_SPCoc(p_j2k, p_j2k->m_current_tile_number, 0,
                                    l_current_data, &l_remaining_size, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error writing COD marker\n");
        return OPJ_FALSE;
    }

    if (l_remaining_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error writing COD marker\n");
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_code_size,
                              p_manager) != l_code_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a COD marker (Coding style defaults)
 * @param       p_header_data   the data contained in the COD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cod(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* loop */
    OPJ_UINT32 i;
    OPJ_UINT32 l_tmp;
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_image_t *l_image = 00;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_image = p_j2k->m_private_image;
    l_cp = &(p_j2k->m_cp);

    /* If we are in the first tile-part header of the current tile */
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH) ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

#if 0
    /* This check was added per https://github.com/uclouvain/openjpeg/commit/daed8cc9195555e101ab708a501af2dfe6d5e001 */
    /* but this is no longer necessary to handle issue476.jp2 */
    /* and this actually cause issues on legit files. See https://github.com/uclouvain/openjpeg/issues/1043 */
    /* Only one COD per tile */
    if (l_tcp->cod) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "COD marker already read. No more than one COD marker per tile.\n");
        return OPJ_FALSE;
    }
#endif
    l_tcp->cod = 1;

    /* Make sure room is sufficient */
    if (p_header_size < 5) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_tcp->csty, 1);         /* Scod */
    ++p_header_data;
    /* Make sure we know how to decode this */
    if ((l_tcp->csty & ~(OPJ_UINT32)(J2K_CP_CSTY_PRT | J2K_CP_CSTY_SOP |
                                     J2K_CP_CSTY_EPH)) != 0U) {
        opj_event_msg(p_manager, EVT_ERROR, "Unknown Scod value in COD marker\n");
        return OPJ_FALSE;
    }
    opj_read_bytes(p_header_data, &l_tmp, 1);                       /* SGcod (A) */
    ++p_header_data;
    l_tcp->prg = (OPJ_PROG_ORDER) l_tmp;
    /* Make sure progression order is valid */
    if (l_tcp->prg > OPJ_CPRL) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Unknown progression order in COD marker\n");
        l_tcp->prg = OPJ_PROG_UNKNOWN;
    }
    opj_read_bytes(p_header_data, &l_tcp->numlayers, 2);    /* SGcod (B) */
    p_header_data += 2;

    if ((l_tcp->numlayers < 1U) || (l_tcp->numlayers > 65535U)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid number of layers in COD marker : %d not in range [1-65535]\n",
                      l_tcp->numlayers);
        return OPJ_FALSE;
    }

    /* If user didn't set a number layer to decode take the max specify in the codestream. */
    if (l_cp->m_specific_param.m_dec.m_layer) {
        l_tcp->num_layers_to_decode = l_cp->m_specific_param.m_dec.m_layer;
    } else {
        l_tcp->num_layers_to_decode = l_tcp->numlayers;
    }

    opj_read_bytes(p_header_data, &l_tcp->mct, 1);          /* SGcod (C) */
    ++p_header_data;

    if (l_tcp->mct > 1) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid multiple component transformation\n");
        return OPJ_FALSE;
    }

    p_header_size -= 5;
    for (i = 0; i < l_image->numcomps; ++i) {
        l_tcp->tccps[i].csty = l_tcp->csty & J2K_CCP_CSTY_PRT;
    }

    if (! opj_j2k_read_SPCod_SPCoc(p_j2k, 0, p_header_data, &p_header_size,
                                   p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
        return OPJ_FALSE;
    }

    if (p_header_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
        return OPJ_FALSE;
    }

    /* Apply the coding style to other components of the current tile or the m_default_tcp*/
    opj_j2k_copy_tile_component_parameters(p_j2k);

    /* Index */
#ifdef WIP_REMOVE_MSD
    if (p_j2k->cstr_info) {
        /*opj_codestream_info_t *l_cstr_info = p_j2k->cstr_info;*/
        p_j2k->cstr_info->prog = l_tcp->prg;
        p_j2k->cstr_info->numlayers = l_tcp->numlayers;
        p_j2k->cstr_info->numdecompos = (OPJ_INT32*) opj_malloc(
                                            l_image->numcomps * sizeof(OPJ_UINT32));
        if (!p_j2k->cstr_info->numdecompos) {
            return OPJ_FALSE;
        }
        for (i = 0; i < l_image->numcomps; ++i) {
            p_j2k->cstr_info->numdecompos[i] = l_tcp->tccps[i].numresolutions - 1;
        }
    }
#endif

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_coc(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_comp_no,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 l_coc_size, l_remaining_size;
    OPJ_UINT32 l_comp_room;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_comp_room = (p_j2k->m_private_image->numcomps <= 256) ? 1 : 2;

    l_coc_size = 5 + l_comp_room + opj_j2k_get_SPCod_SPCoc_size(p_j2k,
                 p_j2k->m_current_tile_number, p_comp_no);

    if (l_coc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data;
        /*p_j2k->m_specific_param.m_encoder.m_header_tile_data
                = (OPJ_BYTE*)opj_realloc(
                        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
                        l_coc_size);*/

        new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                   p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_coc_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write COC marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_coc_size;
    }

    opj_j2k_write_coc_in_memory(p_j2k, p_comp_no,
                                p_j2k->m_specific_param.m_encoder.m_header_tile_data, &l_remaining_size,
                                p_manager);

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_coc_size,
                              p_manager) != l_coc_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_compare_coc(opj_j2k_t *p_j2k,
                                    OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no)
{
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];

    if (l_tcp->tccps[p_first_comp_no].csty != l_tcp->tccps[p_second_comp_no].csty) {
        return OPJ_FALSE;
    }


    return opj_j2k_compare_SPCod_SPCoc(p_j2k, p_j2k->m_current_tile_number,
                                       p_first_comp_no, p_second_comp_no);
}

static void opj_j2k_write_coc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager
                                       )
{
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_coc_size, l_remaining_size;
    OPJ_BYTE * l_current_data = 00;
    opj_image_t *l_image = 00;
    OPJ_UINT32 l_comp_room;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
    l_image = p_j2k->m_private_image;
    l_comp_room = (l_image->numcomps <= 256) ? 1 : 2;

    l_coc_size = 5 + l_comp_room + opj_j2k_get_SPCod_SPCoc_size(p_j2k,
                 p_j2k->m_current_tile_number, p_comp_no);
    l_remaining_size = l_coc_size;

    l_current_data = p_data;

    opj_write_bytes(l_current_data, J2K_MS_COC,
                    2);                         /* COC */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_coc_size - 2,
                    2);                     /* L_COC */
    l_current_data += 2;

    opj_write_bytes(l_current_data, p_comp_no, l_comp_room);        /* Ccoc */
    l_current_data += l_comp_room;

    opj_write_bytes(l_current_data, l_tcp->tccps[p_comp_no].csty,
                    1);               /* Scoc */
    ++l_current_data;

    l_remaining_size -= (5 + l_comp_room);
    opj_j2k_write_SPCod_SPCoc(p_j2k, p_j2k->m_current_tile_number, 0,
                              l_current_data, &l_remaining_size, p_manager);
    * p_data_written = l_coc_size;
}

static OPJ_UINT32 opj_j2k_get_max_coc_size(opj_j2k_t *p_j2k)
{
    OPJ_UINT32 i, j;
    OPJ_UINT32 l_nb_comp;
    OPJ_UINT32 l_nb_tiles;
    OPJ_UINT32 l_max = 0;

    /* preconditions */

    l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th ;
    l_nb_comp = p_j2k->m_private_image->numcomps;

    for (i = 0; i < l_nb_tiles; ++i) {
        for (j = 0; j < l_nb_comp; ++j) {
            l_max = opj_uint_max(l_max, opj_j2k_get_SPCod_SPCoc_size(p_j2k, i, j));
        }
    }

    return 6 + l_max;
}

/**
 * Reads a COC marker (Coding Style Component)
 * @param       p_header_data   the data contained in the COC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_coc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_image_t *l_image = NULL;
    OPJ_UINT32 l_comp_room;
    OPJ_UINT32 l_comp_no;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH)
            ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;
    l_image = p_j2k->m_private_image;

    l_comp_room = l_image->numcomps <= 256 ? 1 : 2;

    /* make sure room is sufficient*/
    if (p_header_size < l_comp_room + 1) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
        return OPJ_FALSE;
    }
    p_header_size -= l_comp_room + 1;

    opj_read_bytes(p_header_data, &l_comp_no,
                   l_comp_room);                 /* Ccoc */
    p_header_data += l_comp_room;
    if (l_comp_no >= l_image->numcomps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error reading COC marker (bad number of components)\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_tcp->tccps[l_comp_no].csty,
                   1);                  /* Scoc */
    ++p_header_data ;

    if (! opj_j2k_read_SPCod_SPCoc(p_j2k, l_comp_no, p_header_data, &p_header_size,
                                   p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
        return OPJ_FALSE;
    }

    if (p_header_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
        return OPJ_FALSE;
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_qcd(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_qcd_size, l_remaining_size;
    OPJ_BYTE * l_current_data = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_qcd_size = 4 + opj_j2k_get_SQcd_SQcc_size(p_j2k, p_j2k->m_current_tile_number,
                 0);
    l_remaining_size = l_qcd_size;

    if (l_qcd_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_qcd_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write QCD marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_qcd_size;
    }

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_QCD, 2);         /* QCD */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_qcd_size - 2, 2);     /* L_QCD */
    l_current_data += 2;

    l_remaining_size -= 4;

    if (! opj_j2k_write_SQcd_SQcc(p_j2k, p_j2k->m_current_tile_number, 0,
                                  l_current_data, &l_remaining_size, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error writing QCD marker\n");
        return OPJ_FALSE;
    }

    if (l_remaining_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error writing QCD marker\n");
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_qcd_size,
                              p_manager) != l_qcd_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a QCD marker (Quantization defaults)
 * @param       p_header_data   the data contained in the QCD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_qcd(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_j2k_read_SQcd_SQcc(p_j2k, 0, p_header_data, &p_header_size,
                                 p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading QCD marker\n");
        return OPJ_FALSE;
    }

    if (p_header_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading QCD marker\n");
        return OPJ_FALSE;
    }

    /* Apply the quantization parameters to other components of the current tile or the m_default_tcp */
    opj_j2k_copy_tile_quantization_parameters(p_j2k);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_qcc(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_comp_no,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_qcc_size, l_remaining_size;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_qcc_size = 5 + opj_j2k_get_SQcd_SQcc_size(p_j2k, p_j2k->m_current_tile_number,
                 p_comp_no);
    l_qcc_size += p_j2k->m_private_image->numcomps <= 256 ? 0 : 1;
    l_remaining_size = l_qcc_size;

    if (l_qcc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_qcc_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write QCC marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_qcc_size;
    }

    opj_j2k_write_qcc_in_memory(p_j2k, p_comp_no,
                                p_j2k->m_specific_param.m_encoder.m_header_tile_data, &l_remaining_size,
                                p_manager);

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_qcc_size,
                              p_manager) != l_qcc_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_compare_qcc(opj_j2k_t *p_j2k,
                                    OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no)
{
    return opj_j2k_compare_SQcd_SQcc(p_j2k, p_j2k->m_current_tile_number,
                                     p_first_comp_no, p_second_comp_no);
}

static void opj_j2k_write_qcc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager
                                       )
{
    OPJ_UINT32 l_qcc_size, l_remaining_size;
    OPJ_BYTE * l_current_data = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_qcc_size = 6 + opj_j2k_get_SQcd_SQcc_size(p_j2k, p_j2k->m_current_tile_number,
                 p_comp_no);
    l_remaining_size = l_qcc_size;

    l_current_data = p_data;

    opj_write_bytes(l_current_data, J2K_MS_QCC, 2);         /* QCC */
    l_current_data += 2;

    if (p_j2k->m_private_image->numcomps <= 256) {
        --l_qcc_size;

        opj_write_bytes(l_current_data, l_qcc_size - 2, 2);     /* L_QCC */
        l_current_data += 2;

        opj_write_bytes(l_current_data, p_comp_no, 1);  /* Cqcc */
        ++l_current_data;

        /* in the case only one byte is sufficient the last byte allocated is useless -> still do -6 for available */
        l_remaining_size -= 6;
    } else {
        opj_write_bytes(l_current_data, l_qcc_size - 2, 2);     /* L_QCC */
        l_current_data += 2;

        opj_write_bytes(l_current_data, p_comp_no, 2);  /* Cqcc */
        l_current_data += 2;

        l_remaining_size -= 6;
    }

    opj_j2k_write_SQcd_SQcc(p_j2k, p_j2k->m_current_tile_number, p_comp_no,
                            l_current_data, &l_remaining_size, p_manager);

    *p_data_written = l_qcc_size;
}

static OPJ_UINT32 opj_j2k_get_max_qcc_size(opj_j2k_t *p_j2k)
{
    return opj_j2k_get_max_coc_size(p_j2k);
}

/**
 * Reads a QCC marker (Quantization component)
 * @param       p_header_data   the data contained in the QCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_qcc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_num_comp, l_comp_no;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_num_comp = p_j2k->m_private_image->numcomps;

    if (l_num_comp <= 256) {
        if (p_header_size < 1) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
            return OPJ_FALSE;
        }
        opj_read_bytes(p_header_data, &l_comp_no, 1);
        ++p_header_data;
        --p_header_size;
    } else {
        if (p_header_size < 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
            return OPJ_FALSE;
        }
        opj_read_bytes(p_header_data, &l_comp_no, 2);
        p_header_data += 2;
        p_header_size -= 2;
    }

#ifdef USE_JPWL
    if (p_j2k->m_cp.correct) {

        static OPJ_UINT32 backup_compno = 0;

        /* compno is negative or larger than the number of components!!! */
        if (/*(l_comp_no < 0) ||*/ (l_comp_no >= l_num_comp)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "JPWL: bad component number in QCC (%d out of a maximum of %d)\n",
                          l_comp_no, l_num_comp);
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
            /* we try to correct */
            l_comp_no = backup_compno % l_num_comp;
            opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust this\n"
                          "- setting component number to %d\n",
                          l_comp_no);
        }

        /* keep your private count of tiles */
        backup_compno++;
    };
#endif /* USE_JPWL */

    if (l_comp_no >= p_j2k->m_private_image->numcomps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid component number: %d, regarding the number of components %d\n",
                      l_comp_no, p_j2k->m_private_image->numcomps);
        return OPJ_FALSE;
    }

    if (! opj_j2k_read_SQcd_SQcc(p_j2k, l_comp_no, p_header_data, &p_header_size,
                                 p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
        return OPJ_FALSE;
    }

    if (p_header_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_poc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_UINT32 l_nb_comp;
    OPJ_UINT32 l_nb_poc;
    OPJ_UINT32 l_poc_size;
    OPJ_UINT32 l_written_size = 0;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_poc_room;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_tcp = &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number];
    l_nb_comp = p_j2k->m_private_image->numcomps;
    l_nb_poc = 1 + l_tcp->numpocs;

    if (l_nb_comp <= 256) {
        l_poc_room = 1;
    } else {
        l_poc_room = 2;
    }
    l_poc_size = 4 + (5 + 2 * l_poc_room) * l_nb_poc;

    if (l_poc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_poc_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write POC marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_poc_size;
    }

    opj_j2k_write_poc_in_memory(p_j2k,
                                p_j2k->m_specific_param.m_encoder.m_header_tile_data, &l_written_size,
                                p_manager);

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_poc_size,
                              p_manager) != l_poc_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static void opj_j2k_write_poc_in_memory(opj_j2k_t *p_j2k,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_data_written,
                                        opj_event_mgr_t * p_manager
                                       )
{
    OPJ_UINT32 i;
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_nb_comp;
    OPJ_UINT32 l_nb_poc;
    OPJ_UINT32 l_poc_size;
    opj_image_t *l_image = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;
    opj_poc_t *l_current_poc = 00;
    OPJ_UINT32 l_poc_room;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_manager);

    l_tcp = &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number];
    l_tccp = &l_tcp->tccps[0];
    l_image = p_j2k->m_private_image;
    l_nb_comp = l_image->numcomps;
    l_nb_poc = 1 + l_tcp->numpocs;

    if (l_nb_comp <= 256) {
        l_poc_room = 1;
    } else {
        l_poc_room = 2;
    }

    l_poc_size = 4 + (5 + 2 * l_poc_room) * l_nb_poc;

    l_current_data = p_data;

    opj_write_bytes(l_current_data, J2K_MS_POC,
                    2);                                   /* POC  */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_poc_size - 2,
                    2);                                 /* Lpoc */
    l_current_data += 2;

    l_current_poc =  l_tcp->pocs;
    for (i = 0; i < l_nb_poc; ++i) {
        opj_write_bytes(l_current_data, l_current_poc->resno0,
                        1);                                /* RSpoc_i */
        ++l_current_data;

        opj_write_bytes(l_current_data, l_current_poc->compno0,
                        l_poc_room);              /* CSpoc_i */
        l_current_data += l_poc_room;

        opj_write_bytes(l_current_data, l_current_poc->layno1,
                        2);                                /* LYEpoc_i */
        l_current_data += 2;

        opj_write_bytes(l_current_data, l_current_poc->resno1,
                        1);                                /* REpoc_i */
        ++l_current_data;

        opj_write_bytes(l_current_data, l_current_poc->compno1,
                        l_poc_room);              /* CEpoc_i */
        l_current_data += l_poc_room;

        opj_write_bytes(l_current_data, (OPJ_UINT32)l_current_poc->prg,
                        1);   /* Ppoc_i */
        ++l_current_data;

        /* change the value of the max layer according to the actual number of layers in the file, components and resolutions*/
        l_current_poc->layno1 = (OPJ_UINT32)opj_int_min((OPJ_INT32)
                                l_current_poc->layno1, (OPJ_INT32)l_tcp->numlayers);
        l_current_poc->resno1 = (OPJ_UINT32)opj_int_min((OPJ_INT32)
                                l_current_poc->resno1, (OPJ_INT32)l_tccp->numresolutions);
        l_current_poc->compno1 = (OPJ_UINT32)opj_int_min((OPJ_INT32)
                                 l_current_poc->compno1, (OPJ_INT32)l_nb_comp);

        ++l_current_poc;
    }

    *p_data_written = l_poc_size;
}

static OPJ_UINT32 opj_j2k_get_max_poc_size(opj_j2k_t *p_j2k)
{
    opj_tcp_t * l_tcp = 00;
    OPJ_UINT32 l_nb_tiles = 0;
    OPJ_UINT32 l_max_poc = 0;
    OPJ_UINT32 i;

    l_tcp = p_j2k->m_cp.tcps;
    l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;

    for (i = 0; i < l_nb_tiles; ++i) {
        l_max_poc = opj_uint_max(l_max_poc, l_tcp->numpocs);
        ++l_tcp;
    }

    ++l_max_poc;

    return 4 + 9 * l_max_poc;
}

static OPJ_UINT32 opj_j2k_get_max_toc_size(opj_j2k_t *p_j2k)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_nb_tiles;
    OPJ_UINT32 l_max = 0;
    opj_tcp_t * l_tcp = 00;

    l_tcp = p_j2k->m_cp.tcps;
    l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th ;

    for (i = 0; i < l_nb_tiles; ++i) {
        l_max = opj_uint_max(l_max, l_tcp->m_nb_tile_parts);

        ++l_tcp;
    }

    return 12 * l_max;
}

static OPJ_UINT32 opj_j2k_get_specific_header_sizes(opj_j2k_t *p_j2k)
{
    OPJ_UINT32 l_nb_bytes = 0;
    OPJ_UINT32 l_nb_comps;
    OPJ_UINT32 l_coc_bytes, l_qcc_bytes;

    l_nb_comps = p_j2k->m_private_image->numcomps - 1;
    l_nb_bytes += opj_j2k_get_max_toc_size(p_j2k);

    if (!(OPJ_IS_CINEMA(p_j2k->m_cp.rsiz))) {
        l_coc_bytes = opj_j2k_get_max_coc_size(p_j2k);
        l_nb_bytes += l_nb_comps * l_coc_bytes;

        l_qcc_bytes = opj_j2k_get_max_qcc_size(p_j2k);
        l_nb_bytes += l_nb_comps * l_qcc_bytes;
    }

    l_nb_bytes += opj_j2k_get_max_poc_size(p_j2k);

    if (p_j2k->m_specific_param.m_encoder.m_PLT) {
        /* Reserve space for PLT markers */

        OPJ_UINT32 i;
        const opj_cp_t * l_cp = &(p_j2k->m_cp);
        OPJ_UINT32 l_max_packet_count = 0;
        for (i = 0; i < l_cp->th * l_cp->tw; ++i) {
            l_max_packet_count = opj_uint_max(l_max_packet_count,
                                              opj_get_encoding_packet_count(p_j2k->m_private_image, l_cp, i));
        }
        /* Minimum 6 bytes per PLT marker, and at a minimum (taking a pessimistic */
        /* estimate of 4 bytes for a packet size), one can write */
        /* (65536-6) / 4 = 16382 paquet sizes per PLT marker */
        p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT =
            6 * opj_uint_ceildiv(l_max_packet_count, 16382);
        /* Maximum 5 bytes per packet to encode a full UINT32 */
        p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT +=
            l_nb_bytes += 5 * l_max_packet_count;
        p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT += 1;
        l_nb_bytes += p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT;
    }

    /*** DEVELOPER CORNER, Add room for your headers ***/

    return l_nb_bytes;
}

/**
 * Reads a POC marker (Progression Order Change)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_poc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 i, l_nb_comp, l_tmp;
    opj_image_t * l_image = 00;
    OPJ_UINT32 l_old_poc_nb, l_current_poc_nb, l_current_poc_remaining;
    OPJ_UINT32 l_chunk_size, l_comp_room;

    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_poc_t *l_current_poc = 00;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_image = p_j2k->m_private_image;
    l_nb_comp = l_image->numcomps;
    if (l_nb_comp <= 256) {
        l_comp_room = 1;
    } else {
        l_comp_room = 2;
    }
    l_chunk_size = 5 + 2 * l_comp_room;
    l_current_poc_nb = p_header_size / l_chunk_size;
    l_current_poc_remaining = p_header_size % l_chunk_size;

    if ((l_current_poc_nb <= 0) || (l_current_poc_remaining != 0)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading POC marker\n");
        return OPJ_FALSE;
    }

    l_cp = &(p_j2k->m_cp);
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH) ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;
    l_old_poc_nb = l_tcp->POC ? l_tcp->numpocs + 1 : 0;
    l_current_poc_nb += l_old_poc_nb;

    if (l_current_poc_nb >= J2K_MAX_POCS) {
        opj_event_msg(p_manager, EVT_ERROR, "Too many POCs %d\n", l_current_poc_nb);
        return OPJ_FALSE;
    }

    /* now poc is in use.*/
    l_tcp->POC = 1;

    l_current_poc = &l_tcp->pocs[l_old_poc_nb];
    for (i = l_old_poc_nb; i < l_current_poc_nb; ++i) {
        opj_read_bytes(p_header_data, &(l_current_poc->resno0),
                       1);                               /* RSpoc_i */
        ++p_header_data;
        opj_read_bytes(p_header_data, &(l_current_poc->compno0),
                       l_comp_room);  /* CSpoc_i */
        p_header_data += l_comp_room;
        opj_read_bytes(p_header_data, &(l_current_poc->layno1),
                       2);                               /* LYEpoc_i */
        /* make sure layer end is in acceptable bounds */
        l_current_poc->layno1 = opj_uint_min(l_current_poc->layno1, l_tcp->numlayers);
        p_header_data += 2;
        opj_read_bytes(p_header_data, &(l_current_poc->resno1),
                       1);                               /* REpoc_i */
        ++p_header_data;
        opj_read_bytes(p_header_data, &(l_current_poc->compno1),
                       l_comp_room);  /* CEpoc_i */
        p_header_data += l_comp_room;
        opj_read_bytes(p_header_data, &l_tmp,
                       1);                                                                 /* Ppoc_i */
        ++p_header_data;
        l_current_poc->prg = (OPJ_PROG_ORDER) l_tmp;
        /* make sure comp is in acceptable bounds */
        l_current_poc->compno1 = opj_uint_min(l_current_poc->compno1, l_nb_comp);
        ++l_current_poc;
    }

    l_tcp->numpocs = l_current_poc_nb - 1;
    return OPJ_TRUE;
}

/**
 * Reads a CRG marker (Component registration)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_crg(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_nb_comp;
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_header_data);

    l_nb_comp = p_j2k->m_private_image->numcomps;

    if (p_header_size != l_nb_comp * 4) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading CRG marker\n");
        return OPJ_FALSE;
    }
    /* Do not care of this at the moment since only local variables are set here */
    /*
    for
            (i = 0; i < l_nb_comp; ++i)
    {
            opj_read_bytes(p_header_data,&l_Xcrg_i,2);                              // Xcrg_i
            p_header_data+=2;
            opj_read_bytes(p_header_data,&l_Ycrg_i,2);                              // Xcrg_i
            p_header_data+=2;
    }
    */
    return OPJ_TRUE;
}

/**
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_tlm(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_Ztlm, l_Stlm, l_ST, l_SP,
               l_Ptlm_size, l_entry_size, l_num_tileparts;
    OPJ_UINT32 i;
    opj_j2k_tlm_tile_part_info_t* l_tile_part_infos;
    opj_j2k_tlm_info_t* l_tlm;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_tlm = &(p_j2k->m_specific_param.m_decoder.m_tlm);

    if (p_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading TLM marker.\n");
        return OPJ_FALSE;
    }
    p_header_size -= 2;

    if (l_tlm->m_is_invalid) {
        return OPJ_TRUE;
    }

    opj_read_bytes(p_header_data, &l_Ztlm,
                   1);                              /* Ztlm */
    ++p_header_data;
    opj_read_bytes(p_header_data, &l_Stlm,
                   1);                              /* Stlm */
    ++p_header_data;

    l_ST = ((l_Stlm >> 4) & 0x3);
    if (l_ST == 3) {
        l_tlm->m_is_invalid = OPJ_TRUE;
        opj_event_msg(p_manager, EVT_WARNING,
                      "opj_j2k_read_tlm(): ST = 3 is invalid.\n");
        return OPJ_TRUE;
    }
    l_SP = (l_Stlm >> 6) & 0x1;

    l_Ptlm_size = (l_SP + 1) * 2;
    l_entry_size = l_Ptlm_size + l_ST;

    if ((p_header_size % l_entry_size) != 0) {
        l_tlm->m_is_invalid = OPJ_TRUE;
        opj_event_msg(p_manager, EVT_WARNING,
                      "opj_j2k_read_tlm(): TLM marker not of expected size.\n");
        return OPJ_TRUE;
    }

    l_num_tileparts = p_header_size / l_entry_size;
    if (l_num_tileparts == 0) {
        /* not totally sure if this is valid... */
        return OPJ_TRUE;
    }

    /* Highly unlikely, unless there are gazillions of TLM markers */
    if (l_tlm->m_entries_count > UINT32_MAX - l_num_tileparts ||
            l_tlm->m_entries_count + l_num_tileparts > UINT32_MAX / sizeof(
                opj_j2k_tlm_tile_part_info_t)) {
        l_tlm->m_is_invalid = OPJ_TRUE;
        opj_event_msg(p_manager, EVT_WARNING,
                      "opj_j2k_read_tlm(): too many TLM markers.\n");
        return OPJ_TRUE;
    }

    l_tile_part_infos = (opj_j2k_tlm_tile_part_info_t*)opj_realloc(
                            l_tlm->m_tile_part_infos,
                            (l_tlm->m_entries_count + l_num_tileparts) * sizeof(
                                opj_j2k_tlm_tile_part_info_t));
    if (!l_tile_part_infos) {
        l_tlm->m_is_invalid = OPJ_TRUE;
        opj_event_msg(p_manager, EVT_WARNING,
                      "opj_j2k_read_tlm(): cannot allocate m_tile_part_infos.\n");
        return OPJ_TRUE;
    }

    l_tlm->m_tile_part_infos = l_tile_part_infos;

    for (i = 0; i < l_num_tileparts; ++ i) {
        OPJ_UINT32 l_tile_index;
        OPJ_UINT32 l_length;

        /* Read Ttlm_i */
        if (l_ST == 0) {
            l_tile_index = l_tlm->m_entries_count;
        } else {
            opj_read_bytes(p_header_data, &l_tile_index, l_ST);
            p_header_data += l_ST;
        }

        if (l_tile_index >= p_j2k->m_cp.tw * p_j2k->m_cp.th) {
            l_tlm->m_is_invalid = OPJ_TRUE;
            opj_event_msg(p_manager, EVT_WARNING,
                          "opj_j2k_read_tlm(): invalid tile number %d\n",
                          l_tile_index);
            return OPJ_TRUE;
        }

        /* Read Ptlm_i */
        opj_read_bytes(p_header_data, &l_length, l_Ptlm_size);
        p_header_data += l_Ptlm_size;

        l_tile_part_infos[l_tlm->m_entries_count].m_tile_index =
            (OPJ_UINT16)l_tile_index;
        l_tile_part_infos[l_tlm->m_entries_count].m_length = l_length;
        ++l_tlm->m_entries_count;
    }

    return OPJ_TRUE;
}

/**
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_plm(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_j2k);
    OPJ_UNUSED(p_header_data);

    if (p_header_size < 1) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading PLM marker\n");
        return OPJ_FALSE;
    }
    /* Do not care of this at the moment since only local variables are set here */
    /*
    opj_read_bytes(p_header_data,&l_Zplm,1);                                        // Zplm
    ++p_header_data;
    --p_header_size;

    while
            (p_header_size > 0)
    {
            opj_read_bytes(p_header_data,&l_Nplm,1);                                // Nplm
            ++p_header_data;
            p_header_size -= (1+l_Nplm);
            if
                    (p_header_size < 0)
            {
                    opj_event_msg(p_manager, EVT_ERROR, "Error reading PLM marker\n");
                    return false;
            }
            for
                    (i = 0; i < l_Nplm; ++i)
            {
                    opj_read_bytes(p_header_data,&l_tmp,1);                         // Iplm_ij
                    ++p_header_data;
                    // take only the last seven bytes
                    l_packet_len |= (l_tmp & 0x7f);
                    if
                            (l_tmp & 0x80)
                    {
                            l_packet_len <<= 7;
                    }
                    else
                    {
            // store packet length and proceed to next packet
                            l_packet_len = 0;
                    }
            }
            if
                    (l_packet_len != 0)
            {
                    opj_event_msg(p_manager, EVT_ERROR, "Error reading PLM marker\n");
                    return false;
            }
    }
    */
    return OPJ_TRUE;
}

/**
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param       p_header_data   the data contained in the PLT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PLT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_plt(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_Zplt, l_tmp, l_packet_len = 0, i;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_j2k);

    if (p_header_size < 1) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading PLT marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_Zplt, 1);              /* Zplt */
    ++p_header_data;
    --p_header_size;

    for (i = 0; i < p_header_size; ++i) {
        opj_read_bytes(p_header_data, &l_tmp, 1);       /* Iplt_ij */
        ++p_header_data;
        /* take only the last seven bytes */
        l_packet_len |= (l_tmp & 0x7f);
        if (l_tmp & 0x80) {
            l_packet_len <<= 7;
        } else {
            /* store packet length and proceed to next packet */
            l_packet_len = 0;
        }
    }

    if (l_packet_len != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading PLT marker\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a PPM marker (Packed packet headers, main header)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
 */

static OPJ_BOOL opj_j2k_read_ppm(
    opj_j2k_t *p_j2k,
    OPJ_BYTE * p_header_data,
    OPJ_UINT32 p_header_size,
    opj_event_mgr_t * p_manager)
{
    opj_cp_t *l_cp = 00;
    OPJ_UINT32 l_Z_ppm;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    /* We need to have the Z_ppm element + 1 byte of Nppm/Ippm at minimum */
    if (p_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading PPM marker\n");
        return OPJ_FALSE;
    }

    l_cp = &(p_j2k->m_cp);
    l_cp->ppm = 1;

    opj_read_bytes(p_header_data, &l_Z_ppm, 1);             /* Z_ppm */
    ++p_header_data;
    --p_header_size;

    /* check allocation needed */
    if (l_cp->ppm_markers == NULL) { /* first PPM marker */
        OPJ_UINT32 l_newCount = l_Z_ppm + 1U; /* can't overflow, l_Z_ppm is UINT8 */
        assert(l_cp->ppm_markers_count == 0U);

        l_cp->ppm_markers = (opj_ppx *) opj_calloc(l_newCount, sizeof(opj_ppx));
        if (l_cp->ppm_markers == NULL) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPM marker\n");
            return OPJ_FALSE;
        }
        l_cp->ppm_markers_count = l_newCount;
    } else if (l_cp->ppm_markers_count <= l_Z_ppm) {
        OPJ_UINT32 l_newCount = l_Z_ppm + 1U; /* can't overflow, l_Z_ppm is UINT8 */
        opj_ppx *new_ppm_markers;
        new_ppm_markers = (opj_ppx *) opj_realloc(l_cp->ppm_markers,
                          l_newCount * sizeof(opj_ppx));
        if (new_ppm_markers == NULL) {
            /* clean up to be done on l_cp destruction */
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPM marker\n");
            return OPJ_FALSE;
        }
        l_cp->ppm_markers = new_ppm_markers;
        memset(l_cp->ppm_markers + l_cp->ppm_markers_count, 0,
               (l_newCount - l_cp->ppm_markers_count) * sizeof(opj_ppx));
        l_cp->ppm_markers_count = l_newCount;
    }

    if (l_cp->ppm_markers[l_Z_ppm].m_data != NULL) {
        /* clean up to be done on l_cp destruction */
        opj_event_msg(p_manager, EVT_ERROR, "Zppm %u already read\n", l_Z_ppm);
        return OPJ_FALSE;
    }

    l_cp->ppm_markers[l_Z_ppm].m_data = (OPJ_BYTE *) opj_malloc(p_header_size);
    if (l_cp->ppm_markers[l_Z_ppm].m_data == NULL) {
        /* clean up to be done on l_cp destruction */
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPM marker\n");
        return OPJ_FALSE;
    }
    l_cp->ppm_markers[l_Z_ppm].m_data_size = p_header_size;
    memcpy(l_cp->ppm_markers[l_Z_ppm].m_data, p_header_data, p_header_size);

    return OPJ_TRUE;
}

/**
 * Merges all PPM markers read (Packed headers, main header)
 *
 * @param       p_cp      main coding parameters.
 * @param       p_manager the user event manager.
 */
static OPJ_BOOL opj_j2k_merge_ppm(opj_cp_t *p_cp, opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, l_ppm_data_size, l_N_ppm_remaining;

    /* preconditions */
    assert(p_cp != 00);
    assert(p_manager != 00);
    assert(p_cp->ppm_buffer == NULL);

    if (p_cp->ppm == 0U) {
        return OPJ_TRUE;
    }

    l_ppm_data_size = 0U;
    l_N_ppm_remaining = 0U;
    for (i = 0U; i < p_cp->ppm_markers_count; ++i) {
        if (p_cp->ppm_markers[i].m_data !=
                NULL) { /* standard doesn't seem to require contiguous Zppm */
            OPJ_UINT32 l_N_ppm;
            OPJ_UINT32 l_data_size = p_cp->ppm_markers[i].m_data_size;
            const OPJ_BYTE* l_data = p_cp->ppm_markers[i].m_data;

            if (l_N_ppm_remaining >= l_data_size) {
                l_N_ppm_remaining -= l_data_size;
                l_data_size = 0U;
            } else {
                l_data += l_N_ppm_remaining;
                l_data_size -= l_N_ppm_remaining;
                l_N_ppm_remaining = 0U;
            }

            if (l_data_size > 0U) {
                do {
                    /* read Nppm */
                    if (l_data_size < 4U) {
                        /* clean up to be done on l_cp destruction */
                        opj_event_msg(p_manager, EVT_ERROR, "Not enough bytes to read Nppm\n");
                        return OPJ_FALSE;
                    }
                    opj_read_bytes(l_data, &l_N_ppm, 4);
                    l_data += 4;
                    l_data_size -= 4;

                    if (l_ppm_data_size > UINT_MAX - l_N_ppm) {
                        opj_event_msg(p_manager, EVT_ERROR, "Too large value for Nppm\n");
                        return OPJ_FALSE;
                    }
                    l_ppm_data_size += l_N_ppm;
                    if (l_data_size >= l_N_ppm) {
                        l_data_size -= l_N_ppm;
                        l_data += l_N_ppm;
                    } else {
                        l_N_ppm_remaining = l_N_ppm - l_data_size;
                        l_data_size = 0U;
                    }
                } while (l_data_size > 0U);
            }
        }
    }

    if (l_N_ppm_remaining != 0U) {
        /* clean up to be done on l_cp destruction */
        opj_event_msg(p_manager, EVT_ERROR, "Corrupted PPM markers\n");
        return OPJ_FALSE;
    }

    p_cp->ppm_buffer = (OPJ_BYTE *) opj_malloc(l_ppm_data_size);
    if (p_cp->ppm_buffer == 00) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPM marker\n");
        return OPJ_FALSE;
    }
    p_cp->ppm_len = l_ppm_data_size;
    l_ppm_data_size = 0U;
    l_N_ppm_remaining = 0U;
    for (i = 0U; i < p_cp->ppm_markers_count; ++i) {
        if (p_cp->ppm_markers[i].m_data !=
                NULL) { /* standard doesn't seem to require contiguous Zppm */
            OPJ_UINT32 l_N_ppm;
            OPJ_UINT32 l_data_size = p_cp->ppm_markers[i].m_data_size;
            const OPJ_BYTE* l_data = p_cp->ppm_markers[i].m_data;

            if (l_N_ppm_remaining >= l_data_size) {
                memcpy(p_cp->ppm_buffer + l_ppm_data_size, l_data, l_data_size);
                l_ppm_data_size += l_data_size;
                l_N_ppm_remaining -= l_data_size;
                l_data_size = 0U;
            } else {
                memcpy(p_cp->ppm_buffer + l_ppm_data_size, l_data, l_N_ppm_remaining);
                l_ppm_data_size += l_N_ppm_remaining;
                l_data += l_N_ppm_remaining;
                l_data_size -= l_N_ppm_remaining;
                l_N_ppm_remaining = 0U;
            }

            if (l_data_size > 0U) {
                do {
                    /* read Nppm */
                    if (l_data_size < 4U) {
                        /* clean up to be done on l_cp destruction */
                        opj_event_msg(p_manager, EVT_ERROR, "Not enough bytes to read Nppm\n");
                        return OPJ_FALSE;
                    }
                    opj_read_bytes(l_data, &l_N_ppm, 4);
                    l_data += 4;
                    l_data_size -= 4;

                    if (l_data_size >= l_N_ppm) {
                        memcpy(p_cp->ppm_buffer + l_ppm_data_size, l_data, l_N_ppm);
                        l_ppm_data_size += l_N_ppm;
                        l_data_size -= l_N_ppm;
                        l_data += l_N_ppm;
                    } else {
                        memcpy(p_cp->ppm_buffer + l_ppm_data_size, l_data, l_data_size);
                        l_ppm_data_size += l_data_size;
                        l_N_ppm_remaining = l_N_ppm - l_data_size;
                        l_data_size = 0U;
                    }
                } while (l_data_size > 0U);
            }
            opj_free(p_cp->ppm_markers[i].m_data);
            p_cp->ppm_markers[i].m_data = NULL;
            p_cp->ppm_markers[i].m_data_size = 0U;
        }
    }

    p_cp->ppm_data = p_cp->ppm_buffer;
    p_cp->ppm_data_size = p_cp->ppm_len;

    p_cp->ppm_markers_count = 0U;
    opj_free(p_cp->ppm_markers);
    p_cp->ppm_markers = NULL;

    return OPJ_TRUE;
}

/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param       p_header_data   the data contained in the PPT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_ppt(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_Z_ppt;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    /* We need to have the Z_ppt element + 1 byte of Ippt at minimum */
    if (p_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading PPT marker\n");
        return OPJ_FALSE;
    }

    l_cp = &(p_j2k->m_cp);
    if (l_cp->ppm) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error reading PPT marker: packet header have been previously found in the main header (PPM marker).\n");
        return OPJ_FALSE;
    }

    l_tcp = &(l_cp->tcps[p_j2k->m_current_tile_number]);
    l_tcp->ppt = 1;

    opj_read_bytes(p_header_data, &l_Z_ppt, 1);             /* Z_ppt */
    ++p_header_data;
    --p_header_size;

    /* check allocation needed */
    if (l_tcp->ppt_markers == NULL) { /* first PPT marker */
        OPJ_UINT32 l_newCount = l_Z_ppt + 1U; /* can't overflow, l_Z_ppt is UINT8 */
        assert(l_tcp->ppt_markers_count == 0U);

        l_tcp->ppt_markers = (opj_ppx *) opj_calloc(l_newCount, sizeof(opj_ppx));
        if (l_tcp->ppt_markers == NULL) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPT marker\n");
            return OPJ_FALSE;
        }
        l_tcp->ppt_markers_count = l_newCount;
    } else if (l_tcp->ppt_markers_count <= l_Z_ppt) {
        OPJ_UINT32 l_newCount = l_Z_ppt + 1U; /* can't overflow, l_Z_ppt is UINT8 */
        opj_ppx *new_ppt_markers;
        new_ppt_markers = (opj_ppx *) opj_realloc(l_tcp->ppt_markers,
                          l_newCount * sizeof(opj_ppx));
        if (new_ppt_markers == NULL) {
            /* clean up to be done on l_tcp destruction */
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPT marker\n");
            return OPJ_FALSE;
        }
        l_tcp->ppt_markers = new_ppt_markers;
        memset(l_tcp->ppt_markers + l_tcp->ppt_markers_count, 0,
               (l_newCount - l_tcp->ppt_markers_count) * sizeof(opj_ppx));
        l_tcp->ppt_markers_count = l_newCount;
    }

    if (l_tcp->ppt_markers[l_Z_ppt].m_data != NULL) {
        /* clean up to be done on l_tcp destruction */
        opj_event_msg(p_manager, EVT_ERROR, "Zppt %u already read\n", l_Z_ppt);
        return OPJ_FALSE;
    }

    l_tcp->ppt_markers[l_Z_ppt].m_data = (OPJ_BYTE *) opj_malloc(p_header_size);
    if (l_tcp->ppt_markers[l_Z_ppt].m_data == NULL) {
        /* clean up to be done on l_tcp destruction */
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPT marker\n");
        return OPJ_FALSE;
    }
    l_tcp->ppt_markers[l_Z_ppt].m_data_size = p_header_size;
    memcpy(l_tcp->ppt_markers[l_Z_ppt].m_data, p_header_data, p_header_size);
    return OPJ_TRUE;
}

/**
 * Merges all PPT markers read (Packed packet headers, tile-part header)
 *
 * @param       p_tcp   the tile.
 * @param       p_manager               the user event manager.
 */
static OPJ_BOOL opj_j2k_merge_ppt(opj_tcp_t *p_tcp, opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, l_ppt_data_size;
    /* preconditions */
    assert(p_tcp != 00);
    assert(p_manager != 00);

    if (p_tcp->ppt_buffer != NULL) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "opj_j2k_merge_ppt() has already been called\n");
        return OPJ_FALSE;
    }

    if (p_tcp->ppt == 0U) {
        return OPJ_TRUE;
    }

    l_ppt_data_size = 0U;
    for (i = 0U; i < p_tcp->ppt_markers_count; ++i) {
        l_ppt_data_size +=
            p_tcp->ppt_markers[i].m_data_size; /* can't overflow, max 256 markers of max 65536 bytes */
    }

    p_tcp->ppt_buffer = (OPJ_BYTE *) opj_malloc(l_ppt_data_size);
    if (p_tcp->ppt_buffer == 00) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read PPT marker\n");
        return OPJ_FALSE;
    }
    p_tcp->ppt_len = l_ppt_data_size;
    l_ppt_data_size = 0U;
    for (i = 0U; i < p_tcp->ppt_markers_count; ++i) {
        if (p_tcp->ppt_markers[i].m_data !=
                NULL) { /* standard doesn't seem to require contiguous Zppt */
            memcpy(p_tcp->ppt_buffer + l_ppt_data_size, p_tcp->ppt_markers[i].m_data,
                   p_tcp->ppt_markers[i].m_data_size);
            l_ppt_data_size +=
                p_tcp->ppt_markers[i].m_data_size; /* can't overflow, max 256 markers of max 65536 bytes */

            opj_free(p_tcp->ppt_markers[i].m_data);
            p_tcp->ppt_markers[i].m_data = NULL;
            p_tcp->ppt_markers[i].m_data_size = 0U;
        }
    }

    p_tcp->ppt_markers_count = 0U;
    opj_free(p_tcp->ppt_markers);
    p_tcp->ppt_markers = NULL;

    p_tcp->ppt_data = p_tcp->ppt_buffer;
    p_tcp->ppt_data_size = p_tcp->ppt_len;
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_tlm(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_tlm_size;
    OPJ_UINT32 size_per_tile_part;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    /* 10921 = (65535 - header_size) / size_per_tile_part where */
    /* header_size = 4 and size_per_tile_part = 6 */
    if (p_j2k->m_specific_param.m_encoder.m_total_tile_parts > 10921) {
        /* We could do more but it would require writing several TLM markers */
        opj_event_msg(p_manager, EVT_ERROR,
                      "A maximum of 10921 tile-parts are supported currently "
                      "when writing TLM marker\n");
        return OPJ_FALSE;
    }

    if (p_j2k->m_specific_param.m_encoder.m_total_tile_parts <= 255) {
        size_per_tile_part = 5;
        p_j2k->m_specific_param.m_encoder.m_Ttlmi_is_byte = OPJ_TRUE;
    } else {
        size_per_tile_part = 6;
        p_j2k->m_specific_param.m_encoder.m_Ttlmi_is_byte = OPJ_FALSE;
    }

    l_tlm_size = 2 + 4 + (size_per_tile_part *
                          p_j2k->m_specific_param.m_encoder.m_total_tile_parts);

    if (l_tlm_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_tlm_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write TLM marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_tlm_size;
    }
    memset(p_j2k->m_specific_param.m_encoder.m_header_tile_data, 0, l_tlm_size);

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    /* change the way data is written to avoid seeking if possible */
    /* TODO */
    p_j2k->m_specific_param.m_encoder.m_tlm_start = opj_stream_tell(p_stream);

    opj_write_bytes(l_current_data, J2K_MS_TLM,
                    2);                                   /* TLM */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_tlm_size - 2,
                    2);                                 /* Lpoc */
    l_current_data += 2;

    opj_write_bytes(l_current_data, 0,
                    1);                                                    /* Ztlm=0*/
    ++l_current_data;

    /* Stlm 0x50= ST=1(8bits-255 tiles max),SP=1(Ptlm=32bits) */
    /* Stlm 0x60= ST=2(16bits-65535 tiles max),SP=1(Ptlm=32bits) */
    opj_write_bytes(l_current_data,
                    size_per_tile_part == 5 ? 0x50 : 0x60,
                    1);
    ++l_current_data;

    /* do nothing on the size_per_tile_part * l_j2k->m_specific_param.m_encoder.m_total_tile_parts remaining data */
    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_tlm_size,
                              p_manager) != l_tlm_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_sot(opj_j2k_t *p_j2k,
                                  OPJ_BYTE * p_data,
                                  OPJ_UINT32 total_data_size,
                                  OPJ_UINT32 * p_data_written,
                                  const opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_stream);

    if (total_data_size < 12) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough bytes in output buffer to write SOT marker\n");
        return OPJ_FALSE;
    }

    opj_write_bytes(p_data, J2K_MS_SOT,
                    2);                                 /* SOT */
    p_data += 2;

    opj_write_bytes(p_data, 10,
                    2);                                                   /* Lsot */
    p_data += 2;

    opj_write_bytes(p_data, p_j2k->m_current_tile_number,
                    2);                        /* Isot */
    p_data += 2;

    /* Psot  */
    p_data += 4;

    opj_write_bytes(p_data,
                    p_j2k->m_specific_param.m_encoder.m_current_tile_part_number,
                    1);                        /* TPsot */
    ++p_data;

    opj_write_bytes(p_data,
                    p_j2k->m_cp.tcps[p_j2k->m_current_tile_number].m_nb_tile_parts,
                    1);                      /* TNsot */
    ++p_data;

    /* UniPG>> */
#ifdef USE_JPWL
    /* update markers struct */
    /*
            OPJ_BOOL res = j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOT, p_j2k->sot_start, len + 2);
    */
    assert(0 && "TODO");
#endif /* USE_JPWL */

    * p_data_written = 12;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_get_sot_values(OPJ_BYTE *  p_header_data,
                                       OPJ_UINT32  p_header_size,
                                       OPJ_UINT32* p_tile_no,
                                       OPJ_UINT32* p_tot_len,
                                       OPJ_UINT32* p_current_part,
                                       OPJ_UINT32* p_num_parts,
                                       opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_manager != 00);

    /* Size of this marker is fixed = 12 (we have already read marker and its size)*/
    if (p_header_size != 8) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading SOT marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, p_tile_no, 2);    /* Isot */
    p_header_data += 2;
    opj_read_bytes(p_header_data, p_tot_len, 4);    /* Psot */
    p_header_data += 4;
    opj_read_bytes(p_header_data, p_current_part, 1); /* TPsot */
    ++p_header_data;
    opj_read_bytes(p_header_data, p_num_parts, 1);  /* TNsot */
    ++p_header_data;
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_sot(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager)
{
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_tot_len, l_num_parts = 0;
    OPJ_UINT32 l_current_part;
    OPJ_UINT32 l_tile_x, l_tile_y;

    /* preconditions */

    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_j2k_get_sot_values(p_header_data, p_header_size,
                                 &(p_j2k->m_current_tile_number), &l_tot_len, &l_current_part, &l_num_parts,
                                 p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading SOT marker\n");
        return OPJ_FALSE;
    }
#ifdef DEBUG_VERBOSE
    fprintf(stderr, "SOT %d %d %d %d\n",
            p_j2k->m_current_tile_number, l_tot_len, l_current_part, l_num_parts);
#endif

    l_cp = &(p_j2k->m_cp);

    /* testcase 2.pdf.SIGFPE.706.1112 */
    if (p_j2k->m_current_tile_number >= l_cp->tw * l_cp->th) {
        opj_event_msg(p_manager, EVT_ERROR, "Invalid tile number %d\n",
                      p_j2k->m_current_tile_number);
        return OPJ_FALSE;
    }

    l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
    l_tile_x = p_j2k->m_current_tile_number % l_cp->tw;
    l_tile_y = p_j2k->m_current_tile_number / l_cp->tw;

    if (p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec < 0 ||
            p_j2k->m_current_tile_number == (OPJ_UINT32)
            p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec) {
        /* Do only this check if we decode all tile part headers, or if */
        /* we decode one precise tile. Otherwise the m_current_tile_part_number */
        /* might not be valid */
        /* Fixes issue with id_000020,sig_06,src_001958,op_flip4,pos_149 */
        /* of https://github.com/uclouvain/openjpeg/issues/939 */
        /* We must avoid reading twice the same tile part number for a given tile */
        /* so as to avoid various issues, like opj_j2k_merge_ppt being called */
        /* several times. */
        /* ISO 15444-1 A.4.2 Start of tile-part (SOT) mandates that tile parts */
        /* should appear in increasing order. */
        if (l_tcp->m_current_tile_part_number + 1 != (OPJ_INT32)l_current_part) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid tile part index for tile number %d. "
                          "Got %d, expected %d\n",
                          p_j2k->m_current_tile_number,
                          l_current_part,
                          l_tcp->m_current_tile_part_number + 1);
            return OPJ_FALSE;
        }
    }

    l_tcp->m_current_tile_part_number = (OPJ_INT32) l_current_part;

#ifdef USE_JPWL
    if (l_cp->correct) {

        OPJ_UINT32 tileno = p_j2k->m_current_tile_number;
        static OPJ_UINT32 backup_tileno = 0;

        /* tileno is negative or larger than the number of tiles!!! */
        if (tileno > (l_cp->tw * l_cp->th)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "JPWL: bad tile number (%d out of a maximum of %d)\n",
                          tileno, (l_cp->tw * l_cp->th));
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
            /* we try to correct */
            tileno = backup_tileno;
            opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust this\n"
                          "- setting tile number to %d\n",
                          tileno);
        }

        /* keep your private count of tiles */
        backup_tileno++;
    };
#endif /* USE_JPWL */

    /* look for the tile in the list of already processed tile (in parts). */
    /* Optimization possible here with a more complex data structure and with the removing of tiles */
    /* since the time taken by this function can only grow at the time */

    /* PSot should be equal to zero or >=14 or <= 2^32-1 */
    if ((l_tot_len != 0) && (l_tot_len < 14)) {
        if (l_tot_len ==
                12) { /* MSD: Special case for the PHR data which are read by kakadu*/
            opj_event_msg(p_manager, EVT_WARNING, "Empty SOT marker detected: Psot=%d.\n",
                          l_tot_len);
        } else {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Psot value is not correct regards to the JPEG2000 norm: %d.\n", l_tot_len);
            return OPJ_FALSE;
        }
    }

#ifdef USE_JPWL
    if (l_cp->correct) {

        /* totlen is negative or larger than the bytes left!!! */
        if (/*(l_tot_len < 0) ||*/ (l_tot_len >
                                    p_header_size)) {   /* FIXME it seems correct; for info in V1 -> (p_stream_numbytesleft(p_stream) + 8))) { */
            opj_event_msg(p_manager, EVT_ERROR,
                          "JPWL: bad tile byte size (%d bytes against %d bytes left)\n",
                          l_tot_len,
                          p_header_size);  /* FIXME it seems correct; for info in V1 -> p_stream_numbytesleft(p_stream) + 8); */
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
            /* we try to correct */
            l_tot_len = 0;
            opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust this\n"
                          "- setting Psot to %d => assuming it is the last tile\n",
                          l_tot_len);
        }
    };
#endif /* USE_JPWL */

    /* Ref A.4.2: Psot could be equal zero if it is the last tile-part of the codestream.*/
    if (!l_tot_len) {
        opj_event_msg(p_manager, EVT_INFO,
                      "Psot value of the current tile-part is equal to zero, "
                      "we assuming it is the last tile-part of the codestream.\n");
        p_j2k->m_specific_param.m_decoder.m_last_tile_part = 1;
    }

    if (l_tcp->m_nb_tile_parts != 0 && l_current_part >= l_tcp->m_nb_tile_parts) {
        /* Fixes https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=2851 */
        opj_event_msg(p_manager, EVT_ERROR,
                      "In SOT marker, TPSot (%d) is not valid regards to the previous "
                      "number of tile-part (%d), giving up\n", l_current_part,
                      l_tcp->m_nb_tile_parts);
        p_j2k->m_specific_param.m_decoder.m_last_tile_part = 1;
        return OPJ_FALSE;
    }

    if (l_num_parts !=
            0) { /* Number of tile-part header is provided by this tile-part header */
        l_num_parts += p_j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction;
        /* Useful to manage the case of textGBR.jp2 file because two values of TNSot are allowed: the correct numbers of
         * tile-parts for that tile and zero (A.4.2 of 15444-1 : 2002). */
        if (l_tcp->m_nb_tile_parts) {
            if (l_current_part >= l_tcp->m_nb_tile_parts) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "In SOT marker, TPSot (%d) is not valid regards to the current "
                              "number of tile-part (%d), giving up\n", l_current_part,
                              l_tcp->m_nb_tile_parts);
                p_j2k->m_specific_param.m_decoder.m_last_tile_part = 1;
                return OPJ_FALSE;
            }
        }
        if (l_current_part >= l_num_parts) {
            /* testcase 451.pdf.SIGSEGV.ce9.3723 */
            opj_event_msg(p_manager, EVT_ERROR,
                          "In SOT marker, TPSot (%d) is not valid regards to the current "
                          "number of tile-part (header) (%d), giving up\n", l_current_part, l_num_parts);
            p_j2k->m_specific_param.m_decoder.m_last_tile_part = 1;
            return OPJ_FALSE;
        }
        l_tcp->m_nb_tile_parts = l_num_parts;
    }

    /* If know the number of tile part header we will check if we didn't read the last*/
    if (l_tcp->m_nb_tile_parts) {
        if (l_tcp->m_nb_tile_parts == (l_current_part + 1)) {
            p_j2k->m_specific_param.m_decoder.m_can_decode =
                1; /* Process the last tile-part header*/
        }
    }

    if (!p_j2k->m_specific_param.m_decoder.m_last_tile_part) {
        /* Keep the size of data to skip after this marker */
        p_j2k->m_specific_param.m_decoder.m_sot_length = l_tot_len -
                12; /* SOT_marker_size = 12 */
    } else {
        /* FIXME: need to be computed from the number of bytes remaining in the codestream */
        p_j2k->m_specific_param.m_decoder.m_sot_length = 0;
    }

    p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_TPH;

    /* Check if the current tile is outside the area we want decode or not corresponding to the tile index*/
    if (p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec == -1) {
        p_j2k->m_specific_param.m_decoder.m_skip_data =
            (l_tile_x < p_j2k->m_specific_param.m_decoder.m_start_tile_x)
            || (l_tile_x >= p_j2k->m_specific_param.m_decoder.m_end_tile_x)
            || (l_tile_y < p_j2k->m_specific_param.m_decoder.m_start_tile_y)
            || (l_tile_y >= p_j2k->m_specific_param.m_decoder.m_end_tile_y);
    } else {
        assert(p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec >= 0);
        p_j2k->m_specific_param.m_decoder.m_skip_data =
            (p_j2k->m_current_tile_number != (OPJ_UINT32)
             p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec);
    }

    /* Index */
    {
        assert(p_j2k->cstr_index->tile_index != 00);
        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tileno =
            p_j2k->m_current_tile_number;
        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_tpsno =
            l_current_part;

        if (!p_j2k->m_specific_param.m_decoder.m_tlm.m_is_invalid &&
                l_num_parts >
                p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].nb_tps) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "SOT marker for tile %u declares more tile-parts than found in TLM marker.",
                          p_j2k->m_current_tile_number);
            p_j2k->m_specific_param.m_decoder.m_tlm.m_is_invalid = OPJ_TRUE;
        }

        if (!p_j2k->m_specific_param.m_decoder.m_tlm.m_is_invalid) {
            /* do nothing */
        } else if (l_num_parts != 0) {

            p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].nb_tps =
                l_num_parts;
            p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps =
                l_num_parts;

            if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index) {
                p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index =
                    (opj_tp_index_t*)opj_calloc(l_num_parts, sizeof(opj_tp_index_t));
                if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Not enough memory to read SOT marker. Tile index allocation failed\n");
                    return OPJ_FALSE;
                }
            } else {
                opj_tp_index_t *new_tp_index = (opj_tp_index_t *) opj_realloc(
                                                   p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index,
                                                   l_num_parts * sizeof(opj_tp_index_t));
                if (! new_tp_index) {
                    opj_free(p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index);
                    p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index = NULL;
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Not enough memory to read SOT marker. Tile index allocation failed\n");
                    return OPJ_FALSE;
                }
                p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index =
                    new_tp_index;
            }
        } else {
            /*if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index)*/ {

                if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index) {
                    p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps = 10;
                    p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index =
                        (opj_tp_index_t*)opj_calloc(
                            p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps,
                            sizeof(opj_tp_index_t));
                    if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index) {
                        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps = 0;
                        opj_event_msg(p_manager, EVT_ERROR,
                                      "Not enough memory to read SOT marker. Tile index allocation failed\n");
                        return OPJ_FALSE;
                    }
                }

                if (l_current_part >=
                        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps) {
                    opj_tp_index_t *new_tp_index;
                    p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps =
                        l_current_part + 1;
                    new_tp_index = (opj_tp_index_t *) opj_realloc(
                                       p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index,
                                       p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps *
                                       sizeof(opj_tp_index_t));
                    if (! new_tp_index) {
                        opj_free(p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index);
                        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index = NULL;
                        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].current_nb_tps = 0;
                        opj_event_msg(p_manager, EVT_ERROR,
                                      "Not enough memory to read SOT marker. Tile index allocation failed\n");
                        return OPJ_FALSE;
                    }
                    p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index =
                        new_tp_index;
                }
            }

        }

    }

    return OPJ_TRUE;
}

/**
 * Write one or more PLT markers in the provided buffer
 */
static OPJ_BOOL opj_j2k_write_plt_in_memory(opj_j2k_t *p_j2k,
        opj_tcd_marker_info_t* marker_info,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_data_written,
        opj_event_mgr_t * p_manager)
{
    OPJ_BYTE Zplt = 0;
    OPJ_UINT16 Lplt;
    OPJ_BYTE* p_data_start = p_data;
    OPJ_BYTE* p_data_Lplt = p_data + 2;
    OPJ_UINT32 i;

    OPJ_UNUSED(p_j2k);

    opj_write_bytes(p_data, J2K_MS_PLT, 2);
    p_data += 2;

    /* Reserve space for Lplt */
    p_data += 2;

    opj_write_bytes(p_data, Zplt, 1);
    p_data += 1;

    Lplt = 3;

    for (i = 0; i < marker_info->packet_count; i++) {
        OPJ_BYTE var_bytes[5];
        OPJ_UINT8 var_bytes_size = 0;
        OPJ_UINT32 packet_size = marker_info->p_packet_size[i];

        /* Packet size written in variable-length way, starting with LSB */
        var_bytes[var_bytes_size] = (OPJ_BYTE)(packet_size & 0x7f);
        var_bytes_size ++;
        packet_size >>= 7;
        while (packet_size > 0) {
            var_bytes[var_bytes_size] = (OPJ_BYTE)((packet_size & 0x7f) | 0x80);
            var_bytes_size ++;
            packet_size >>= 7;
        }

        /* Check if that can fit in the current PLT marker. If not, finish */
        /* current one, and start a new one */
        if (Lplt + var_bytes_size > 65535) {
            if (Zplt == 255) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "More than 255 PLT markers would be needed for current tile-part !\n");
                return OPJ_FALSE;
            }

            /* Patch Lplt */
            opj_write_bytes(p_data_Lplt, Lplt, 2);

            /* Start new segment */
            opj_write_bytes(p_data, J2K_MS_PLT, 2);
            p_data += 2;

            /* Reserve space for Lplt */
            p_data_Lplt = p_data;
            p_data += 2;

            Zplt ++;
            opj_write_bytes(p_data, Zplt, 1);
            p_data += 1;

            Lplt = 3;
        }

        Lplt = (OPJ_UINT16)(Lplt + var_bytes_size);

        /* Serialize variable-length packet size, starting with MSB */
        for (; var_bytes_size > 0; --var_bytes_size) {
            opj_write_bytes(p_data, var_bytes[var_bytes_size - 1], 1);
            p_data += 1;
        }
    }

    *p_data_written = (OPJ_UINT32)(p_data - p_data_start);

    /* Patch Lplt */
    opj_write_bytes(p_data_Lplt, Lplt, 2);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_sod(opj_j2k_t *p_j2k,
                                  opj_tcd_t * p_tile_coder,
                                  OPJ_BYTE * p_data,
                                  OPJ_UINT32 * p_data_written,
                                  OPJ_UINT32 total_data_size,
                                  const opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    opj_codestream_info_t *l_cstr_info = 00;
    OPJ_UINT32 l_remaining_data;
    opj_tcd_marker_info_t* marker_info = NULL;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_stream);

    if (total_data_size < 4) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough bytes in output buffer to write SOD marker\n");
        return OPJ_FALSE;
    }

    opj_write_bytes(p_data, J2K_MS_SOD,
                    2);                                 /* SOD */

    /* make room for the EOF marker */
    l_remaining_data =  total_data_size - 4;

    /* update tile coder */
    p_tile_coder->tp_num =
        p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number ;
    p_tile_coder->cur_tp_num =
        p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;

    /* INDEX >> */
    /* TODO mergeV2: check this part which use cstr_info */
    /*l_cstr_info = p_j2k->cstr_info;
    if (l_cstr_info) {
            if (!p_j2k->m_specific_param.m_encoder.m_current_tile_part_number ) {
                    //TODO cstr_info->tile[p_j2k->m_current_tile_number].end_header = p_stream_tell(p_stream) + p_j2k->pos_correction - 1;
                    l_cstr_info->tile[p_j2k->m_current_tile_number].tileno = p_j2k->m_current_tile_number;
            }
            else {*/
    /*
    TODO
    if
            (cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno - 1].end_pos < p_stream_tell(p_stream))
    {
            cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno].start_pos = p_stream_tell(p_stream);
    }*/
    /*}*/
    /* UniPG>> */
#ifdef USE_JPWL
    /* update markers struct */
    /*OPJ_BOOL res = j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOD, p_j2k->sod_start, 2);
    */
    assert(0 && "TODO");
#endif /* USE_JPWL */
    /* <<UniPG */
    /*}*/
    /* << INDEX */

    if (p_j2k->m_specific_param.m_encoder.m_current_tile_part_number == 0) {
        p_tile_coder->tcd_image->tiles->packno = 0;
#ifdef deadcode
        if (l_cstr_info) {
            l_cstr_info->packno = 0;
        }
#endif
    }

    *p_data_written = 0;

    if (p_j2k->m_specific_param.m_encoder.m_PLT) {
        marker_info = opj_tcd_marker_info_create(
                          p_j2k->m_specific_param.m_encoder.m_PLT);
        if (marker_info == NULL) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Cannot encode tile: opj_tcd_marker_info_create() failed\n");
            return OPJ_FALSE;
        }
    }

    if (l_remaining_data <
            p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough bytes in output buffer to write SOD marker\n");
        opj_tcd_marker_info_destroy(marker_info);
        return OPJ_FALSE;
    }
    l_remaining_data -= p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT;

    if (! opj_tcd_encode_tile(p_tile_coder, p_j2k->m_current_tile_number,
                              p_data + 2,
                              p_data_written, l_remaining_data, l_cstr_info,
                              marker_info,
                              p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Cannot encode tile\n");
        opj_tcd_marker_info_destroy(marker_info);
        return OPJ_FALSE;
    }

    /* For SOD */
    *p_data_written += 2;

    if (p_j2k->m_specific_param.m_encoder.m_PLT) {
        OPJ_UINT32 l_data_written_PLT = 0;
        OPJ_BYTE* p_PLT_buffer = (OPJ_BYTE*)opj_malloc(
                                     p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT);
        if (!p_PLT_buffer) {
            opj_event_msg(p_manager, EVT_ERROR, "Cannot allocate memory\n");
            opj_tcd_marker_info_destroy(marker_info);
            return OPJ_FALSE;
        }
        if (!opj_j2k_write_plt_in_memory(p_j2k,
                                         marker_info,
                                         p_PLT_buffer,
                                         &l_data_written_PLT,
                                         p_manager)) {
            opj_tcd_marker_info_destroy(marker_info);
            opj_free(p_PLT_buffer);
            return OPJ_FALSE;
        }

        assert(l_data_written_PLT <=
               p_j2k->m_specific_param.m_encoder.m_reserved_bytes_for_PLT);

        /* Move PLT marker(s) before SOD */
        memmove(p_data + l_data_written_PLT, p_data, *p_data_written);
        memcpy(p_data, p_PLT_buffer, l_data_written_PLT);
        opj_free(p_PLT_buffer);
        *p_data_written += l_data_written_PLT;
    }

    opj_tcd_marker_info_destroy(marker_info);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_sod(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_SIZE_T l_current_read_size;
    opj_codestream_index_t * l_cstr_index = 00;
    OPJ_BYTE ** l_current_data = 00;
    opj_tcp_t * l_tcp = 00;
    OPJ_UINT32 * l_tile_len = 00;
    OPJ_BOOL l_sot_length_pb_detected = OPJ_FALSE;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_tcp = &(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);

    if (p_j2k->m_specific_param.m_decoder.m_last_tile_part) {
        /* opj_stream_get_number_byte_left returns OPJ_OFF_T
        // but we are in the last tile part,
        // so its result will fit on OPJ_UINT32 unless we find
        // a file with a single tile part of more than 4 GB...*/
        p_j2k->m_specific_param.m_decoder.m_sot_length = (OPJ_UINT32)(
                    opj_stream_get_number_byte_left(p_stream) - 2);
    } else {
        /* Check to avoid pass the limit of OPJ_UINT32 */
        if (p_j2k->m_specific_param.m_decoder.m_sot_length >= 2) {
            p_j2k->m_specific_param.m_decoder.m_sot_length -= 2;
        } else {
            /* MSD: case commented to support empty SOT marker (PHR data) */
        }
    }

    l_current_data = &(l_tcp->m_data);
    l_tile_len = &l_tcp->m_data_size;

    /* Patch to support new PHR data */
    if (p_j2k->m_specific_param.m_decoder.m_sot_length) {
        /* If we are here, we'll try to read the data after allocation */
        /* Check enough bytes left in stream before allocation */
        if ((OPJ_OFF_T)p_j2k->m_specific_param.m_decoder.m_sot_length >
                opj_stream_get_number_byte_left(p_stream)) {
            if (p_j2k->m_cp.strict) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Tile part length size inconsistent with stream length\n");
                return OPJ_FALSE;
            } else {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Tile part length size inconsistent with stream length\n");
            }
        }
        if (p_j2k->m_specific_param.m_decoder.m_sot_length >
                UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "p_j2k->m_specific_param.m_decoder.m_sot_length > "
                          "UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA");
            return OPJ_FALSE;
        }
        /* Add a margin of OPJ_COMMON_CBLK_DATA_EXTRA to the allocation we */
        /* do so that opj_mqc_init_dec_common() can safely add a synthetic */
        /* 0xFFFF marker. */
        if (! *l_current_data) {
            /* LH: oddly enough, in this path, l_tile_len!=0.
             * TODO: If this was consistent, we could simplify the code to only use realloc(), as realloc(0,...) default to malloc(0,...).
             */
            *l_current_data = (OPJ_BYTE*) opj_malloc(
                                  p_j2k->m_specific_param.m_decoder.m_sot_length + OPJ_COMMON_CBLK_DATA_EXTRA);
        } else {
            OPJ_BYTE *l_new_current_data;
            if (*l_tile_len > UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA -
                    p_j2k->m_specific_param.m_decoder.m_sot_length) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "*l_tile_len > UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA - "
                              "p_j2k->m_specific_param.m_decoder.m_sot_length");
                return OPJ_FALSE;
            }

            l_new_current_data = (OPJ_BYTE *) opj_realloc(*l_current_data,
                                 *l_tile_len + p_j2k->m_specific_param.m_decoder.m_sot_length +
                                 OPJ_COMMON_CBLK_DATA_EXTRA);
            if (! l_new_current_data) {
                opj_free(*l_current_data);
                /*nothing more is done as l_current_data will be set to null, and just
                  afterward we enter in the error path
                  and the actual tile_len is updated (committed) at the end of the
                  function. */
            }
            *l_current_data = l_new_current_data;
        }

        if (*l_current_data == 00) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to decode tile\n");
            return OPJ_FALSE;
        }
    } else {
        l_sot_length_pb_detected = OPJ_TRUE;
    }

    /* Index */
    l_cstr_index = p_j2k->cstr_index;
    {
        OPJ_OFF_T l_current_pos = opj_stream_tell(p_stream) - 2;

        OPJ_UINT32 l_current_tile_part =
            l_cstr_index->tile_index[p_j2k->m_current_tile_number].current_tpsno;
        l_cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index[l_current_tile_part].end_header
            =
                l_current_pos;
        l_cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index[l_current_tile_part].end_pos
            =
                l_current_pos + p_j2k->m_specific_param.m_decoder.m_sot_length + 2;

        if (OPJ_FALSE == opj_j2k_add_tlmarker(p_j2k->m_current_tile_number,
                                              l_cstr_index,
                                              J2K_MS_SOD,
                                              l_current_pos,
                                              p_j2k->m_specific_param.m_decoder.m_sot_length + 2)) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n");
            return OPJ_FALSE;
        }

        /*l_cstr_index->packno = 0;*/
    }

    /* Patch to support new PHR data */
    if (!l_sot_length_pb_detected) {
        l_current_read_size = opj_stream_read_data(
                                  p_stream,
                                  *l_current_data + *l_tile_len,
                                  p_j2k->m_specific_param.m_decoder.m_sot_length,
                                  p_manager);
    } else {
        l_current_read_size = 0;
    }

    if (l_current_read_size != p_j2k->m_specific_param.m_decoder.m_sot_length) {
        if (l_current_read_size == (OPJ_SIZE_T)(-1)) {
            /* Avoid issue of https://github.com/uclouvain/openjpeg/issues/1533 */
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_NEOC;
    } else {
        p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_TPHSOT;
    }

    *l_tile_len += (OPJ_UINT32)l_current_read_size;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_rgn(opj_j2k_t *p_j2k,
                                  OPJ_UINT32 p_tile_no,
                                  OPJ_UINT32 p_comp_no,
                                  OPJ_UINT32 nb_comps,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_rgn_size;
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;
    OPJ_UINT32 l_comp_room;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp = &l_tcp->tccps[p_comp_no];

    if (nb_comps <= 256) {
        l_comp_room = 1;
    } else {
        l_comp_room = 2;
    }

    l_rgn_size = 6 + l_comp_room;

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_RGN,
                    2);                                   /* RGN  */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_rgn_size - 2,
                    2);                                 /* Lrgn */
    l_current_data += 2;

    opj_write_bytes(l_current_data, p_comp_no,
                    l_comp_room);                          /* Crgn */
    l_current_data += l_comp_room;

    opj_write_bytes(l_current_data, 0,
                    1);                                           /* Srgn */
    ++l_current_data;

    opj_write_bytes(l_current_data, (OPJ_UINT32)l_tccp->roishift,
                    1);                            /* SPrgn */
    ++l_current_data;

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_rgn_size,
                              p_manager) != l_rgn_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_eoc(opj_j2k_t *p_j2k,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager
                                 )
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_header_tile_data,
                    J2K_MS_EOC, 2);                                    /* EOC */

    /* UniPG>> */
#ifdef USE_JPWL
    /* update markers struct */
    /*
    OPJ_BOOL res = j2k_add_marker(p_j2k->cstr_info, J2K_MS_EOC, p_stream_tell(p_stream) - 2, 2);
    */
#endif /* USE_JPWL */

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, 2, p_manager) != 2) {
        return OPJ_FALSE;
    }

    if (! opj_stream_flush(p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a RGN marker (Region Of Interest)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_rgn(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_nb_comp;
    opj_image_t * l_image = 00;

    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_comp_room, l_comp_no, l_roi_sty;

    /* preconditions*/
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_image = p_j2k->m_private_image;
    l_nb_comp = l_image->numcomps;

    if (l_nb_comp <= 256) {
        l_comp_room = 1;
    } else {
        l_comp_room = 2;
    }

    if (p_header_size != 2 + l_comp_room) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading RGN marker\n");
        return OPJ_FALSE;
    }

    l_cp = &(p_j2k->m_cp);
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH) ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    opj_read_bytes(p_header_data, &l_comp_no, l_comp_room);         /* Crgn */
    p_header_data += l_comp_room;
    opj_read_bytes(p_header_data, &l_roi_sty,
                   1);                                     /* Srgn */
    ++p_header_data;

#ifdef USE_JPWL
    if (l_cp->correct) {
        /* totlen is negative or larger than the bytes left!!! */
        if (l_comp_room >= l_nb_comp) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "JPWL: bad component number in RGN (%d when there are only %d)\n",
                          l_comp_room, l_nb_comp);
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
        }
    };
#endif /* USE_JPWL */

    /* testcase 3635.pdf.asan.77.2930 */
    if (l_comp_no >= l_nb_comp) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "bad component number in RGN (%d when there are only %d)\n",
                      l_comp_no, l_nb_comp);
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data,
                   (OPJ_UINT32 *)(&(l_tcp->tccps[l_comp_no].roishift)), 1);  /* SPrgn */
    ++p_header_data;

    return OPJ_TRUE;

}

static OPJ_FLOAT32 opj_j2k_get_tp_stride(opj_tcp_t * p_tcp)
{
    return (OPJ_FLOAT32)((p_tcp->m_nb_tile_parts - 1) * 14);
}

static OPJ_FLOAT32 opj_j2k_get_default_stride(opj_tcp_t * p_tcp)
{
    (void)p_tcp;
    return 0;
}

static OPJ_BOOL opj_j2k_update_rates(opj_j2k_t *p_j2k,
                                     opj_stream_private_t *p_stream,
                                     opj_event_mgr_t * p_manager)
{
    opj_cp_t * l_cp = 00;
    opj_image_t * l_image = 00;
    opj_tcp_t * l_tcp = 00;
    opj_image_comp_t * l_img_comp = 00;

    OPJ_UINT32 i, j, k;
    OPJ_INT32 l_x0, l_y0, l_x1, l_y1;
    OPJ_FLOAT32 * l_rates = 0;
    OPJ_FLOAT32 l_sot_remove;
    OPJ_UINT32 l_bits_empty, l_size_pixel;
    OPJ_UINT64 l_tile_size = 0;
    OPJ_UINT32 l_last_res;
    OPJ_FLOAT32(* l_tp_stride_func)(opj_tcp_t *) = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_manager);

    l_cp = &(p_j2k->m_cp);
    l_image = p_j2k->m_private_image;
    l_tcp = l_cp->tcps;

    l_bits_empty = 8 * l_image->comps->dx * l_image->comps->dy;
    l_size_pixel = l_image->numcomps * l_image->comps->prec;
    l_sot_remove = (OPJ_FLOAT32) opj_stream_tell(p_stream) / (OPJ_FLOAT32)(
                       l_cp->th * l_cp->tw);

    if (l_cp->m_specific_param.m_enc.m_tp_on) {
        l_tp_stride_func = opj_j2k_get_tp_stride;
    } else {
        l_tp_stride_func = opj_j2k_get_default_stride;
    }

    for (i = 0; i < l_cp->th; ++i) {
        for (j = 0; j < l_cp->tw; ++j) {
            OPJ_FLOAT32 l_offset = (OPJ_FLOAT32)(*l_tp_stride_func)(l_tcp) /
                                   (OPJ_FLOAT32)l_tcp->numlayers;

            /* 4 borders of the tile rescale on the image if necessary */
            l_x0 = opj_int_max((OPJ_INT32)(l_cp->tx0 + j * l_cp->tdx),
                               (OPJ_INT32)l_image->x0);
            l_y0 = opj_int_max((OPJ_INT32)(l_cp->ty0 + i * l_cp->tdy),
                               (OPJ_INT32)l_image->y0);
            l_x1 = opj_int_min((OPJ_INT32)(l_cp->tx0 + (j + 1) * l_cp->tdx),
                               (OPJ_INT32)l_image->x1);
            l_y1 = opj_int_min((OPJ_INT32)(l_cp->ty0 + (i + 1) * l_cp->tdy),
                               (OPJ_INT32)l_image->y1);

            l_rates = l_tcp->rates;

            /* Modification of the RATE >> */
            for (k = 0; k < l_tcp->numlayers; ++k) {
                if (*l_rates > 0.0f) {
                    *l_rates = (OPJ_FLOAT32)(((OPJ_FLOAT64)l_size_pixel * (OPJ_UINT32)(
                                                  l_x1 - l_x0) *
                                              (OPJ_UINT32)(l_y1 - l_y0))
                                             / ((*l_rates) * (OPJ_FLOAT32)l_bits_empty))
                               -
                               l_offset;
                }

                ++l_rates;
            }

            ++l_tcp;

        }
    }

    l_tcp = l_cp->tcps;

    for (i = 0; i < l_cp->th; ++i) {
        for (j = 0; j < l_cp->tw; ++j) {
            l_rates = l_tcp->rates;

            if (*l_rates > 0.0f) {
                *l_rates -= l_sot_remove;

                if (*l_rates < 30.0f) {
                    *l_rates = 30.0f;
                }
            }

            ++l_rates;

            l_last_res = l_tcp->numlayers - 1;

            for (k = 1; k < l_last_res; ++k) {

                if (*l_rates > 0.0f) {
                    *l_rates -= l_sot_remove;

                    if (*l_rates < * (l_rates - 1) + 10.0f) {
                        *l_rates  = (*(l_rates - 1)) + 20.0f;
                    }
                }

                ++l_rates;
            }

            if (*l_rates > 0.0f) {
                *l_rates -= (l_sot_remove + 2.f);

                if (*l_rates < * (l_rates - 1) + 10.0f) {
                    *l_rates  = (*(l_rates - 1)) + 20.0f;
                }
            }

            ++l_tcp;
        }
    }

    l_img_comp = l_image->comps;
    l_tile_size = 0;

    for (i = 0; i < l_image->numcomps; ++i) {
        l_tile_size += (OPJ_UINT64)opj_uint_ceildiv(l_cp->tdx, l_img_comp->dx)
                       *
                       opj_uint_ceildiv(l_cp->tdy, l_img_comp->dy)
                       *
                       l_img_comp->prec;

        ++l_img_comp;
    }

    /* TODO: where does this magic value come from ? */
    /* This used to be 1.3 / 8, but with random data and very small code */
    /* block sizes, this is not enough. For example with */
    /* bin/test_tile_encoder 1 256 256 32 32 8 0 reversible_with_precinct.j2k 4 4 3 0 0 1 16 16 */
    /* TODO revise this to take into account the overhead linked to the */
    /* number of packets and number of code blocks in packets */
    l_tile_size = (OPJ_UINT64)((double)l_tile_size * 1.4 / 8);

    /* Arbitrary amount to make the following work: */
    /* bin/test_tile_encoder 1 256 256 17 16 8 0 reversible_no_precinct.j2k 4 4 3 0 0 1 */
    l_tile_size += 500;

    l_tile_size += opj_j2k_get_specific_header_sizes(p_j2k);

    if (l_tile_size > UINT_MAX) {
        l_tile_size = UINT_MAX;
    }

    p_j2k->m_specific_param.m_encoder.m_encoded_tile_size = (OPJ_UINT32)l_tile_size;
    p_j2k->m_specific_param.m_encoder.m_encoded_tile_data =
        (OPJ_BYTE *) opj_malloc(p_j2k->m_specific_param.m_encoder.m_encoded_tile_size);
    if (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data == 00) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to allocate m_encoded_tile_data. %u MB required\n",
                      (OPJ_UINT32)(l_tile_size / 1024 / 1024));
        return OPJ_FALSE;
    }

    if (p_j2k->m_specific_param.m_encoder.m_TLM) {
        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer =
            (OPJ_BYTE *) opj_malloc(6 *
                                    p_j2k->m_specific_param.m_encoder.m_total_tile_parts);
        if (! p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer) {
            return OPJ_FALSE;
        }

        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current =
            p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer;
    }

    return OPJ_TRUE;
}

#if 0
static OPJ_BOOL opj_j2k_read_eoc(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i;
    opj_tcd_t * l_tcd = 00;
    OPJ_UINT32 l_nb_tiles;
    opj_tcp_t * l_tcp = 00;
    OPJ_BOOL l_success;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
    l_tcp = p_j2k->m_cp.tcps;

    l_tcd = opj_tcd_create(OPJ_TRUE);
    if (l_tcd == 00) {
        opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
        return OPJ_FALSE;
    }

    for (i = 0; i < l_nb_tiles; ++i) {
        if (l_tcp->m_data) {
            if (! opj_tcd_init_decode_tile(l_tcd, i)) {
                opj_tcd_destroy(l_tcd);
                opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
                return OPJ_FALSE;
            }

            l_success = opj_tcd_decode_tile(l_tcd, l_tcp->m_data, l_tcp->m_data_size, i,
                                            p_j2k->cstr_index);
            /* cleanup */

            if (! l_success) {
                p_j2k->m_specific_param.m_decoder.m_state |= J2K_STATE_ERR;
                break;
            }
        }

        opj_j2k_tcp_destroy(l_tcp);
        ++l_tcp;
    }

    opj_tcd_destroy(l_tcd);
    return OPJ_TRUE;
}
#endif

static OPJ_BOOL opj_j2k_get_end_header(opj_j2k_t *p_j2k,
                                       struct opj_stream_private *p_stream,
                                       struct opj_event_mgr * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_manager);

    p_j2k->cstr_index->main_head_end = opj_stream_tell(p_stream);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_mct_data_group(opj_j2k_t *p_j2k,
        struct opj_stream_private *p_stream,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 i;
    opj_simple_mcc_decorrelation_data_t * l_mcc_record;
    opj_mct_data_t * l_mct_record;
    opj_tcp_t * l_tcp;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    if (! opj_j2k_write_cbd(p_j2k, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    l_tcp = &(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);
    l_mct_record = l_tcp->m_mct_records;

    for (i = 0; i < l_tcp->m_nb_mct_records; ++i) {

        if (! opj_j2k_write_mct_record(p_j2k, l_mct_record, p_stream, p_manager)) {
            return OPJ_FALSE;
        }

        ++l_mct_record;
    }

    l_mcc_record = l_tcp->m_mcc_records;

    for (i = 0; i < l_tcp->m_nb_mcc_records; ++i) {

        if (! opj_j2k_write_mcc_record(p_j2k, l_mcc_record, p_stream, p_manager)) {
            return OPJ_FALSE;
        }

        ++l_mcc_record;
    }

    if (! opj_j2k_write_mco(p_j2k, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_all_coc(
    opj_j2k_t *p_j2k,
    struct opj_stream_private *p_stream,
    struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 compno;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    for (compno = 1; compno < p_j2k->m_private_image->numcomps; ++compno) {
        /* cod is first component of first tile */
        if (! opj_j2k_compare_coc(p_j2k, 0, compno)) {
            if (! opj_j2k_write_coc(p_j2k, compno, p_stream, p_manager)) {
                return OPJ_FALSE;
            }
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_all_qcc(
    opj_j2k_t *p_j2k,
    struct opj_stream_private *p_stream,
    struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 compno;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    for (compno = 1; compno < p_j2k->m_private_image->numcomps; ++compno) {
        /* qcd is first component of first tile */
        if (! opj_j2k_compare_qcc(p_j2k, 0, compno)) {
            if (! opj_j2k_write_qcc(p_j2k, compno, p_stream, p_manager)) {
                return OPJ_FALSE;
            }
        }
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_regions(opj_j2k_t *p_j2k,
                                      struct opj_stream_private *p_stream,
                                      struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 compno;
    const opj_tccp_t *l_tccp = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_tccp = p_j2k->m_cp.tcps->tccps;

    for (compno = 0; compno < p_j2k->m_private_image->numcomps; ++compno)  {
        if (l_tccp->roishift) {

            if (! opj_j2k_write_rgn(p_j2k, 0, compno, p_j2k->m_private_image->numcomps,
                                    p_stream, p_manager)) {
                return OPJ_FALSE;
            }
        }

        ++l_tccp;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_epc(opj_j2k_t *p_j2k,
                                  struct opj_stream_private *p_stream,
                                  struct opj_event_mgr * p_manager)
{
    opj_codestream_index_t * l_cstr_index = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_manager);

    l_cstr_index = p_j2k->cstr_index;
    if (l_cstr_index) {
        l_cstr_index->codestream_size = (OPJ_UINT64)opj_stream_tell(p_stream);
        /* UniPG>> */
        /* The following adjustment is done to adjust the codestream size */
        /* if SOD is not at 0 in the buffer. Useful in case of JP2, where */
        /* the first bunch of bytes is not in the codestream              */
        l_cstr_index->codestream_size -= (OPJ_UINT64)l_cstr_index->main_head_start;
        /* <<UniPG */
    }

#ifdef USE_JPWL
    /* preparation of JPWL marker segments */
#if 0
    if (cp->epc_on) {

        /* encode according to JPWL */
        jpwl_encode(p_j2k, p_stream, image);

    }
#endif
    assert(0 && "TODO");
#endif /* USE_JPWL */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_unk(opj_j2k_t *p_j2k,
                                 opj_stream_private_t *p_stream,
                                 OPJ_UINT32 *output_marker,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_unknown_marker;
    const opj_dec_memory_marker_handler_t * l_marker_handler;
    OPJ_UINT32 l_size_unk = 2;

    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    opj_event_msg(p_manager, EVT_WARNING, "Unknown marker\n");

    for (;;) {
        /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer*/
        if (opj_stream_read_data(p_stream,
                                 p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        /* read 2 bytes as the new marker ID*/
        opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                       &l_unknown_marker, 2);

        if (!(l_unknown_marker < 0xff00)) {

            /* Get the marker handler from the marker ID*/
            l_marker_handler = opj_j2k_get_marker_handler(l_unknown_marker);

            if (!(p_j2k->m_specific_param.m_decoder.m_state & l_marker_handler->states)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Marker is not compliant with its position\n");
                return OPJ_FALSE;
            } else {
                if (l_marker_handler->id != J2K_MS_UNK) {
                    /* Add the marker to the codestream index*/
                    if (l_marker_handler->id != J2K_MS_SOT) {
                        OPJ_BOOL res = opj_j2k_add_mhmarker(p_j2k->cstr_index, J2K_MS_UNK,
                                                            (OPJ_UINT32) opj_stream_tell(p_stream) - l_size_unk,
                                                            l_size_unk);
                        if (res == OPJ_FALSE) {
                            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n");
                            return OPJ_FALSE;
                        }
                    }
                    break; /* next marker is known and well located */
                } else {
                    l_size_unk += 2;
                }
            }
        }
    }

    *output_marker = l_marker_handler->id ;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_mct_record(opj_j2k_t *p_j2k,
        opj_mct_data_t * p_mct_record,
        struct opj_stream_private *p_stream,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 l_mct_size;
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_tmp;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_mct_size = 10 + p_mct_record->m_data_size;

    if (l_mct_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mct_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write MCT marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mct_size;
    }

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_MCT,
                    2);                                   /* MCT */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_mct_size - 2,
                    2);                                 /* Lmct */
    l_current_data += 2;

    opj_write_bytes(l_current_data, 0,
                    2);                                                    /* Zmct */
    l_current_data += 2;

    /* only one marker atm */
    l_tmp = (p_mct_record->m_index & 0xff) | (p_mct_record->m_array_type << 8) |
            (p_mct_record->m_element_type << 10);

    opj_write_bytes(l_current_data, l_tmp, 2);
    l_current_data += 2;

    opj_write_bytes(l_current_data, 0,
                    2);                                                    /* Ymct */
    l_current_data += 2;

    memcpy(l_current_data, p_mct_record->m_data, p_mct_record->m_data_size);

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mct_size,
                              p_manager) != l_mct_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param       p_header_data   the data contained in the MCT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCT marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_mct(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 i;
    opj_tcp_t *l_tcp = 00;
    OPJ_UINT32 l_tmp;
    OPJ_UINT32 l_indix;
    opj_mct_data_t * l_mct_data;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);

    l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH ?
            &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    if (p_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
        return OPJ_FALSE;
    }

    /* first marker */
    opj_read_bytes(p_header_data, &l_tmp, 2);                       /* Zmct */
    p_header_data += 2;
    if (l_tmp != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge mct data within multiple MCT records\n");
        return OPJ_TRUE;
    }

    if (p_header_size <= 6) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
        return OPJ_FALSE;
    }

    /* Imct -> no need for other values, take the first, type is double with decorrelation x0000 1101 0000 0000*/
    opj_read_bytes(p_header_data, &l_tmp, 2);                       /* Imct */
    p_header_data += 2;

    l_indix = l_tmp & 0xff;
    l_mct_data = l_tcp->m_mct_records;

    for (i = 0; i < l_tcp->m_nb_mct_records; ++i) {
        if (l_mct_data->m_index == l_indix) {
            break;
        }
        ++l_mct_data;
    }

    /* NOT FOUND */
    if (i == l_tcp->m_nb_mct_records) {
        if (l_tcp->m_nb_mct_records == l_tcp->m_nb_max_mct_records) {
            opj_mct_data_t *new_mct_records;
            l_tcp->m_nb_max_mct_records += OPJ_J2K_MCT_DEFAULT_NB_RECORDS;

            new_mct_records = (opj_mct_data_t *) opj_realloc(l_tcp->m_mct_records,
                              l_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
            if (! new_mct_records) {
                opj_free(l_tcp->m_mct_records);
                l_tcp->m_mct_records = NULL;
                l_tcp->m_nb_max_mct_records = 0;
                l_tcp->m_nb_mct_records = 0;
                opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read MCT marker\n");
                return OPJ_FALSE;
            }

            /* Update m_mcc_records[].m_offset_array and m_decorrelation_array
             * to point to the new addresses */
            if (new_mct_records != l_tcp->m_mct_records) {
                for (i = 0; i < l_tcp->m_nb_mcc_records; ++i) {
                    opj_simple_mcc_decorrelation_data_t* l_mcc_record =
                        &(l_tcp->m_mcc_records[i]);
                    if (l_mcc_record->m_decorrelation_array) {
                        l_mcc_record->m_decorrelation_array =
                            new_mct_records +
                            (l_mcc_record->m_decorrelation_array -
                             l_tcp->m_mct_records);
                    }
                    if (l_mcc_record->m_offset_array) {
                        l_mcc_record->m_offset_array =
                            new_mct_records +
                            (l_mcc_record->m_offset_array -
                             l_tcp->m_mct_records);
                    }
                }
            }

            l_tcp->m_mct_records = new_mct_records;
            l_mct_data = l_tcp->m_mct_records + l_tcp->m_nb_mct_records;
            memset(l_mct_data, 0, (l_tcp->m_nb_max_mct_records - l_tcp->m_nb_mct_records) *
                   sizeof(opj_mct_data_t));
        }

        l_mct_data = l_tcp->m_mct_records + l_tcp->m_nb_mct_records;
        ++l_tcp->m_nb_mct_records;
    }

    if (l_mct_data->m_data) {
        opj_free(l_mct_data->m_data);
        l_mct_data->m_data = 00;
        l_mct_data->m_data_size = 0;
    }

    l_mct_data->m_index = l_indix;
    l_mct_data->m_array_type = (J2K_MCT_ARRAY_TYPE)((l_tmp  >> 8) & 3);
    l_mct_data->m_element_type = (J2K_MCT_ELEMENT_TYPE)((l_tmp  >> 10) & 3);

    opj_read_bytes(p_header_data, &l_tmp, 2);                       /* Ymct */
    p_header_data += 2;
    if (l_tmp != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge multiple MCT markers\n");
        return OPJ_TRUE;
    }

    p_header_size -= 6;

    l_mct_data->m_data = (OPJ_BYTE*)opj_malloc(p_header_size);
    if (! l_mct_data->m_data) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
        return OPJ_FALSE;
    }
    memcpy(l_mct_data->m_data, p_header_data, p_header_size);

    l_mct_data->m_data_size = p_header_size;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_mcc_record(opj_j2k_t *p_j2k,
        struct opj_simple_mcc_decorrelation_data * p_mcc_record,
        struct opj_stream_private *p_stream,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_mcc_size;
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_nb_bytes_for_comp;
    OPJ_UINT32 l_mask;
    OPJ_UINT32 l_tmcc;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    if (p_mcc_record->m_nb_comps > 255) {
        l_nb_bytes_for_comp = 2;
        l_mask = 0x8000;
    } else {
        l_nb_bytes_for_comp = 1;
        l_mask = 0;
    }

    l_mcc_size = p_mcc_record->m_nb_comps * 2 * l_nb_bytes_for_comp + 19;
    if (l_mcc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mcc_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write MCC marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mcc_size;
    }

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_MCC,
                    2);                                   /* MCC */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_mcc_size - 2,
                    2);                                 /* Lmcc */
    l_current_data += 2;

    /* first marker */
    opj_write_bytes(l_current_data, 0,
                    2);                                  /* Zmcc */
    l_current_data += 2;

    opj_write_bytes(l_current_data, p_mcc_record->m_index,
                    1);                                        /* Imcc -> no need for other values, take the first */
    ++l_current_data;

    /* only one marker atm */
    opj_write_bytes(l_current_data, 0,
                    2);                                  /* Ymcc */
    l_current_data += 2;

    opj_write_bytes(l_current_data, 1,
                    2);                                  /* Qmcc -> number of collections -> 1 */
    l_current_data += 2;

    opj_write_bytes(l_current_data, 0x1,
                    1);                                /* Xmcci type of component transformation -> array based decorrelation */
    ++l_current_data;

    opj_write_bytes(l_current_data, p_mcc_record->m_nb_comps | l_mask,
                    2);  /* Nmcci number of input components involved and size for each component offset = 8 bits */
    l_current_data += 2;

    for (i = 0; i < p_mcc_record->m_nb_comps; ++i) {
        opj_write_bytes(l_current_data, i,
                        l_nb_bytes_for_comp);                          /* Cmccij Component offset*/
        l_current_data += l_nb_bytes_for_comp;
    }

    opj_write_bytes(l_current_data, p_mcc_record->m_nb_comps | l_mask,
                    2);  /* Mmcci number of output components involved and size for each component offset = 8 bits */
    l_current_data += 2;

    for (i = 0; i < p_mcc_record->m_nb_comps; ++i) {
        opj_write_bytes(l_current_data, i,
                        l_nb_bytes_for_comp);                          /* Wmccij Component offset*/
        l_current_data += l_nb_bytes_for_comp;
    }

    l_tmcc = ((!p_mcc_record->m_is_irreversible) & 1U) << 16;

    if (p_mcc_record->m_decorrelation_array) {
        l_tmcc |= p_mcc_record->m_decorrelation_array->m_index;
    }

    if (p_mcc_record->m_offset_array) {
        l_tmcc |= ((p_mcc_record->m_offset_array->m_index) << 8);
    }

    opj_write_bytes(l_current_data, l_tmcc,
                    3);     /* Tmcci : use MCT defined as number 1 and irreversible array based. */
    l_current_data += 3;

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mcc_size,
                              p_manager) != l_mcc_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_mcc(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, j;
    OPJ_UINT32 l_tmp;
    OPJ_UINT32 l_indix;
    opj_tcp_t * l_tcp;
    opj_simple_mcc_decorrelation_data_t * l_mcc_record;
    opj_mct_data_t * l_mct_data;
    OPJ_UINT32 l_nb_collections;
    OPJ_UINT32 l_nb_comps;
    OPJ_UINT32 l_nb_bytes_by_comp;
    OPJ_BOOL l_new_mcc = OPJ_FALSE;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH ?
            &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    if (p_header_size < 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return OPJ_FALSE;
    }

    /* first marker */
    opj_read_bytes(p_header_data, &l_tmp, 2);                       /* Zmcc */
    p_header_data += 2;
    if (l_tmp != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge multiple data spanning\n");
        return OPJ_TRUE;
    }

    if (p_header_size < 7) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_indix,
                   1); /* Imcc -> no need for other values, take the first */
    ++p_header_data;

    l_mcc_record = l_tcp->m_mcc_records;

    for (i = 0; i < l_tcp->m_nb_mcc_records; ++i) {
        if (l_mcc_record->m_index == l_indix) {
            break;
        }
        ++l_mcc_record;
    }

    /** NOT FOUND */
    if (i == l_tcp->m_nb_mcc_records) {
        if (l_tcp->m_nb_mcc_records == l_tcp->m_nb_max_mcc_records) {
            opj_simple_mcc_decorrelation_data_t *new_mcc_records;
            l_tcp->m_nb_max_mcc_records += OPJ_J2K_MCC_DEFAULT_NB_RECORDS;

            new_mcc_records = (opj_simple_mcc_decorrelation_data_t *) opj_realloc(
                                  l_tcp->m_mcc_records, l_tcp->m_nb_max_mcc_records * sizeof(
                                      opj_simple_mcc_decorrelation_data_t));
            if (! new_mcc_records) {
                opj_free(l_tcp->m_mcc_records);
                l_tcp->m_mcc_records = NULL;
                l_tcp->m_nb_max_mcc_records = 0;
                l_tcp->m_nb_mcc_records = 0;
                opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read MCC marker\n");
                return OPJ_FALSE;
            }
            l_tcp->m_mcc_records = new_mcc_records;
            l_mcc_record = l_tcp->m_mcc_records + l_tcp->m_nb_mcc_records;
            memset(l_mcc_record, 0, (l_tcp->m_nb_max_mcc_records - l_tcp->m_nb_mcc_records)
                   * sizeof(opj_simple_mcc_decorrelation_data_t));
        }
        l_mcc_record = l_tcp->m_mcc_records + l_tcp->m_nb_mcc_records;
        l_new_mcc = OPJ_TRUE;
    }
    l_mcc_record->m_index = l_indix;

    /* only one marker atm */
    opj_read_bytes(p_header_data, &l_tmp, 2);                       /* Ymcc */
    p_header_data += 2;
    if (l_tmp != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge multiple data spanning\n");
        return OPJ_TRUE;
    }

    opj_read_bytes(p_header_data, &l_nb_collections,
                   2);                              /* Qmcc -> number of collections -> 1 */
    p_header_data += 2;

    if (l_nb_collections > 1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge multiple collections\n");
        return OPJ_TRUE;
    }

    p_header_size -= 7;

    for (i = 0; i < l_nb_collections; ++i) {
        if (p_header_size < 3) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
            return OPJ_FALSE;
        }

        opj_read_bytes(p_header_data, &l_tmp,
                       1); /* Xmcci type of component transformation -> array based decorrelation */
        ++p_header_data;

        if (l_tmp != 1) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Cannot take in charge collections other than array decorrelation\n");
            return OPJ_TRUE;
        }

        opj_read_bytes(p_header_data, &l_nb_comps, 2);

        p_header_data += 2;
        p_header_size -= 3;

        l_nb_bytes_by_comp = 1 + (l_nb_comps >> 15);
        l_mcc_record->m_nb_comps = l_nb_comps & 0x7fff;

        if (p_header_size < (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 2)) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
            return OPJ_FALSE;
        }

        p_header_size -= (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 2);

        for (j = 0; j < l_mcc_record->m_nb_comps; ++j) {
            opj_read_bytes(p_header_data, &l_tmp,
                           l_nb_bytes_by_comp);      /* Cmccij Component offset*/
            p_header_data += l_nb_bytes_by_comp;

            if (l_tmp != j) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Cannot take in charge collections with indix shuffle\n");
                return OPJ_TRUE;
            }
        }

        opj_read_bytes(p_header_data, &l_nb_comps, 2);
        p_header_data += 2;

        l_nb_bytes_by_comp = 1 + (l_nb_comps >> 15);
        l_nb_comps &= 0x7fff;

        if (l_nb_comps != l_mcc_record->m_nb_comps) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Cannot take in charge collections without same number of indixes\n");
            return OPJ_TRUE;
        }

        if (p_header_size < (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 3)) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
            return OPJ_FALSE;
        }

        p_header_size -= (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 3);

        for (j = 0; j < l_mcc_record->m_nb_comps; ++j) {
            opj_read_bytes(p_header_data, &l_tmp,
                           l_nb_bytes_by_comp);      /* Wmccij Component offset*/
            p_header_data += l_nb_bytes_by_comp;

            if (l_tmp != j) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "Cannot take in charge collections with indix shuffle\n");
                return OPJ_TRUE;
            }
        }

        opj_read_bytes(p_header_data, &l_tmp, 3); /* Wmccij Component offset*/
        p_header_data += 3;

        l_mcc_record->m_is_irreversible = !((l_tmp >> 16) & 1);
        l_mcc_record->m_decorrelation_array = 00;
        l_mcc_record->m_offset_array = 00;

        l_indix = l_tmp & 0xff;
        if (l_indix != 0) {
            l_mct_data = l_tcp->m_mct_records;
            for (j = 0; j < l_tcp->m_nb_mct_records; ++j) {
                if (l_mct_data->m_index == l_indix) {
                    l_mcc_record->m_decorrelation_array = l_mct_data;
                    break;
                }
                ++l_mct_data;
            }

            if (l_mcc_record->m_decorrelation_array == 00) {
                opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
                return OPJ_FALSE;
            }
        }

        l_indix = (l_tmp >> 8) & 0xff;
        if (l_indix != 0) {
            l_mct_data = l_tcp->m_mct_records;
            for (j = 0; j < l_tcp->m_nb_mct_records; ++j) {
                if (l_mct_data->m_index == l_indix) {
                    l_mcc_record->m_offset_array = l_mct_data;
                    break;
                }
                ++l_mct_data;
            }

            if (l_mcc_record->m_offset_array == 00) {
                opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
                return OPJ_FALSE;
            }
        }
    }

    if (p_header_size != 0) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return OPJ_FALSE;
    }

    if (l_new_mcc) {
        ++l_tcp->m_nb_mcc_records;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_mco(opj_j2k_t *p_j2k,
                                  struct opj_stream_private *p_stream,
                                  struct opj_event_mgr * p_manager
                                 )
{
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_mco_size;
    opj_tcp_t * l_tcp = 00;
    opj_simple_mcc_decorrelation_data_t * l_mcc_record;
    OPJ_UINT32 i;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_tcp = &(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);

    l_mco_size = 5 + l_tcp->m_nb_mcc_records;
    if (l_mco_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {

        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mco_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write MCO marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mco_size;
    }
    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;


    opj_write_bytes(l_current_data, J2K_MS_MCO, 2);                 /* MCO */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_mco_size - 2, 2);             /* Lmco */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_tcp->m_nb_mcc_records,
                    1);    /* Nmco : only one transform stage*/
    ++l_current_data;

    l_mcc_record = l_tcp->m_mcc_records;
    for (i = 0; i < l_tcp->m_nb_mcc_records; ++i) {
        opj_write_bytes(l_current_data, l_mcc_record->m_index,
                        1); /* Imco -> use the mcc indicated by 1*/
        ++l_current_data;
        ++l_mcc_record;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_mco_size,
                              p_manager) != l_mco_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param       p_header_data   the data contained in the MCO box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCO marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_mco(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_tmp, i;
    OPJ_UINT32 l_nb_stages;
    opj_tcp_t * l_tcp;
    opj_tccp_t * l_tccp;
    opj_image_t * l_image;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_image = p_j2k->m_private_image;
    l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH ?
            &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    if (p_header_size < 1) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCO marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_nb_stages,
                   1);                         /* Nmco : only one transform stage*/
    ++p_header_data;

    if (l_nb_stages > 1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Cannot take in charge multiple transformation stages.\n");
        return OPJ_TRUE;
    }

    if (p_header_size != l_nb_stages + 1) {
        opj_event_msg(p_manager, EVT_WARNING, "Error reading MCO marker\n");
        return OPJ_FALSE;
    }

    l_tccp = l_tcp->tccps;

    for (i = 0; i < l_image->numcomps; ++i) {
        l_tccp->m_dc_level_shift = 0;
        ++l_tccp;
    }

    if (l_tcp->m_mct_decoding_matrix) {
        opj_free(l_tcp->m_mct_decoding_matrix);
        l_tcp->m_mct_decoding_matrix = 00;
    }

    for (i = 0; i < l_nb_stages; ++i) {
        opj_read_bytes(p_header_data, &l_tmp, 1);
        ++p_header_data;

        if (! opj_j2k_add_mct(l_tcp, p_j2k->m_private_image, l_tmp)) {
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_add_mct(opj_tcp_t * p_tcp, opj_image_t * p_image,
                                OPJ_UINT32 p_index)
{
    OPJ_UINT32 i;
    opj_simple_mcc_decorrelation_data_t * l_mcc_record;
    opj_mct_data_t * l_deco_array, * l_offset_array;
    OPJ_UINT32 l_data_size, l_mct_size, l_offset_size;
    OPJ_UINT32 l_nb_elem;
    OPJ_UINT32 * l_offset_data, * l_current_offset_data;
    opj_tccp_t * l_tccp;

    /* preconditions */
    assert(p_tcp != 00);

    l_mcc_record = p_tcp->m_mcc_records;

    for (i = 0; i < p_tcp->m_nb_mcc_records; ++i) {
        if (l_mcc_record->m_index == p_index) {
            break;
        }
    }

    if (i == p_tcp->m_nb_mcc_records) {
        /** element discarded **/
        return OPJ_TRUE;
    }

    if (l_mcc_record->m_nb_comps != p_image->numcomps) {
        /** do not support number of comps != image */
        return OPJ_TRUE;
    }

    l_deco_array = l_mcc_record->m_decorrelation_array;

    if (l_deco_array) {
        l_data_size = MCT_ELEMENT_SIZE[l_deco_array->m_element_type] * p_image->numcomps
                      * p_image->numcomps;
        if (l_deco_array->m_data_size != l_data_size) {
            return OPJ_FALSE;
        }

        l_nb_elem = p_image->numcomps * p_image->numcomps;
        l_mct_size = l_nb_elem * (OPJ_UINT32)sizeof(OPJ_FLOAT32);
        p_tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(l_mct_size);

        if (! p_tcp->m_mct_decoding_matrix) {
            return OPJ_FALSE;
        }

        j2k_mct_read_functions_to_float[l_deco_array->m_element_type](
            l_deco_array->m_data, p_tcp->m_mct_decoding_matrix, l_nb_elem);
    }

    l_offset_array = l_mcc_record->m_offset_array;

    if (l_offset_array) {
        l_data_size = MCT_ELEMENT_SIZE[l_offset_array->m_element_type] *
                      p_image->numcomps;
        if (l_offset_array->m_data_size != l_data_size) {
            return OPJ_FALSE;
        }

        l_nb_elem = p_image->numcomps;
        l_offset_size = l_nb_elem * (OPJ_UINT32)sizeof(OPJ_UINT32);
        l_offset_data = (OPJ_UINT32*)opj_malloc(l_offset_size);

        if (! l_offset_data) {
            return OPJ_FALSE;
        }

        j2k_mct_read_functions_to_int32[l_offset_array->m_element_type](
            l_offset_array->m_data, l_offset_data, l_nb_elem);

        l_tccp = p_tcp->tccps;
        l_current_offset_data = l_offset_data;

        for (i = 0; i < p_image->numcomps; ++i) {
            l_tccp->m_dc_level_shift = (OPJ_INT32) * (l_current_offset_data++);
            ++l_tccp;
        }

        opj_free(l_offset_data);
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_cbd(opj_j2k_t *p_j2k,
                                  struct opj_stream_private *p_stream,
                                  struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_cbd_size;
    OPJ_BYTE * l_current_data = 00;
    opj_image_t *l_image = 00;
    opj_image_comp_t * l_comp = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    l_image = p_j2k->m_private_image;
    l_cbd_size = 6 + p_j2k->m_private_image->numcomps;

    if (l_cbd_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size) {
        OPJ_BYTE *new_header_tile_data = (OPJ_BYTE *) opj_realloc(
                                             p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_cbd_size);
        if (! new_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = NULL;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to write CBD marker\n");
            return OPJ_FALSE;
        }
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
        p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_cbd_size;
    }

    l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

    opj_write_bytes(l_current_data, J2K_MS_CBD, 2);                 /* CBD */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_cbd_size - 2, 2);             /* L_CBD */
    l_current_data += 2;

    opj_write_bytes(l_current_data, l_image->numcomps, 2);          /* Ncbd */
    l_current_data += 2;

    l_comp = l_image->comps;

    for (i = 0; i < l_image->numcomps; ++i) {
        opj_write_bytes(l_current_data, (l_comp->sgnd << 7) | (l_comp->prec - 1),
                        1);           /* Component bit depth */
        ++l_current_data;

        ++l_comp;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_header_tile_data, l_cbd_size,
                              p_manager) != l_cbd_size) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Reads a CBD marker (Component bit depth definition)
 * @param       p_header_data   the data contained in the CBD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CBD marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cbd(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    OPJ_UINT32 l_nb_comp, l_num_comp;
    OPJ_UINT32 l_comp_def;
    OPJ_UINT32 i;
    opj_image_comp_t * l_comp = 00;

    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    l_num_comp = p_j2k->m_private_image->numcomps;

    if (p_header_size != (p_j2k->m_private_image->numcomps + 2)) {
        opj_event_msg(p_manager, EVT_ERROR, "Crror reading CBD marker\n");
        return OPJ_FALSE;
    }

    opj_read_bytes(p_header_data, &l_nb_comp,
                   2);                           /* Ncbd */
    p_header_data += 2;

    if (l_nb_comp != l_num_comp) {
        opj_event_msg(p_manager, EVT_ERROR, "Crror reading CBD marker\n");
        return OPJ_FALSE;
    }

    l_comp = p_j2k->m_private_image->comps;
    for (i = 0; i < l_num_comp; ++i) {
        opj_read_bytes(p_header_data, &l_comp_def,
                       1);                  /* Component bit depth */
        ++p_header_data;
        l_comp->sgnd = (l_comp_def >> 7) & 1;
        l_comp->prec = (l_comp_def & 0x7f) + 1;

        if (l_comp->prec > 31) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : prec=%u (should be between 1 and 38 according to the JPEG2000 norm. OpenJpeg only supports up to 31)\n",
                          i, l_comp->prec);
            return OPJ_FALSE;
        }
        ++l_comp;
    }

    return OPJ_TRUE;
}

/**
 * Reads a CAP marker (extended capabilities definition). Empty implementation.
 * Found in HTJ2K files.
 *
 * @param       p_header_data   the data contained in the CAP box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CAP marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cap(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    (void)p_j2k;
    (void)p_header_data;
    (void)p_header_size;
    (void)p_manager;

    return OPJ_TRUE;
}

/**
 * Reads a CPF marker (corresponding profile). Empty implementation. Found in HTJ2K files
 * @param       p_header_data   the data contained in the CPF box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CPF marker.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_read_cpf(opj_j2k_t *p_j2k,
                                 OPJ_BYTE * p_header_data,
                                 OPJ_UINT32 p_header_size,
                                 opj_event_mgr_t * p_manager
                                )
{
    /* preconditions */
    assert(p_header_data != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    (void)p_j2k;
    (void)p_header_data;
    (void)p_header_size;
    (void)p_manager;

    return OPJ_TRUE;
}

/* ----------------------------------------------------------------------- */
/* J2K / JPT decoder interface                                             */
/* ----------------------------------------------------------------------- */

void opj_j2k_setup_decoder(opj_j2k_t *j2k, opj_dparameters_t *parameters)
{
    if (j2k && parameters) {
        j2k->m_cp.m_specific_param.m_dec.m_layer = parameters->cp_layer;
        j2k->m_cp.m_specific_param.m_dec.m_reduce = parameters->cp_reduce;

        j2k->dump_state = (parameters->flags & OPJ_DPARAMETERS_DUMP_FLAG);
#ifdef USE_JPWL
        j2k->m_cp.correct = parameters->jpwl_correct;
        j2k->m_cp.exp_comps = parameters->jpwl_exp_comps;
        j2k->m_cp.max_tiles = parameters->jpwl_max_tiles;
#endif /* USE_JPWL */
    }
}

void opj_j2k_decoder_set_strict_mode(opj_j2k_t *j2k, OPJ_BOOL strict)
{
    if (j2k) {
        j2k->m_cp.strict = strict;
        if (strict) {
            j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction_checked = 1;
        }
    }
}

OPJ_BOOL opj_j2k_set_threads(opj_j2k_t *j2k, OPJ_UINT32 num_threads)
{
    /* Currently we pass the thread-pool to the tcd, so we cannot re-set it */
    /* afterwards */
    if (opj_has_thread_support() && j2k->m_tcd == NULL) {
        opj_thread_pool_destroy(j2k->m_tp);
        j2k->m_tp = NULL;
        if (num_threads <= (OPJ_UINT32)INT_MAX) {
            j2k->m_tp = opj_thread_pool_create((int)num_threads);
        }
        if (j2k->m_tp == NULL) {
            j2k->m_tp = opj_thread_pool_create(0);
            return OPJ_FALSE;
        }
        return OPJ_TRUE;
    }
    return OPJ_FALSE;
}

static int opj_j2k_get_default_thread_count(void)
{
    const char* num_threads_str = getenv("OPJ_NUM_THREADS");
    int num_cpus;
    int num_threads;

    if (num_threads_str == NULL || !opj_has_thread_support()) {
        return 0;
    }
    num_cpus = opj_get_num_cpus();
    if (strcmp(num_threads_str, "ALL_CPUS") == 0) {
        return num_cpus;
    }
    if (num_cpus == 0) {
        num_cpus = 32;
    }
    num_threads = atoi(num_threads_str);
    if (num_threads < 0) {
        num_threads = 0;
    } else if (num_threads > 2 * num_cpus) {
        num_threads = 2 * num_cpus;
    }
    return num_threads;
}

/* ----------------------------------------------------------------------- */
/* J2K encoder interface                                                       */
/* ----------------------------------------------------------------------- */

opj_j2k_t* opj_j2k_create_compress(void)
{
    opj_j2k_t *l_j2k = (opj_j2k_t*) opj_calloc(1, sizeof(opj_j2k_t));
    if (!l_j2k) {
        return NULL;
    }


    l_j2k->m_is_decoder = 0;
    l_j2k->m_cp.m_is_decoder = 0;

    l_j2k->m_specific_param.m_encoder.m_header_tile_data = (OPJ_BYTE *) opj_malloc(
                OPJ_J2K_DEFAULT_HEADER_SIZE);
    if (! l_j2k->m_specific_param.m_encoder.m_header_tile_data) {
        opj_j2k_destroy(l_j2k);
        return NULL;
    }

    l_j2k->m_specific_param.m_encoder.m_header_tile_data_size =
        OPJ_J2K_DEFAULT_HEADER_SIZE;

    /* validation list creation*/
    l_j2k->m_validation_list = opj_procedure_list_create();
    if (! l_j2k->m_validation_list) {
        opj_j2k_destroy(l_j2k);
        return NULL;
    }

    /* execution list creation*/
    l_j2k->m_procedure_list = opj_procedure_list_create();
    if (! l_j2k->m_procedure_list) {
        opj_j2k_destroy(l_j2k);
        return NULL;
    }

    l_j2k->m_tp = opj_thread_pool_create(opj_j2k_get_default_thread_count());
    if (!l_j2k->m_tp) {
        l_j2k->m_tp = opj_thread_pool_create(0);
    }
    if (!l_j2k->m_tp) {
        opj_j2k_destroy(l_j2k);
        return NULL;
    }

    return l_j2k;
}

static int opj_j2k_initialise_4K_poc(opj_poc_t *POC, int numres)
{
    POC[0].tile  = 1;
    POC[0].resno0  = 0;
    POC[0].compno0 = 0;
    POC[0].layno1  = 1;
    POC[0].resno1  = (OPJ_UINT32)(numres - 1);
    POC[0].compno1 = 3;
    POC[0].prg1 = OPJ_CPRL;
    POC[1].tile  = 1;
    POC[1].resno0  = (OPJ_UINT32)(numres - 1);
    POC[1].compno0 = 0;
    POC[1].layno1  = 1;
    POC[1].resno1  = (OPJ_UINT32)numres;
    POC[1].compno1 = 3;
    POC[1].prg1 = OPJ_CPRL;
    return 2;
}

static void opj_j2k_set_cinema_parameters(opj_cparameters_t *parameters,
        opj_image_t *image, opj_event_mgr_t *p_manager)
{
    /* Configure cinema parameters */
    int i;

    /* No tiling */
    parameters->tile_size_on = OPJ_FALSE;
    parameters->cp_tdx = 1;
    parameters->cp_tdy = 1;

    /* One tile part for each component */
    parameters->tp_flag = 'C';
    parameters->tp_on = 1;

    /* Tile and Image shall be at (0,0) */
    parameters->cp_tx0 = 0;
    parameters->cp_ty0 = 0;
    parameters->image_offset_x0 = 0;
    parameters->image_offset_y0 = 0;

    /* Codeblock size= 32*32 */
    parameters->cblockw_init = 32;
    parameters->cblockh_init = 32;

    /* Codeblock style: no mode switch enabled */
    parameters->mode = 0;

    /* No ROI */
    parameters->roi_compno = -1;

    /* No subsampling */
    parameters->subsampling_dx = 1;
    parameters->subsampling_dy = 1;

    /* 9-7 transform */
    parameters->irreversible = 1;

    /* Number of layers */
    if (parameters->tcp_numlayers > 1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n"
                      "1 single quality layer"
                      "-> Number of layers forced to 1 (rather than %d)\n"
                      "-> Rate of the last layer (%3.1f) will be used",
                      parameters->tcp_numlayers,
                      parameters->tcp_rates[parameters->tcp_numlayers - 1]);
        parameters->tcp_rates[0] = parameters->tcp_rates[parameters->tcp_numlayers - 1];
        parameters->tcp_numlayers = 1;
    }

    /* Resolution levels */
    switch (parameters->rsiz) {
    case OPJ_PROFILE_CINEMA_2K:
        if (parameters->numresolution > 6) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-3 (2k dc profile) requires:\n"
                          "Number of decomposition levels <= 5\n"
                          "-> Number of decomposition levels forced to 5 (rather than %d)\n",
                          parameters->numresolution + 1);
            parameters->numresolution = 6;
        }
        break;
    case OPJ_PROFILE_CINEMA_4K:
        if (parameters->numresolution < 2) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-4 (4k dc profile) requires:\n"
                          "Number of decomposition levels >= 1 && <= 6\n"
                          "-> Number of decomposition levels forced to 1 (rather than %d)\n",
                          parameters->numresolution + 1);
            parameters->numresolution = 1;
        } else if (parameters->numresolution > 7) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-4 (4k dc profile) requires:\n"
                          "Number of decomposition levels >= 1 && <= 6\n"
                          "-> Number of decomposition levels forced to 6 (rather than %d)\n",
                          parameters->numresolution + 1);
            parameters->numresolution = 7;
        }
        break;
    default :
        break;
    }

    /* Precincts */
    parameters->csty |= J2K_CP_CSTY_PRT;
    if (parameters->numresolution == 1) {
        parameters->res_spec = 1;
        parameters->prcw_init[0] = 128;
        parameters->prch_init[0] = 128;
    } else {
        parameters->res_spec = parameters->numresolution - 1;
        for (i = 0; i < parameters->res_spec; i++) {
            parameters->prcw_init[i] = 256;
            parameters->prch_init[i] = 256;
        }
    }

    /* The progression order shall be CPRL */
    parameters->prog_order = OPJ_CPRL;

    /* Progression order changes for 4K, disallowed for 2K */
    if (parameters->rsiz == OPJ_PROFILE_CINEMA_4K) {
        parameters->numpocs = (OPJ_UINT32)opj_j2k_initialise_4K_poc(parameters->POC,
                              parameters->numresolution);
    } else {
        parameters->numpocs = 0;
    }

    /* Limited bit-rate */
    parameters->cp_disto_alloc = 1;
    if (parameters->max_cs_size <= 0) {
        /* No rate has been introduced, 24 fps is assumed */
        parameters->max_cs_size = OPJ_CINEMA_24_CS;
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n"
                      "Maximum 1302083 compressed bytes @ 24fps\n"
                      "As no rate has been given, this limit will be used.\n");
    } else if (parameters->max_cs_size > OPJ_CINEMA_24_CS) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n"
                      "Maximum 1302083 compressed bytes @ 24fps\n"
                      "-> Specified rate exceeds this limit. Rate will be forced to 1302083 bytes.\n");
        parameters->max_cs_size = OPJ_CINEMA_24_CS;
    }

    if (parameters->max_comp_size <= 0) {
        /* No rate has been introduced, 24 fps is assumed */
        parameters->max_comp_size = OPJ_CINEMA_24_COMP;
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n"
                      "Maximum 1041666 compressed bytes @ 24fps\n"
                      "As no rate has been given, this limit will be used.\n");
    } else if (parameters->max_comp_size > OPJ_CINEMA_24_COMP) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n"
                      "Maximum 1041666 compressed bytes @ 24fps\n"
                      "-> Specified rate exceeds this limit. Rate will be forced to 1041666 bytes.\n");
        parameters->max_comp_size = OPJ_CINEMA_24_COMP;
    }

    parameters->tcp_rates[0] = (OPJ_FLOAT32)(image->numcomps * image->comps[0].w *
                               image->comps[0].h * image->comps[0].prec) /
                               (OPJ_FLOAT32)(((OPJ_UINT32)parameters->max_cs_size) * 8 * image->comps[0].dx *
                                       image->comps[0].dy);

}

static OPJ_BOOL opj_j2k_is_cinema_compliant(opj_image_t *image, OPJ_UINT16 rsiz,
        opj_event_mgr_t *p_manager)
{
    OPJ_UINT32 i;

    /* Number of components */
    if (image->numcomps != 3) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 (2k dc profile) requires:\n"
                      "3 components"
                      "-> Number of components of input image (%d) is not compliant\n"
                      "-> Non-profile-3 codestream will be generated\n",
                      image->numcomps);
        return OPJ_FALSE;
    }

    /* Bitdepth */
    for (i = 0; i < image->numcomps; i++) {
        if ((image->comps[i].prec != 12) | (image->comps[i].sgnd)) {
            char signed_str[] = "signed";
            char unsigned_str[] = "unsigned";
            char *tmp_str = image->comps[i].sgnd ? signed_str : unsigned_str;
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-3 (2k dc profile) requires:\n"
                          "Precision of each component shall be 12 bits unsigned"
                          "-> At least component %d of input image (%d bits, %s) is not compliant\n"
                          "-> Non-profile-3 codestream will be generated\n",
                          i, image->comps[i].prec, tmp_str);
            return OPJ_FALSE;
        }
    }

    /* Image size */
    switch (rsiz) {
    case OPJ_PROFILE_CINEMA_2K:
        if (((image->comps[0].w > 2048) | (image->comps[0].h > 1080))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-3 (2k dc profile) requires:\n"
                          "width <= 2048 and height <= 1080\n"
                          "-> Input image size %d x %d is not compliant\n"
                          "-> Non-profile-3 codestream will be generated\n",
                          image->comps[0].w, image->comps[0].h);
            return OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_CINEMA_4K:
        if (((image->comps[0].w > 4096) | (image->comps[0].h > 2160))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-4 (4k dc profile) requires:\n"
                          "width <= 4096 and height <= 2160\n"
                          "-> Image size %d x %d is not compliant\n"
                          "-> Non-profile-4 codestream will be generated\n",
                          image->comps[0].w, image->comps[0].h);
            return OPJ_FALSE;
        }
        break;
    default :
        break;
    }

    return OPJ_TRUE;
}

static int opj_j2k_get_imf_max_NL(opj_cparameters_t *parameters,
                                  opj_image_t *image)
{
    /* Decomposition levels */
    const OPJ_UINT16 rsiz = parameters->rsiz;
    const OPJ_UINT16 profile = OPJ_GET_IMF_PROFILE(rsiz);
    const OPJ_UINT32 XTsiz = parameters->tile_size_on ? (OPJ_UINT32)
                             parameters->cp_tdx : image->x1;
    switch (profile) {
    case OPJ_PROFILE_IMF_2K:
        return 5;
    case OPJ_PROFILE_IMF_4K:
        return 6;
    case OPJ_PROFILE_IMF_8K:
        return 7;
    case OPJ_PROFILE_IMF_2K_R: {
        if (XTsiz >= 2048) {
            return 5;
        } else if (XTsiz >= 1024) {
            return 4;
        }
        break;
    }
    case OPJ_PROFILE_IMF_4K_R: {
        if (XTsiz >= 4096) {
            return 6;
        } else if (XTsiz >= 2048) {
            return 5;
        } else if (XTsiz >= 1024) {
            return 4;
        }
        break;
    }
    case OPJ_PROFILE_IMF_8K_R: {
        if (XTsiz >= 8192) {
            return 7;
        } else if (XTsiz >= 4096) {
            return 6;
        } else if (XTsiz >= 2048) {
            return 5;
        } else if (XTsiz >= 1024) {
            return 4;
        }
        break;
    }
    default:
        break;
    }
    return -1;
}

static void opj_j2k_set_imf_parameters(opj_cparameters_t *parameters,
                                       opj_image_t *image, opj_event_mgr_t *p_manager)
{
    const OPJ_UINT16 rsiz = parameters->rsiz;
    const OPJ_UINT16 profile = OPJ_GET_IMF_PROFILE(rsiz);

    OPJ_UNUSED(p_manager);

    /* Override defaults set by opj_set_default_encoder_parameters */
    if (parameters->cblockw_init == OPJ_COMP_PARAM_DEFAULT_CBLOCKW &&
            parameters->cblockh_init == OPJ_COMP_PARAM_DEFAULT_CBLOCKH) {
        parameters->cblockw_init = 32;
        parameters->cblockh_init = 32;
    }

    /* One tile part for each component */
    parameters->tp_flag = 'C';
    parameters->tp_on = 1;

    if (parameters->prog_order == OPJ_COMP_PARAM_DEFAULT_PROG_ORDER) {
        parameters->prog_order = OPJ_CPRL;
    }

    if (profile == OPJ_PROFILE_IMF_2K ||
            profile == OPJ_PROFILE_IMF_4K ||
            profile == OPJ_PROFILE_IMF_8K) {
        /* 9-7 transform */
        parameters->irreversible = 1;
    }

    /* Adjust the number of resolutions if set to its defaults */
    if (parameters->numresolution == OPJ_COMP_PARAM_DEFAULT_NUMRESOLUTION &&
            image->x0 == 0 &&
            image->y0 == 0) {
        const int max_NL = opj_j2k_get_imf_max_NL(parameters, image);
        if (max_NL >= 0 && parameters->numresolution > max_NL) {
            parameters->numresolution = max_NL + 1;
        }

        /* Note: below is generic logic */
        if (!parameters->tile_size_on) {
            while (parameters->numresolution > 0) {
                if (image->x1 < (1U << ((OPJ_UINT32)parameters->numresolution - 1U))) {
                    parameters->numresolution --;
                    continue;
                }
                if (image->y1 < (1U << ((OPJ_UINT32)parameters->numresolution - 1U))) {
                    parameters->numresolution --;
                    continue;
                }
                break;
            }
        }
    }

    /* Set defaults precincts */
    if (parameters->csty == 0) {
        parameters->csty |= J2K_CP_CSTY_PRT;
        if (parameters->numresolution == 1) {
            parameters->res_spec = 1;
            parameters->prcw_init[0] = 128;
            parameters->prch_init[0] = 128;
        } else {
            int i;
            parameters->res_spec = parameters->numresolution - 1;
            for (i = 0; i < parameters->res_spec; i++) {
                parameters->prcw_init[i] = 256;
                parameters->prch_init[i] = 256;
            }
        }
    }
}

/* Table A.53 from JPEG2000 standard */
static const OPJ_UINT16 tabMaxSubLevelFromMainLevel[] = {
    15, /* unspecified */
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
};

static OPJ_BOOL opj_j2k_is_imf_compliant(opj_cparameters_t *parameters,
        opj_image_t *image,
        opj_event_mgr_t *p_manager)
{
    OPJ_UINT32 i;
    const OPJ_UINT16 rsiz = parameters->rsiz;
    const OPJ_UINT16 profile = OPJ_GET_IMF_PROFILE(rsiz);
    const OPJ_UINT16 mainlevel = OPJ_GET_IMF_MAINLEVEL(rsiz);
    const OPJ_UINT16 sublevel = OPJ_GET_IMF_SUBLEVEL(rsiz);
    const int NL = parameters->numresolution - 1;
    const OPJ_UINT32 XTsiz = parameters->tile_size_on ? (OPJ_UINT32)
                             parameters->cp_tdx : image->x1;
    OPJ_BOOL ret = OPJ_TRUE;

    /* Validate mainlevel */
    if (mainlevel > OPJ_IMF_MAINLEVEL_MAX) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile require mainlevel <= 11.\n"
                      "-> %d is thus not compliant\n"
                      "-> Non-IMF codestream will be generated\n",
                      mainlevel);
        ret = OPJ_FALSE;
    } else {
        /* Validate sublevel */
        assert(sizeof(tabMaxSubLevelFromMainLevel) ==
               (OPJ_IMF_MAINLEVEL_MAX + 1) * sizeof(tabMaxSubLevelFromMainLevel[0]));
        if (sublevel > tabMaxSubLevelFromMainLevel[mainlevel]) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profile require sublevel <= %d for mainlevel = %d.\n"
                          "-> %d is thus not compliant\n"
                          "-> Non-IMF codestream will be generated\n",
                          tabMaxSubLevelFromMainLevel[mainlevel],
                          mainlevel,
                          sublevel);
            ret = OPJ_FALSE;
        }
    }

    /* Number of components */
    if (image->numcomps > 3) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profiles require at most 3 components.\n"
                      "-> Number of components of input image (%d) is not compliant\n"
                      "-> Non-IMF codestream will be generated\n",
                      image->numcomps);
        ret = OPJ_FALSE;
    }

    if (image->x0 != 0 || image->y0 != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profiles require image origin to be at 0,0.\n"
                      "-> %d,%d is not compliant\n"
                      "-> Non-IMF codestream will be generated\n",
                      image->x0, image->y0 != 0);
        ret = OPJ_FALSE;
    }

    if (parameters->cp_tx0 != 0 || parameters->cp_ty0 != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profiles require tile origin to be at 0,0.\n"
                      "-> %d,%d is not compliant\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->cp_tx0, parameters->cp_ty0);
        ret = OPJ_FALSE;
    }

    if (parameters->tile_size_on) {
        if (profile == OPJ_PROFILE_IMF_2K ||
                profile == OPJ_PROFILE_IMF_4K ||
                profile == OPJ_PROFILE_IMF_8K) {
            if ((OPJ_UINT32)parameters->cp_tdx < image->x1 ||
                    (OPJ_UINT32)parameters->cp_tdy < image->y1) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 2K/4K/8K single tile profiles require tile to be greater or equal to image size.\n"
                              "-> %d,%d is lesser than %d,%d\n"
                              "-> Non-IMF codestream will be generated\n",
                              parameters->cp_tdx,
                              parameters->cp_tdy,
                              image->x1,
                              image->y1);
                ret = OPJ_FALSE;
            }
        } else {
            if ((OPJ_UINT32)parameters->cp_tdx >= image->x1 &&
                    (OPJ_UINT32)parameters->cp_tdy >= image->y1) {
                /* ok */
            } else if (parameters->cp_tdx == 1024 &&
                       parameters->cp_tdy == 1024) {
                /* ok */
            } else if (parameters->cp_tdx == 2048 &&
                       parameters->cp_tdy == 2048 &&
                       (profile == OPJ_PROFILE_IMF_4K ||
                        profile == OPJ_PROFILE_IMF_8K)) {
                /* ok */
            } else if (parameters->cp_tdx == 4096 &&
                       parameters->cp_tdy == 4096 &&
                       profile == OPJ_PROFILE_IMF_8K) {
                /* ok */
            } else {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 2K_R/4K_R/8K_R single/multiple tile profiles "
                              "require tile to be greater or equal to image size,\n"
                              "or to be (1024,1024), or (2048,2048) for 4K_R/8K_R "
                              "or (4096,4096) for 8K_R.\n"
                              "-> %d,%d is non conformant\n"
                              "-> Non-IMF codestream will be generated\n",
                              parameters->cp_tdx,
                              parameters->cp_tdy);
                ret = OPJ_FALSE;
            }
        }
    }

    /* Bitdepth */
    for (i = 0; i < image->numcomps; i++) {
        if (!(image->comps[i].prec >= 8 && image->comps[i].prec <= 16) ||
                (image->comps[i].sgnd)) {
            char signed_str[] = "signed";
            char unsigned_str[] = "unsigned";
            char *tmp_str = image->comps[i].sgnd ? signed_str : unsigned_str;
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require precision of each component to b in [8-16] bits unsigned"
                          "-> At least component %d of input image (%d bits, %s) is not compliant\n"
                          "-> Non-IMF codestream will be generated\n",
                          i, image->comps[i].prec, tmp_str);
            ret = OPJ_FALSE;
        }
    }

    /* Sub-sampling */
    for (i = 0; i < image->numcomps; i++) {
        if (i == 0 && image->comps[i].dx != 1) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz1 == 1. Here it is set to %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[i].dx);
            ret = OPJ_FALSE;
        }
        if (i == 1 && image->comps[i].dx != 1 && image->comps[i].dx != 2) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz2 == 1 or 2. Here it is set to %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[i].dx);
            ret = OPJ_FALSE;
        }
        if (i > 1 && image->comps[i].dx != image->comps[i - 1].dx) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz%d to be the same as XRSiz2. "
                          "Here it is set to %d instead of %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          i + 1, image->comps[i].dx, image->comps[i - 1].dx);
            ret = OPJ_FALSE;
        }
        if (image->comps[i].dy != 1) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require YRsiz == 1. "
                          "Here it is set to %d for component %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[i].dy, i);
            ret = OPJ_FALSE;
        }
    }

    /* Image size */
    switch (profile) {
    case OPJ_PROFILE_IMF_2K:
    case OPJ_PROFILE_IMF_2K_R:
        if (((image->comps[0].w > 2048) | (image->comps[0].h > 1556))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 2K/2K_R profile require:\n"
                          "width <= 2048 and height <= 1556\n"
                          "-> Input image size %d x %d is not compliant\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[0].w, image->comps[0].h);
            ret = OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_IMF_4K:
    case OPJ_PROFILE_IMF_4K_R:
        if (((image->comps[0].w > 4096) | (image->comps[0].h > 3112))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 4K/4K_R profile require:\n"
                          "width <= 4096 and height <= 3112\n"
                          "-> Input image size %d x %d is not compliant\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[0].w, image->comps[0].h);
            ret = OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_IMF_8K:
    case OPJ_PROFILE_IMF_8K_R:
        if (((image->comps[0].w > 8192) | (image->comps[0].h > 6224))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 8K/8K_R profile require:\n"
                          "width <= 8192 and height <= 6224\n"
                          "-> Input image size %d x %d is not compliant\n"
                          "-> Non-IMF codestream will be generated\n",
                          image->comps[0].w, image->comps[0].h);
            ret = OPJ_FALSE;
        }
        break;
    default :
        assert(0);
        return OPJ_FALSE;
    }

    if (parameters->roi_compno != -1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile forbid RGN / region of interest marker.\n"
                      "-> Compression parameters specify a ROI\n"
                      "-> Non-IMF codestream will be generated\n");
        ret = OPJ_FALSE;
    }

    if (parameters->cblockw_init != 32 || parameters->cblockh_init != 32) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile require code block size to be 32x32.\n"
                      "-> Compression parameters set it to %dx%d.\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->cblockw_init,
                      parameters->cblockh_init);
        ret = OPJ_FALSE;
    }

    if (parameters->prog_order != OPJ_CPRL) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile require progression order to be CPRL.\n"
                      "-> Compression parameters set it to %d.\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->prog_order);
        ret = OPJ_FALSE;
    }

    if (parameters->numpocs != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile forbid POC markers.\n"
                      "-> Compression parameters set %d POC.\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->numpocs);
        ret = OPJ_FALSE;
    }

    /* Codeblock style: no mode switch enabled */
    if (parameters->mode != 0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF profile forbid mode switch in code block style.\n"
                      "-> Compression parameters set code block style to %d.\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->mode);
        ret = OPJ_FALSE;
    }

    if (profile == OPJ_PROFILE_IMF_2K ||
            profile == OPJ_PROFILE_IMF_4K ||
            profile == OPJ_PROFILE_IMF_8K) {
        /* Expect 9-7 transform */
        if (parameters->irreversible != 1) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 2K/4K/8K profiles require 9-7 Irreversible Transform.\n"
                          "-> Compression parameters set it to reversible.\n"
                          "-> Non-IMF codestream will be generated\n");
            ret = OPJ_FALSE;
        }
    } else {
        /* Expect 5-3 transform */
        if (parameters->irreversible != 0) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 2K/4K/8K profiles require 5-3 reversible Transform.\n"
                          "-> Compression parameters set it to irreversible.\n"
                          "-> Non-IMF codestream will be generated\n");
            ret = OPJ_FALSE;
        }
    }

    /* Number of layers */
    if (parameters->tcp_numlayers != 1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "IMF 2K/4K/8K profiles require 1 single quality layer.\n"
                      "-> Number of layers is %d.\n"
                      "-> Non-IMF codestream will be generated\n",
                      parameters->tcp_numlayers);
        ret = OPJ_FALSE;
    }

    /* Decomposition levels */
    switch (profile) {
    case OPJ_PROFILE_IMF_2K:
        if (!(NL >= 1 && NL <= 5)) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 2K profile requires 1 <= NL <= 5:\n"
                          "-> Number of decomposition levels is %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          NL);
            ret = OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_IMF_4K:
        if (!(NL >= 1 && NL <= 6)) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 4K profile requires 1 <= NL <= 6:\n"
                          "-> Number of decomposition levels is %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          NL);
            ret = OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_IMF_8K:
        if (!(NL >= 1 && NL <= 7)) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF 8K profile requires 1 <= NL <= 7:\n"
                          "-> Number of decomposition levels is %d.\n"
                          "-> Non-IMF codestream will be generated\n",
                          NL);
            ret = OPJ_FALSE;
        }
        break;
    case OPJ_PROFILE_IMF_2K_R: {
        if (XTsiz >= 2048) {
            if (!(NL >= 1 && NL <= 5)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 2K_R profile requires 1 <= NL <= 5 for XTsiz >= 2048:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 1024) {
            if (!(NL >= 1 && NL <= 4)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 2K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        }
        break;
    }
    case OPJ_PROFILE_IMF_4K_R: {
        if (XTsiz >= 4096) {
            if (!(NL >= 1 && NL <= 6)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 6 for XTsiz >= 4096:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 2048) {
            if (!(NL >= 1 && NL <= 5)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 5 for XTsiz in [2048,4096[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 1024) {
            if (!(NL >= 1 && NL <= 4)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        }
        break;
    }
    case OPJ_PROFILE_IMF_8K_R: {
        if (XTsiz >= 8192) {
            if (!(NL >= 1 && NL <= 7)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 7 for XTsiz >= 8192:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 4096) {
            if (!(NL >= 1 && NL <= 6)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 6 for XTsiz in [4096,8192[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 2048) {
            if (!(NL >= 1 && NL <= 5)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 5 for XTsiz in [2048,4096[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        } else if (XTsiz >= 1024) {
            if (!(NL >= 1 && NL <= 4)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF 4K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n"
                              "-> Number of decomposition levels is %d.\n"
                              "-> Non-IMF codestream will be generated\n",
                              NL);
                ret = OPJ_FALSE;
            }
        }
        break;
    }
    default:
        break;
    }

    if (parameters->numresolution == 1) {
        if (parameters->res_spec != 1 ||
                parameters->prcw_init[0] != 128 ||
                parameters->prch_init[0] != 128) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "IMF profiles require PPx = PPy = 7 for NLLL band, else 8.\n"
                          "-> Supplied values are different from that.\n"
                          "-> Non-IMF codestream will be generated\n");
            ret = OPJ_FALSE;
        }
    } else {
        int i;
        for (i = 0; i < parameters->res_spec; i++) {
            if (parameters->prcw_init[i] != 256 ||
                    parameters->prch_init[i] != 256) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "IMF profiles require PPx = PPy = 7 for NLLL band, else 8.\n"
                              "-> Supplied values are different from that.\n"
                              "-> Non-IMF codestream will be generated\n");
                ret = OPJ_FALSE;
            }
        }
    }

    return ret;
}


OPJ_BOOL opj_j2k_setup_encoder(opj_j2k_t *p_j2k,
                               opj_cparameters_t *parameters,
                               opj_image_t *image,
                               opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, j, tileno, numpocs_tile;
    opj_cp_t *cp = 00;
    OPJ_UINT32 cblkw, cblkh;

    if (!p_j2k || !parameters || ! image) {
        return OPJ_FALSE;
    }

    if ((parameters->numresolution <= 0) ||
            (parameters->numresolution > OPJ_J2K_MAXRLVLS)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid number of resolutions : %d not in range [1,%d]\n",
                      parameters->numresolution, OPJ_J2K_MAXRLVLS);
        return OPJ_FALSE;
    }

    if (parameters->cblockw_init < 4 || parameters->cblockw_init > 1024) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
                      parameters->cblockw_init);
        return OPJ_FALSE;
    }
    if (parameters->cblockh_init < 4 || parameters->cblockh_init > 1024) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for cblockh_init: %d not a power of 2 not in range [4,1024]\n",
                      parameters->cblockh_init);
        return OPJ_FALSE;
    }
    if (parameters->cblockw_init * parameters->cblockh_init > 4096) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for cblockw_init * cblockh_init: should be <= 4096\n");
        return OPJ_FALSE;
    }
    cblkw = (OPJ_UINT32)opj_int_floorlog2(parameters->cblockw_init);
    cblkh = (OPJ_UINT32)opj_int_floorlog2(parameters->cblockh_init);
    if (parameters->cblockw_init != (1 << cblkw)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
                      parameters->cblockw_init);
        return OPJ_FALSE;
    }
    if (parameters->cblockh_init != (1 << cblkh)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
                      parameters->cblockh_init);
        return OPJ_FALSE;
    }

    if (parameters->cp_fixed_alloc) {
        if (parameters->cp_matrice == NULL) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "cp_fixed_alloc set, but cp_matrice missing\n");
            return OPJ_FALSE;
        }

        if (parameters->tcp_numlayers > J2K_TCD_MATRIX_MAX_LAYER_COUNT) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "tcp_numlayers when cp_fixed_alloc set should not exceed %d\n",
                          J2K_TCD_MATRIX_MAX_LAYER_COUNT);
            return OPJ_FALSE;
        }
        if (parameters->numresolution > J2K_TCD_MATRIX_MAX_RESOLUTION_COUNT) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "numresolution when cp_fixed_alloc set should not exceed %d\n",
                          J2K_TCD_MATRIX_MAX_RESOLUTION_COUNT);
            return OPJ_FALSE;
        }
    }

    p_j2k->m_specific_param.m_encoder.m_nb_comps = image->numcomps;

    /* keep a link to cp so that we can destroy it later in j2k_destroy_compress */
    cp = &(p_j2k->m_cp);

    /* set default values for cp */
    cp->tw = 1;
    cp->th = 1;

    /* FIXME ADE: to be removed once deprecated cp_cinema and cp_rsiz have been removed */
    if (parameters->rsiz ==
            OPJ_PROFILE_NONE) { /* consider deprecated fields only if RSIZ has not been set */
        OPJ_BOOL deprecated_used = OPJ_FALSE;
        switch (parameters->cp_cinema) {
        case OPJ_CINEMA2K_24:
            parameters->rsiz = OPJ_PROFILE_CINEMA_2K;
            parameters->max_cs_size = OPJ_CINEMA_24_CS;
            parameters->max_comp_size = OPJ_CINEMA_24_COMP;
            deprecated_used = OPJ_TRUE;
            break;
        case OPJ_CINEMA2K_48:
            parameters->rsiz = OPJ_PROFILE_CINEMA_2K;
            parameters->max_cs_size = OPJ_CINEMA_48_CS;
            parameters->max_comp_size = OPJ_CINEMA_48_COMP;
            deprecated_used = OPJ_TRUE;
            break;
        case OPJ_CINEMA4K_24:
            parameters->rsiz = OPJ_PROFILE_CINEMA_4K;
            parameters->max_cs_size = OPJ_CINEMA_24_CS;
            parameters->max_comp_size = OPJ_CINEMA_24_COMP;
            deprecated_used = OPJ_TRUE;
            break;
        case OPJ_OFF:
        default:
            break;
        }
        switch (parameters->cp_rsiz) {
        case OPJ_CINEMA2K:
            parameters->rsiz = OPJ_PROFILE_CINEMA_2K;
            deprecated_used = OPJ_TRUE;
            break;
        case OPJ_CINEMA4K:
            parameters->rsiz = OPJ_PROFILE_CINEMA_4K;
            deprecated_used = OPJ_TRUE;
            break;
        case OPJ_MCT:
            parameters->rsiz = OPJ_PROFILE_PART2 | OPJ_EXTENSION_MCT;
            deprecated_used = OPJ_TRUE;
        case OPJ_STD_RSIZ:
        default:
            break;
        }
        if (deprecated_used) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Deprecated fields cp_cinema or cp_rsiz are used\n"
                          "Please consider using only the rsiz field\n"
                          "See openjpeg.h documentation for more details\n");
        }
    }

    /* If no explicit layers are provided, use lossless settings */
    if (parameters->tcp_numlayers == 0) {
        parameters->tcp_numlayers = 1;
        parameters->cp_disto_alloc = 1;
        parameters->tcp_rates[0] = 0;
    }

    if (parameters->cp_disto_alloc) {
        /* Emit warnings if tcp_rates are not decreasing */
        for (i = 1; i < (OPJ_UINT32) parameters->tcp_numlayers; i++) {
            OPJ_FLOAT32 rate_i_corr = parameters->tcp_rates[i];
            OPJ_FLOAT32 rate_i_m_1_corr = parameters->tcp_rates[i - 1];
            if (rate_i_corr <= 1.0) {
                rate_i_corr = 1.0;
            }
            if (rate_i_m_1_corr <= 1.0) {
                rate_i_m_1_corr = 1.0;
            }
            if (rate_i_corr >= rate_i_m_1_corr) {
                if (rate_i_corr != parameters->tcp_rates[i] &&
                        rate_i_m_1_corr != parameters->tcp_rates[i - 1]) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "tcp_rates[%d]=%f (corrected as %f) should be strictly lesser "
                                  "than tcp_rates[%d]=%f (corrected as %f)\n",
                                  i, parameters->tcp_rates[i], rate_i_corr,
                                  i - 1, parameters->tcp_rates[i - 1], rate_i_m_1_corr);
                } else if (rate_i_corr != parameters->tcp_rates[i]) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "tcp_rates[%d]=%f (corrected as %f) should be strictly lesser "
                                  "than tcp_rates[%d]=%f\n",
                                  i, parameters->tcp_rates[i], rate_i_corr,
                                  i - 1, parameters->tcp_rates[i - 1]);
                } else if (rate_i_m_1_corr != parameters->tcp_rates[i - 1]) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "tcp_rates[%d]=%f should be strictly lesser "
                                  "than tcp_rates[%d]=%f (corrected as %f)\n",
                                  i, parameters->tcp_rates[i],
                                  i - 1, parameters->tcp_rates[i - 1], rate_i_m_1_corr);
                } else {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "tcp_rates[%d]=%f should be strictly lesser "
                                  "than tcp_rates[%d]=%f\n",
                                  i, parameters->tcp_rates[i],
                                  i - 1, parameters->tcp_rates[i - 1]);
                }
            }
        }
    } else if (parameters->cp_fixed_quality) {
        /* Emit warnings if tcp_distoratio are not increasing */
        for (i = 1; i < (OPJ_UINT32) parameters->tcp_numlayers; i++) {
            if (parameters->tcp_distoratio[i] < parameters->tcp_distoratio[i - 1] &&
                    !(i == (OPJ_UINT32)parameters->tcp_numlayers - 1 &&
                      parameters->tcp_distoratio[i] == 0)) {
                opj_event_msg(p_manager, EVT_WARNING,
                              "tcp_distoratio[%d]=%f should be strictly greater "
                              "than tcp_distoratio[%d]=%f\n",
                              i, parameters->tcp_distoratio[i], i - 1,
                              parameters->tcp_distoratio[i - 1]);
            }
        }
    }

    /* see if max_codestream_size does limit input rate */
    if (parameters->max_cs_size <= 0) {
        if (parameters->tcp_rates[parameters->tcp_numlayers - 1] > 0) {
            OPJ_FLOAT32 temp_size;
            temp_size = (OPJ_FLOAT32)(((double)image->numcomps * image->comps[0].w *
                                       image->comps[0].h * image->comps[0].prec) /
                                      ((double)parameters->tcp_rates[parameters->tcp_numlayers - 1] * 8 *
                                       image->comps[0].dx * image->comps[0].dy));
            if (temp_size > (OPJ_FLOAT32)INT_MAX) {
                parameters->max_cs_size = INT_MAX;
            } else {
                parameters->max_cs_size = (int) floor(temp_size);
            }
        } else {
            parameters->max_cs_size = 0;
        }
    } else {
        OPJ_FLOAT32 temp_rate;
        OPJ_BOOL cap = OPJ_FALSE;

        if (OPJ_IS_IMF(parameters->rsiz) && parameters->max_cs_size > 0 &&
                parameters->tcp_numlayers == 1 && parameters->tcp_rates[0] == 0) {
            parameters->tcp_rates[0] = (OPJ_FLOAT32)(image->numcomps * image->comps[0].w *
                                       image->comps[0].h * image->comps[0].prec) /
                                       (OPJ_FLOAT32)(((OPJ_UINT32)parameters->max_cs_size) * 8 * image->comps[0].dx *
                                               image->comps[0].dy);
        }

        temp_rate = (OPJ_FLOAT32)(((double)image->numcomps * image->comps[0].w *
                                   image->comps[0].h * image->comps[0].prec) /
                                  (((double)parameters->max_cs_size) * 8 * image->comps[0].dx *
                                   image->comps[0].dy));
        for (i = 0; i < (OPJ_UINT32) parameters->tcp_numlayers; i++) {
            if (parameters->tcp_rates[i] < temp_rate) {
                parameters->tcp_rates[i] = temp_rate;
                cap = OPJ_TRUE;
            }
        }
        if (cap) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "The desired maximum codestream size has limited\n"
                          "at least one of the desired quality layers\n");
        }
    }

    if (OPJ_IS_CINEMA(parameters->rsiz) || OPJ_IS_IMF(parameters->rsiz)) {
        p_j2k->m_specific_param.m_encoder.m_TLM = OPJ_TRUE;
    }

    /* Manage profiles and applications and set RSIZ */
    /* set cinema parameters if required */
    if (OPJ_IS_CINEMA(parameters->rsiz)) {
        if ((parameters->rsiz == OPJ_PROFILE_CINEMA_S2K)
                || (parameters->rsiz == OPJ_PROFILE_CINEMA_S4K)) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Scalable Digital Cinema profiles not yet supported\n");
            parameters->rsiz = OPJ_PROFILE_NONE;
        } else {
            opj_j2k_set_cinema_parameters(parameters, image, p_manager);
            if (!opj_j2k_is_cinema_compliant(image, parameters->rsiz, p_manager)) {
                parameters->rsiz = OPJ_PROFILE_NONE;
            }
        }
    } else if (OPJ_IS_STORAGE(parameters->rsiz)) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Long Term Storage profile not yet supported\n");
        parameters->rsiz = OPJ_PROFILE_NONE;
    } else if (OPJ_IS_BROADCAST(parameters->rsiz)) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "JPEG 2000 Broadcast profiles not yet supported\n");
        parameters->rsiz = OPJ_PROFILE_NONE;
    } else if (OPJ_IS_IMF(parameters->rsiz)) {
        opj_j2k_set_imf_parameters(parameters, image, p_manager);
        if (!opj_j2k_is_imf_compliant(parameters, image, p_manager)) {
            parameters->rsiz = OPJ_PROFILE_NONE;
        }
    } else if (OPJ_IS_PART2(parameters->rsiz)) {
        if (parameters->rsiz == ((OPJ_PROFILE_PART2) | (OPJ_EXTENSION_NONE))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "JPEG 2000 Part-2 profile defined\n"
                          "but no Part-2 extension enabled.\n"
                          "Profile set to NONE.\n");
            parameters->rsiz = OPJ_PROFILE_NONE;
        } else if (parameters->rsiz != ((OPJ_PROFILE_PART2) | (OPJ_EXTENSION_MCT))) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Unsupported Part-2 extension enabled\n"
                          "Profile set to NONE.\n");
            parameters->rsiz = OPJ_PROFILE_NONE;
        }
    }

    /*
    copy user encoding parameters
    */
    cp->m_specific_param.m_enc.m_max_comp_size = (OPJ_UINT32)
            parameters->max_comp_size;
    cp->rsiz = parameters->rsiz;
    if (parameters->cp_fixed_alloc) {
        cp->m_specific_param.m_enc.m_quality_layer_alloc_strategy = FIXED_LAYER;
    } else if (parameters->cp_fixed_quality) {
        cp->m_specific_param.m_enc.m_quality_layer_alloc_strategy =
            FIXED_DISTORTION_RATIO;
    } else {
        cp->m_specific_param.m_enc.m_quality_layer_alloc_strategy =
            RATE_DISTORTION_RATIO;
    }

    if (parameters->cp_fixed_alloc) {
        size_t array_size = (size_t)parameters->tcp_numlayers *
                            (size_t)parameters->numresolution * 3 * sizeof(OPJ_INT32);
        cp->m_specific_param.m_enc.m_matrice = (OPJ_INT32 *) opj_malloc(array_size);
        if (!cp->m_specific_param.m_enc.m_matrice) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to allocate copy of user encoding parameters matrix \n");
            return OPJ_FALSE;
        }
        memcpy(cp->m_specific_param.m_enc.m_matrice, parameters->cp_matrice,
               array_size);
    }

    /* tiles */
    cp->tdx = (OPJ_UINT32)parameters->cp_tdx;
    cp->tdy = (OPJ_UINT32)parameters->cp_tdy;

    /* tile offset */
    cp->tx0 = (OPJ_UINT32)parameters->cp_tx0;
    cp->ty0 = (OPJ_UINT32)parameters->cp_ty0;

    /* comment string */
    if (parameters->cp_comment) {
        cp->comment = (char*)opj_malloc(strlen(parameters->cp_comment) + 1U);
        if (!cp->comment) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to allocate copy of comment string\n");
            return OPJ_FALSE;
        }
        strcpy(cp->comment, parameters->cp_comment);
    } else {
        /* Create default comment for codestream */
        const char comment[] = "Created by OpenJPEG version ";
        const size_t clen = strlen(comment);
        const char *version = opj_version();

        /* UniPG>> */
#ifdef USE_JPWL
        const size_t cp_comment_buf_size = clen + strlen(version) + 11;
        cp->comment = (char*)opj_malloc(cp_comment_buf_size);
        if (!cp->comment) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to allocate comment string\n");
            return OPJ_FALSE;
        }
        snprintf(cp->comment, cp_comment_buf_size, "%s%s with JPWL",
                 comment, version);
#else
        const size_t cp_comment_buf_size = clen + strlen(version) + 1;
        cp->comment = (char*)opj_malloc(cp_comment_buf_size);
        if (!cp->comment) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to allocate comment string\n");
            return OPJ_FALSE;
        }
        snprintf(cp->comment, cp_comment_buf_size, "%s%s", comment, version);
#endif
        /* <<UniPG */
    }

    /*
    calculate other encoding parameters
    */

    if (parameters->tile_size_on) {
        if (cp->tdx == 0) {
            opj_event_msg(p_manager, EVT_ERROR, "Invalid tile width\n");
            return OPJ_FALSE;
        }
        if (cp->tdy == 0) {
            opj_event_msg(p_manager, EVT_ERROR, "Invalid tile height\n");
            return OPJ_FALSE;
        }
        cp->tw = opj_uint_ceildiv(image->x1 - cp->tx0, cp->tdx);
        cp->th = opj_uint_ceildiv(image->y1 - cp->ty0, cp->tdy);
        /* Check that the number of tiles is valid */
        if (cp->tw > 65535 / cp->th) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid number of tiles : %u x %u (maximum fixed by jpeg2000 norm is 65535 tiles)\n",
                          cp->tw, cp->th);
            return OPJ_FALSE;
        }
    } else {
        cp->tdx = image->x1 - cp->tx0;
        cp->tdy = image->y1 - cp->ty0;
    }

    if (parameters->tp_on) {
        cp->m_specific_param.m_enc.m_tp_flag = (OPJ_BYTE)parameters->tp_flag;
        cp->m_specific_param.m_enc.m_tp_on = 1;
    }

#ifdef USE_JPWL
    /*
    calculate JPWL encoding parameters
    */

    if (parameters->jpwl_epc_on) {
        OPJ_INT32 i;

        /* set JPWL on */
        cp->epc_on = OPJ_TRUE;
        cp->info_on = OPJ_FALSE; /* no informative technique */

        /* set EPB on */
        if ((parameters->jpwl_hprot_MH > 0) || (parameters->jpwl_hprot_TPH[0] > 0)) {
            cp->epb_on = OPJ_TRUE;

            cp->hprot_MH = parameters->jpwl_hprot_MH;
            for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
                cp->hprot_TPH_tileno[i] = parameters->jpwl_hprot_TPH_tileno[i];
                cp->hprot_TPH[i] = parameters->jpwl_hprot_TPH[i];
            }
            /* if tile specs are not specified, copy MH specs */
            if (cp->hprot_TPH[0] == -1) {
                cp->hprot_TPH_tileno[0] = 0;
                cp->hprot_TPH[0] = parameters->jpwl_hprot_MH;
            }
            for (i = 0; i < JPWL_MAX_NO_PACKSPECS; i++) {
                cp->pprot_tileno[i] = parameters->jpwl_pprot_tileno[i];
                cp->pprot_packno[i] = parameters->jpwl_pprot_packno[i];
                cp->pprot[i] = parameters->jpwl_pprot[i];
            }
        }

        /* set ESD writing */
        if ((parameters->jpwl_sens_size == 1) || (parameters->jpwl_sens_size == 2)) {
            cp->esd_on = OPJ_TRUE;

            cp->sens_size = parameters->jpwl_sens_size;
            cp->sens_addr = parameters->jpwl_sens_addr;
            cp->sens_range = parameters->jpwl_sens_range;

            cp->sens_MH = parameters->jpwl_sens_MH;
            for (i = 0; i < JPWL_MAX_NO_TILESPECS; i++) {
                cp->sens_TPH_tileno[i] = parameters->jpwl_sens_TPH_tileno[i];
                cp->sens_TPH[i] = parameters->jpwl_sens_TPH[i];
            }
        }

        /* always set RED writing to false: we are at the encoder */
        cp->red_on = OPJ_FALSE;

    } else {
        cp->epc_on = OPJ_FALSE;
    }
#endif /* USE_JPWL */

    /* initialize the multiple tiles */
    /* ---------------------------- */
    cp->tcps = (opj_tcp_t*) opj_calloc(cp->tw * cp->th, sizeof(opj_tcp_t));
    if (!cp->tcps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Not enough memory to allocate tile coding parameters\n");
        return OPJ_FALSE;
    }

    for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
        opj_tcp_t *tcp = &cp->tcps[tileno];
        const OPJ_BOOL fixed_distoratio =
            cp->m_specific_param.m_enc.m_quality_layer_alloc_strategy ==
            FIXED_DISTORTION_RATIO;
        tcp->numlayers = (OPJ_UINT32)parameters->tcp_numlayers;

        for (j = 0; j < tcp->numlayers; j++) {
            if (OPJ_IS_CINEMA(cp->rsiz) || OPJ_IS_IMF(cp->rsiz)) {
                if (fixed_distoratio) {
                    tcp->distoratio[j] = parameters->tcp_distoratio[j];
                }
                tcp->rates[j] = parameters->tcp_rates[j];
            } else {
                if (fixed_distoratio) {
                    tcp->distoratio[j] = parameters->tcp_distoratio[j];
                } else {
                    tcp->rates[j] = parameters->tcp_rates[j];
                }
            }
            if (!fixed_distoratio &&
                    tcp->rates[j] <= 1.0) {
                tcp->rates[j] = 0.0;    /* force lossless */
            }
        }

        tcp->csty = (OPJ_UINT32)parameters->csty;
        tcp->prg = parameters->prog_order;
        tcp->mct = (OPJ_UINT32)parameters->tcp_mct;

        numpocs_tile = 0;
        tcp->POC = 0;

        if (parameters->numpocs) {
            /* initialisation of POC */
            for (i = 0; i < parameters->numpocs; i++) {
                if (tileno + 1 == parameters->POC[i].tile)  {
                    opj_poc_t *tcp_poc = &tcp->pocs[numpocs_tile];

                    if (parameters->POC[numpocs_tile].compno0 >= image->numcomps) {
                        opj_event_msg(p_manager, EVT_ERROR,
                                      "Invalid compno0 for POC %d\n", i);
                        return OPJ_FALSE;
                    }

                    tcp_poc->resno0         = parameters->POC[numpocs_tile].resno0;
                    tcp_poc->compno0        = parameters->POC[numpocs_tile].compno0;
                    tcp_poc->layno1         = parameters->POC[numpocs_tile].layno1;
                    tcp_poc->resno1         = parameters->POC[numpocs_tile].resno1;
                    tcp_poc->compno1        = opj_uint_min(parameters->POC[numpocs_tile].compno1,
                                                           image->numcomps);
                    tcp_poc->prg1           = parameters->POC[numpocs_tile].prg1;
                    tcp_poc->tile           = parameters->POC[numpocs_tile].tile;

                    numpocs_tile++;
                }
            }

            if (numpocs_tile) {

                /* TODO MSD use the return value*/
                opj_j2k_check_poc_val(parameters->POC, tileno, parameters->numpocs,
                                      (OPJ_UINT32)parameters->numresolution, image->numcomps,
                                      (OPJ_UINT32)parameters->tcp_numlayers, p_manager);

                tcp->POC = 1;
                tcp->numpocs = numpocs_tile - 1 ;
            }
        } else {
            tcp->numpocs = 0;
        }

        tcp->tccps = (opj_tccp_t*) opj_calloc(image->numcomps, sizeof(opj_tccp_t));
        if (!tcp->tccps) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Not enough memory to allocate tile component coding parameters\n");
            return OPJ_FALSE;
        }
        if (parameters->mct_data) {

            OPJ_UINT32 lMctSize = image->numcomps * image->numcomps * (OPJ_UINT32)sizeof(
                                      OPJ_FLOAT32);
            OPJ_FLOAT32 * lTmpBuf = (OPJ_FLOAT32*)opj_malloc(lMctSize);
            OPJ_INT32 * l_dc_shift = (OPJ_INT32 *)((OPJ_BYTE *) parameters->mct_data +
                                                   lMctSize);

            if (!lTmpBuf) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Not enough memory to allocate temp buffer\n");
                return OPJ_FALSE;
            }

            tcp->mct = 2;
            tcp->m_mct_coding_matrix = (OPJ_FLOAT32*)opj_malloc(lMctSize);
            if (! tcp->m_mct_coding_matrix) {
                opj_free(lTmpBuf);
                lTmpBuf = NULL;
                opj_event_msg(p_manager, EVT_ERROR,
                              "Not enough memory to allocate encoder MCT coding matrix \n");
                return OPJ_FALSE;
            }
            memcpy(tcp->m_mct_coding_matrix, parameters->mct_data, lMctSize);
            memcpy(lTmpBuf, parameters->mct_data, lMctSize);

            tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(lMctSize);
            if (! tcp->m_mct_decoding_matrix) {
                opj_free(lTmpBuf);
                lTmpBuf = NULL;
                opj_event_msg(p_manager, EVT_ERROR,
                              "Not enough memory to allocate encoder MCT decoding matrix \n");
                return OPJ_FALSE;
            }
            if (opj_matrix_inversion_f(lTmpBuf, (tcp->m_mct_decoding_matrix),
                                       image->numcomps) == OPJ_FALSE) {
                opj_free(lTmpBuf);
                lTmpBuf = NULL;
                opj_event_msg(p_manager, EVT_ERROR,
                              "Failed to inverse encoder MCT decoding matrix \n");
                return OPJ_FALSE;
            }

            tcp->mct_norms = (OPJ_FLOAT64*)
                             opj_malloc(image->numcomps * sizeof(OPJ_FLOAT64));
            if (! tcp->mct_norms) {
                opj_free(lTmpBuf);
                lTmpBuf = NULL;
                opj_event_msg(p_manager, EVT_ERROR,
                              "Not enough memory to allocate encoder MCT norms \n");
                return OPJ_FALSE;
            }
            opj_calculate_norms(tcp->mct_norms, image->numcomps,
                                tcp->m_mct_decoding_matrix);
            opj_free(lTmpBuf);

            for (i = 0; i < image->numcomps; i++) {
                opj_tccp_t *tccp = &tcp->tccps[i];
                tccp->m_dc_level_shift = l_dc_shift[i];
            }

            if (opj_j2k_setup_mct_encoding(tcp, image) == OPJ_FALSE) {
                /* free will be handled by opj_j2k_destroy */
                opj_event_msg(p_manager, EVT_ERROR, "Failed to setup j2k mct encoding\n");
                return OPJ_FALSE;
            }
        } else {
            if (tcp->mct == 1 && image->numcomps >= 3) { /* RGB->YCC MCT is enabled */
                if ((image->comps[0].dx != image->comps[1].dx) ||
                        (image->comps[0].dx != image->comps[2].dx) ||
                        (image->comps[0].dy != image->comps[1].dy) ||
                        (image->comps[0].dy != image->comps[2].dy)) {
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "Cannot perform MCT on components with different sizes. Disabling MCT.\n");
                    tcp->mct = 0;
                }
            }
            for (i = 0; i < image->numcomps; i++) {
                opj_tccp_t *tccp = &tcp->tccps[i];
                opj_image_comp_t * l_comp = &(image->comps[i]);

                if (! l_comp->sgnd) {
                    tccp->m_dc_level_shift = 1 << (l_comp->prec - 1);
                }
            }
        }

        for (i = 0; i < image->numcomps; i++) {
            opj_tccp_t *tccp = &tcp->tccps[i];

            tccp->csty = parameters->csty &
                         0x01;   /* 0 => one precinct || 1 => custom precinct  */
            tccp->numresolutions = (OPJ_UINT32)parameters->numresolution;
            tccp->cblkw = (OPJ_UINT32)opj_int_floorlog2(parameters->cblockw_init);
            tccp->cblkh = (OPJ_UINT32)opj_int_floorlog2(parameters->cblockh_init);
            tccp->cblksty = (OPJ_UINT32)parameters->mode;
            tccp->qmfbid = parameters->irreversible ? 0 : 1;
            tccp->qntsty = parameters->irreversible ? J2K_CCP_QNTSTY_SEQNT :
                           J2K_CCP_QNTSTY_NOQNT;

            if (OPJ_IS_CINEMA(parameters->rsiz) &&
                    parameters->rsiz == OPJ_PROFILE_CINEMA_2K) {
                /* From https://github.com/uclouvain/openjpeg/issues/1340 */
                tccp->numgbits = 1;
            } else {
                tccp->numgbits = 2;
            }

            if ((OPJ_INT32)i == parameters->roi_compno) {
                tccp->roishift = parameters->roi_shift;
            } else {
                tccp->roishift = 0;
            }

            if (parameters->csty & J2K_CCP_CSTY_PRT) {
                OPJ_INT32 p = 0, it_res;
                assert(tccp->numresolutions > 0);
                for (it_res = (OPJ_INT32)tccp->numresolutions - 1; it_res >= 0; it_res--) {
                    if (p < parameters->res_spec) {

                        if (parameters->prcw_init[p] < 1) {
                            tccp->prcw[it_res] = 1;
                        } else {
                            tccp->prcw[it_res] = (OPJ_UINT32)opj_int_floorlog2(parameters->prcw_init[p]);
                        }

                        if (parameters->prch_init[p] < 1) {
                            tccp->prch[it_res] = 1;
                        } else {
                            tccp->prch[it_res] = (OPJ_UINT32)opj_int_floorlog2(parameters->prch_init[p]);
                        }

                    } else {
                        OPJ_INT32 res_spec = parameters->res_spec;
                        OPJ_INT32 size_prcw = 0;
                        OPJ_INT32 size_prch = 0;

                        assert(res_spec > 0); /* issue 189 */
                        size_prcw = parameters->prcw_init[res_spec - 1] >> (p - (res_spec - 1));
                        size_prch = parameters->prch_init[res_spec - 1] >> (p - (res_spec - 1));


                        if (size_prcw < 1) {
                            tccp->prcw[it_res] = 1;
                        } else {
                            tccp->prcw[it_res] = (OPJ_UINT32)opj_int_floorlog2(size_prcw);
                        }

                        if (size_prch < 1) {
                            tccp->prch[it_res] = 1;
                        } else {
                            tccp->prch[it_res] = (OPJ_UINT32)opj_int_floorlog2(size_prch);
                        }
                    }
                    p++;
                    /*printf("\nsize precinct for level %d : %d,%d\n", it_res,tccp->prcw[it_res], tccp->prch[it_res]); */
                }       /*end for*/
            } else {
                for (j = 0; j < tccp->numresolutions; j++) {
                    tccp->prcw[j] = 15;
                    tccp->prch[j] = 15;
                }
            }

            opj_dwt_calc_explicit_stepsizes(tccp, image->comps[i].prec);
        }
    }

    if (parameters->mct_data) {
        opj_free(parameters->mct_data);
        parameters->mct_data = 00;
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_add_mhmarker(opj_codestream_index_t *cstr_index,
                                     OPJ_UINT32 type, OPJ_OFF_T pos, OPJ_UINT32 len)
{
    assert(cstr_index != 00);

    /* expand the list? */
    if ((cstr_index->marknum + 1) > cstr_index->maxmarknum) {
        opj_marker_info_t *new_marker;
        cstr_index->maxmarknum = (OPJ_UINT32)(100 + (OPJ_FLOAT32)
                                              cstr_index->maxmarknum);
        new_marker = (opj_marker_info_t *) opj_realloc(cstr_index->marker,
                     cstr_index->maxmarknum * sizeof(opj_marker_info_t));
        if (! new_marker) {
            opj_free(cstr_index->marker);
            cstr_index->marker = NULL;
            cstr_index->maxmarknum = 0;
            cstr_index->marknum = 0;
            /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n"); */
            return OPJ_FALSE;
        }
        cstr_index->marker = new_marker;
    }

    /* add the marker */
    cstr_index->marker[cstr_index->marknum].type = (OPJ_UINT16)type;
    cstr_index->marker[cstr_index->marknum].pos = (OPJ_INT32)pos;
    cstr_index->marker[cstr_index->marknum].len = (OPJ_INT32)len;
    cstr_index->marknum++;
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_add_tlmarker(OPJ_UINT32 tileno,
                                     opj_codestream_index_t *cstr_index, OPJ_UINT32 type, OPJ_OFF_T pos,
                                     OPJ_UINT32 len)
{
    assert(cstr_index != 00);
    assert(cstr_index->tile_index != 00);

    /* expand the list? */
    if ((cstr_index->tile_index[tileno].marknum + 1) >
            cstr_index->tile_index[tileno].maxmarknum) {
        opj_marker_info_t *new_marker;
        cstr_index->tile_index[tileno].maxmarknum = (OPJ_UINT32)(100 +
                (OPJ_FLOAT32) cstr_index->tile_index[tileno].maxmarknum);
        new_marker = (opj_marker_info_t *) opj_realloc(
                         cstr_index->tile_index[tileno].marker,
                         cstr_index->tile_index[tileno].maxmarknum * sizeof(opj_marker_info_t));
        if (! new_marker) {
            opj_free(cstr_index->tile_index[tileno].marker);
            cstr_index->tile_index[tileno].marker = NULL;
            cstr_index->tile_index[tileno].maxmarknum = 0;
            cstr_index->tile_index[tileno].marknum = 0;
            /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n"); */
            return OPJ_FALSE;
        }
        cstr_index->tile_index[tileno].marker = new_marker;
    }

    /* add the marker */
    cstr_index->tile_index[tileno].marker[cstr_index->tile_index[tileno].marknum].type
        = (OPJ_UINT16)type;
    cstr_index->tile_index[tileno].marker[cstr_index->tile_index[tileno].marknum].pos
        = (OPJ_INT32)pos;
    cstr_index->tile_index[tileno].marker[cstr_index->tile_index[tileno].marknum].len
        = (OPJ_INT32)len;
    cstr_index->tile_index[tileno].marknum++;

    if (type == J2K_MS_SOT) {
        OPJ_UINT32 l_current_tile_part = cstr_index->tile_index[tileno].current_tpsno;

        if (cstr_index->tile_index[tileno].tp_index &&
                l_current_tile_part < cstr_index->tile_index[tileno].nb_tps) {
            cstr_index->tile_index[tileno].tp_index[l_current_tile_part].start_pos = pos;
        }

    }
    return OPJ_TRUE;
}

/*
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 */

OPJ_BOOL opj_j2k_end_decompress(opj_j2k_t *p_j2k,
                                opj_stream_private_t *p_stream,
                                opj_event_mgr_t * p_manager
                               )
{
    (void)p_j2k;
    (void)p_stream;
    (void)p_manager;
    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_read_header(opj_stream_private_t *p_stream,
                             opj_j2k_t* p_j2k,
                             opj_image_t** p_image,
                             opj_event_mgr_t* p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    /* create an empty image header */
    p_j2k->m_private_image = opj_image_create0();
    if (! p_j2k->m_private_image) {
        return OPJ_FALSE;
    }

    /* customization of the validation */
    if (! opj_j2k_setup_decoding_validation(p_j2k, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    /* validation of the parameters codec */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_validation_list, p_stream, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    /* customization of the encoding */
    if (! opj_j2k_setup_header_reading(p_j2k, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    /* read header */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_procedure_list, p_stream, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    *p_image = opj_image_create0();
    if (!(*p_image)) {
        return OPJ_FALSE;
    }

    /* Copy codestream image information to the output image */
    opj_copy_image_header(p_j2k->m_private_image, *p_image);

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_setup_header_reading(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_read_header_procedure, p_manager)) {
        return OPJ_FALSE;
    }

    /* DEVELOPER CORNER, add your custom procedures */
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_copy_default_tcp_and_create_tcd, p_manager))  {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_setup_decoding_validation(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_validation_list,
                                           (opj_procedure)opj_j2k_build_decoder, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_validation_list,
                                           (opj_procedure)opj_j2k_decoding_validation, p_manager)) {
        return OPJ_FALSE;
    }

    /* DEVELOPER CORNER, add your custom validation procedure */
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_mct_validation(opj_j2k_t * p_j2k,
                                       opj_stream_private_t *p_stream,
                                       opj_event_mgr_t * p_manager)
{
    OPJ_BOOL l_is_valid = OPJ_TRUE;
    OPJ_UINT32 i, j;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_stream);
    OPJ_UNUSED(p_manager);

    if ((p_j2k->m_cp.rsiz & 0x8200) == 0x8200) {
        OPJ_UINT32 l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
        opj_tcp_t * l_tcp = p_j2k->m_cp.tcps;

        for (i = 0; i < l_nb_tiles; ++i) {
            if (l_tcp->mct == 2) {
                opj_tccp_t * l_tccp = l_tcp->tccps;
                l_is_valid &= (l_tcp->m_mct_coding_matrix != 00);

                for (j = 0; j < p_j2k->m_private_image->numcomps; ++j) {
                    l_is_valid &= !(l_tccp->qmfbid & 1);
                    ++l_tccp;
                }
            }
            ++l_tcp;
        }
    }

    return l_is_valid;
}

OPJ_BOOL opj_j2k_setup_mct_encoding(opj_tcp_t * p_tcp, opj_image_t * p_image)
{
    OPJ_UINT32 i;
    OPJ_UINT32 l_indix = 1;
    opj_mct_data_t * l_mct_deco_data = 00, * l_mct_offset_data = 00;
    opj_simple_mcc_decorrelation_data_t * l_mcc_data;
    OPJ_UINT32 l_mct_size, l_nb_elem;
    OPJ_FLOAT32 * l_data, * l_current_data;
    opj_tccp_t * l_tccp;

    /* preconditions */
    assert(p_tcp != 00);

    if (p_tcp->mct != 2) {
        return OPJ_TRUE;
    }

    if (p_tcp->m_mct_decoding_matrix) {
        if (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records) {
            opj_mct_data_t *new_mct_records;
            p_tcp->m_nb_max_mct_records += OPJ_J2K_MCT_DEFAULT_NB_RECORDS;

            new_mct_records = (opj_mct_data_t *) opj_realloc(p_tcp->m_mct_records,
                              p_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
            if (! new_mct_records) {
                opj_free(p_tcp->m_mct_records);
                p_tcp->m_mct_records = NULL;
                p_tcp->m_nb_max_mct_records = 0;
                p_tcp->m_nb_mct_records = 0;
                /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
                return OPJ_FALSE;
            }
            p_tcp->m_mct_records = new_mct_records;
            l_mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

            memset(l_mct_deco_data, 0,
                   (p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records) * sizeof(
                       opj_mct_data_t));
        }
        l_mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

        if (l_mct_deco_data->m_data) {
            opj_free(l_mct_deco_data->m_data);
            l_mct_deco_data->m_data = 00;
        }

        l_mct_deco_data->m_index = l_indix++;
        l_mct_deco_data->m_array_type = MCT_TYPE_DECORRELATION;
        l_mct_deco_data->m_element_type = MCT_TYPE_FLOAT;
        l_nb_elem = p_image->numcomps * p_image->numcomps;
        l_mct_size = l_nb_elem * MCT_ELEMENT_SIZE[l_mct_deco_data->m_element_type];
        l_mct_deco_data->m_data = (OPJ_BYTE*)opj_malloc(l_mct_size);

        if (! l_mct_deco_data->m_data) {
            return OPJ_FALSE;
        }

        j2k_mct_write_functions_from_float[l_mct_deco_data->m_element_type](
            p_tcp->m_mct_decoding_matrix, l_mct_deco_data->m_data, l_nb_elem);

        l_mct_deco_data->m_data_size = l_mct_size;
        ++p_tcp->m_nb_mct_records;
    }

    if (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records) {
        opj_mct_data_t *new_mct_records;
        p_tcp->m_nb_max_mct_records += OPJ_J2K_MCT_DEFAULT_NB_RECORDS;
        new_mct_records = (opj_mct_data_t *) opj_realloc(p_tcp->m_mct_records,
                          p_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
        if (! new_mct_records) {
            opj_free(p_tcp->m_mct_records);
            p_tcp->m_mct_records = NULL;
            p_tcp->m_nb_max_mct_records = 0;
            p_tcp->m_nb_mct_records = 0;
            /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
            return OPJ_FALSE;
        }
        p_tcp->m_mct_records = new_mct_records;
        l_mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

        memset(l_mct_offset_data, 0,
               (p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records) * sizeof(
                   opj_mct_data_t));

        if (l_mct_deco_data) {
            l_mct_deco_data = l_mct_offset_data - 1;
        }
    }

    l_mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

    if (l_mct_offset_data->m_data) {
        opj_free(l_mct_offset_data->m_data);
        l_mct_offset_data->m_data = 00;
    }

    l_mct_offset_data->m_index = l_indix++;
    l_mct_offset_data->m_array_type = MCT_TYPE_OFFSET;
    l_mct_offset_data->m_element_type = MCT_TYPE_FLOAT;
    l_nb_elem = p_image->numcomps;
    l_mct_size = l_nb_elem * MCT_ELEMENT_SIZE[l_mct_offset_data->m_element_type];
    l_mct_offset_data->m_data = (OPJ_BYTE*)opj_malloc(l_mct_size);

    if (! l_mct_offset_data->m_data) {
        return OPJ_FALSE;
    }

    l_data = (OPJ_FLOAT32*)opj_malloc(l_nb_elem * sizeof(OPJ_FLOAT32));
    if (! l_data) {
        opj_free(l_mct_offset_data->m_data);
        l_mct_offset_data->m_data = 00;
        return OPJ_FALSE;
    }

    l_tccp = p_tcp->tccps;
    l_current_data = l_data;

    for (i = 0; i < l_nb_elem; ++i) {
        *(l_current_data++) = (OPJ_FLOAT32)(l_tccp->m_dc_level_shift);
        ++l_tccp;
    }

    j2k_mct_write_functions_from_float[l_mct_offset_data->m_element_type](l_data,
            l_mct_offset_data->m_data, l_nb_elem);

    opj_free(l_data);

    l_mct_offset_data->m_data_size = l_mct_size;

    ++p_tcp->m_nb_mct_records;

    if (p_tcp->m_nb_mcc_records == p_tcp->m_nb_max_mcc_records) {
        opj_simple_mcc_decorrelation_data_t *new_mcc_records;
        p_tcp->m_nb_max_mcc_records += OPJ_J2K_MCT_DEFAULT_NB_RECORDS;
        new_mcc_records = (opj_simple_mcc_decorrelation_data_t *) opj_realloc(
                              p_tcp->m_mcc_records, p_tcp->m_nb_max_mcc_records * sizeof(
                                  opj_simple_mcc_decorrelation_data_t));
        if (! new_mcc_records) {
            opj_free(p_tcp->m_mcc_records);
            p_tcp->m_mcc_records = NULL;
            p_tcp->m_nb_max_mcc_records = 0;
            p_tcp->m_nb_mcc_records = 0;
            /* opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
            return OPJ_FALSE;
        }
        p_tcp->m_mcc_records = new_mcc_records;
        l_mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
        memset(l_mcc_data, 0, (p_tcp->m_nb_max_mcc_records - p_tcp->m_nb_mcc_records) *
               sizeof(opj_simple_mcc_decorrelation_data_t));

    }

    l_mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
    l_mcc_data->m_decorrelation_array = l_mct_deco_data;
    l_mcc_data->m_is_irreversible = 1;
    l_mcc_data->m_nb_comps = p_image->numcomps;
    l_mcc_data->m_index = l_indix++;
    l_mcc_data->m_offset_array = l_mct_offset_data;
    ++p_tcp->m_nb_mcc_records;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_build_decoder(opj_j2k_t * p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager)
{
    /* add here initialization of cp
       copy paste of setup_decoder */
    (void)p_j2k;
    (void)p_stream;
    (void)p_manager;
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_build_encoder(opj_j2k_t * p_j2k,
                                      opj_stream_private_t *p_stream,
                                      opj_event_mgr_t * p_manager)
{
    /* add here initialization of cp
       copy paste of setup_encoder */
    (void)p_j2k;
    (void)p_stream;
    (void)p_manager;
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_encoding_validation(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager)
{
    OPJ_BOOL l_is_valid = OPJ_TRUE;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_stream);

    /* STATE checking */
    /* make sure the state is at 0 */
    l_is_valid &= (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_NONE);

    /* POINTER validation */
    /* make sure a p_j2k codec is present */
    l_is_valid &= (p_j2k->m_procedure_list != 00);
    /* make sure a validation list is present */
    l_is_valid &= (p_j2k->m_validation_list != 00);

    /* ISO 15444-1:2004 states between 1 & 33 (0 -> 32) */
    /* 33 (32) would always fail the check below (if a cast to 64bits was done) */
    /* FIXME Shall we change OPJ_J2K_MAXRLVLS to 32 ? */
    if ((p_j2k->m_cp.tcps->tccps->numresolutions <= 0) ||
            (p_j2k->m_cp.tcps->tccps->numresolutions > 32)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Number of resolutions is too high in comparison to the size of tiles\n");
        return OPJ_FALSE;
    }

    if ((p_j2k->m_cp.tdx) < (OPJ_UINT32)(1 <<
                                         (p_j2k->m_cp.tcps->tccps->numresolutions - 1U))) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Number of resolutions is too high in comparison to the size of tiles\n");
        return OPJ_FALSE;
    }

    if ((p_j2k->m_cp.tdy) < (OPJ_UINT32)(1 <<
                                         (p_j2k->m_cp.tcps->tccps->numresolutions - 1U))) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Number of resolutions is too high in comparison to the size of tiles\n");
        return OPJ_FALSE;
    }

    /* PARAMETER VALIDATION */
    return l_is_valid;
}

static OPJ_BOOL opj_j2k_decoding_validation(opj_j2k_t *p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager
                                           )
{
    OPJ_BOOL l_is_valid = OPJ_TRUE;

    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_stream);
    OPJ_UNUSED(p_manager);

    /* STATE checking */
    /* make sure the state is at 0 */
#ifdef TODO_MSD
    l_is_valid &= (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_NONE);
#endif
    l_is_valid &= (p_j2k->m_specific_param.m_decoder.m_state == 0x0000);

    /* POINTER validation */
    /* make sure a p_j2k codec is present */
    /* make sure a procedure list is present */
    l_is_valid &= (p_j2k->m_procedure_list != 00);
    /* make sure a validation list is present */
    l_is_valid &= (p_j2k->m_validation_list != 00);

    /* PARAMETER VALIDATION */
    return l_is_valid;
}

/** Fill p_j2k->cstr_index->tp_index[].start_pos/end_pos fields from TLM marker segments */
static void opj_j2k_build_tp_index_from_tlm(opj_j2k_t* p_j2k,
        opj_event_mgr_t * p_manager)
{
    opj_j2k_tlm_info_t* l_tlm;
    OPJ_UINT32 i;
    OPJ_OFF_T l_cur_offset;

    assert(p_j2k->cstr_index->main_head_end > 0);
    assert(p_j2k->cstr_index->nb_of_tiles > 0);
    assert(p_j2k->cstr_index->tile_index != NULL);

    l_tlm = &(p_j2k->m_specific_param.m_decoder.m_tlm);

    if (l_tlm->m_entries_count == 0) {
        l_tlm->m_is_invalid = OPJ_TRUE;
        return;
    }

    if (l_tlm->m_is_invalid) {
        return;
    }

    /* Initial pass to count the number of tile-parts per tile */
    for (i = 0; i < l_tlm->m_entries_count; ++i) {
        OPJ_UINT32 l_tile_index_no = l_tlm->m_tile_part_infos[i].m_tile_index;
        assert(l_tile_index_no < p_j2k->cstr_index->nb_of_tiles);
        p_j2k->cstr_index->tile_index[l_tile_index_no].tileno = l_tile_index_no;
        ++p_j2k->cstr_index->tile_index[l_tile_index_no].current_nb_tps;
    }

    /* Now check that all tiles have at least one tile-part */
    for (i = 0; i < p_j2k->cstr_index->nb_of_tiles; ++i) {
        if (p_j2k->cstr_index->tile_index[i].current_nb_tps == 0) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "opj_j2k_build_tp_index_from_tlm(): tile %d has no "
                          "registered tile-part in TLM marker segments.\n", i);
            goto error;
        }
    }

    /* Final pass to fill p_j2k->cstr_index */
    l_cur_offset = p_j2k->cstr_index->main_head_end;
    for (i = 0; i < l_tlm->m_entries_count; ++i) {
        OPJ_UINT32 l_tile_index_no = l_tlm->m_tile_part_infos[i].m_tile_index;
        opj_tile_index_t* l_tile_index = &
                                         (p_j2k->cstr_index->tile_index[l_tile_index_no]);
        if (!l_tile_index->tp_index) {
            l_tile_index->tp_index = (opj_tp_index_t *) opj_calloc(
                                         l_tile_index->current_nb_tps, sizeof(opj_tp_index_t));
            if (! l_tile_index->tp_index) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "opj_j2k_build_tp_index_from_tlm(): tile index allocation failed\n");
                goto error;
            }
        }

        assert(l_tile_index->nb_tps < l_tile_index->current_nb_tps);
        l_tile_index->tp_index[l_tile_index->nb_tps].start_pos = l_cur_offset;
        /* We don't know how to set the tp_index[].end_header field, but this is not really needed */
        /* If there would be no markers between SOT and SOD, that would be : */
        /* l_tile_index->tp_index[l_tile_index->nb_tps].end_header = l_cur_offset + 12; */
        l_tile_index->tp_index[l_tile_index->nb_tps].end_pos = l_cur_offset +
                l_tlm->m_tile_part_infos[i].m_length;
        ++l_tile_index->nb_tps;

        l_cur_offset += l_tlm->m_tile_part_infos[i].m_length;
    }

    return;

error:
    l_tlm->m_is_invalid = OPJ_TRUE;
    for (i = 0; i < l_tlm->m_entries_count; ++i) {
        OPJ_UINT32 l_tile_index = l_tlm->m_tile_part_infos[i].m_tile_index;
        p_j2k->cstr_index->tile_index[l_tile_index].current_nb_tps = 0;
        opj_free(p_j2k->cstr_index->tile_index[l_tile_index].tp_index);
        p_j2k->cstr_index->tile_index[l_tile_index].tp_index = NULL;
    }
}

static OPJ_BOOL opj_j2k_read_header_procedure(opj_j2k_t *p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 l_current_marker;
    OPJ_UINT32 l_marker_size;
    const opj_dec_memory_marker_handler_t * l_marker_handler = 00;
    OPJ_BOOL l_has_siz = 0;
    OPJ_BOOL l_has_cod = 0;
    OPJ_BOOL l_has_qcd = 0;

    /* preconditions */
    assert(p_stream != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    /*  We enter in the main header */
    p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_MHSOC;

    /* Try to read the SOC marker, the codestream must begin with SOC marker */
    if (! opj_j2k_read_soc(p_j2k, p_stream, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Expected a SOC marker \n");
        return OPJ_FALSE;
    }

    /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
    if (opj_stream_read_data(p_stream,
                             p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
        opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
        return OPJ_FALSE;
    }

    /* Read 2 bytes as the new marker ID */
    opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                   &l_current_marker, 2);

    /* Try to read until the SOT is detected */
    while (l_current_marker != J2K_MS_SOT) {

        /* Check if the current marker ID is valid */
        if (l_current_marker < 0xff00) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "A marker ID was expected (0xff--) instead of %.8x\n", l_current_marker);
            return OPJ_FALSE;
        }

        /* Get the marker handler from the marker ID */
        l_marker_handler = opj_j2k_get_marker_handler(l_current_marker);

        /* Manage case where marker is unknown */
        if (l_marker_handler->id == J2K_MS_UNK) {
            if (! opj_j2k_read_unk(p_j2k, p_stream, &l_current_marker, p_manager)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Unknown marker has been detected and generated error.\n");
                return OPJ_FALSE;
            }

            if (l_current_marker == J2K_MS_SOT) {
                break;    /* SOT marker is detected main header is completely read */
            } else { /* Get the marker handler from the marker ID */
                l_marker_handler = opj_j2k_get_marker_handler(l_current_marker);
            }
        }

        if (l_marker_handler->id == J2K_MS_SIZ) {
            /* Mark required SIZ marker as found */
            l_has_siz = 1;
        }
        if (l_marker_handler->id == J2K_MS_COD) {
            /* Mark required COD marker as found */
            l_has_cod = 1;
        }
        if (l_marker_handler->id == J2K_MS_QCD) {
            /* Mark required QCD marker as found */
            l_has_qcd = 1;
        }

        /* Check if the marker is known and if it is the right place to find it */
        if (!(p_j2k->m_specific_param.m_decoder.m_state & l_marker_handler->states)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Marker is not compliant with its position\n");
            return OPJ_FALSE;
        }

        /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
        if (opj_stream_read_data(p_stream,
                                 p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        /* read 2 bytes as the marker size */
        opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data, &l_marker_size,
                       2);
        if (l_marker_size < 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Invalid marker size\n");
            return OPJ_FALSE;
        }
        l_marker_size -= 2; /* Subtract the size of the marker ID already read */

        /* Check if the marker size is compatible with the header data size */
        if (l_marker_size > p_j2k->m_specific_param.m_decoder.m_header_data_size) {
            OPJ_BYTE *new_header_data = (OPJ_BYTE *) opj_realloc(
                                            p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size);
            if (! new_header_data) {
                opj_free(p_j2k->m_specific_param.m_decoder.m_header_data);
                p_j2k->m_specific_param.m_decoder.m_header_data = NULL;
                p_j2k->m_specific_param.m_decoder.m_header_data_size = 0;
                opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read header\n");
                return OPJ_FALSE;
            }
            p_j2k->m_specific_param.m_decoder.m_header_data = new_header_data;
            p_j2k->m_specific_param.m_decoder.m_header_data_size = l_marker_size;
        }

        /* Try to read the rest of the marker segment from stream and copy them into the buffer */
        if (opj_stream_read_data(p_stream,
                                 p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size,
                                 p_manager) != l_marker_size) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        /* Read the marker segment with the correct marker handler */
        if (!(*(l_marker_handler->handler))(p_j2k,
                                            p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size, p_manager)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Marker handler function failed to read the marker segment\n");
            return OPJ_FALSE;
        }

        /* Add the marker to the codestream index*/
        if (OPJ_FALSE == opj_j2k_add_mhmarker(
                    p_j2k->cstr_index,
                    l_marker_handler->id,
                    (OPJ_UINT32) opj_stream_tell(p_stream) - l_marker_size - 4,
                    l_marker_size + 4)) {
            opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n");
            return OPJ_FALSE;
        }

        /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
        if (opj_stream_read_data(p_stream,
                                 p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        /* read 2 bytes as the new marker ID */
        opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                       &l_current_marker, 2);
    }

    if (l_has_siz == 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "required SIZ marker not found in main header\n");
        return OPJ_FALSE;
    }
    if (l_has_cod == 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "required COD marker not found in main header\n");
        return OPJ_FALSE;
    }
    if (l_has_qcd == 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "required QCD marker not found in main header\n");
        return OPJ_FALSE;
    }

    if (! opj_j2k_merge_ppm(&(p_j2k->m_cp), p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to merge PPM data\n");
        return OPJ_FALSE;
    }

    opj_event_msg(p_manager, EVT_INFO, "Main header has been correctly decoded.\n");

    /* Position of the last element if the main header */
    p_j2k->cstr_index->main_head_end = (OPJ_UINT32) opj_stream_tell(p_stream) - 2;

    /* Build tile-part index from TLM information */
    opj_j2k_build_tp_index_from_tlm(p_j2k, p_manager);

    /* Next step: read a tile-part header */
    p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_TPHSOT;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_exec(opj_j2k_t * p_j2k,
                             opj_procedure_list_t * p_procedure_list,
                             opj_stream_private_t *p_stream,
                             opj_event_mgr_t * p_manager)
{
    OPJ_BOOL(** l_procedure)(opj_j2k_t *, opj_stream_private_t *,
                             opj_event_mgr_t *) = 00;
    OPJ_BOOL l_result = OPJ_TRUE;
    OPJ_UINT32 l_nb_proc, i;

    /* preconditions*/
    assert(p_procedure_list != 00);
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    l_nb_proc = opj_procedure_list_get_nb_procedures(p_procedure_list);
    l_procedure = (OPJ_BOOL(**)(opj_j2k_t *, opj_stream_private_t *,
                                opj_event_mgr_t *)) opj_procedure_list_get_first_procedure(p_procedure_list);

    for (i = 0; i < l_nb_proc; ++i) {
        l_result = l_result && ((*l_procedure)(p_j2k, p_stream, p_manager));
        ++l_procedure;
    }

    /* and clear the procedure list at the end.*/
    opj_procedure_list_clear(p_procedure_list);
    return l_result;
}

/* FIXME DOC*/
static OPJ_BOOL opj_j2k_copy_default_tcp_and_create_tcd(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager
                                                       )
{
    opj_tcp_t * l_tcp = 00;
    opj_tcp_t * l_default_tcp = 00;
    OPJ_UINT32 l_nb_tiles;
    OPJ_UINT32 i, j;
    opj_tccp_t *l_current_tccp = 00;
    OPJ_UINT32 l_tccp_size;
    OPJ_UINT32 l_mct_size;
    opj_image_t * l_image;
    OPJ_UINT32 l_mcc_records_size, l_mct_records_size;
    opj_mct_data_t * l_src_mct_rec, *l_dest_mct_rec;
    opj_simple_mcc_decorrelation_data_t * l_src_mcc_rec, *l_dest_mcc_rec;
    OPJ_UINT32 l_offset;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_stream);

    l_image = p_j2k->m_private_image;
    l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
    l_tcp = p_j2k->m_cp.tcps;
    l_tccp_size = l_image->numcomps * (OPJ_UINT32)sizeof(opj_tccp_t);
    l_default_tcp = p_j2k->m_specific_param.m_decoder.m_default_tcp;
    l_mct_size = l_image->numcomps * l_image->numcomps * (OPJ_UINT32)sizeof(
                     OPJ_FLOAT32);

    /* For each tile */
    for (i = 0; i < l_nb_tiles; ++i) {
        /* keep the tile-compo coding parameters pointer of the current tile coding parameters*/
        l_current_tccp = l_tcp->tccps;
        /*Copy default coding parameters into the current tile coding parameters*/
        memcpy(l_tcp, l_default_tcp, sizeof(opj_tcp_t));
        /* Initialize some values of the current tile coding parameters*/
        l_tcp->cod = 0;
        l_tcp->ppt = 0;
        l_tcp->ppt_data = 00;
        l_tcp->m_current_tile_part_number = -1;
        /* Remove memory not owned by this tile in case of early error return. */
        l_tcp->m_mct_decoding_matrix = 00;
        l_tcp->m_nb_max_mct_records = 0;
        l_tcp->m_mct_records = 00;
        l_tcp->m_nb_max_mcc_records = 0;
        l_tcp->m_mcc_records = 00;
        /* Reconnect the tile-compo coding parameters pointer to the current tile coding parameters*/
        l_tcp->tccps = l_current_tccp;

        /* Get the mct_decoding_matrix of the dflt_tile_cp and copy them into the current tile cp*/
        if (l_default_tcp->m_mct_decoding_matrix) {
            l_tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(l_mct_size);
            if (! l_tcp->m_mct_decoding_matrix) {
                return OPJ_FALSE;
            }
            memcpy(l_tcp->m_mct_decoding_matrix, l_default_tcp->m_mct_decoding_matrix,
                   l_mct_size);
        }

        /* Get the mct_record of the dflt_tile_cp and copy them into the current tile cp*/
        l_mct_records_size = l_default_tcp->m_nb_max_mct_records * (OPJ_UINT32)sizeof(
                                 opj_mct_data_t);
        l_tcp->m_mct_records = (opj_mct_data_t*)opj_malloc(l_mct_records_size);
        if (! l_tcp->m_mct_records) {
            return OPJ_FALSE;
        }
        memcpy(l_tcp->m_mct_records, l_default_tcp->m_mct_records, l_mct_records_size);

        /* Copy the mct record data from dflt_tile_cp to the current tile*/
        l_src_mct_rec = l_default_tcp->m_mct_records;
        l_dest_mct_rec = l_tcp->m_mct_records;

        for (j = 0; j < l_default_tcp->m_nb_mct_records; ++j) {

            if (l_src_mct_rec->m_data) {

                l_dest_mct_rec->m_data = (OPJ_BYTE*) opj_malloc(l_src_mct_rec->m_data_size);
                if (! l_dest_mct_rec->m_data) {
                    return OPJ_FALSE;
                }
                memcpy(l_dest_mct_rec->m_data, l_src_mct_rec->m_data,
                       l_src_mct_rec->m_data_size);
            }

            ++l_src_mct_rec;
            ++l_dest_mct_rec;
            /* Update with each pass to free exactly what has been allocated on early return. */
            l_tcp->m_nb_max_mct_records += 1;
        }

        /* Get the mcc_record of the dflt_tile_cp and copy them into the current tile cp*/
        l_mcc_records_size = l_default_tcp->m_nb_max_mcc_records * (OPJ_UINT32)sizeof(
                                 opj_simple_mcc_decorrelation_data_t);
        l_tcp->m_mcc_records = (opj_simple_mcc_decorrelation_data_t*) opj_malloc(
                                   l_mcc_records_size);
        if (! l_tcp->m_mcc_records) {
            return OPJ_FALSE;
        }
        memcpy(l_tcp->m_mcc_records, l_default_tcp->m_mcc_records, l_mcc_records_size);
        l_tcp->m_nb_max_mcc_records = l_default_tcp->m_nb_max_mcc_records;

        /* Copy the mcc record data from dflt_tile_cp to the current tile*/
        l_src_mcc_rec = l_default_tcp->m_mcc_records;
        l_dest_mcc_rec = l_tcp->m_mcc_records;

        for (j = 0; j < l_default_tcp->m_nb_max_mcc_records; ++j) {

            if (l_src_mcc_rec->m_decorrelation_array) {
                l_offset = (OPJ_UINT32)(l_src_mcc_rec->m_decorrelation_array -
                                        l_default_tcp->m_mct_records);
                l_dest_mcc_rec->m_decorrelation_array = l_tcp->m_mct_records + l_offset;
            }

            if (l_src_mcc_rec->m_offset_array) {
                l_offset = (OPJ_UINT32)(l_src_mcc_rec->m_offset_array -
                                        l_default_tcp->m_mct_records);
                l_dest_mcc_rec->m_offset_array = l_tcp->m_mct_records + l_offset;
            }

            ++l_src_mcc_rec;
            ++l_dest_mcc_rec;
        }

        /* Copy all the dflt_tile_compo_cp to the current tile cp */
        memcpy(l_current_tccp, l_default_tcp->tccps, l_tccp_size);

        /* Move to next tile cp*/
        ++l_tcp;
    }

    /* Create the current tile decoder*/
    p_j2k->m_tcd = opj_tcd_create(OPJ_TRUE);
    if (! p_j2k->m_tcd) {
        return OPJ_FALSE;
    }

    if (!opj_tcd_init(p_j2k->m_tcd, l_image, &(p_j2k->m_cp), p_j2k->m_tp)) {
        opj_tcd_destroy(p_j2k->m_tcd);
        p_j2k->m_tcd = 00;
        opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static const opj_dec_memory_marker_handler_t * opj_j2k_get_marker_handler(
    OPJ_UINT32 p_id)
{
    const opj_dec_memory_marker_handler_t *e;
    for (e = j2k_memory_marker_handler_tab; e->id != 0; ++e) {
        if (e->id == p_id) {
            break; /* we find a handler corresponding to the marker ID*/
        }
    }
    return e;
}

void opj_j2k_destroy(opj_j2k_t *p_j2k)
{
    if (p_j2k == 00) {
        return;
    }

    if (p_j2k->m_is_decoder) {

        if (p_j2k->m_specific_param.m_decoder.m_default_tcp != 00) {
            opj_j2k_tcp_destroy(p_j2k->m_specific_param.m_decoder.m_default_tcp);
            opj_free(p_j2k->m_specific_param.m_decoder.m_default_tcp);
            p_j2k->m_specific_param.m_decoder.m_default_tcp = 00;
        }

        if (p_j2k->m_specific_param.m_decoder.m_header_data != 00) {
            opj_free(p_j2k->m_specific_param.m_decoder.m_header_data);
            p_j2k->m_specific_param.m_decoder.m_header_data = 00;
            p_j2k->m_specific_param.m_decoder.m_header_data_size = 0;
        }

        opj_free(p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode);
        p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode = 00;
        p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode = 0;

        opj_free(p_j2k->m_specific_param.m_decoder.m_tlm.m_tile_part_infos);
        p_j2k->m_specific_param.m_decoder.m_tlm.m_tile_part_infos = NULL;

        opj_free(p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset);
        p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset = NULL;

    } else {

        if (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);
            p_j2k->m_specific_param.m_encoder.m_encoded_tile_data = 00;
        }

        if (p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer);
            p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = 00;
            p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current = 00;
        }

        if (p_j2k->m_specific_param.m_encoder.m_header_tile_data) {
            opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
            p_j2k->m_specific_param.m_encoder.m_header_tile_data = 00;
            p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
        }
    }

    opj_tcd_destroy(p_j2k->m_tcd);

    opj_j2k_cp_destroy(&(p_j2k->m_cp));
    memset(&(p_j2k->m_cp), 0, sizeof(opj_cp_t));

    opj_procedure_list_destroy(p_j2k->m_procedure_list);
    p_j2k->m_procedure_list = 00;

    opj_procedure_list_destroy(p_j2k->m_validation_list);
    p_j2k->m_procedure_list = 00;

    j2k_destroy_cstr_index(p_j2k->cstr_index);
    p_j2k->cstr_index = NULL;

    opj_image_destroy(p_j2k->m_private_image);
    p_j2k->m_private_image = NULL;

    opj_image_destroy(p_j2k->m_output_image);
    p_j2k->m_output_image = NULL;

    opj_thread_pool_destroy(p_j2k->m_tp);
    p_j2k->m_tp = NULL;

    opj_free(p_j2k);
}

void j2k_destroy_cstr_index(opj_codestream_index_t *p_cstr_ind)
{
    if (p_cstr_ind) {

        if (p_cstr_ind->marker) {
            opj_free(p_cstr_ind->marker);
            p_cstr_ind->marker = NULL;
        }

        if (p_cstr_ind->tile_index) {
            OPJ_UINT32 it_tile = 0;

            for (it_tile = 0; it_tile < p_cstr_ind->nb_of_tiles; it_tile++) {

                if (p_cstr_ind->tile_index[it_tile].packet_index) {
                    opj_free(p_cstr_ind->tile_index[it_tile].packet_index);
                    p_cstr_ind->tile_index[it_tile].packet_index = NULL;
                }

                if (p_cstr_ind->tile_index[it_tile].tp_index) {
                    opj_free(p_cstr_ind->tile_index[it_tile].tp_index);
                    p_cstr_ind->tile_index[it_tile].tp_index = NULL;
                }

                if (p_cstr_ind->tile_index[it_tile].marker) {
                    opj_free(p_cstr_ind->tile_index[it_tile].marker);
                    p_cstr_ind->tile_index[it_tile].marker = NULL;

                }
            }

            opj_free(p_cstr_ind->tile_index);
            p_cstr_ind->tile_index = NULL;
        }

        opj_free(p_cstr_ind);
    }
}

static void opj_j2k_tcp_destroy(opj_tcp_t *p_tcp)
{
    if (p_tcp == 00) {
        return;
    }

    if (p_tcp->ppt_markers != 00) {
        OPJ_UINT32 i;
        for (i = 0U; i < p_tcp->ppt_markers_count; ++i) {
            if (p_tcp->ppt_markers[i].m_data != NULL) {
                opj_free(p_tcp->ppt_markers[i].m_data);
            }
        }
        p_tcp->ppt_markers_count = 0U;
        opj_free(p_tcp->ppt_markers);
        p_tcp->ppt_markers = NULL;
    }

    if (p_tcp->ppt_buffer != 00) {
        opj_free(p_tcp->ppt_buffer);
        p_tcp->ppt_buffer = 00;
    }

    if (p_tcp->tccps != 00) {
        opj_free(p_tcp->tccps);
        p_tcp->tccps = 00;
    }

    if (p_tcp->m_mct_coding_matrix != 00) {
        opj_free(p_tcp->m_mct_coding_matrix);
        p_tcp->m_mct_coding_matrix = 00;
    }

    if (p_tcp->m_mct_decoding_matrix != 00) {
        opj_free(p_tcp->m_mct_decoding_matrix);
        p_tcp->m_mct_decoding_matrix = 00;
    }

    if (p_tcp->m_mcc_records) {
        opj_free(p_tcp->m_mcc_records);
        p_tcp->m_mcc_records = 00;
        p_tcp->m_nb_max_mcc_records = 0;
        p_tcp->m_nb_mcc_records = 0;
    }

    if (p_tcp->m_mct_records) {
        opj_mct_data_t * l_mct_data = p_tcp->m_mct_records;
        OPJ_UINT32 i;

        for (i = 0; i < p_tcp->m_nb_mct_records; ++i) {
            if (l_mct_data->m_data) {
                opj_free(l_mct_data->m_data);
                l_mct_data->m_data = 00;
            }

            ++l_mct_data;
        }

        opj_free(p_tcp->m_mct_records);
        p_tcp->m_mct_records = 00;
    }

    if (p_tcp->mct_norms != 00) {
        opj_free(p_tcp->mct_norms);
        p_tcp->mct_norms = 00;
    }

    opj_j2k_tcp_data_destroy(p_tcp);

}

static void opj_j2k_tcp_data_destroy(opj_tcp_t *p_tcp)
{
    if (p_tcp->m_data) {
        opj_free(p_tcp->m_data);
        p_tcp->m_data = NULL;
        p_tcp->m_data_size = 0;
    }
}

static void opj_j2k_cp_destroy(opj_cp_t *p_cp)
{
    OPJ_UINT32 l_nb_tiles;
    opj_tcp_t * l_current_tile = 00;

    if (p_cp == 00) {
        return;
    }
    if (p_cp->tcps != 00) {
        OPJ_UINT32 i;
        l_current_tile = p_cp->tcps;
        l_nb_tiles = p_cp->th * p_cp->tw;

        for (i = 0U; i < l_nb_tiles; ++i) {
            opj_j2k_tcp_destroy(l_current_tile);
            ++l_current_tile;
        }
        opj_free(p_cp->tcps);
        p_cp->tcps = 00;
    }
    if (p_cp->ppm_markers != 00) {
        OPJ_UINT32 i;
        for (i = 0U; i < p_cp->ppm_markers_count; ++i) {
            if (p_cp->ppm_markers[i].m_data != NULL) {
                opj_free(p_cp->ppm_markers[i].m_data);
            }
        }
        p_cp->ppm_markers_count = 0U;
        opj_free(p_cp->ppm_markers);
        p_cp->ppm_markers = NULL;
    }
    opj_free(p_cp->ppm_buffer);
    p_cp->ppm_buffer = 00;
    p_cp->ppm_data =
        NULL; /* ppm_data belongs to the allocated buffer pointed by ppm_buffer */
    opj_free(p_cp->comment);
    p_cp->comment = 00;
    if (! p_cp->m_is_decoder) {
        opj_free(p_cp->m_specific_param.m_enc.m_matrice);
        p_cp->m_specific_param.m_enc.m_matrice = 00;
    }
}

static OPJ_BOOL opj_j2k_need_nb_tile_parts_correction(opj_stream_private_t
        *p_stream, OPJ_UINT32 tile_no, OPJ_BOOL* p_correction_needed,
        opj_event_mgr_t * p_manager)
{
    OPJ_BYTE   l_header_data[10];
    OPJ_OFF_T  l_stream_pos_backup;
    OPJ_UINT32 l_current_marker;
    OPJ_UINT32 l_marker_size;
    OPJ_UINT32 l_tile_no, l_tot_len, l_current_part, l_num_parts;

    /* initialize to no correction needed */
    *p_correction_needed = OPJ_FALSE;

    if (!opj_stream_has_seek(p_stream)) {
        /* We can't do much in this case, seek is needed */
        return OPJ_TRUE;
    }

    l_stream_pos_backup = opj_stream_tell(p_stream);
    if (l_stream_pos_backup == -1) {
        /* let's do nothing */
        return OPJ_TRUE;
    }

    for (;;) {
        /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
        if (opj_stream_read_data(p_stream, l_header_data, 2, p_manager) != 2) {
            /* assume all is OK */
            if (! opj_stream_seek(p_stream, l_stream_pos_backup, p_manager)) {
                return OPJ_FALSE;
            }
            return OPJ_TRUE;
        }

        /* Read 2 bytes from buffer as the new marker ID */
        opj_read_bytes(l_header_data, &l_current_marker, 2);

        if (l_current_marker != J2K_MS_SOT) {
            /* assume all is OK */
            if (! opj_stream_seek(p_stream, l_stream_pos_backup, p_manager)) {
                return OPJ_FALSE;
            }
            return OPJ_TRUE;
        }

        /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
        if (opj_stream_read_data(p_stream, l_header_data, 2, p_manager) != 2) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        /* Read 2 bytes from the buffer as the marker size */
        opj_read_bytes(l_header_data, &l_marker_size, 2);

        /* Check marker size for SOT Marker */
        if (l_marker_size != 10) {
            opj_event_msg(p_manager, EVT_ERROR, "Inconsistent marker size\n");
            return OPJ_FALSE;
        }
        l_marker_size -= 2;

        if (opj_stream_read_data(p_stream, l_header_data, l_marker_size,
                                 p_manager) != l_marker_size) {
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
            return OPJ_FALSE;
        }

        if (! opj_j2k_get_sot_values(l_header_data, l_marker_size, &l_tile_no,
                                     &l_tot_len, &l_current_part, &l_num_parts, p_manager)) {
            return OPJ_FALSE;
        }

        if (l_tile_no == tile_no) {
            /* we found what we were looking for */
            break;
        }

        if (l_tot_len < 14U) {
            /* last SOT until EOC or invalid Psot value */
            /* assume all is OK */
            if (! opj_stream_seek(p_stream, l_stream_pos_backup, p_manager)) {
                return OPJ_FALSE;
            }
            return OPJ_TRUE;
        }
        l_tot_len -= 12U;
        /* look for next SOT marker */
        if (opj_stream_skip(p_stream, (OPJ_OFF_T)(l_tot_len),
                            p_manager) != (OPJ_OFF_T)(l_tot_len)) {
            /* assume all is OK */
            if (! opj_stream_seek(p_stream, l_stream_pos_backup, p_manager)) {
                return OPJ_FALSE;
            }
            return OPJ_TRUE;
        }
    }

    /* check for correction */
    if (l_current_part == l_num_parts) {
        *p_correction_needed = OPJ_TRUE;
    }

    if (! opj_stream_seek(p_stream, l_stream_pos_backup, p_manager)) {
        return OPJ_FALSE;
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_read_tile_header(opj_j2k_t * p_j2k,
                                  OPJ_UINT32 * p_tile_index,
                                  OPJ_UINT32 * p_data_size,
                                  OPJ_INT32 * p_tile_x0, OPJ_INT32 * p_tile_y0,
                                  OPJ_INT32 * p_tile_x1, OPJ_INT32 * p_tile_y1,
                                  OPJ_UINT32 * p_nb_comps,
                                  OPJ_BOOL * p_go_on,
                                  opj_stream_private_t *p_stream,
                                  opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 l_current_marker = J2K_MS_SOT;
    OPJ_UINT32 l_marker_size;
    const opj_dec_memory_marker_handler_t * l_marker_handler = 00;
    opj_tcp_t * l_tcp = NULL;
    const OPJ_UINT32 l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th;

    /* preconditions */
    assert(p_stream != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    /* Reach the End Of Codestream ?*/
    if (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_EOC) {
        l_current_marker = J2K_MS_EOC;
    }
    /* We need to encounter a SOT marker (a new tile-part header) */
    else if (p_j2k->m_specific_param.m_decoder.m_state != J2K_STATE_TPHSOT) {
        return OPJ_FALSE;
    }

    /* Read into the codestream until reach the EOC or ! can_decode ??? FIXME */
    while ((!p_j2k->m_specific_param.m_decoder.m_can_decode) &&
            (l_current_marker != J2K_MS_EOC)) {

        if (p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts > 0 &&
                p_j2k->m_specific_param.m_decoder.m_idx_intersecting_tile_parts <
                p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts) {
            OPJ_OFF_T next_tp_sot_pos;

            next_tp_sot_pos =
                p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset[p_j2k->m_specific_param.m_decoder.m_idx_intersecting_tile_parts];
            ++p_j2k->m_specific_param.m_decoder.m_idx_intersecting_tile_parts;
            if (!(opj_stream_read_seek(p_stream,
                                       next_tp_sot_pos,
                                       p_manager))) {
                opj_event_msg(p_manager, EVT_ERROR, "Problem with seek function\n");
                return OPJ_FALSE;
            }

            /* Try to read 2 bytes (the marker ID) from stream and copy them into the buffer */
            if (opj_stream_read_data(p_stream,
                                     p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
                opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                return OPJ_FALSE;
            }

            /* Read 2 bytes from the buffer as the marker ID */
            opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                           &l_current_marker,
                           2);

            if (l_current_marker != J2K_MS_SOT) {
                opj_event_msg(p_manager, EVT_ERROR, "Did not get expected SOT marker\n");
                return OPJ_FALSE;
            }
        }

        /* Try to read until the Start Of Data is detected */
        while (l_current_marker != J2K_MS_SOD) {

            if (opj_stream_get_number_byte_left(p_stream) == 0) {
                p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_NEOC;
                break;
            }

            /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
            if (opj_stream_read_data(p_stream,
                                     p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
                opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                return OPJ_FALSE;
            }

            /* Read 2 bytes from the buffer as the marker size */
            opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data, &l_marker_size,
                           2);

            /* Check marker size (does not include marker ID but includes marker size) */
            if (l_marker_size < 2) {
                opj_event_msg(p_manager, EVT_ERROR, "Inconsistent marker size\n");
                return OPJ_FALSE;
            }

            /* cf. https://code.google.com/p/openjpeg/issues/detail?id=226 */
            if (l_current_marker == 0x8080 &&
                    opj_stream_get_number_byte_left(p_stream) == 0) {
                p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_NEOC;
                break;
            }

            /* Why this condition? FIXME */
            if ((p_j2k->m_specific_param.m_decoder.m_state & J2K_STATE_TPH) &&
                    p_j2k->m_specific_param.m_decoder.m_sot_length != 0) {
                if (p_j2k->m_specific_param.m_decoder.m_sot_length < l_marker_size + 2) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Sot length is less than marker size + marker ID\n");
                    return OPJ_FALSE;
                }
                p_j2k->m_specific_param.m_decoder.m_sot_length -= (l_marker_size + 2);
            }
            l_marker_size -= 2; /* Subtract the size of the marker ID already read */

            /* Get the marker handler from the marker ID */
            l_marker_handler = opj_j2k_get_marker_handler(l_current_marker);

            /* Check if the marker is known and if it is the right place to find it */
            if (!(p_j2k->m_specific_param.m_decoder.m_state & l_marker_handler->states)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Marker is not compliant with its position\n");
                return OPJ_FALSE;
            }
            /* FIXME manage case of unknown marker as in the main header ? */

            /* Check if the marker size is compatible with the header data size */
            if (l_marker_size > p_j2k->m_specific_param.m_decoder.m_header_data_size) {
                OPJ_BYTE *new_header_data = NULL;
                /* If we are here, this means we consider this marker as known & we will read it */
                /* Check enough bytes left in stream before allocation */
                if ((OPJ_OFF_T)l_marker_size >  opj_stream_get_number_byte_left(p_stream)) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "Marker size inconsistent with stream length\n");
                    return OPJ_FALSE;
                }
                new_header_data = (OPJ_BYTE *) opj_realloc(
                                      p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size);
                if (! new_header_data) {
                    opj_free(p_j2k->m_specific_param.m_decoder.m_header_data);
                    p_j2k->m_specific_param.m_decoder.m_header_data = NULL;
                    p_j2k->m_specific_param.m_decoder.m_header_data_size = 0;
                    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to read header\n");
                    return OPJ_FALSE;
                }
                p_j2k->m_specific_param.m_decoder.m_header_data = new_header_data;
                p_j2k->m_specific_param.m_decoder.m_header_data_size = l_marker_size;
            }

            /* Try to read the rest of the marker segment from stream and copy them into the buffer */
            if (opj_stream_read_data(p_stream,
                                     p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size,
                                     p_manager) != l_marker_size) {
                opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                return OPJ_FALSE;
            }

            if (!l_marker_handler->handler) {
                /* See issue #175 */
                opj_event_msg(p_manager, EVT_ERROR, "Not sure how that happened.\n");
                return OPJ_FALSE;
            }
            /* Read the marker segment with the correct marker handler */
            if (!(*(l_marker_handler->handler))(p_j2k,
                                                p_j2k->m_specific_param.m_decoder.m_header_data, l_marker_size, p_manager)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Fail to read the current marker segment (%#x)\n", l_current_marker);
                return OPJ_FALSE;
            }

            /* Add the marker to the codestream index*/
            if (OPJ_FALSE == opj_j2k_add_tlmarker(p_j2k->m_current_tile_number,
                                                  p_j2k->cstr_index,
                                                  l_marker_handler->id,
                                                  (OPJ_UINT32) opj_stream_tell(p_stream) - l_marker_size - 4,
                                                  l_marker_size + 4)) {
                opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n");
                return OPJ_FALSE;
            }

            /* Keep the position of the last SOT marker read */
            if (l_marker_handler->id == J2K_MS_SOT) {
                OPJ_UINT32 sot_pos = (OPJ_UINT32) opj_stream_tell(p_stream) - l_marker_size - 4
                                     ;
                if (sot_pos > p_j2k->m_specific_param.m_decoder.m_last_sot_read_pos) {
                    p_j2k->m_specific_param.m_decoder.m_last_sot_read_pos = sot_pos;
                }
            }

            if (p_j2k->m_specific_param.m_decoder.m_skip_data) {
                /* Skip the rest of the tile part header*/
                if (opj_stream_skip(p_stream, p_j2k->m_specific_param.m_decoder.m_sot_length,
                                    p_manager) != p_j2k->m_specific_param.m_decoder.m_sot_length) {
                    opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                    return OPJ_FALSE;
                }
                l_current_marker = J2K_MS_SOD; /* Normally we reached a SOD */
            } else {
                /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer*/
                if (opj_stream_read_data(p_stream,
                                         p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
                    opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                    return OPJ_FALSE;
                }
                /* Read 2 bytes from the buffer as the new marker ID */
                opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                               &l_current_marker, 2);
            }
        }
        if (opj_stream_get_number_byte_left(p_stream) == 0
                && p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_NEOC) {
            break;
        }

        /* If we didn't skip data before, we need to read the SOD marker*/
        if (! p_j2k->m_specific_param.m_decoder.m_skip_data) {
            /* Try to read the SOD marker and skip data ? FIXME */
            if (! opj_j2k_read_sod(p_j2k, p_stream, p_manager)) {
                return OPJ_FALSE;
            }

            /* Check if we can use the TLM index to access the next tile-part */
            if (!p_j2k->m_specific_param.m_decoder.m_can_decode &&
                    p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec >= 0 &&
                    p_j2k->m_current_tile_number == (OPJ_UINT32)
                    p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec &&
                    !p_j2k->m_specific_param.m_decoder.m_tlm.m_is_invalid &&
                    opj_stream_has_seek(p_stream)) {
                l_tcp = p_j2k->m_cp.tcps + p_j2k->m_current_tile_number;
                if (l_tcp->m_nb_tile_parts ==
                        p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].nb_tps &&
                        (OPJ_UINT32)l_tcp->m_current_tile_part_number + 1 < l_tcp->m_nb_tile_parts) {
                    const OPJ_OFF_T next_tp_sot_pos = p_j2k->cstr_index->tile_index[
                                                          p_j2k->m_current_tile_number].tp_index[l_tcp->m_current_tile_part_number +
                                                                  1].start_pos;

                    if (next_tp_sot_pos != opj_stream_tell(p_stream)) {
#if 0
                        opj_event_msg(p_manager, EVT_INFO,
                                      "opj_j2k_read_tile_header(tile=%u): seek to tile part %u at %" PRId64 "\n",
                                      p_j2k->m_current_tile_number,
                                      l_tcp->m_current_tile_part_number + 1,
                                      next_tp_sot_pos);
#endif

                        if (!(opj_stream_read_seek(p_stream,
                                                   next_tp_sot_pos,
                                                   p_manager))) {
                            opj_event_msg(p_manager, EVT_ERROR, "Problem with seek function\n");
                            return OPJ_FALSE;
                        }
                    }

                    /* Try to read 2 bytes (the marker ID) from stream and copy them into the buffer */
                    if (opj_stream_read_data(p_stream,
                                             p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
                        opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                        return OPJ_FALSE;
                    }

                    /* Read 2 bytes from the buffer as the marker ID */
                    opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                                   &l_current_marker,
                                   2);

                    if (l_current_marker != J2K_MS_SOT) {
                        opj_event_msg(p_manager, EVT_ERROR, "Did not get expected SOT marker\n");
                        return OPJ_FALSE;
                    }

                    continue;
                }
            }

            if (p_j2k->m_specific_param.m_decoder.m_can_decode &&
                    !p_j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction_checked) {
                /* Issue 254 */
                OPJ_BOOL l_correction_needed = OPJ_FALSE;

                p_j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction_checked = 1;
                if (p_j2k->m_cp.tcps[p_j2k->m_current_tile_number].m_nb_tile_parts == 1) {
                    /* Skip opj_j2k_need_nb_tile_parts_correction() if there is
                     * only a single tile part declared. The
                     * opj_j2k_need_nb_tile_parts_correction() hack was needed
                     * for files with 5 declared tileparts (where they were
                     * actually 6).
                     * Doing it systematically hurts performance when reading
                     * Sentinel2 L1C JPEG2000 files as explained in
                     * https://lists.osgeo.org/pipermail/gdal-dev/2024-November/059805.html
                     */
                } else if (!opj_j2k_need_nb_tile_parts_correction(p_stream,
                           p_j2k->m_current_tile_number, &l_correction_needed, p_manager)) {
                    opj_event_msg(p_manager, EVT_ERROR,
                                  "opj_j2k_apply_nb_tile_parts_correction error\n");
                    return OPJ_FALSE;
                }
                if (l_correction_needed) {
                    OPJ_UINT32 l_tile_no;

                    p_j2k->m_specific_param.m_decoder.m_can_decode = 0;
                    p_j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction = 1;
                    /* correct tiles */
                    for (l_tile_no = 0U; l_tile_no < l_nb_tiles; ++l_tile_no) {
                        if (p_j2k->m_cp.tcps[l_tile_no].m_nb_tile_parts != 0U) {
                            p_j2k->m_cp.tcps[l_tile_no].m_nb_tile_parts += 1;
                        }
                    }
                    opj_event_msg(p_manager, EVT_WARNING,
                                  "Non conformant codestream TPsot==TNsot.\n");
                }
            }
        } else {
            /* Indicate we will try to read a new tile-part header*/
            p_j2k->m_specific_param.m_decoder.m_skip_data = 0;
            p_j2k->m_specific_param.m_decoder.m_can_decode = 0;
            p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_TPHSOT;
        }

        if (! p_j2k->m_specific_param.m_decoder.m_can_decode) {
            /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
            if (opj_stream_read_data(p_stream,
                                     p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {

                /* Deal with likely non conformant SPOT6 files, where the last */
                /* row of tiles have TPsot == 0 and TNsot == 0, and missing EOC, */
                /* but no other tile-parts were found. */
                if (p_j2k->m_current_tile_number + 1 == l_nb_tiles) {
                    OPJ_UINT32 l_tile_no;
                    for (l_tile_no = 0U; l_tile_no < l_nb_tiles; ++l_tile_no) {
                        if (p_j2k->m_cp.tcps[l_tile_no].m_current_tile_part_number == 0 &&
                                p_j2k->m_cp.tcps[l_tile_no].m_nb_tile_parts == 0) {
                            break;
                        }
                    }
                    if (l_tile_no < l_nb_tiles) {
                        opj_event_msg(p_manager, EVT_INFO,
                                      "Tile %u has TPsot == 0 and TNsot == 0, "
                                      "but no other tile-parts were found. "
                                      "EOC is also missing.\n",
                                      l_tile_no);
                        p_j2k->m_current_tile_number = l_tile_no;
                        l_current_marker = J2K_MS_EOC;
                        p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_EOC;
                        break;
                    }
                }

                opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                return OPJ_FALSE;
            }

            /* Read 2 bytes from buffer as the new marker ID */
            opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,
                           &l_current_marker, 2);
        }
    }

    /* Current marker is the EOC marker ?*/
    if (l_current_marker == J2K_MS_EOC) {
        if (p_j2k->m_specific_param.m_decoder.m_state != J2K_STATE_EOC) {
            p_j2k->m_current_tile_number = 0;
            p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_EOC;
        }
    }

    /* Deal with tiles that have a single tile-part with TPsot == 0 and TNsot == 0 */
    if (! p_j2k->m_specific_param.m_decoder.m_can_decode) {
        l_tcp = p_j2k->m_cp.tcps + p_j2k->m_current_tile_number;

        while ((p_j2k->m_current_tile_number < l_nb_tiles) && (l_tcp->m_data == 00)) {
            ++p_j2k->m_current_tile_number;
            ++l_tcp;
        }

        if (p_j2k->m_current_tile_number == l_nb_tiles) {
            *p_go_on = OPJ_FALSE;
            return OPJ_TRUE;
        }
    }

    if (! opj_j2k_merge_ppt(p_j2k->m_cp.tcps + p_j2k->m_current_tile_number,
                            p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to merge PPT data\n");
        return OPJ_FALSE;
    }
    /*FIXME ???*/
    if (! opj_tcd_init_decode_tile(p_j2k->m_tcd, p_j2k->m_current_tile_number,
                                   p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
        return OPJ_FALSE;
    }

    opj_event_msg(p_manager, EVT_INFO, "Header of tile %d / %d has been read.\n",
                  p_j2k->m_current_tile_number + 1, (p_j2k->m_cp.th * p_j2k->m_cp.tw));

    *p_tile_index = p_j2k->m_current_tile_number;
    *p_go_on = OPJ_TRUE;
    if (p_data_size) {
        /* For internal use in j2k.c, we don't need this */
        /* This is just needed for folks using the opj_read_tile_header() / opj_decode_tile_data() combo */
        *p_data_size = opj_tcd_get_decoded_tile_size(p_j2k->m_tcd, OPJ_FALSE);
        if (*p_data_size == UINT_MAX) {
            return OPJ_FALSE;
        }
    }
    *p_tile_x0 = p_j2k->m_tcd->tcd_image->tiles->x0;
    *p_tile_y0 = p_j2k->m_tcd->tcd_image->tiles->y0;
    *p_tile_x1 = p_j2k->m_tcd->tcd_image->tiles->x1;
    *p_tile_y1 = p_j2k->m_tcd->tcd_image->tiles->y1;
    *p_nb_comps = p_j2k->m_tcd->tcd_image->tiles->numcomps;

    p_j2k->m_specific_param.m_decoder.m_state |= J2K_STATE_DATA;

    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_decode_tile(opj_j2k_t * p_j2k,
                             OPJ_UINT32 p_tile_index,
                             OPJ_BYTE * p_data,
                             OPJ_UINT32 p_data_size,
                             opj_stream_private_t *p_stream,
                             opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 l_current_marker;
    OPJ_BYTE l_data [2];
    opj_tcp_t * l_tcp;
    opj_image_t* l_image_for_bounds;

    /* preconditions */
    assert(p_stream != 00);
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (!(p_j2k->m_specific_param.m_decoder.m_state & J2K_STATE_DATA)
            || (p_tile_index != p_j2k->m_current_tile_number)) {
        return OPJ_FALSE;
    }

    l_tcp = &(p_j2k->m_cp.tcps[p_tile_index]);
    if (! l_tcp->m_data) {
        opj_j2k_tcp_destroy(l_tcp);
        return OPJ_FALSE;
    }

    /* When using the opj_read_tile_header / opj_decode_tile_data API */
    /* such as in test_tile_decoder, m_output_image is NULL, so fall back */
    /* to the full image dimension. This is a bit surprising that */
    /* opj_set_decode_area() is only used to determine intersecting tiles, */
    /* but full tile decoding is done */
    l_image_for_bounds = p_j2k->m_output_image ? p_j2k->m_output_image :
                         p_j2k->m_private_image;
    if (! opj_tcd_decode_tile(p_j2k->m_tcd,
                              l_image_for_bounds->x0,
                              l_image_for_bounds->y0,
                              l_image_for_bounds->x1,
                              l_image_for_bounds->y1,
                              p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode,
                              p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode,
                              l_tcp->m_data,
                              l_tcp->m_data_size,
                              p_tile_index,
                              p_j2k->cstr_index, p_manager)) {
        opj_j2k_tcp_destroy(l_tcp);
        p_j2k->m_specific_param.m_decoder.m_state |= J2K_STATE_ERR;
        opj_event_msg(p_manager, EVT_ERROR, "Failed to decode.\n");
        return OPJ_FALSE;
    }

    /* p_data can be set to NULL when the call will take care of using */
    /* itself the TCD data. This is typically the case for whole single */
    /* tile decoding optimization. */
    if (p_data != NULL) {
        if (! opj_tcd_update_tile_data(p_j2k->m_tcd, p_data, p_data_size)) {
            return OPJ_FALSE;
        }

        /* To avoid to destroy the tcp which can be useful when we try to decode a tile decoded before (cf j2k_random_tile_access)
        * we destroy just the data which will be re-read in read_tile_header*/
        /*opj_j2k_tcp_destroy(l_tcp);
        p_j2k->m_tcd->tcp = 0;*/
        opj_j2k_tcp_data_destroy(l_tcp);
    }

    p_j2k->m_specific_param.m_decoder.m_can_decode = 0;
    p_j2k->m_specific_param.m_decoder.m_state &= (~(OPJ_UINT32)J2K_STATE_DATA);

    if (opj_stream_get_number_byte_left(p_stream) == 0
            && p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_NEOC) {
        return OPJ_TRUE;
    }

    if (p_j2k->m_specific_param.m_decoder.m_state != J2K_STATE_EOC) {
        if (opj_stream_read_data(p_stream, l_data, 2, p_manager) != 2) {
            opj_event_msg(p_manager, p_j2k->m_cp.strict ? EVT_ERROR : EVT_WARNING,
                          "Stream too short\n");
            return p_j2k->m_cp.strict ? OPJ_FALSE : OPJ_TRUE;
        }
        opj_read_bytes(l_data, &l_current_marker, 2);

        if (l_current_marker == J2K_MS_EOC) {
            p_j2k->m_current_tile_number = 0;
            p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_EOC;
        } else if (l_current_marker != J2K_MS_SOT) {
            if (opj_stream_get_number_byte_left(p_stream) == 0) {
                p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_NEOC;
                opj_event_msg(p_manager, EVT_WARNING, "Stream does not end with EOC\n");
                return OPJ_TRUE;
            }
            opj_event_msg(p_manager, EVT_ERROR, "Stream too short, expected SOT\n");
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_update_image_data(opj_tcd_t * p_tcd,
        opj_image_t* p_output_image)
{
    OPJ_UINT32 i, j;
    OPJ_UINT32 l_width_src, l_height_src;
    OPJ_UINT32 l_width_dest, l_height_dest;
    OPJ_INT32 l_offset_x0_src, l_offset_y0_src, l_offset_x1_src, l_offset_y1_src;
    OPJ_SIZE_T l_start_offset_src;
    OPJ_UINT32 l_start_x_dest, l_start_y_dest;
    OPJ_UINT32 l_x0_dest, l_y0_dest, l_x1_dest, l_y1_dest;
    OPJ_SIZE_T l_start_offset_dest;

    opj_image_comp_t * l_img_comp_src = 00;
    opj_image_comp_t * l_img_comp_dest = 00;

    opj_tcd_tilecomp_t * l_tilec = 00;
    opj_image_t * l_image_src = 00;
    OPJ_INT32 * l_dest_ptr;

    l_tilec = p_tcd->tcd_image->tiles->comps;
    l_image_src = p_tcd->image;
    l_img_comp_src = l_image_src->comps;

    l_img_comp_dest = p_output_image->comps;

    for (i = 0; i < l_image_src->numcomps;
            i++, ++l_img_comp_dest, ++l_img_comp_src,  ++l_tilec) {
        OPJ_INT32 res_x0, res_x1, res_y0, res_y1;
        OPJ_UINT32 src_data_stride;
        const OPJ_INT32* p_src_data;

        /* Copy info from decoded comp image to output image */
        l_img_comp_dest->resno_decoded = l_img_comp_src->resno_decoded;

        if (p_tcd->whole_tile_decoding) {
            opj_tcd_resolution_t* l_res = l_tilec->resolutions +
                                          l_img_comp_src->resno_decoded;
            res_x0 = l_res->x0;
            res_y0 = l_res->y0;
            res_x1 = l_res->x1;
            res_y1 = l_res->y1;
            src_data_stride = (OPJ_UINT32)(
                                  l_tilec->resolutions[l_tilec->minimum_num_resolutions - 1].x1 -
                                  l_tilec->resolutions[l_tilec->minimum_num_resolutions - 1].x0);
            p_src_data = l_tilec->data;
        } else {
            opj_tcd_resolution_t* l_res = l_tilec->resolutions +
                                          l_img_comp_src->resno_decoded;
            res_x0 = (OPJ_INT32)l_res->win_x0;
            res_y0 = (OPJ_INT32)l_res->win_y0;
            res_x1 = (OPJ_INT32)l_res->win_x1;
            res_y1 = (OPJ_INT32)l_res->win_y1;
            src_data_stride = l_res->win_x1 - l_res->win_x0;
            p_src_data = l_tilec->data_win;
        }

        if (p_src_data == NULL) {
            /* Happens for partial component decoding */
            continue;
        }

        l_width_src = (OPJ_UINT32)(res_x1 - res_x0);
        l_height_src = (OPJ_UINT32)(res_y1 - res_y0);


        /* Current tile component size*/
        /*if (i == 0) {
        fprintf(stdout, "SRC: l_res_x0=%d, l_res_x1=%d, l_res_y0=%d, l_res_y1=%d\n",
                        res_x0, res_x1, res_y0, res_y1);
        }*/


        /* Border of the current output component*/
        l_x0_dest = opj_uint_ceildivpow2(l_img_comp_dest->x0, l_img_comp_dest->factor);
        l_y0_dest = opj_uint_ceildivpow2(l_img_comp_dest->y0, l_img_comp_dest->factor);
        l_x1_dest = l_x0_dest +
                    l_img_comp_dest->w; /* can't overflow given that image->x1 is uint32 */
        l_y1_dest = l_y0_dest + l_img_comp_dest->h;

        /*if (i == 0) {
        fprintf(stdout, "DEST: l_x0_dest=%d, l_x1_dest=%d, l_y0_dest=%d, l_y1_dest=%d (%d)\n",
                        l_x0_dest, l_x1_dest, l_y0_dest, l_y1_dest, l_img_comp_dest->factor );
        }*/

        /*-----*/
        /* Compute the area (l_offset_x0_src, l_offset_y0_src, l_offset_x1_src, l_offset_y1_src)
         * of the input buffer (decoded tile component) which will be move
         * in the output buffer. Compute the area of the output buffer (l_start_x_dest,
         * l_start_y_dest, l_width_dest, l_height_dest)  which will be modified
         * by this input area.
         * */
        assert(res_x0 >= 0);
        assert(res_x1 >= 0);
        if (l_x0_dest < (OPJ_UINT32)res_x0) {
            l_start_x_dest = (OPJ_UINT32)res_x0 - l_x0_dest;
            l_offset_x0_src = 0;

            if (l_x1_dest >= (OPJ_UINT32)res_x1) {
                l_width_dest = l_width_src;
                l_offset_x1_src = 0;
            } else {
                l_width_dest = l_x1_dest - (OPJ_UINT32)res_x0 ;
                l_offset_x1_src = (OPJ_INT32)(l_width_src - l_width_dest);
            }
        } else {
            l_start_x_dest = 0U;
            l_offset_x0_src = (OPJ_INT32)l_x0_dest - res_x0;

            if (l_x1_dest >= (OPJ_UINT32)res_x1) {
                l_width_dest = l_width_src - (OPJ_UINT32)l_offset_x0_src;
                l_offset_x1_src = 0;
            } else {
                l_width_dest = l_img_comp_dest->w ;
                l_offset_x1_src = res_x1 - (OPJ_INT32)l_x1_dest;
            }
        }

        if (l_y0_dest < (OPJ_UINT32)res_y0) {
            l_start_y_dest = (OPJ_UINT32)res_y0 - l_y0_dest;
            l_offset_y0_src = 0;

            if (l_y1_dest >= (OPJ_UINT32)res_y1) {
                l_height_dest = l_height_src;
                l_offset_y1_src = 0;
            } else {
                l_height_dest = l_y1_dest - (OPJ_UINT32)res_y0 ;
                l_offset_y1_src = (OPJ_INT32)(l_height_src - l_height_dest);
            }
        } else {
            l_start_y_dest = 0U;
            l_offset_y0_src = (OPJ_INT32)l_y0_dest - res_y0;

            if (l_y1_dest >= (OPJ_UINT32)res_y1) {
                l_height_dest = l_height_src - (OPJ_UINT32)l_offset_y0_src;
                l_offset_y1_src = 0;
            } else {
                l_height_dest = l_img_comp_dest->h ;
                l_offset_y1_src = res_y1 - (OPJ_INT32)l_y1_dest;
            }
        }

        if ((l_offset_x0_src < 0) || (l_offset_y0_src < 0) || (l_offset_x1_src < 0) ||
                (l_offset_y1_src < 0)) {
            return OPJ_FALSE;
        }
        /* testcase 2977.pdf.asan.67.2198 */
        if ((OPJ_INT32)l_width_dest < 0 || (OPJ_INT32)l_height_dest < 0) {
            return OPJ_FALSE;
        }
        /*-----*/

        /* Compute the input buffer offset */
        l_start_offset_src = (OPJ_SIZE_T)l_offset_x0_src + (OPJ_SIZE_T)l_offset_y0_src
                             * (OPJ_SIZE_T)src_data_stride;

        /* Compute the output buffer offset */
        l_start_offset_dest = (OPJ_SIZE_T)l_start_x_dest + (OPJ_SIZE_T)l_start_y_dest
                              * (OPJ_SIZE_T)l_img_comp_dest->w;

        /* Allocate output component buffer if necessary */
        if (l_img_comp_dest->data == NULL &&
                l_start_offset_src == 0 && l_start_offset_dest == 0 &&
                src_data_stride == l_img_comp_dest->w &&
                l_width_dest == l_img_comp_dest->w &&
                l_height_dest == l_img_comp_dest->h) {
            /* If the final image matches the tile buffer, then borrow it */
            /* directly to save a copy */
            if (p_tcd->whole_tile_decoding) {
                l_img_comp_dest->data = l_tilec->data;
                l_tilec->data = NULL;
            } else {
                l_img_comp_dest->data = l_tilec->data_win;
                l_tilec->data_win = NULL;
            }
            continue;
        } else if (l_img_comp_dest->data == NULL) {
            OPJ_SIZE_T l_width = l_img_comp_dest->w;
            OPJ_SIZE_T l_height = l_img_comp_dest->h;

            if ((l_height == 0U) || (l_width > (SIZE_MAX / l_height)) ||
                    l_width * l_height > SIZE_MAX / sizeof(OPJ_INT32)) {
                /* would overflow */
                return OPJ_FALSE;
            }
            l_img_comp_dest->data = (OPJ_INT32*) opj_image_data_alloc(l_width * l_height *
                                    sizeof(OPJ_INT32));
            if (! l_img_comp_dest->data) {
                return OPJ_FALSE;
            }

            if (l_img_comp_dest->w != l_width_dest ||
                    l_img_comp_dest->h != l_height_dest) {
                memset(l_img_comp_dest->data, 0,
                       (OPJ_SIZE_T)l_img_comp_dest->w * l_img_comp_dest->h * sizeof(OPJ_INT32));
            }
        }

        /* Move the output buffer to the first place where we will write*/
        l_dest_ptr = l_img_comp_dest->data + l_start_offset_dest;

        {
            const OPJ_INT32 * l_src_ptr = p_src_data;
            l_src_ptr += l_start_offset_src;

            for (j = 0; j < l_height_dest; ++j) {
                memcpy(l_dest_ptr, l_src_ptr, l_width_dest * sizeof(OPJ_INT32));
                l_dest_ptr += l_img_comp_dest->w;
                l_src_ptr += src_data_stride;
            }
        }


    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_update_image_dimensions(opj_image_t* p_image,
        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 it_comp;
    OPJ_INT32 l_comp_x1, l_comp_y1;
    opj_image_comp_t* l_img_comp = NULL;

    l_img_comp = p_image->comps;
    for (it_comp = 0; it_comp < p_image->numcomps; ++it_comp) {
        OPJ_INT32 l_h, l_w;
        if (p_image->x0 > (OPJ_UINT32)INT_MAX ||
                p_image->y0 > (OPJ_UINT32)INT_MAX ||
                p_image->x1 > (OPJ_UINT32)INT_MAX ||
                p_image->y1 > (OPJ_UINT32)INT_MAX) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Image coordinates above INT_MAX are not supported\n");
            return OPJ_FALSE;
        }

        l_img_comp->x0 = opj_uint_ceildiv(p_image->x0, l_img_comp->dx);
        l_img_comp->y0 = opj_uint_ceildiv(p_image->y0, l_img_comp->dy);
        l_comp_x1 = opj_int_ceildiv((OPJ_INT32)p_image->x1, (OPJ_INT32)l_img_comp->dx);
        l_comp_y1 = opj_int_ceildiv((OPJ_INT32)p_image->y1, (OPJ_INT32)l_img_comp->dy);

        l_w = opj_int_ceildivpow2(l_comp_x1, (OPJ_INT32)l_img_comp->factor)
              - opj_int_ceildivpow2((OPJ_INT32)l_img_comp->x0, (OPJ_INT32)l_img_comp->factor);
        if (l_w < 0) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Size x of the decoded component image is incorrect (comp[%d].w=%d).\n",
                          it_comp, l_w);
            return OPJ_FALSE;
        }
        l_img_comp->w = (OPJ_UINT32)l_w;

        l_h = opj_int_ceildivpow2(l_comp_y1, (OPJ_INT32)l_img_comp->factor)
              - opj_int_ceildivpow2((OPJ_INT32)l_img_comp->y0, (OPJ_INT32)l_img_comp->factor);
        if (l_h < 0) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Size y of the decoded component image is incorrect (comp[%d].h=%d).\n",
                          it_comp, l_h);
            return OPJ_FALSE;
        }
        l_img_comp->h = (OPJ_UINT32)l_h;

        l_img_comp++;
    }

    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_set_decoded_components(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 numcomps,
                                        const OPJ_UINT32* comps_indices,
                                        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i;
    OPJ_BOOL* already_mapped;

    if (p_j2k->m_private_image == NULL) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "opj_read_header() should be called before "
                      "opj_set_decoded_components().\n");
        return OPJ_FALSE;
    }

    already_mapped = (OPJ_BOOL*) opj_calloc(sizeof(OPJ_BOOL),
                                            p_j2k->m_private_image->numcomps);
    if (already_mapped == NULL) {
        return OPJ_FALSE;
    }

    for (i = 0; i < numcomps; i++) {
        if (comps_indices[i] >= p_j2k->m_private_image->numcomps) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid component index: %u\n",
                          comps_indices[i]);
            opj_free(already_mapped);
            return OPJ_FALSE;
        }
        if (already_mapped[comps_indices[i]]) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Component index %u used several times\n",
                          comps_indices[i]);
            opj_free(already_mapped);
            return OPJ_FALSE;
        }
        already_mapped[comps_indices[i]] = OPJ_TRUE;
    }
    opj_free(already_mapped);

    opj_free(p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode);
    if (numcomps) {
        p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode =
            (OPJ_UINT32*) opj_malloc(numcomps * sizeof(OPJ_UINT32));
        if (p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode == NULL) {
            p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode = 0;
            return OPJ_FALSE;
        }
        memcpy(p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode,
               comps_indices,
               numcomps * sizeof(OPJ_UINT32));
    } else {
        p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode = NULL;
    }
    p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode = numcomps;

    return OPJ_TRUE;
}


OPJ_BOOL opj_j2k_set_decode_area(opj_j2k_t *p_j2k,
                                 opj_image_t* p_image,
                                 OPJ_INT32 p_start_x, OPJ_INT32 p_start_y,
                                 OPJ_INT32 p_end_x, OPJ_INT32 p_end_y,
                                 opj_event_mgr_t * p_manager)
{
    opj_cp_t * l_cp = &(p_j2k->m_cp);
    opj_image_t * l_image = p_j2k->m_private_image;
    OPJ_BOOL ret;
    OPJ_UINT32 it_comp;

    if (p_j2k->m_cp.tw == 1 && p_j2k->m_cp.th == 1 &&
            p_j2k->m_cp.tcps[0].m_data != NULL) {
        /* In the case of a single-tiled image whose codestream we have already */
        /* ingested, go on */
    }
    /* Check if we are read the main header */
    else if (p_j2k->m_specific_param.m_decoder.m_state != J2K_STATE_TPHSOT) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Need to decode the main header before begin to decode the remaining codestream.\n");
        return OPJ_FALSE;
    }

    /* Update the comps[].factor member of the output image with the one */
    /* of m_reduce */
    for (it_comp = 0; it_comp < p_image->numcomps; ++it_comp) {
        p_image->comps[it_comp].factor = p_j2k->m_cp.m_specific_param.m_dec.m_reduce;
    }

    if (!p_start_x && !p_start_y && !p_end_x && !p_end_y) {
        opj_event_msg(p_manager, EVT_INFO,
                      "No decoded area parameters, set the decoded area to the whole image\n");

        p_j2k->m_specific_param.m_decoder.m_start_tile_x = 0;
        p_j2k->m_specific_param.m_decoder.m_start_tile_y = 0;
        p_j2k->m_specific_param.m_decoder.m_end_tile_x = l_cp->tw;
        p_j2k->m_specific_param.m_decoder.m_end_tile_y = l_cp->th;

        p_image->x0 = l_image->x0;
        p_image->y0 = l_image->y0;
        p_image->x1 = l_image->x1;
        p_image->y1 = l_image->y1;

        return opj_j2k_update_image_dimensions(p_image, p_manager);
    }

    /* ----- */
    /* Check if the positions provided by the user are correct */

    /* Left */
    if (p_start_x < 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Left position of the decoded area (region_x0=%d) should be >= 0.\n",
                      p_start_x);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_start_x > l_image->x1) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Left position of the decoded area (region_x0=%d) is outside the image area (Xsiz=%d).\n",
                      p_start_x, l_image->x1);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_start_x < l_image->x0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Left position of the decoded area (region_x0=%d) is outside the image area (XOsiz=%d).\n",
                      p_start_x, l_image->x0);
        p_j2k->m_specific_param.m_decoder.m_start_tile_x = 0;
        p_image->x0 = l_image->x0;
    } else {
        p_j2k->m_specific_param.m_decoder.m_start_tile_x = ((OPJ_UINT32)p_start_x -
                l_cp->tx0) / l_cp->tdx;
        p_image->x0 = (OPJ_UINT32)p_start_x;
    }

    /* Up */
    if (p_start_y < 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Up position of the decoded area (region_y0=%d) should be >= 0.\n",
                      p_start_y);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_start_y > l_image->y1) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Up position of the decoded area (region_y0=%d) is outside the image area (Ysiz=%d).\n",
                      p_start_y, l_image->y1);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_start_y < l_image->y0) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Up position of the decoded area (region_y0=%d) is outside the image area (YOsiz=%d).\n",
                      p_start_y, l_image->y0);
        p_j2k->m_specific_param.m_decoder.m_start_tile_y = 0;
        p_image->y0 = l_image->y0;
    } else {
        p_j2k->m_specific_param.m_decoder.m_start_tile_y = ((OPJ_UINT32)p_start_y -
                l_cp->ty0) / l_cp->tdy;
        p_image->y0 = (OPJ_UINT32)p_start_y;
    }

    /* Right */
    if (p_end_x <= 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Right position of the decoded area (region_x1=%d) should be > 0.\n",
                      p_end_x);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_end_x < l_image->x0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Right position of the decoded area (region_x1=%d) is outside the image area (XOsiz=%d).\n",
                      p_end_x, l_image->x0);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_end_x > l_image->x1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Right position of the decoded area (region_x1=%d) is outside the image area (Xsiz=%d).\n",
                      p_end_x, l_image->x1);
        p_j2k->m_specific_param.m_decoder.m_end_tile_x = l_cp->tw;
        p_image->x1 = l_image->x1;
    } else {
        p_j2k->m_specific_param.m_decoder.m_end_tile_x = opj_uint_ceildiv((
                    OPJ_UINT32)p_end_x - l_cp->tx0, l_cp->tdx);
        p_image->x1 = (OPJ_UINT32)p_end_x;
    }

    /* Bottom */
    if (p_end_y <= 0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Bottom position of the decoded area (region_y1=%d) should be > 0.\n",
                      p_end_y);
        return OPJ_FALSE;
    } else if ((OPJ_UINT32)p_end_y < l_image->y0) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Bottom position of the decoded area (region_y1=%d) is outside the image area (YOsiz=%d).\n",
                      p_end_y, l_image->y0);
        return OPJ_FALSE;
    }
    if ((OPJ_UINT32)p_end_y > l_image->y1) {
        opj_event_msg(p_manager, EVT_WARNING,
                      "Bottom position of the decoded area (region_y1=%d) is outside the image area (Ysiz=%d).\n",
                      p_end_y, l_image->y1);
        p_j2k->m_specific_param.m_decoder.m_end_tile_y = l_cp->th;
        p_image->y1 = l_image->y1;
    } else {
        p_j2k->m_specific_param.m_decoder.m_end_tile_y = opj_uint_ceildiv((
                    OPJ_UINT32)p_end_y - l_cp->ty0, l_cp->tdy);
        p_image->y1 = (OPJ_UINT32)p_end_y;
    }
    /* ----- */

    p_j2k->m_specific_param.m_decoder.m_discard_tiles = 1;

    ret = opj_j2k_update_image_dimensions(p_image, p_manager);

    if (ret) {
        opj_event_msg(p_manager, EVT_INFO, "Setting decoding area to %d,%d,%d,%d\n",
                      p_image->x0, p_image->y0, p_image->x1, p_image->y1);
    }

    return ret;
}

opj_j2k_t* opj_j2k_create_decompress(void)
{
    opj_j2k_t *l_j2k = (opj_j2k_t*) opj_calloc(1, sizeof(opj_j2k_t));
    if (!l_j2k) {
        return 00;
    }

    l_j2k->m_is_decoder = 1;
    l_j2k->m_cp.m_is_decoder = 1;
    /* in the absence of JP2 boxes, consider different bit depth / sign */
    /* per component is allowed */
    l_j2k->m_cp.allow_different_bit_depth_sign = 1;

    /* Default to using strict mode. */
    l_j2k->m_cp.strict = OPJ_TRUE;

#ifdef OPJ_DISABLE_TPSOT_FIX
    l_j2k->m_specific_param.m_decoder.m_nb_tile_parts_correction_checked = 1;
#endif

    l_j2k->m_specific_param.m_decoder.m_default_tcp = (opj_tcp_t*) opj_calloc(1,
            sizeof(opj_tcp_t));
    if (!l_j2k->m_specific_param.m_decoder.m_default_tcp) {
        opj_j2k_destroy(l_j2k);
        return 00;
    }

    l_j2k->m_specific_param.m_decoder.m_header_data = (OPJ_BYTE *) opj_calloc(1,
            OPJ_J2K_DEFAULT_HEADER_SIZE);
    if (! l_j2k->m_specific_param.m_decoder.m_header_data) {
        opj_j2k_destroy(l_j2k);
        return 00;
    }

    l_j2k->m_specific_param.m_decoder.m_header_data_size =
        OPJ_J2K_DEFAULT_HEADER_SIZE;

    l_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec = -1 ;

    l_j2k->m_specific_param.m_decoder.m_last_sot_read_pos = 0 ;

    /* codestream index creation */
    l_j2k->cstr_index = opj_j2k_create_cstr_index();
    if (!l_j2k->cstr_index) {
        opj_j2k_destroy(l_j2k);
        return 00;
    }

    /* validation list creation */
    l_j2k->m_validation_list = opj_procedure_list_create();
    if (! l_j2k->m_validation_list) {
        opj_j2k_destroy(l_j2k);
        return 00;
    }

    /* execution list creation */
    l_j2k->m_procedure_list = opj_procedure_list_create();
    if (! l_j2k->m_procedure_list) {
        opj_j2k_destroy(l_j2k);
        return 00;
    }

    l_j2k->m_tp = opj_thread_pool_create(opj_j2k_get_default_thread_count());
    if (!l_j2k->m_tp) {
        l_j2k->m_tp = opj_thread_pool_create(0);
    }
    if (!l_j2k->m_tp) {
        opj_j2k_destroy(l_j2k);
        return NULL;
    }

    return l_j2k;
}

static opj_codestream_index_t* opj_j2k_create_cstr_index(void)
{
    opj_codestream_index_t* cstr_index = (opj_codestream_index_t*)
                                         opj_calloc(1, sizeof(opj_codestream_index_t));
    if (!cstr_index) {
        return NULL;
    }

    cstr_index->maxmarknum = 100;
    cstr_index->marknum = 0;
    cstr_index->marker = (opj_marker_info_t*)
                         opj_calloc(cstr_index->maxmarknum, sizeof(opj_marker_info_t));
    if (!cstr_index-> marker) {
        opj_free(cstr_index);
        return NULL;
    }

    cstr_index->tile_index = NULL;

    return cstr_index;
}

static OPJ_UINT32 opj_j2k_get_SPCod_SPCoc_size(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no)
{
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp = &l_tcp->tccps[p_comp_no];

    /* preconditions again */
    assert(p_tile_no < (l_cp->tw * l_cp->th));
    assert(p_comp_no < p_j2k->m_private_image->numcomps);

    if (l_tccp->csty & J2K_CCP_CSTY_PRT) {
        return 5 + l_tccp->numresolutions;
    } else {
        return 5;
    }
}

static OPJ_BOOL opj_j2k_compare_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no, OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no)
{
    OPJ_UINT32 i;
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_tccp_t *l_tccp0 = NULL;
    opj_tccp_t *l_tccp1 = NULL;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp0 = &l_tcp->tccps[p_first_comp_no];
    l_tccp1 = &l_tcp->tccps[p_second_comp_no];

    if (l_tccp0->numresolutions != l_tccp1->numresolutions) {
        return OPJ_FALSE;
    }
    if (l_tccp0->cblkw != l_tccp1->cblkw) {
        return OPJ_FALSE;
    }
    if (l_tccp0->cblkh != l_tccp1->cblkh) {
        return OPJ_FALSE;
    }
    if (l_tccp0->cblksty != l_tccp1->cblksty) {
        return OPJ_FALSE;
    }
    if (l_tccp0->qmfbid != l_tccp1->qmfbid) {
        return OPJ_FALSE;
    }
    if ((l_tccp0->csty & J2K_CCP_CSTY_PRT) != (l_tccp1->csty & J2K_CCP_CSTY_PRT)) {
        return OPJ_FALSE;
    }

    for (i = 0U; i < l_tccp0->numresolutions; ++i) {
        if (l_tccp0->prcw[i] != l_tccp1->prcw[i]) {
            return OPJ_FALSE;
        }
        if (l_tccp0->prch[i] != l_tccp1->prch[i]) {
            return OPJ_FALSE;
        }
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_header_size,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 i;
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_header_size != 00);
    assert(p_manager != 00);
    assert(p_data != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp = &l_tcp->tccps[p_comp_no];

    /* preconditions again */
    assert(p_tile_no < (l_cp->tw * l_cp->th));
    assert(p_comp_no < (p_j2k->m_private_image->numcomps));

    if (*p_header_size < 5) {
        opj_event_msg(p_manager, EVT_ERROR, "Error writing SPCod SPCoc element\n");
        return OPJ_FALSE;
    }

    opj_write_bytes(p_data, l_tccp->numresolutions - 1, 1); /* SPcoc (D) */
    ++p_data;

    opj_write_bytes(p_data, l_tccp->cblkw - 2, 1);                  /* SPcoc (E) */
    ++p_data;

    opj_write_bytes(p_data, l_tccp->cblkh - 2, 1);                  /* SPcoc (F) */
    ++p_data;

    opj_write_bytes(p_data, l_tccp->cblksty,
                    1);                            /* SPcoc (G) */
    ++p_data;

    opj_write_bytes(p_data, l_tccp->qmfbid,
                    1);                             /* SPcoc (H) */
    ++p_data;

    *p_header_size = *p_header_size - 5;

    if (l_tccp->csty & J2K_CCP_CSTY_PRT) {

        if (*p_header_size < l_tccp->numresolutions) {
            opj_event_msg(p_manager, EVT_ERROR, "Error writing SPCod SPCoc element\n");
            return OPJ_FALSE;
        }

        for (i = 0; i < l_tccp->numresolutions; ++i) {
            opj_write_bytes(p_data, l_tccp->prcw[i] + (l_tccp->prch[i] << 4),
                            1);   /* SPcoc (I_i) */
            ++p_data;
        }

        *p_header_size = *p_header_size - l_tccp->numresolutions;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_SPCod_SPCoc(opj_j2k_t *p_j2k,
        OPJ_UINT32 compno,
        OPJ_BYTE * p_header_data,
        OPJ_UINT32 * p_header_size,
        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, l_tmp;
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_tccp_t *l_tccp = NULL;
    OPJ_BYTE * l_current_ptr = NULL;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_header_data != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH) ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    /* precondition again */
    assert(compno < p_j2k->m_private_image->numcomps);

    l_tccp = &l_tcp->tccps[compno];
    l_current_ptr = p_header_data;

    /* make sure room is sufficient */
    if (*p_header_size < 5) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n");
        return OPJ_FALSE;
    }

    /* SPcod (D) / SPcoc (A) */
    opj_read_bytes(l_current_ptr, &l_tccp->numresolutions, 1);
    ++l_tccp->numresolutions;  /* tccp->numresolutions = read() + 1 */
    if (l_tccp->numresolutions > OPJ_J2K_MAXRLVLS) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Invalid value for numresolutions : %d, max value is set in openjpeg.h at %d\n",
                      l_tccp->numresolutions, OPJ_J2K_MAXRLVLS);
        return OPJ_FALSE;
    }
    ++l_current_ptr;

    /* If user wants to remove more resolutions than the codestream contains, return error */
    if (l_cp->m_specific_param.m_dec.m_reduce >= l_tccp->numresolutions) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error decoding component %d.\nThe number of resolutions "
                      "to remove (%d) is greater or equal than the number "
                      "of resolutions of this component (%d)\nModify the cp_reduce parameter.\n\n",
                      compno, l_cp->m_specific_param.m_dec.m_reduce, l_tccp->numresolutions);
        p_j2k->m_specific_param.m_decoder.m_state |=
            0x8000;/* FIXME J2K_DEC_STATE_ERR;*/
        return OPJ_FALSE;
    }

    /* SPcod (E) / SPcoc (B) */
    opj_read_bytes(l_current_ptr, &l_tccp->cblkw, 1);
    ++l_current_ptr;
    l_tccp->cblkw += 2;

    /* SPcod (F) / SPcoc (C) */
    opj_read_bytes(l_current_ptr, &l_tccp->cblkh, 1);
    ++l_current_ptr;
    l_tccp->cblkh += 2;

    if ((l_tccp->cblkw > 10) || (l_tccp->cblkh > 10) ||
            ((l_tccp->cblkw + l_tccp->cblkh) > 12)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error reading SPCod SPCoc element, Invalid cblkw/cblkh combination\n");
        return OPJ_FALSE;
    }

    /* SPcod (G) / SPcoc (D) */
    opj_read_bytes(l_current_ptr, &l_tccp->cblksty, 1);
    ++l_current_ptr;
    if ((l_tccp->cblksty & J2K_CCP_CBLKSTY_HTMIXED) != 0) {
        /* We do not support HT mixed mode yet.  For conformance, it should be supported.*/
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error reading SPCod SPCoc element. Unsupported Mixed HT code-block style found\n");
        return OPJ_FALSE;
    }

    /* SPcod (H) / SPcoc (E) */
    opj_read_bytes(l_current_ptr, &l_tccp->qmfbid, 1);
    ++l_current_ptr;

    if (l_tccp->qmfbid > 1) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error reading SPCod SPCoc element, Invalid transformation found\n");
        return OPJ_FALSE;
    }

    *p_header_size = *p_header_size - 5;

    /* use custom precinct size ? */
    if (l_tccp->csty & J2K_CCP_CSTY_PRT) {
        if (*p_header_size < l_tccp->numresolutions) {
            opj_event_msg(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n");
            return OPJ_FALSE;
        }

        /* SPcod (I_i) / SPcoc (F_i) */
        for (i = 0; i < l_tccp->numresolutions; ++i) {
            opj_read_bytes(l_current_ptr, &l_tmp, 1);
            ++l_current_ptr;
            /* Precinct exponent 0 is only allowed for lowest resolution level (Table A.21) */
            if ((i != 0) && (((l_tmp & 0xf) == 0) || ((l_tmp >> 4) == 0))) {
                opj_event_msg(p_manager, EVT_ERROR, "Invalid precinct size\n");
                return OPJ_FALSE;
            }
            l_tccp->prcw[i] = l_tmp & 0xf;
            l_tccp->prch[i] = l_tmp >> 4;
        }

        *p_header_size = *p_header_size - l_tccp->numresolutions;
    } else {
        /* set default size for the precinct width and height */
        for (i = 0; i < l_tccp->numresolutions; ++i) {
            l_tccp->prcw[i] = 15;
            l_tccp->prch[i] = 15;
        }
    }

#ifdef WIP_REMOVE_MSD
    /* INDEX >> */
    if (p_j2k->cstr_info && compno == 0) {
        OPJ_UINT32 l_data_size = l_tccp->numresolutions * sizeof(OPJ_UINT32);

        p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].tccp_info[compno].cblkh =
            l_tccp->cblkh;
        p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].tccp_info[compno].cblkw =
            l_tccp->cblkw;
        p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].tccp_info[compno].numresolutions
            = l_tccp->numresolutions;
        p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].tccp_info[compno].cblksty =
            l_tccp->cblksty;
        p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].tccp_info[compno].qmfbid =
            l_tccp->qmfbid;

        memcpy(p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].pdx, l_tccp->prcw,
               l_data_size);
        memcpy(p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].pdy, l_tccp->prch,
               l_data_size);
    }
    /* << INDEX */
#endif

    return OPJ_TRUE;
}

static void opj_j2k_copy_tile_component_parameters(opj_j2k_t *p_j2k)
{
    /* loop */
    OPJ_UINT32 i;
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_tccp_t *l_ref_tccp = NULL, *l_copied_tccp = NULL;
    OPJ_UINT32 l_prc_size;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH)
            ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    l_ref_tccp = &l_tcp->tccps[0];
    l_copied_tccp = l_ref_tccp + 1;
    l_prc_size = l_ref_tccp->numresolutions * (OPJ_UINT32)sizeof(OPJ_UINT32);

    for (i = 1; i < p_j2k->m_private_image->numcomps; ++i) {
        l_copied_tccp->numresolutions = l_ref_tccp->numresolutions;
        l_copied_tccp->cblkw = l_ref_tccp->cblkw;
        l_copied_tccp->cblkh = l_ref_tccp->cblkh;
        l_copied_tccp->cblksty = l_ref_tccp->cblksty;
        l_copied_tccp->qmfbid = l_ref_tccp->qmfbid;
        memcpy(l_copied_tccp->prcw, l_ref_tccp->prcw, l_prc_size);
        memcpy(l_copied_tccp->prch, l_ref_tccp->prch, l_prc_size);
        ++l_copied_tccp;
    }
}

static OPJ_UINT32 opj_j2k_get_SQcd_SQcc_size(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no,
        OPJ_UINT32 p_comp_no)
{
    OPJ_UINT32 l_num_bands;

    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp = &l_tcp->tccps[p_comp_no];

    /* preconditions again */
    assert(p_tile_no < l_cp->tw * l_cp->th);
    assert(p_comp_no < p_j2k->m_private_image->numcomps);

    l_num_bands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 :
                  (l_tccp->numresolutions * 3 - 2);

    if (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT)  {
        return 1 + l_num_bands;
    } else {
        return 1 + 2 * l_num_bands;
    }
}

static OPJ_BOOL opj_j2k_compare_SQcd_SQcc(opj_j2k_t *p_j2k,
        OPJ_UINT32 p_tile_no, OPJ_UINT32 p_first_comp_no, OPJ_UINT32 p_second_comp_no)
{
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_tccp_t *l_tccp0 = NULL;
    opj_tccp_t *l_tccp1 = NULL;
    OPJ_UINT32 l_band_no, l_num_bands;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp0 = &l_tcp->tccps[p_first_comp_no];
    l_tccp1 = &l_tcp->tccps[p_second_comp_no];

    if (l_tccp0->qntsty != l_tccp1->qntsty) {
        return OPJ_FALSE;
    }
    if (l_tccp0->numgbits != l_tccp1->numgbits) {
        return OPJ_FALSE;
    }
    if (l_tccp0->qntsty == J2K_CCP_QNTSTY_SIQNT) {
        l_num_bands = 1U;
    } else {
        l_num_bands = l_tccp0->numresolutions * 3U - 2U;
        if (l_num_bands != (l_tccp1->numresolutions * 3U - 2U)) {
            return OPJ_FALSE;
        }
    }

    for (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no) {
        if (l_tccp0->stepsizes[l_band_no].expn != l_tccp1->stepsizes[l_band_no].expn) {
            return OPJ_FALSE;
        }
    }
    if (l_tccp0->qntsty != J2K_CCP_QNTSTY_NOQNT) {
        for (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no) {
            if (l_tccp0->stepsizes[l_band_no].mant != l_tccp1->stepsizes[l_band_no].mant) {
                return OPJ_FALSE;
            }
        }
    }
    return OPJ_TRUE;
}


static OPJ_BOOL opj_j2k_write_SQcd_SQcc(opj_j2k_t *p_j2k,
                                        OPJ_UINT32 p_tile_no,
                                        OPJ_UINT32 p_comp_no,
                                        OPJ_BYTE * p_data,
                                        OPJ_UINT32 * p_header_size,
                                        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 l_header_size;
    OPJ_UINT32 l_band_no, l_num_bands;
    OPJ_UINT32 l_expn, l_mant;

    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_header_size != 00);
    assert(p_manager != 00);
    assert(p_data != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = &l_cp->tcps[p_tile_no];
    l_tccp = &l_tcp->tccps[p_comp_no];

    /* preconditions again */
    assert(p_tile_no < l_cp->tw * l_cp->th);
    assert(p_comp_no < p_j2k->m_private_image->numcomps);

    l_num_bands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 :
                  (l_tccp->numresolutions * 3 - 2);

    if (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT)  {
        l_header_size = 1 + l_num_bands;

        if (*p_header_size < l_header_size) {
            opj_event_msg(p_manager, EVT_ERROR, "Error writing SQcd SQcc element\n");
            return OPJ_FALSE;
        }

        opj_write_bytes(p_data, l_tccp->qntsty + (l_tccp->numgbits << 5),
                        1);   /* Sqcx */
        ++p_data;

        for (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no) {
            l_expn = (OPJ_UINT32)l_tccp->stepsizes[l_band_no].expn;
            opj_write_bytes(p_data, l_expn << 3, 1);        /* SPqcx_i */
            ++p_data;
        }
    } else {
        l_header_size = 1 + 2 * l_num_bands;

        if (*p_header_size < l_header_size) {
            opj_event_msg(p_manager, EVT_ERROR, "Error writing SQcd SQcc element\n");
            return OPJ_FALSE;
        }

        opj_write_bytes(p_data, l_tccp->qntsty + (l_tccp->numgbits << 5),
                        1);   /* Sqcx */
        ++p_data;

        for (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no) {
            l_expn = (OPJ_UINT32)l_tccp->stepsizes[l_band_no].expn;
            l_mant = (OPJ_UINT32)l_tccp->stepsizes[l_band_no].mant;

            opj_write_bytes(p_data, (l_expn << 11) + l_mant, 2);    /* SPqcx_i */
            p_data += 2;
        }
    }

    *p_header_size = *p_header_size - l_header_size;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_read_SQcd_SQcc(opj_j2k_t *p_j2k,
                                       OPJ_UINT32 p_comp_no,
                                       OPJ_BYTE* p_header_data,
                                       OPJ_UINT32 * p_header_size,
                                       opj_event_mgr_t * p_manager
                                      )
{
    /* loop*/
    OPJ_UINT32 l_band_no;
    opj_cp_t *l_cp = 00;
    opj_tcp_t *l_tcp = 00;
    opj_tccp_t *l_tccp = 00;
    OPJ_BYTE * l_current_ptr = 00;
    OPJ_UINT32 l_tmp, l_num_band;

    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_header_data != 00);

    l_cp = &(p_j2k->m_cp);
    /* come from tile part header or main header ?*/
    l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH)
            ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    /* precondition again*/
    assert(p_comp_no <  p_j2k->m_private_image->numcomps);

    l_tccp = &l_tcp->tccps[p_comp_no];
    l_current_ptr = p_header_data;

    if (*p_header_size < 1) {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading SQcd or SQcc element\n");
        return OPJ_FALSE;
    }
    *p_header_size -= 1;

    opj_read_bytes(l_current_ptr, &l_tmp, 1);                       /* Sqcx */
    ++l_current_ptr;

    l_tccp->qntsty = l_tmp & 0x1f;
    l_tccp->numgbits = l_tmp >> 5;
    if (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) {
        l_num_band = 1;
    } else {
        l_num_band = (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) ?
                     (*p_header_size) :
                     (*p_header_size) / 2;

        if (l_num_band > OPJ_J2K_MAXBANDS) {
            opj_event_msg(p_manager, EVT_WARNING,
                          "While reading CCP_QNTSTY element inside QCD or QCC marker segment, "
                          "number of subbands (%d) is greater to OPJ_J2K_MAXBANDS (%d). So we limit the number of elements stored to "
                          "OPJ_J2K_MAXBANDS (%d) and skip the rest. \n", l_num_band, OPJ_J2K_MAXBANDS,
                          OPJ_J2K_MAXBANDS);
            /*return OPJ_FALSE;*/
        }
    }

#ifdef USE_JPWL
    if (l_cp->correct) {

        /* if JPWL is on, we check whether there are too many subbands */
        if (/*(l_num_band < 0) ||*/ (l_num_band >= OPJ_J2K_MAXBANDS)) {
            opj_event_msg(p_manager, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
                          "JPWL: bad number of subbands in Sqcx (%d)\n",
                          l_num_band);
            if (!JPWL_ASSUME) {
                opj_event_msg(p_manager, EVT_ERROR, "JPWL: giving up\n");
                return OPJ_FALSE;
            }
            /* we try to correct */
            l_num_band = 1;
            opj_event_msg(p_manager, EVT_WARNING, "- trying to adjust them\n"
                          "- setting number of bands to %d => HYPOTHESIS!!!\n",
                          l_num_band);
        };

    };
#endif /* USE_JPWL */

    if (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) {
        for (l_band_no = 0; l_band_no < l_num_band; l_band_no++) {
            opj_read_bytes(l_current_ptr, &l_tmp, 1);                       /* SPqcx_i */
            ++l_current_ptr;
            if (l_band_no < OPJ_J2K_MAXBANDS) {
                l_tccp->stepsizes[l_band_no].expn = (OPJ_INT32)(l_tmp >> 3);
                l_tccp->stepsizes[l_band_no].mant = 0;
            }
        }

        if (*p_header_size < l_num_band) {
            return OPJ_FALSE;
        }
        *p_header_size = *p_header_size - l_num_band;
    } else {
        for (l_band_no = 0; l_band_no < l_num_band; l_band_no++) {
            opj_read_bytes(l_current_ptr, &l_tmp, 2);                       /* SPqcx_i */
            l_current_ptr += 2;
            if (l_band_no < OPJ_J2K_MAXBANDS) {
                l_tccp->stepsizes[l_band_no].expn = (OPJ_INT32)(l_tmp >> 11);
                l_tccp->stepsizes[l_band_no].mant = l_tmp & 0x7ff;
            }
        }

        if (*p_header_size < 2 * l_num_band) {
            return OPJ_FALSE;
        }
        *p_header_size = *p_header_size - 2 * l_num_band;
    }

    /* Add Antonin : if scalar_derived -> compute other stepsizes */
    if (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) {
        for (l_band_no = 1; l_band_no < OPJ_J2K_MAXBANDS; l_band_no++) {
            l_tccp->stepsizes[l_band_no].expn =
                ((OPJ_INT32)(l_tccp->stepsizes[0].expn) - (OPJ_INT32)((l_band_no - 1) / 3) > 0)
                ?
                (OPJ_INT32)(l_tccp->stepsizes[0].expn) - (OPJ_INT32)((l_band_no - 1) / 3) : 0;
            l_tccp->stepsizes[l_band_no].mant = l_tccp->stepsizes[0].mant;
        }
    }

    return OPJ_TRUE;
}

static void opj_j2k_copy_tile_quantization_parameters(opj_j2k_t *p_j2k)
{
    OPJ_UINT32 i;
    opj_cp_t *l_cp = NULL;
    opj_tcp_t *l_tcp = NULL;
    opj_tccp_t *l_ref_tccp = NULL;
    opj_tccp_t *l_copied_tccp = NULL;
    OPJ_UINT32 l_size;

    /* preconditions */
    assert(p_j2k != 00);

    l_cp = &(p_j2k->m_cp);
    l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_TPH ?
            &l_cp->tcps[p_j2k->m_current_tile_number] :
            p_j2k->m_specific_param.m_decoder.m_default_tcp;

    l_ref_tccp = &l_tcp->tccps[0];
    l_copied_tccp = l_ref_tccp + 1;
    l_size = OPJ_J2K_MAXBANDS * sizeof(opj_stepsize_t);

    for (i = 1; i < p_j2k->m_private_image->numcomps; ++i) {
        l_copied_tccp->qntsty = l_ref_tccp->qntsty;
        l_copied_tccp->numgbits = l_ref_tccp->numgbits;
        memcpy(l_copied_tccp->stepsizes, l_ref_tccp->stepsizes, l_size);
        ++l_copied_tccp;
    }
}

static void opj_j2k_dump_tile_info(opj_tcp_t * l_default_tile,
                                   OPJ_INT32 numcomps, FILE* out_stream)
{
    if (l_default_tile) {
        OPJ_INT32 compno;

        fprintf(out_stream, "\t default tile {\n");
        fprintf(out_stream, "\t\t csty=%#x\n", l_default_tile->csty);
        fprintf(out_stream, "\t\t prg=%#x\n", l_default_tile->prg);
        fprintf(out_stream, "\t\t numlayers=%d\n", l_default_tile->numlayers);
        fprintf(out_stream, "\t\t mct=%x\n", l_default_tile->mct);

        for (compno = 0; compno < numcomps; compno++) {
            opj_tccp_t *l_tccp = &(l_default_tile->tccps[compno]);
            OPJ_UINT32 resno;
            OPJ_INT32 bandno, numbands;

            /* coding style*/
            fprintf(out_stream, "\t\t comp %d {\n", compno);
            fprintf(out_stream, "\t\t\t csty=%#x\n", l_tccp->csty);
            fprintf(out_stream, "\t\t\t numresolutions=%d\n", l_tccp->numresolutions);
            fprintf(out_stream, "\t\t\t cblkw=2^%d\n", l_tccp->cblkw);
            fprintf(out_stream, "\t\t\t cblkh=2^%d\n", l_tccp->cblkh);
            fprintf(out_stream, "\t\t\t cblksty=%#x\n", l_tccp->cblksty);
            fprintf(out_stream, "\t\t\t qmfbid=%d\n", l_tccp->qmfbid);

            fprintf(out_stream, "\t\t\t preccintsize (w,h)=");
            for (resno = 0; resno < l_tccp->numresolutions; resno++) {
                fprintf(out_stream, "(%d,%d) ", l_tccp->prcw[resno], l_tccp->prch[resno]);
            }
            fprintf(out_stream, "\n");

            /* quantization style*/
            fprintf(out_stream, "\t\t\t qntsty=%d\n", l_tccp->qntsty);
            fprintf(out_stream, "\t\t\t numgbits=%d\n", l_tccp->numgbits);
            fprintf(out_stream, "\t\t\t stepsizes (m,e)=");
            numbands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 :
                       (OPJ_INT32)l_tccp->numresolutions * 3 - 2;
            for (bandno = 0; bandno < numbands; bandno++) {
                fprintf(out_stream, "(%d,%d) ", l_tccp->stepsizes[bandno].mant,
                        l_tccp->stepsizes[bandno].expn);
            }
            fprintf(out_stream, "\n");

            /* RGN value*/
            fprintf(out_stream, "\t\t\t roishift=%d\n", l_tccp->roishift);

            fprintf(out_stream, "\t\t }\n");
        } /*end of component of default tile*/
        fprintf(out_stream, "\t }\n"); /*end of default tile*/
    }
}

void j2k_dump(opj_j2k_t* p_j2k, OPJ_INT32 flag, FILE* out_stream)
{
    /* Check if the flag is compatible with j2k file*/
    if ((flag & OPJ_JP2_INFO) || (flag & OPJ_JP2_IND)) {
        fprintf(out_stream, "Wrong flag\n");
        return;
    }

    /* Dump the image_header */
    if (flag & OPJ_IMG_INFO) {
        if (p_j2k->m_private_image) {
            j2k_dump_image_header(p_j2k->m_private_image, 0, out_stream);
        }
    }

    /* Dump the codestream info from main header */
    if (flag & OPJ_J2K_MH_INFO) {
        if (p_j2k->m_private_image) {
            opj_j2k_dump_MH_info(p_j2k, out_stream);
        }
    }
    /* Dump all tile/codestream info */
    if (flag & OPJ_J2K_TCH_INFO) {
        OPJ_UINT32 l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
        OPJ_UINT32 i;
        opj_tcp_t * l_tcp = p_j2k->m_cp.tcps;
        if (p_j2k->m_private_image) {
            for (i = 0; i < l_nb_tiles; ++i) {
                opj_j2k_dump_tile_info(l_tcp, (OPJ_INT32)p_j2k->m_private_image->numcomps,
                                       out_stream);
                ++l_tcp;
            }
        }
    }

    /* Dump the codestream info of the current tile */
    if (flag & OPJ_J2K_TH_INFO) {

    }

    /* Dump the codestream index from main header */
    if (flag & OPJ_J2K_MH_IND) {
        opj_j2k_dump_MH_index(p_j2k, out_stream);
    }

    /* Dump the codestream index of the current tile */
    if (flag & OPJ_J2K_TH_IND) {

    }

}

static void opj_j2k_dump_MH_index(opj_j2k_t* p_j2k, FILE* out_stream)
{
    opj_codestream_index_t* cstr_index = p_j2k->cstr_index;
    OPJ_UINT32 it_marker, it_tile, it_tile_part;

    fprintf(out_stream, "Codestream index from main header: {\n");

    fprintf(out_stream, "\t Main header start position=%" PRIi64 "\n"
            "\t Main header end position=%" PRIi64 "\n",
            cstr_index->main_head_start, cstr_index->main_head_end);

    fprintf(out_stream, "\t Marker list: {\n");

    if (cstr_index->marker) {
        for (it_marker = 0; it_marker < cstr_index->marknum ; it_marker++) {
            fprintf(out_stream, "\t\t type=%#x, pos=%" PRIi64 ", len=%d\n",
                    cstr_index->marker[it_marker].type,
                    cstr_index->marker[it_marker].pos,
                    cstr_index->marker[it_marker].len);
        }
    }

    fprintf(out_stream, "\t }\n");

    if (cstr_index->tile_index) {

        /* Simple test to avoid to write empty information*/
        OPJ_UINT32 l_acc_nb_of_tile_part = 0;
        for (it_tile = 0; it_tile < cstr_index->nb_of_tiles ; it_tile++) {
            l_acc_nb_of_tile_part += cstr_index->tile_index[it_tile].nb_tps;

            /* To avoid regenerating expected opj_dump results from the test */
            /* suite when there is a TLM marker present */
            if (cstr_index->tile_index[it_tile].nb_tps &&
                    cstr_index->tile_index[it_tile].tp_index &&
                    cstr_index->tile_index[it_tile].tp_index[0].start_pos > 0 &&
                    cstr_index->tile_index[it_tile].tp_index[0].end_header == 0 &&
                    getenv("OJP_DO_NOT_DISPLAY_TILE_INDEX_IF_TLM") != NULL) {
                l_acc_nb_of_tile_part = 0;
                break;
            }
        }

        if (l_acc_nb_of_tile_part) {
            fprintf(out_stream, "\t Tile index: {\n");

            for (it_tile = 0; it_tile < cstr_index->nb_of_tiles ; it_tile++) {
                OPJ_UINT32 nb_of_tile_part = cstr_index->tile_index[it_tile].nb_tps;

                fprintf(out_stream, "\t\t nb of tile-part in tile [%d]=%d\n", it_tile,
                        nb_of_tile_part);

                if (cstr_index->tile_index[it_tile].tp_index) {
                    for (it_tile_part = 0; it_tile_part < nb_of_tile_part; it_tile_part++) {
                        fprintf(out_stream, "\t\t\t tile-part[%d]: star_pos=%" PRIi64 ", end_header=%"
                                PRIi64 ", end_pos=%" PRIi64 ".\n",
                                it_tile_part,
                                cstr_index->tile_index[it_tile].tp_index[it_tile_part].start_pos,
                                cstr_index->tile_index[it_tile].tp_index[it_tile_part].end_header,
                                cstr_index->tile_index[it_tile].tp_index[it_tile_part].end_pos);
                    }
                }

                if (cstr_index->tile_index[it_tile].marker) {
                    for (it_marker = 0; it_marker < cstr_index->tile_index[it_tile].marknum ;
                            it_marker++) {
                        fprintf(out_stream, "\t\t type=%#x, pos=%" PRIi64 ", len=%d\n",
                                cstr_index->tile_index[it_tile].marker[it_marker].type,
                                cstr_index->tile_index[it_tile].marker[it_marker].pos,
                                cstr_index->tile_index[it_tile].marker[it_marker].len);
                    }
                }
            }
            fprintf(out_stream, "\t }\n");
        }
    }

    fprintf(out_stream, "}\n");

}


static void opj_j2k_dump_MH_info(opj_j2k_t* p_j2k, FILE* out_stream)
{

    fprintf(out_stream, "Codestream info from main header: {\n");

    fprintf(out_stream, "\t tx0=%" PRIu32 ", ty0=%" PRIu32 "\n", p_j2k->m_cp.tx0,
            p_j2k->m_cp.ty0);
    fprintf(out_stream, "\t tdx=%" PRIu32 ", tdy=%" PRIu32 "\n", p_j2k->m_cp.tdx,
            p_j2k->m_cp.tdy);
    fprintf(out_stream, "\t tw=%" PRIu32 ", th=%" PRIu32 "\n", p_j2k->m_cp.tw,
            p_j2k->m_cp.th);
    opj_j2k_dump_tile_info(p_j2k->m_specific_param.m_decoder.m_default_tcp,
                           (OPJ_INT32)p_j2k->m_private_image->numcomps, out_stream);
    fprintf(out_stream, "}\n");
}

void j2k_dump_image_header(opj_image_t* img_header, OPJ_BOOL dev_dump_flag,
                           FILE* out_stream)
{
    char tab[2];

    if (dev_dump_flag) {
        fprintf(stdout, "[DEV] Dump an image_header struct {\n");
        tab[0] = '\0';
    } else {
        fprintf(out_stream, "Image info {\n");
        tab[0] = '\t';
        tab[1] = '\0';
    }

    fprintf(out_stream, "%s x0=%d, y0=%d\n", tab, img_header->x0, img_header->y0);
    fprintf(out_stream,     "%s x1=%d, y1=%d\n", tab, img_header->x1,
            img_header->y1);
    fprintf(out_stream, "%s numcomps=%d\n", tab, img_header->numcomps);

    if (img_header->comps) {
        OPJ_UINT32 compno;
        for (compno = 0; compno < img_header->numcomps; compno++) {
            fprintf(out_stream, "%s\t component %d {\n", tab, compno);
            j2k_dump_image_comp_header(&(img_header->comps[compno]), dev_dump_flag,
                                       out_stream);
            fprintf(out_stream, "%s}\n", tab);
        }
    }

    fprintf(out_stream, "}\n");
}

void j2k_dump_image_comp_header(opj_image_comp_t* comp_header,
                                OPJ_BOOL dev_dump_flag, FILE* out_stream)
{
    char tab[3];

    if (dev_dump_flag) {
        fprintf(stdout, "[DEV] Dump an image_comp_header struct {\n");
        tab[0] = '\0';
    }       else {
        tab[0] = '\t';
        tab[1] = '\t';
        tab[2] = '\0';
    }

    fprintf(out_stream, "%s dx=%d, dy=%d\n", tab, comp_header->dx, comp_header->dy);
    fprintf(out_stream, "%s prec=%d\n", tab, comp_header->prec);
    fprintf(out_stream, "%s sgnd=%d\n", tab, comp_header->sgnd);

    if (dev_dump_flag) {
        fprintf(out_stream, "}\n");
    }
}

opj_codestream_info_v2_t* j2k_get_cstr_info(opj_j2k_t* p_j2k)
{
    OPJ_UINT32 compno;
    OPJ_UINT32 numcomps = p_j2k->m_private_image->numcomps;
    opj_tcp_t *l_default_tile;
    opj_codestream_info_v2_t* cstr_info = (opj_codestream_info_v2_t*) opj_calloc(1,
                                          sizeof(opj_codestream_info_v2_t));
    if (!cstr_info) {
        return NULL;
    }

    cstr_info->nbcomps = p_j2k->m_private_image->numcomps;

    cstr_info->tx0 = p_j2k->m_cp.tx0;
    cstr_info->ty0 = p_j2k->m_cp.ty0;
    cstr_info->tdx = p_j2k->m_cp.tdx;
    cstr_info->tdy = p_j2k->m_cp.tdy;
    cstr_info->tw = p_j2k->m_cp.tw;
    cstr_info->th = p_j2k->m_cp.th;

    cstr_info->tile_info = NULL; /* Not fill from the main header*/

    l_default_tile = p_j2k->m_specific_param.m_decoder.m_default_tcp;

    cstr_info->m_default_tile_info.csty = l_default_tile->csty;
    cstr_info->m_default_tile_info.prg = l_default_tile->prg;
    cstr_info->m_default_tile_info.numlayers = l_default_tile->numlayers;
    cstr_info->m_default_tile_info.mct = l_default_tile->mct;

    cstr_info->m_default_tile_info.tccp_info = (opj_tccp_info_t*) opj_calloc(
                cstr_info->nbcomps, sizeof(opj_tccp_info_t));
    if (!cstr_info->m_default_tile_info.tccp_info) {
        opj_destroy_cstr_info(&cstr_info);
        return NULL;
    }

    for (compno = 0; compno < numcomps; compno++) {
        opj_tccp_t *l_tccp = &(l_default_tile->tccps[compno]);
        opj_tccp_info_t *l_tccp_info = &
                                       (cstr_info->m_default_tile_info.tccp_info[compno]);
        OPJ_INT32 bandno, numbands;

        /* coding style*/
        l_tccp_info->csty = l_tccp->csty;
        l_tccp_info->numresolutions = l_tccp->numresolutions;
        l_tccp_info->cblkw = l_tccp->cblkw;
        l_tccp_info->cblkh = l_tccp->cblkh;
        l_tccp_info->cblksty = l_tccp->cblksty;
        l_tccp_info->qmfbid = l_tccp->qmfbid;
        if (l_tccp->numresolutions < OPJ_J2K_MAXRLVLS) {
            memcpy(l_tccp_info->prch, l_tccp->prch, l_tccp->numresolutions);
            memcpy(l_tccp_info->prcw, l_tccp->prcw, l_tccp->numresolutions);
        }

        /* quantization style*/
        l_tccp_info->qntsty = l_tccp->qntsty;
        l_tccp_info->numgbits = l_tccp->numgbits;

        numbands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 :
                   (OPJ_INT32)l_tccp->numresolutions * 3 - 2;
        if (numbands < OPJ_J2K_MAXBANDS) {
            for (bandno = 0; bandno < numbands; bandno++) {
                l_tccp_info->stepsizes_mant[bandno] = (OPJ_UINT32)
                                                      l_tccp->stepsizes[bandno].mant;
                l_tccp_info->stepsizes_expn[bandno] = (OPJ_UINT32)
                                                      l_tccp->stepsizes[bandno].expn;
            }
        }

        /* RGN value*/
        l_tccp_info->roishift = l_tccp->roishift;
    }

    return cstr_info;
}

opj_codestream_index_t* j2k_get_cstr_index(opj_j2k_t* p_j2k)
{
    opj_codestream_index_t* l_cstr_index = (opj_codestream_index_t*)
                                           opj_calloc(1, sizeof(opj_codestream_index_t));
    if (!l_cstr_index) {
        return NULL;
    }

    l_cstr_index->main_head_start = p_j2k->cstr_index->main_head_start;
    l_cstr_index->main_head_end = p_j2k->cstr_index->main_head_end;
    l_cstr_index->codestream_size = p_j2k->cstr_index->codestream_size;

    l_cstr_index->marknum = p_j2k->cstr_index->marknum;
    l_cstr_index->marker = (opj_marker_info_t*)opj_malloc(l_cstr_index->marknum *
                           sizeof(opj_marker_info_t));
    if (!l_cstr_index->marker) {
        opj_free(l_cstr_index);
        return NULL;
    }

    if (p_j2k->cstr_index->marker) {
        memcpy(l_cstr_index->marker, p_j2k->cstr_index->marker,
               l_cstr_index->marknum * sizeof(opj_marker_info_t));
    } else {
        opj_free(l_cstr_index->marker);
        l_cstr_index->marker = NULL;
    }

    l_cstr_index->nb_of_tiles = p_j2k->cstr_index->nb_of_tiles;
    l_cstr_index->tile_index = (opj_tile_index_t*)opj_calloc(
                                   l_cstr_index->nb_of_tiles, sizeof(opj_tile_index_t));
    if (!l_cstr_index->tile_index) {
        opj_free(l_cstr_index->marker);
        opj_free(l_cstr_index);
        return NULL;
    }

    if (!p_j2k->cstr_index->tile_index) {
        opj_free(l_cstr_index->tile_index);
        l_cstr_index->tile_index = NULL;
    } else {
        OPJ_UINT32 it_tile = 0;
        for (it_tile = 0; it_tile < l_cstr_index->nb_of_tiles; it_tile++) {

            /* Tile Marker*/
            l_cstr_index->tile_index[it_tile].marknum =
                p_j2k->cstr_index->tile_index[it_tile].marknum;

            l_cstr_index->tile_index[it_tile].marker =
                (opj_marker_info_t*)opj_malloc(l_cstr_index->tile_index[it_tile].marknum *
                                               sizeof(opj_marker_info_t));

            if (!l_cstr_index->tile_index[it_tile].marker) {
                OPJ_UINT32 it_tile_free;

                for (it_tile_free = 0; it_tile_free < it_tile; it_tile_free++) {
                    opj_free(l_cstr_index->tile_index[it_tile_free].marker);
                }

                opj_free(l_cstr_index->tile_index);
                opj_free(l_cstr_index->marker);
                opj_free(l_cstr_index);
                return NULL;
            }

            if (p_j2k->cstr_index->tile_index[it_tile].marker)
                memcpy(l_cstr_index->tile_index[it_tile].marker,
                       p_j2k->cstr_index->tile_index[it_tile].marker,
                       l_cstr_index->tile_index[it_tile].marknum * sizeof(opj_marker_info_t));
            else {
                opj_free(l_cstr_index->tile_index[it_tile].marker);
                l_cstr_index->tile_index[it_tile].marker = NULL;
            }

            /* Tile part index*/
            l_cstr_index->tile_index[it_tile].nb_tps =
                p_j2k->cstr_index->tile_index[it_tile].nb_tps;

            l_cstr_index->tile_index[it_tile].tp_index =
                (opj_tp_index_t*)opj_malloc(l_cstr_index->tile_index[it_tile].nb_tps * sizeof(
                                                opj_tp_index_t));

            if (!l_cstr_index->tile_index[it_tile].tp_index) {
                OPJ_UINT32 it_tile_free;

                for (it_tile_free = 0; it_tile_free < it_tile; it_tile_free++) {
                    opj_free(l_cstr_index->tile_index[it_tile_free].marker);
                    opj_free(l_cstr_index->tile_index[it_tile_free].tp_index);
                }

                opj_free(l_cstr_index->tile_index);
                opj_free(l_cstr_index->marker);
                opj_free(l_cstr_index);
                return NULL;
            }

            if (p_j2k->cstr_index->tile_index[it_tile].tp_index) {
                memcpy(l_cstr_index->tile_index[it_tile].tp_index,
                       p_j2k->cstr_index->tile_index[it_tile].tp_index,
                       l_cstr_index->tile_index[it_tile].nb_tps * sizeof(opj_tp_index_t));
            } else {
                opj_free(l_cstr_index->tile_index[it_tile].tp_index);
                l_cstr_index->tile_index[it_tile].tp_index = NULL;
            }

            /* Packet index (NOT USED)*/
            l_cstr_index->tile_index[it_tile].nb_packet = 0;
            l_cstr_index->tile_index[it_tile].packet_index = NULL;

        }
    }

    return l_cstr_index;
}

static OPJ_BOOL opj_j2k_allocate_tile_element_cstr_index(opj_j2k_t *p_j2k)
{
    OPJ_UINT32 it_tile = 0;

    p_j2k->cstr_index->nb_of_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th;
    p_j2k->cstr_index->tile_index = (opj_tile_index_t*)opj_calloc(
                                        p_j2k->cstr_index->nb_of_tiles, sizeof(opj_tile_index_t));
    if (!p_j2k->cstr_index->tile_index) {
        return OPJ_FALSE;
    }

    for (it_tile = 0; it_tile < p_j2k->cstr_index->nb_of_tiles; it_tile++) {
        p_j2k->cstr_index->tile_index[it_tile].maxmarknum = 100;
        p_j2k->cstr_index->tile_index[it_tile].marknum = 0;
        p_j2k->cstr_index->tile_index[it_tile].marker = (opj_marker_info_t*)
                opj_calloc(p_j2k->cstr_index->tile_index[it_tile].maxmarknum,
                           sizeof(opj_marker_info_t));
        if (!p_j2k->cstr_index->tile_index[it_tile].marker) {
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_are_all_used_components_decoded(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 compno;
    OPJ_BOOL decoded_all_used_components = OPJ_TRUE;

    if (p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode) {
        for (compno = 0;
                compno < p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode; compno++) {
            OPJ_UINT32 dec_compno =
                p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode[compno];
            if (p_j2k->m_output_image->comps[dec_compno].data == NULL) {
                opj_event_msg(p_manager, EVT_WARNING, "Failed to decode component %d\n",
                              dec_compno);
                decoded_all_used_components = OPJ_FALSE;
            }
        }
    } else {
        for (compno = 0; compno < p_j2k->m_output_image->numcomps; compno++) {
            if (p_j2k->m_output_image->comps[compno].data == NULL) {
                opj_event_msg(p_manager, EVT_WARNING, "Failed to decode component %d\n",
                              compno);
                decoded_all_used_components = OPJ_FALSE;
            }
        }
    }

    if (decoded_all_used_components == OPJ_FALSE) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to decode all used components\n");
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static int CompareOffT(const void* a, const void* b)
{
    const OPJ_OFF_T offA = *(const OPJ_OFF_T*)a;
    const OPJ_OFF_T offB = *(const OPJ_OFF_T*)b;
    if (offA < offB) {
        return -1;
    }
    if (offA == offB) {
        return 0;
    }
    return 1;
}

static OPJ_BOOL opj_j2k_decode_tiles(opj_j2k_t *p_j2k,
                                     opj_stream_private_t *p_stream,
                                     opj_event_mgr_t * p_manager)
{
    OPJ_BOOL l_go_on = OPJ_TRUE;
    OPJ_UINT32 l_current_tile_no;
    OPJ_INT32 l_tile_x0, l_tile_y0, l_tile_x1, l_tile_y1;
    OPJ_UINT32 l_nb_comps;
    OPJ_UINT32 nr_tiles = 0;
    OPJ_OFF_T end_pos = 0;

    /* Particular case for whole single tile decoding */
    /* We can avoid allocating intermediate tile buffers */
    if (p_j2k->m_cp.tw == 1 && p_j2k->m_cp.th == 1 &&
            p_j2k->m_cp.tx0 == 0 && p_j2k->m_cp.ty0 == 0 &&
            p_j2k->m_output_image->x0 == 0 &&
            p_j2k->m_output_image->y0 == 0 &&
            p_j2k->m_output_image->x1 == p_j2k->m_cp.tdx &&
            p_j2k->m_output_image->y1 == p_j2k->m_cp.tdy) {
        OPJ_UINT32 i;
        if (! opj_j2k_read_tile_header(p_j2k,
                                       &l_current_tile_no,
                                       NULL,
                                       &l_tile_x0, &l_tile_y0,
                                       &l_tile_x1, &l_tile_y1,
                                       &l_nb_comps,
                                       &l_go_on,
                                       p_stream,
                                       p_manager)) {
            return OPJ_FALSE;
        }

        if (!l_go_on ||
                ! opj_j2k_decode_tile(p_j2k, l_current_tile_no, NULL, 0,
                                      p_stream, p_manager)) {
            opj_event_msg(p_manager, EVT_ERROR, "Failed to decode tile 1/1\n");
            return OPJ_FALSE;
        }

        /* Transfer TCD data to output image data */
        for (i = 0; i < p_j2k->m_output_image->numcomps; i++) {
            opj_image_data_free(p_j2k->m_output_image->comps[i].data);
            p_j2k->m_output_image->comps[i].data =
                p_j2k->m_tcd->tcd_image->tiles->comps[i].data;
            p_j2k->m_output_image->comps[i].resno_decoded =
                p_j2k->m_tcd->image->comps[i].resno_decoded;
            p_j2k->m_tcd->tcd_image->tiles->comps[i].data = NULL;
        }

        return OPJ_TRUE;
    }

    p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts = 0;
    p_j2k->m_specific_param.m_decoder.m_idx_intersecting_tile_parts = 0;
    opj_free(p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset);
    p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset = NULL;

    /* If the area to decode only intersects a subset of tiles, and we have
     * valid TLM information, then use it to plan the tilepart offsets to
     * seek to.
     */
    if (!(p_j2k->m_specific_param.m_decoder.m_start_tile_x == 0 &&
            p_j2k->m_specific_param.m_decoder.m_start_tile_y == 0 &&
            p_j2k->m_specific_param.m_decoder.m_end_tile_x == p_j2k->m_cp.tw &&
            p_j2k->m_specific_param.m_decoder.m_end_tile_y == p_j2k->m_cp.th) &&
            !p_j2k->m_specific_param.m_decoder.m_tlm.m_is_invalid &&
            opj_stream_has_seek(p_stream)) {
        OPJ_UINT32 m_num_intersecting_tile_parts = 0;

        OPJ_UINT32 j;
        for (j = 0; j < p_j2k->m_cp.tw * p_j2k->m_cp.th; ++j) {
            if (p_j2k->cstr_index->tile_index[j].nb_tps > 0 &&
                    p_j2k->cstr_index->tile_index[j].tp_index[
                        p_j2k->cstr_index->tile_index[j].nb_tps - 1].end_pos > end_pos) {
                end_pos = p_j2k->cstr_index->tile_index[j].tp_index[
                              p_j2k->cstr_index->tile_index[j].nb_tps - 1].end_pos;
            }
        }

        for (j = p_j2k->m_specific_param.m_decoder.m_start_tile_y;
                j < p_j2k->m_specific_param.m_decoder.m_end_tile_y; ++j) {
            OPJ_UINT32 i;
            for (i = p_j2k->m_specific_param.m_decoder.m_start_tile_x;
                    i < p_j2k->m_specific_param.m_decoder.m_end_tile_x; ++i) {
                const OPJ_UINT32 tile_number = j * p_j2k->m_cp.tw + i;
                m_num_intersecting_tile_parts +=
                    p_j2k->cstr_index->tile_index[tile_number].nb_tps;
            }
        }

        p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset =
            (OPJ_OFF_T*)
            opj_malloc(m_num_intersecting_tile_parts * sizeof(OPJ_OFF_T));
        if (m_num_intersecting_tile_parts > 0 &&
                p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset) {
            OPJ_UINT32 idx = 0;
            for (j = p_j2k->m_specific_param.m_decoder.m_start_tile_y;
                    j < p_j2k->m_specific_param.m_decoder.m_end_tile_y; ++j) {
                OPJ_UINT32 i;
                for (i = p_j2k->m_specific_param.m_decoder.m_start_tile_x;
                        i < p_j2k->m_specific_param.m_decoder.m_end_tile_x; ++i) {
                    const OPJ_UINT32 tile_number = j * p_j2k->m_cp.tw + i;
                    OPJ_UINT32 k;
                    for (k = 0; k < p_j2k->cstr_index->tile_index[tile_number].nb_tps; ++k) {
                        const OPJ_OFF_T next_tp_sot_pos =
                            p_j2k->cstr_index->tile_index[tile_number].tp_index[k].start_pos;
                        p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset[idx] =
                            next_tp_sot_pos;
                        ++idx;
                    }
                }
            }

            p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts = idx;

            /* Sort by increasing offset */
            qsort(p_j2k->m_specific_param.m_decoder.m_intersecting_tile_parts_offset,
                  p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts,
                  sizeof(OPJ_OFF_T),
                  CompareOffT);
        }
    }

    for (;;) {
        if (p_j2k->m_cp.tw == 1 && p_j2k->m_cp.th == 1 &&
                p_j2k->m_cp.tcps[0].m_data != NULL) {
            l_current_tile_no = 0;
            p_j2k->m_current_tile_number = 0;
            p_j2k->m_specific_param.m_decoder.m_state |= J2K_STATE_DATA;
        } else {
            if (! opj_j2k_read_tile_header(p_j2k,
                                           &l_current_tile_no,
                                           NULL,
                                           &l_tile_x0, &l_tile_y0,
                                           &l_tile_x1, &l_tile_y1,
                                           &l_nb_comps,
                                           &l_go_on,
                                           p_stream,
                                           p_manager)) {
                return OPJ_FALSE;
            }

            if (! l_go_on) {
                break;
            }
        }

        if (! opj_j2k_decode_tile(p_j2k, l_current_tile_no, NULL, 0,
                                  p_stream, p_manager)) {
            opj_event_msg(p_manager, EVT_ERROR, "Failed to decode tile %d/%d\n",
                          l_current_tile_no + 1, p_j2k->m_cp.th * p_j2k->m_cp.tw);
            return OPJ_FALSE;
        }

        opj_event_msg(p_manager, EVT_INFO, "Tile %d/%d has been decoded.\n",
                      l_current_tile_no + 1, p_j2k->m_cp.th * p_j2k->m_cp.tw);

        if (! opj_j2k_update_image_data(p_j2k->m_tcd,
                                        p_j2k->m_output_image)) {
            return OPJ_FALSE;
        }

        if (p_j2k->m_cp.tw == 1 && p_j2k->m_cp.th == 1 &&
                !(p_j2k->m_output_image->x0 == p_j2k->m_private_image->x0 &&
                  p_j2k->m_output_image->y0 == p_j2k->m_private_image->y0 &&
                  p_j2k->m_output_image->x1 == p_j2k->m_private_image->x1 &&
                  p_j2k->m_output_image->y1 == p_j2k->m_private_image->y1)) {
            /* Keep current tcp data */
        } else {
            opj_j2k_tcp_data_destroy(&p_j2k->m_cp.tcps[l_current_tile_no]);
        }

        opj_event_msg(p_manager, EVT_INFO,
                      "Image data has been updated with tile %d.\n\n", l_current_tile_no + 1);

        if (opj_stream_get_number_byte_left(p_stream) == 0
                && p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_NEOC) {
            break;
        }
        if (++nr_tiles ==  p_j2k->m_cp.th * p_j2k->m_cp.tw) {
            break;
        }
        if (p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts > 0 &&
                p_j2k->m_specific_param.m_decoder.m_idx_intersecting_tile_parts ==
                p_j2k->m_specific_param.m_decoder.m_num_intersecting_tile_parts) {
            opj_stream_seek(p_stream, end_pos + 2, p_manager);
            break;
        }
    }

    if (! opj_j2k_are_all_used_components_decoded(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Sets up the procedures to do on decoding data. Developers wanting to extend the library can add their own reading procedures.
 */
static OPJ_BOOL opj_j2k_setup_decoding(opj_j2k_t *p_j2k,
                                       opj_event_mgr_t * p_manager)
{
    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_decode_tiles, p_manager)) {
        return OPJ_FALSE;
    }
    /* DEVELOPER CORNER, add your custom procedures */

    return OPJ_TRUE;
}

/*
 * Read and decode one tile.
 */
static OPJ_BOOL opj_j2k_decode_one_tile(opj_j2k_t *p_j2k,
                                        opj_stream_private_t *p_stream,
                                        opj_event_mgr_t * p_manager)
{
    OPJ_BOOL l_go_on = OPJ_TRUE;
    OPJ_UINT32 l_current_tile_no;
    OPJ_UINT32 l_tile_no_to_dec;
    OPJ_INT32 l_tile_x0, l_tile_y0, l_tile_x1, l_tile_y1;
    OPJ_UINT32 l_nb_comps;
    OPJ_UINT32 l_nb_tiles;
    OPJ_UINT32 i;

    /* Move into the codestream to the first SOT used to decode the desired tile */
    l_tile_no_to_dec = (OPJ_UINT32)
                       p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec;
    if (p_j2k->cstr_index->tile_index)
        if (p_j2k->cstr_index->tile_index->tp_index) {
            if (! p_j2k->cstr_index->tile_index[l_tile_no_to_dec].nb_tps) {
                /* the index for this tile has not been built,
                 *  so move to the last SOT read */
                if (!(opj_stream_read_seek(p_stream,
                                           p_j2k->m_specific_param.m_decoder.m_last_sot_read_pos + 2, p_manager))) {
                    opj_event_msg(p_manager, EVT_ERROR, "Problem with seek function\n");
                    return OPJ_FALSE;
                }
            } else {
                OPJ_OFF_T sot_pos =
                    p_j2k->cstr_index->tile_index[l_tile_no_to_dec].tp_index[0].start_pos;
                OPJ_UINT32 l_marker;

#if 0
                opj_event_msg(p_manager, EVT_INFO,
                              "opj_j2k_decode_one_tile(%u): seek to %" PRId64 "\n",
                              l_tile_no_to_dec,
                              sot_pos);
#endif
                if (!(opj_stream_read_seek(p_stream,
                                           sot_pos,
                                           p_manager))) {
                    opj_event_msg(p_manager, EVT_ERROR, "Problem with seek function\n");
                    return OPJ_FALSE;
                }

                /* Try to read 2 bytes (the marker ID) from stream and copy them into the buffer */
                if (opj_stream_read_data(p_stream,
                                         p_j2k->m_specific_param.m_decoder.m_header_data, 2, p_manager) != 2) {
                    opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
                    return OPJ_FALSE;
                }

                /* Read 2 bytes from the buffer as the marker ID */
                opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data, &l_marker,
                               2);

                if (l_marker != J2K_MS_SOT) {
                    opj_event_msg(p_manager, EVT_ERROR, "Did not get expected SOT marker\n");
                    return OPJ_FALSE;
                }
            }
            /* Special case if we have previously read the EOC marker (if the previous tile getted is the last ) */
            if (p_j2k->m_specific_param.m_decoder.m_state == J2K_STATE_EOC) {
                p_j2k->m_specific_param.m_decoder.m_state = J2K_STATE_TPHSOT;
            }
        }

    /* Reset current tile part number for all tiles, and not only the one */
    /* of interest. */
    /* Not completely sure this is always correct but required for */
    /* ./build/bin/j2k_random_tile_access ./build/tests/tte1.j2k */
    l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th;
    for (i = 0; i < l_nb_tiles; ++i) {
        p_j2k->m_cp.tcps[i].m_current_tile_part_number = -1;
    }

    for (;;) {
        if (! opj_j2k_read_tile_header(p_j2k,
                                       &l_current_tile_no,
                                       NULL,
                                       &l_tile_x0, &l_tile_y0,
                                       &l_tile_x1, &l_tile_y1,
                                       &l_nb_comps,
                                       &l_go_on,
                                       p_stream,
                                       p_manager)) {
            return OPJ_FALSE;
        }

        if (! l_go_on) {
            break;
        }

        if (! opj_j2k_decode_tile(p_j2k, l_current_tile_no, NULL, 0,
                                  p_stream, p_manager)) {
            return OPJ_FALSE;
        }
        opj_event_msg(p_manager, EVT_INFO, "Tile %d/%d has been decoded.\n",
                      l_current_tile_no + 1, p_j2k->m_cp.th * p_j2k->m_cp.tw);

        if (! opj_j2k_update_image_data(p_j2k->m_tcd,
                                        p_j2k->m_output_image)) {
            return OPJ_FALSE;
        }
        opj_j2k_tcp_data_destroy(&p_j2k->m_cp.tcps[l_current_tile_no]);

        opj_event_msg(p_manager, EVT_INFO,
                      "Image data has been updated with tile %d.\n\n", l_current_tile_no + 1);

        if (l_current_tile_no == l_tile_no_to_dec) {
            /* move into the codestream to the first SOT (FIXME or not move?)*/
            if (!(opj_stream_read_seek(p_stream, p_j2k->cstr_index->main_head_end + 2,
                                       p_manager))) {
                opj_event_msg(p_manager, EVT_ERROR, "Problem with seek function\n");
                return OPJ_FALSE;
            }
            break;
        } else {
            opj_event_msg(p_manager, EVT_WARNING,
                          "Tile read, decoded and updated is not the desired one (%d vs %d).\n",
                          l_current_tile_no + 1, l_tile_no_to_dec + 1);
        }

    }

    if (! opj_j2k_are_all_used_components_decoded(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

/**
 * Sets up the procedures to do on decoding one tile. Developers wanting to extend the library can add their own reading procedures.
 */
static OPJ_BOOL opj_j2k_setup_decoding_tile(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions*/
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_decode_one_tile, p_manager)) {
        return OPJ_FALSE;
    }
    /* DEVELOPER CORNER, add your custom procedures */

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_move_data_from_codec_to_output_image(opj_j2k_t * p_j2k,
        opj_image_t * p_image)
{
    OPJ_UINT32 compno;

    /* Move data and copy one information from codec to output image*/
    if (p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode > 0) {
        opj_image_comp_t* newcomps =
            (opj_image_comp_t*) opj_malloc(
                p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode *
                sizeof(opj_image_comp_t));
        if (newcomps == NULL) {
            opj_image_destroy(p_j2k->m_private_image);
            p_j2k->m_private_image = NULL;
            return OPJ_FALSE;
        }
        for (compno = 0; compno < p_image->numcomps; compno++) {
            opj_image_data_free(p_image->comps[compno].data);
            p_image->comps[compno].data = NULL;
        }
        for (compno = 0;
                compno < p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode; compno++) {
            OPJ_UINT32 src_compno =
                p_j2k->m_specific_param.m_decoder.m_comps_indices_to_decode[compno];
            memcpy(&(newcomps[compno]),
                   &(p_j2k->m_output_image->comps[src_compno]),
                   sizeof(opj_image_comp_t));
            newcomps[compno].resno_decoded =
                p_j2k->m_output_image->comps[src_compno].resno_decoded;
            newcomps[compno].data = p_j2k->m_output_image->comps[src_compno].data;
            p_j2k->m_output_image->comps[src_compno].data = NULL;
        }
        for (compno = 0; compno < p_image->numcomps; compno++) {
            assert(p_j2k->m_output_image->comps[compno].data == NULL);
            opj_image_data_free(p_j2k->m_output_image->comps[compno].data);
            p_j2k->m_output_image->comps[compno].data = NULL;
        }
        p_image->numcomps = p_j2k->m_specific_param.m_decoder.m_numcomps_to_decode;
        opj_free(p_image->comps);
        p_image->comps = newcomps;
    } else {
        for (compno = 0; compno < p_image->numcomps; compno++) {
            p_image->comps[compno].resno_decoded =
                p_j2k->m_output_image->comps[compno].resno_decoded;
            opj_image_data_free(p_image->comps[compno].data);
            p_image->comps[compno].data = p_j2k->m_output_image->comps[compno].data;
#if 0
            char fn[256];
            snprintf(fn, sizeof fn, "/tmp/%d.raw", compno);
            FILE *debug = fopen(fn, "wb");
            fwrite(p_image->comps[compno].data, sizeof(OPJ_INT32),
                   p_image->comps[compno].w * p_image->comps[compno].h, debug);
            fclose(debug);
#endif
            p_j2k->m_output_image->comps[compno].data = NULL;
        }
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_decode(opj_j2k_t * p_j2k,
                        opj_stream_private_t * p_stream,
                        opj_image_t * p_image,
                        opj_event_mgr_t * p_manager)
{
    if (!p_image) {
        return OPJ_FALSE;
    }

    /* Heuristics to detect sequence opj_read_header(), opj_set_decoded_resolution_factor() */
    /* and finally opj_decode_image() without manual setting of comps[].factor */
    /* We could potentially always execute it, if we don't allow people to do */
    /* opj_read_header(), modify x0,y0,x1,y1 of returned image an call opj_decode_image() */
    if (p_j2k->m_cp.m_specific_param.m_dec.m_reduce > 0 &&
            p_j2k->m_private_image != NULL &&
            p_j2k->m_private_image->numcomps > 0 &&
            p_j2k->m_private_image->comps[0].factor ==
            p_j2k->m_cp.m_specific_param.m_dec.m_reduce &&
            p_image->numcomps > 0 &&
            p_image->comps[0].factor == 0 &&
            /* Don't mess with image dimension if the user has allocated it */
            p_image->comps[0].data == NULL) {
        OPJ_UINT32 it_comp;

        /* Update the comps[].factor member of the output image with the one */
        /* of m_reduce */
        for (it_comp = 0; it_comp < p_image->numcomps; ++it_comp) {
            p_image->comps[it_comp].factor = p_j2k->m_cp.m_specific_param.m_dec.m_reduce;
        }
        if (!opj_j2k_update_image_dimensions(p_image, p_manager)) {
            return OPJ_FALSE;
        }
    }

    if (p_j2k->m_output_image == NULL) {
        p_j2k->m_output_image = opj_image_create0();
        if (!(p_j2k->m_output_image)) {
            return OPJ_FALSE;
        }
    }
    opj_copy_image_header(p_image, p_j2k->m_output_image);

    /* customization of the decoding */
    if (!opj_j2k_setup_decoding(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    /* Decode the codestream */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_procedure_list, p_stream, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    /* Move data and copy one information from codec to output image*/
    return opj_j2k_move_data_from_codec_to_output_image(p_j2k, p_image);
}

OPJ_BOOL opj_j2k_get_tile(opj_j2k_t *p_j2k,
                          opj_stream_private_t *p_stream,
                          opj_image_t* p_image,
                          opj_event_mgr_t * p_manager,
                          OPJ_UINT32 tile_index)
{
    OPJ_UINT32 compno;
    OPJ_UINT32 l_tile_x, l_tile_y;
    opj_image_comp_t* l_img_comp;

    if (!p_image) {
        opj_event_msg(p_manager, EVT_ERROR, "We need an image previously created.\n");
        return OPJ_FALSE;
    }

    if (p_image->numcomps < p_j2k->m_private_image->numcomps) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Image has less components than codestream.\n");
        return OPJ_FALSE;
    }

    if (/*(tile_index < 0) &&*/ (tile_index >= p_j2k->m_cp.tw * p_j2k->m_cp.th)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Tile index provided by the user is incorrect %d (max = %d) \n", tile_index,
                      (p_j2k->m_cp.tw * p_j2k->m_cp.th) - 1);
        return OPJ_FALSE;
    }

    /* Compute the dimension of the desired tile*/
    l_tile_x = tile_index % p_j2k->m_cp.tw;
    l_tile_y = tile_index / p_j2k->m_cp.tw;

    p_image->x0 = l_tile_x * p_j2k->m_cp.tdx + p_j2k->m_cp.tx0;
    if (p_image->x0 < p_j2k->m_private_image->x0) {
        p_image->x0 = p_j2k->m_private_image->x0;
    }
    p_image->x1 = (l_tile_x + 1) * p_j2k->m_cp.tdx + p_j2k->m_cp.tx0;
    if (p_image->x1 > p_j2k->m_private_image->x1) {
        p_image->x1 = p_j2k->m_private_image->x1;
    }

    p_image->y0 = l_tile_y * p_j2k->m_cp.tdy + p_j2k->m_cp.ty0;
    if (p_image->y0 < p_j2k->m_private_image->y0) {
        p_image->y0 = p_j2k->m_private_image->y0;
    }
    p_image->y1 = (l_tile_y + 1) * p_j2k->m_cp.tdy + p_j2k->m_cp.ty0;
    if (p_image->y1 > p_j2k->m_private_image->y1) {
        p_image->y1 = p_j2k->m_private_image->y1;
    }

    l_img_comp = p_image->comps;
    for (compno = 0; compno < p_j2k->m_private_image->numcomps; ++compno) {
        OPJ_INT32 l_comp_x1, l_comp_y1;

        l_img_comp->factor = p_j2k->m_private_image->comps[compno].factor;

        l_img_comp->x0 = opj_uint_ceildiv(p_image->x0, l_img_comp->dx);
        l_img_comp->y0 = opj_uint_ceildiv(p_image->y0, l_img_comp->dy);
        l_comp_x1 = opj_int_ceildiv((OPJ_INT32)p_image->x1, (OPJ_INT32)l_img_comp->dx);
        l_comp_y1 = opj_int_ceildiv((OPJ_INT32)p_image->y1, (OPJ_INT32)l_img_comp->dy);

        l_img_comp->w = (OPJ_UINT32)(opj_int_ceildivpow2(l_comp_x1,
                                     (OPJ_INT32)l_img_comp->factor) - opj_int_ceildivpow2((OPJ_INT32)l_img_comp->x0,
                                             (OPJ_INT32)l_img_comp->factor));
        l_img_comp->h = (OPJ_UINT32)(opj_int_ceildivpow2(l_comp_y1,
                                     (OPJ_INT32)l_img_comp->factor) - opj_int_ceildivpow2((OPJ_INT32)l_img_comp->y0,
                                             (OPJ_INT32)l_img_comp->factor));

        l_img_comp++;
    }

    if (p_image->numcomps > p_j2k->m_private_image->numcomps) {
        /* Can happen when calling repeatdly opj_get_decoded_tile() on an
         * image with a color palette, where color palette expansion is done
         * later in jp2.c */
        for (compno = p_j2k->m_private_image->numcomps; compno < p_image->numcomps;
                ++compno) {
            opj_image_data_free(p_image->comps[compno].data);
            p_image->comps[compno].data = NULL;
        }
        p_image->numcomps = p_j2k->m_private_image->numcomps;
    }

    /* Destroy the previous output image*/
    if (p_j2k->m_output_image) {
        opj_image_destroy(p_j2k->m_output_image);
    }

    /* Create the output image from the information previously computed*/
    p_j2k->m_output_image = opj_image_create0();
    if (!(p_j2k->m_output_image)) {
        return OPJ_FALSE;
    }
    opj_copy_image_header(p_image, p_j2k->m_output_image);

    p_j2k->m_specific_param.m_decoder.m_tile_ind_to_dec = (OPJ_INT32)tile_index;

    /* customization of the decoding */
    if (!opj_j2k_setup_decoding_tile(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    /* Decode the codestream */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_procedure_list, p_stream, p_manager)) {
        opj_image_destroy(p_j2k->m_private_image);
        p_j2k->m_private_image = NULL;
        return OPJ_FALSE;
    }

    /* Move data and copy one information from codec to output image*/
    return opj_j2k_move_data_from_codec_to_output_image(p_j2k, p_image);
}

OPJ_BOOL opj_j2k_set_decoded_resolution_factor(opj_j2k_t *p_j2k,
        OPJ_UINT32 res_factor,
        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 it_comp;

    p_j2k->m_cp.m_specific_param.m_dec.m_reduce = res_factor;

    if (p_j2k->m_private_image) {
        if (p_j2k->m_private_image->comps) {
            if (p_j2k->m_specific_param.m_decoder.m_default_tcp) {
                if (p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps) {
                    for (it_comp = 0 ; it_comp < p_j2k->m_private_image->numcomps; it_comp++) {
                        OPJ_UINT32 max_res =
                            p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps[it_comp].numresolutions;
                        if (res_factor >= max_res) {
                            opj_event_msg(p_manager, EVT_ERROR,
                                          "Resolution factor is greater than the maximum resolution in the component.\n");
                            return OPJ_FALSE;
                        }
                        p_j2k->m_private_image->comps[it_comp].factor = res_factor;
                    }
                    return OPJ_TRUE;
                }
            }
        }
    }

    return OPJ_FALSE;
}

/* ----------------------------------------------------------------------- */

OPJ_BOOL opj_j2k_encoder_set_extra_options(
    opj_j2k_t *p_j2k,
    const char* const* p_options,
    opj_event_mgr_t * p_manager)
{
    const char* const* p_option_iter;

    if (p_options == NULL) {
        return OPJ_TRUE;
    }

    for (p_option_iter = p_options; *p_option_iter != NULL; ++p_option_iter) {
        if (strncmp(*p_option_iter, "PLT=", 4) == 0) {
            if (strcmp(*p_option_iter, "PLT=YES") == 0) {
                p_j2k->m_specific_param.m_encoder.m_PLT = OPJ_TRUE;
            } else if (strcmp(*p_option_iter, "PLT=NO") == 0) {
                p_j2k->m_specific_param.m_encoder.m_PLT = OPJ_FALSE;
            } else {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid value for option: %s.\n", *p_option_iter);
                return OPJ_FALSE;
            }
        } else if (strncmp(*p_option_iter, "TLM=", 4) == 0) {
            if (strcmp(*p_option_iter, "TLM=YES") == 0) {
                p_j2k->m_specific_param.m_encoder.m_TLM = OPJ_TRUE;
            } else if (strcmp(*p_option_iter, "TLM=NO") == 0) {
                p_j2k->m_specific_param.m_encoder.m_TLM = OPJ_FALSE;
            } else {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid value for option: %s.\n", *p_option_iter);
                return OPJ_FALSE;
            }
        } else if (strncmp(*p_option_iter, "GUARD_BITS=", strlen("GUARD_BITS=")) == 0) {
            OPJ_UINT32 tileno;
            opj_cp_t *cp = cp = &(p_j2k->m_cp);

            int numgbits = atoi(*p_option_iter + strlen("GUARD_BITS="));
            if (numgbits < 0 || numgbits > 7) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Invalid value for option: %s. Should be in [0,7]\n", *p_option_iter);
                return OPJ_FALSE;
            }

            for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
                OPJ_UINT32 i;
                opj_tcp_t *tcp = &cp->tcps[tileno];
                for (i = 0; i < p_j2k->m_specific_param.m_encoder.m_nb_comps; i++) {
                    opj_tccp_t *tccp = &tcp->tccps[i];
                    tccp->numgbits = (OPJ_UINT32)numgbits;
                }
            }
        } else {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Invalid option: %s.\n", *p_option_iter);
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}

/* ----------------------------------------------------------------------- */

OPJ_BOOL opj_j2k_encode(opj_j2k_t * p_j2k,
                        opj_stream_private_t *p_stream,
                        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 i, j;
    OPJ_UINT32 l_nb_tiles;
    OPJ_SIZE_T l_max_tile_size = 0, l_current_tile_size;
    OPJ_BYTE * l_current_data = 00;
    OPJ_BOOL l_reuse_data = OPJ_FALSE;
    opj_tcd_t* p_tcd = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    p_tcd = p_j2k->m_tcd;

    l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
    if (l_nb_tiles == 1) {
        l_reuse_data = OPJ_TRUE;
#ifdef __SSE__
        for (j = 0; j < p_j2k->m_tcd->image->numcomps; ++j) {
            opj_image_comp_t * l_img_comp = p_tcd->image->comps + j;
            if (((size_t)l_img_comp->data & 0xFU) !=
                    0U) { /* tile data shall be aligned on 16 bytes */
                l_reuse_data = OPJ_FALSE;
            }
        }
#endif
    }
    for (i = 0; i < l_nb_tiles; ++i) {
        if (! opj_j2k_pre_write_tile(p_j2k, i, p_stream, p_manager)) {
            if (l_current_data) {
                opj_free(l_current_data);
            }
            return OPJ_FALSE;
        }

        /* if we only have one tile, then simply set tile component data equal to image component data */
        /* otherwise, allocate the data */
        for (j = 0; j < p_j2k->m_tcd->image->numcomps; ++j) {
            opj_tcd_tilecomp_t* l_tilec = p_tcd->tcd_image->tiles->comps + j;
            if (l_reuse_data) {
                opj_image_comp_t * l_img_comp = p_tcd->image->comps + j;
                l_tilec->data  =  l_img_comp->data;
                l_tilec->ownsData = OPJ_FALSE;
            } else {
                if (! opj_alloc_tile_component_data(l_tilec)) {
                    opj_event_msg(p_manager, EVT_ERROR, "Error allocating tile component data.");
                    if (l_current_data) {
                        opj_free(l_current_data);
                    }
                    return OPJ_FALSE;
                }
            }
        }
        l_current_tile_size = opj_tcd_get_encoder_input_buffer_size(p_j2k->m_tcd);
        if (!l_reuse_data) {
            if (l_current_tile_size > l_max_tile_size) {
                OPJ_BYTE *l_new_current_data = (OPJ_BYTE *) opj_realloc(l_current_data,
                                               l_current_tile_size);
                if (! l_new_current_data) {
                    if (l_current_data) {
                        opj_free(l_current_data);
                    }
                    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to encode all tiles\n");
                    return OPJ_FALSE;
                }
                l_current_data = l_new_current_data;
                l_max_tile_size = l_current_tile_size;
            }
            if (l_current_data == NULL) {
                /* Should not happen in practice, but will avoid Coverity to */
                /* complain about a null pointer dereference */
                assert(0);
                return OPJ_FALSE;
            }

            /* copy image data (32 bit) to l_current_data as contiguous, all-component, zero offset buffer */
            /* 32 bit components @ 8 bit precision get converted to 8 bit */
            /* 32 bit components @ 16 bit precision get converted to 16 bit */
            opj_j2k_get_tile_data(p_j2k->m_tcd, l_current_data);

            /* now copy this data into the tile component */
            if (! opj_tcd_copy_tile_data(p_j2k->m_tcd, l_current_data,
                                         l_current_tile_size)) {
                opj_event_msg(p_manager, EVT_ERROR,
                              "Size mismatch between tile data and sent data.");
                opj_free(l_current_data);
                return OPJ_FALSE;
            }
        }

        if (! opj_j2k_post_write_tile(p_j2k, p_stream, p_manager)) {
            if (l_current_data) {
                opj_free(l_current_data);
            }
            return OPJ_FALSE;
        }
    }

    if (l_current_data) {
        opj_free(l_current_data);
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_end_compress(opj_j2k_t *p_j2k,
                              opj_stream_private_t *p_stream,
                              opj_event_mgr_t * p_manager)
{
    /* customization of the encoding */
    if (! opj_j2k_setup_end_compress(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    if (! opj_j2k_exec(p_j2k, p_j2k->m_procedure_list, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_start_compress(opj_j2k_t *p_j2k,
                                opj_stream_private_t *p_stream,
                                opj_image_t * p_image,
                                opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    p_j2k->m_private_image = opj_image_create0();
    if (! p_j2k->m_private_image) {
        opj_event_msg(p_manager, EVT_ERROR, "Failed to allocate image header.");
        return OPJ_FALSE;
    }
    opj_copy_image_header(p_image, p_j2k->m_private_image);

    /* TODO_MSD: Find a better way */
    if (p_image->comps) {
        OPJ_UINT32 it_comp;
        for (it_comp = 0 ; it_comp < p_image->numcomps; it_comp++) {
            if (p_image->comps[it_comp].data) {
                p_j2k->m_private_image->comps[it_comp].data = p_image->comps[it_comp].data;
                p_image->comps[it_comp].data = NULL;

            }
        }
    }

    /* customization of the validation */
    if (! opj_j2k_setup_encoding_validation(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    /* validation of the parameters codec */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_validation_list, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    /* customization of the encoding */
    if (! opj_j2k_setup_header_writing(p_j2k, p_manager)) {
        return OPJ_FALSE;
    }

    /* write header */
    if (! opj_j2k_exec(p_j2k, p_j2k->m_procedure_list, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_pre_write_tile(opj_j2k_t * p_j2k,
                                       OPJ_UINT32 p_tile_index,
                                       opj_stream_private_t *p_stream,
                                       opj_event_mgr_t * p_manager)
{
    (void)p_stream;
    if (p_tile_index != p_j2k->m_current_tile_number) {
        opj_event_msg(p_manager, EVT_ERROR, "The given tile index does not match.");
        return OPJ_FALSE;
    }

    opj_event_msg(p_manager, EVT_INFO, "tile number %d / %d\n",
                  p_j2k->m_current_tile_number + 1, p_j2k->m_cp.tw * p_j2k->m_cp.th);

    p_j2k->m_specific_param.m_encoder.m_current_tile_part_number = 0;
    p_j2k->m_tcd->cur_totnum_tp = p_j2k->m_cp.tcps[p_tile_index].m_nb_tile_parts;
    p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = 0;

    /* initialisation before tile encoding  */
    if (! opj_tcd_init_encode_tile(p_j2k->m_tcd, p_j2k->m_current_tile_number,
                                   p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static void opj_get_tile_dimensions(opj_image_t * l_image,
                                    opj_tcd_tilecomp_t * l_tilec,
                                    opj_image_comp_t * l_img_comp,
                                    OPJ_UINT32* l_size_comp,
                                    OPJ_UINT32* l_width,
                                    OPJ_UINT32* l_height,
                                    OPJ_UINT32* l_offset_x,
                                    OPJ_UINT32* l_offset_y,
                                    OPJ_UINT32* l_image_width,
                                    OPJ_UINT32* l_stride,
                                    OPJ_UINT32* l_tile_offset)
{
    OPJ_UINT32 l_remaining;
    *l_size_comp = l_img_comp->prec >> 3; /* (/8) */
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    if (l_remaining) {
        *l_size_comp += 1;
    }

    if (*l_size_comp == 3) {
        *l_size_comp = 4;
    }

    *l_width  = (OPJ_UINT32)(l_tilec->x1 - l_tilec->x0);
    *l_height = (OPJ_UINT32)(l_tilec->y1 - l_tilec->y0);
    *l_offset_x = opj_uint_ceildiv(l_image->x0, l_img_comp->dx);
    *l_offset_y = opj_uint_ceildiv(l_image->y0, l_img_comp->dy);
    *l_image_width = opj_uint_ceildiv(l_image->x1 - l_image->x0, l_img_comp->dx);
    *l_stride = *l_image_width - *l_width;
    *l_tile_offset = ((OPJ_UINT32)l_tilec->x0 - *l_offset_x) + ((
                         OPJ_UINT32)l_tilec->y0 - *l_offset_y) * *l_image_width;
}

static void opj_j2k_get_tile_data(opj_tcd_t * p_tcd, OPJ_BYTE * p_data)
{
    OPJ_UINT32 i, j, k = 0;

    for (i = 0; i < p_tcd->image->numcomps; ++i) {
        opj_image_t * l_image =  p_tcd->image;
        OPJ_INT32 * l_src_ptr;
        opj_tcd_tilecomp_t * l_tilec = p_tcd->tcd_image->tiles->comps + i;
        opj_image_comp_t * l_img_comp = l_image->comps + i;
        OPJ_UINT32 l_size_comp, l_width, l_height, l_offset_x, l_offset_y,
                   l_image_width, l_stride, l_tile_offset;

        opj_get_tile_dimensions(l_image,
                                l_tilec,
                                l_img_comp,
                                &l_size_comp,
                                &l_width,
                                &l_height,
                                &l_offset_x,
                                &l_offset_y,
                                &l_image_width,
                                &l_stride,
                                &l_tile_offset);

        l_src_ptr = l_img_comp->data + l_tile_offset;

        switch (l_size_comp) {
        case 1: {
            OPJ_CHAR * l_dest_ptr = (OPJ_CHAR*) p_data;
            if (l_img_comp->sgnd) {
                for (j = 0; j < l_height; ++j) {
                    for (k = 0; k < l_width; ++k) {
                        *(l_dest_ptr) = (OPJ_CHAR)(*l_src_ptr);
                        ++l_dest_ptr;
                        ++l_src_ptr;
                    }
                    l_src_ptr += l_stride;
                }
            } else {
                for (j = 0; j < l_height; ++j) {
                    for (k = 0; k < l_width; ++k) {
                        *(l_dest_ptr) = (OPJ_CHAR)((*l_src_ptr) & 0xff);
                        ++l_dest_ptr;
                        ++l_src_ptr;
                    }
                    l_src_ptr += l_stride;
                }
            }

            p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
        case 2: {
            OPJ_INT16 * l_dest_ptr = (OPJ_INT16 *) p_data;
            if (l_img_comp->sgnd) {
                for (j = 0; j < l_height; ++j) {
                    for (k = 0; k < l_width; ++k) {
                        *(l_dest_ptr++) = (OPJ_INT16)(*(l_src_ptr++));
                    }
                    l_src_ptr += l_stride;
                }
            } else {
                for (j = 0; j < l_height; ++j) {
                    for (k = 0; k < l_width; ++k) {
                        *(l_dest_ptr++) = (OPJ_INT16)((*(l_src_ptr++)) & 0xffff);
                    }
                    l_src_ptr += l_stride;
                }
            }

            p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
        case 4: {
            OPJ_INT32 * l_dest_ptr = (OPJ_INT32 *) p_data;
            for (j = 0; j < l_height; ++j) {
                for (k = 0; k < l_width; ++k) {
                    *(l_dest_ptr++) = *(l_src_ptr++);
                }
                l_src_ptr += l_stride;
            }

            p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
        }
    }
}

static OPJ_BOOL opj_j2k_post_write_tile(opj_j2k_t * p_j2k,
                                        opj_stream_private_t *p_stream,
                                        opj_event_mgr_t * p_manager)
{
    OPJ_UINT32 l_nb_bytes_written;
    OPJ_BYTE * l_current_data = 00;
    OPJ_UINT32 l_tile_size = 0;
    OPJ_UINT32 l_available_data;

    /* preconditions */
    assert(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);

    l_tile_size = p_j2k->m_specific_param.m_encoder.m_encoded_tile_size;
    l_available_data = l_tile_size;
    l_current_data = p_j2k->m_specific_param.m_encoder.m_encoded_tile_data;

    l_nb_bytes_written = 0;
    if (! opj_j2k_write_first_tile_part(p_j2k, l_current_data, &l_nb_bytes_written,
                                        l_available_data, p_stream, p_manager)) {
        return OPJ_FALSE;
    }
    l_current_data += l_nb_bytes_written;
    l_available_data -= l_nb_bytes_written;

    l_nb_bytes_written = 0;
    if (! opj_j2k_write_all_tile_parts(p_j2k, l_current_data, &l_nb_bytes_written,
                                       l_available_data, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    l_available_data -= l_nb_bytes_written;
    l_nb_bytes_written = l_tile_size - l_available_data;

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_encoded_tile_data,
                              l_nb_bytes_written, p_manager) != l_nb_bytes_written) {
        return OPJ_FALSE;
    }

    ++p_j2k->m_current_tile_number;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_setup_end_compress(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    /* DEVELOPER CORNER, insert your custom procedures */
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_eoc, p_manager)) {
        return OPJ_FALSE;
    }

    if (p_j2k->m_specific_param.m_encoder.m_TLM) {
        if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                               (opj_procedure)opj_j2k_write_updated_tlm, p_manager)) {
            return OPJ_FALSE;
        }
    }

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_epc, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_end_encoding, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_destroy_header_memory, p_manager)) {
        return OPJ_FALSE;
    }
    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_setup_encoding_validation(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_validation_list,
                                           (opj_procedure)opj_j2k_build_encoder, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_validation_list,
                                           (opj_procedure)opj_j2k_encoding_validation, p_manager)) {
        return OPJ_FALSE;
    }

    /* DEVELOPER CORNER, add your custom validation procedure */
    if (! opj_procedure_list_add_procedure(p_j2k->m_validation_list,
                                           (opj_procedure)opj_j2k_mct_validation, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_setup_header_writing(opj_j2k_t *p_j2k,
        opj_event_mgr_t * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_init_info, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_soc, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_siz, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_cod, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_qcd, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_all_coc, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_all_qcc, p_manager)) {
        return OPJ_FALSE;
    }

    if (p_j2k->m_specific_param.m_encoder.m_TLM) {
        if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                               (opj_procedure)opj_j2k_write_tlm, p_manager)) {
            return OPJ_FALSE;
        }

        if (p_j2k->m_cp.rsiz == OPJ_PROFILE_CINEMA_4K) {
            if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                                   (opj_procedure)opj_j2k_write_poc, p_manager)) {
                return OPJ_FALSE;
            }
        }
    }

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_write_regions, p_manager)) {
        return OPJ_FALSE;
    }

    if (p_j2k->m_cp.comment != 00)  {
        if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                               (opj_procedure)opj_j2k_write_com, p_manager)) {
            return OPJ_FALSE;
        }
    }

    /* DEVELOPER CORNER, insert your custom procedures */
    if ((p_j2k->m_cp.rsiz & (OPJ_PROFILE_PART2 | OPJ_EXTENSION_MCT)) ==
            (OPJ_PROFILE_PART2 | OPJ_EXTENSION_MCT)) {
        if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                               (opj_procedure)opj_j2k_write_mct_data_group, p_manager)) {
            return OPJ_FALSE;
        }
    }
    /* End of Developer Corner */

    if (p_j2k->cstr_index) {
        if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                               (opj_procedure)opj_j2k_get_end_header, p_manager)) {
            return OPJ_FALSE;
        }
    }

    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_create_tcd, p_manager)) {
        return OPJ_FALSE;
    }
    if (! opj_procedure_list_add_procedure(p_j2k->m_procedure_list,
                                           (opj_procedure)opj_j2k_update_rates, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_first_tile_part(opj_j2k_t *p_j2k,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_data_written,
        OPJ_UINT32 total_data_size,
        opj_stream_private_t *p_stream,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 l_nb_bytes_written = 0;
    OPJ_UINT32 l_current_nb_bytes_written;
    OPJ_BYTE * l_begin_data = 00;

    opj_tcd_t * l_tcd = 00;
    opj_cp_t * l_cp = 00;

    l_tcd = p_j2k->m_tcd;
    l_cp = &(p_j2k->m_cp);

    l_tcd->cur_pino = 0;

    /*Get number of tile parts*/
    p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = 0;

    /* INDEX >> */
    /* << INDEX */

    l_current_nb_bytes_written = 0;
    l_begin_data = p_data;
    if (! opj_j2k_write_sot(p_j2k, p_data, total_data_size,
                            &l_current_nb_bytes_written, p_stream,
                            p_manager)) {
        return OPJ_FALSE;
    }

    l_nb_bytes_written += l_current_nb_bytes_written;
    p_data += l_current_nb_bytes_written;
    total_data_size -= l_current_nb_bytes_written;

    if (!OPJ_IS_CINEMA(l_cp->rsiz)) {
#if 0
        for (compno = 1; compno < p_j2k->m_private_image->numcomps; compno++) {
            l_current_nb_bytes_written = 0;
            opj_j2k_write_coc_in_memory(p_j2k, compno, p_data, &l_current_nb_bytes_written,
                                        p_manager);
            l_nb_bytes_written += l_current_nb_bytes_written;
            p_data += l_current_nb_bytes_written;
            total_data_size -= l_current_nb_bytes_written;

            l_current_nb_bytes_written = 0;
            opj_j2k_write_qcc_in_memory(p_j2k, compno, p_data, &l_current_nb_bytes_written,
                                        p_manager);
            l_nb_bytes_written += l_current_nb_bytes_written;
            p_data += l_current_nb_bytes_written;
            total_data_size -= l_current_nb_bytes_written;
        }
#endif
        if (l_cp->tcps[p_j2k->m_current_tile_number].POC) {
            l_current_nb_bytes_written = 0;
            opj_j2k_write_poc_in_memory(p_j2k, p_data, &l_current_nb_bytes_written,
                                        p_manager);
            l_nb_bytes_written += l_current_nb_bytes_written;
            p_data += l_current_nb_bytes_written;
            total_data_size -= l_current_nb_bytes_written;
        }
    }

    l_current_nb_bytes_written = 0;
    if (! opj_j2k_write_sod(p_j2k, l_tcd, p_data, &l_current_nb_bytes_written,
                            total_data_size, p_stream, p_manager)) {
        return OPJ_FALSE;
    }

    l_nb_bytes_written += l_current_nb_bytes_written;
    * p_data_written = l_nb_bytes_written;

    /* Writing Psot in SOT marker */
    opj_write_bytes(l_begin_data + 6, l_nb_bytes_written,
                    4);                                 /* PSOT */

    if (p_j2k->m_specific_param.m_encoder.m_TLM) {
        opj_j2k_update_tlm(p_j2k, l_nb_bytes_written);
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_all_tile_parts(opj_j2k_t *p_j2k,
        OPJ_BYTE * p_data,
        OPJ_UINT32 * p_data_written,
        OPJ_UINT32 total_data_size,
        opj_stream_private_t *p_stream,
        struct opj_event_mgr * p_manager
                                            )
{
    OPJ_UINT32 tilepartno = 0;
    OPJ_UINT32 l_nb_bytes_written = 0;
    OPJ_UINT32 l_current_nb_bytes_written;
    OPJ_UINT32 l_part_tile_size;
    OPJ_UINT32 tot_num_tp;
    OPJ_UINT32 pino;

    OPJ_BYTE * l_begin_data;
    opj_tcp_t *l_tcp = 00;
    opj_tcd_t * l_tcd = 00;
    opj_cp_t * l_cp = 00;

    l_tcd = p_j2k->m_tcd;
    l_cp = &(p_j2k->m_cp);
    l_tcp = l_cp->tcps + p_j2k->m_current_tile_number;

    /*Get number of tile parts*/
    tot_num_tp = opj_j2k_get_num_tp(l_cp, 0, p_j2k->m_current_tile_number);

    /* start writing remaining tile parts */
    ++p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
    for (tilepartno = 1; tilepartno < tot_num_tp ; ++tilepartno) {
        p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = tilepartno;
        l_current_nb_bytes_written = 0;
        l_part_tile_size = 0;
        l_begin_data = p_data;

        if (! opj_j2k_write_sot(p_j2k, p_data,
                                total_data_size,
                                &l_current_nb_bytes_written,
                                p_stream,
                                p_manager)) {
            return OPJ_FALSE;
        }

        l_nb_bytes_written += l_current_nb_bytes_written;
        p_data += l_current_nb_bytes_written;
        total_data_size -= l_current_nb_bytes_written;
        l_part_tile_size += l_current_nb_bytes_written;

        l_current_nb_bytes_written = 0;
        if (! opj_j2k_write_sod(p_j2k, l_tcd, p_data, &l_current_nb_bytes_written,
                                total_data_size, p_stream, p_manager)) {
            return OPJ_FALSE;
        }

        p_data += l_current_nb_bytes_written;
        l_nb_bytes_written += l_current_nb_bytes_written;
        total_data_size -= l_current_nb_bytes_written;
        l_part_tile_size += l_current_nb_bytes_written;

        /* Writing Psot in SOT marker */
        opj_write_bytes(l_begin_data + 6, l_part_tile_size,
                        4);                                   /* PSOT */

        if (p_j2k->m_specific_param.m_encoder.m_TLM) {
            opj_j2k_update_tlm(p_j2k, l_part_tile_size);
        }

        ++p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
    }

    for (pino = 1; pino <= l_tcp->numpocs; ++pino) {
        l_tcd->cur_pino = pino;

        /*Get number of tile parts*/
        tot_num_tp = opj_j2k_get_num_tp(l_cp, pino, p_j2k->m_current_tile_number);
        for (tilepartno = 0; tilepartno < tot_num_tp ; ++tilepartno) {
            p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = tilepartno;
            l_current_nb_bytes_written = 0;
            l_part_tile_size = 0;
            l_begin_data = p_data;

            if (! opj_j2k_write_sot(p_j2k, p_data,
                                    total_data_size,
                                    &l_current_nb_bytes_written, p_stream,
                                    p_manager)) {
                return OPJ_FALSE;
            }

            l_nb_bytes_written += l_current_nb_bytes_written;
            p_data += l_current_nb_bytes_written;
            total_data_size -= l_current_nb_bytes_written;
            l_part_tile_size += l_current_nb_bytes_written;

            l_current_nb_bytes_written = 0;

            if (! opj_j2k_write_sod(p_j2k, l_tcd, p_data, &l_current_nb_bytes_written,
                                    total_data_size, p_stream, p_manager)) {
                return OPJ_FALSE;
            }

            l_nb_bytes_written += l_current_nb_bytes_written;
            p_data += l_current_nb_bytes_written;
            total_data_size -= l_current_nb_bytes_written;
            l_part_tile_size += l_current_nb_bytes_written;

            /* Writing Psot in SOT marker */
            opj_write_bytes(l_begin_data + 6, l_part_tile_size,
                            4);                                   /* PSOT */

            if (p_j2k->m_specific_param.m_encoder.m_TLM) {
                opj_j2k_update_tlm(p_j2k, l_part_tile_size);
            }

            ++p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
        }
    }

    *p_data_written = l_nb_bytes_written;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_write_updated_tlm(opj_j2k_t *p_j2k,
        struct opj_stream_private *p_stream,
        struct opj_event_mgr * p_manager)
{
    OPJ_UINT32 l_tlm_size;
    OPJ_OFF_T l_tlm_position, l_current_position;
    OPJ_UINT32 size_per_tile_part;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    size_per_tile_part = p_j2k->m_specific_param.m_encoder.m_Ttlmi_is_byte ? 5 : 6;
    l_tlm_size = size_per_tile_part *
                 p_j2k->m_specific_param.m_encoder.m_total_tile_parts;
    l_tlm_position = 6 + p_j2k->m_specific_param.m_encoder.m_tlm_start;
    l_current_position = opj_stream_tell(p_stream);

    if (! opj_stream_seek(p_stream, l_tlm_position, p_manager)) {
        return OPJ_FALSE;
    }

    if (opj_stream_write_data(p_stream,
                              p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer, l_tlm_size,
                              p_manager) != l_tlm_size) {
        return OPJ_FALSE;
    }

    if (! opj_stream_seek(p_stream, l_current_position, p_manager)) {
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_end_encoding(opj_j2k_t *p_j2k,
                                     struct opj_stream_private *p_stream,
                                     struct opj_event_mgr * p_manager)
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_stream);
    OPJ_UNUSED(p_manager);

    opj_tcd_destroy(p_j2k->m_tcd);
    p_j2k->m_tcd = 00;

    if (p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer) {
        opj_free(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer);
        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = 0;
        p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current = 0;
    }

    if (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data) {
        opj_free(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);
        p_j2k->m_specific_param.m_encoder.m_encoded_tile_data = 0;
    }

    p_j2k->m_specific_param.m_encoder.m_encoded_tile_size = 0;

    return OPJ_TRUE;
}

/**
 * Destroys the memory associated with the decoding of headers.
 */
static OPJ_BOOL opj_j2k_destroy_header_memory(opj_j2k_t * p_j2k,
        opj_stream_private_t *p_stream,
        opj_event_mgr_t * p_manager
                                             )
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_stream != 00);
    assert(p_manager != 00);

    OPJ_UNUSED(p_stream);
    OPJ_UNUSED(p_manager);

    if (p_j2k->m_specific_param.m_encoder.m_header_tile_data) {
        opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
        p_j2k->m_specific_param.m_encoder.m_header_tile_data = 0;
    }

    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;

    return OPJ_TRUE;
}

static OPJ_BOOL opj_j2k_init_info(opj_j2k_t *p_j2k,
                                  struct opj_stream_private *p_stream,
                                  struct opj_event_mgr * p_manager)
{
    opj_codestream_info_t * l_cstr_info = 00;

    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);
    (void)l_cstr_info;

    OPJ_UNUSED(p_stream);

    /* TODO mergeV2: check this part which use cstr_info */
    /*l_cstr_info = p_j2k->cstr_info;

    if (l_cstr_info)  {
            OPJ_UINT32 compno;
            l_cstr_info->tile = (opj_tile_info_t *) opj_malloc(p_j2k->m_cp.tw * p_j2k->m_cp.th * sizeof(opj_tile_info_t));

            l_cstr_info->image_w = p_j2k->m_image->x1 - p_j2k->m_image->x0;
            l_cstr_info->image_h = p_j2k->m_image->y1 - p_j2k->m_image->y0;

            l_cstr_info->prog = (&p_j2k->m_cp.tcps[0])->prg;

            l_cstr_info->tw = p_j2k->m_cp.tw;
            l_cstr_info->th = p_j2k->m_cp.th;

            l_cstr_info->tile_x = p_j2k->m_cp.tdx;*/        /* new version parser */
    /*l_cstr_info->tile_y = p_j2k->m_cp.tdy;*/      /* new version parser */
    /*l_cstr_info->tile_Ox = p_j2k->m_cp.tx0;*/     /* new version parser */
    /*l_cstr_info->tile_Oy = p_j2k->m_cp.ty0;*/     /* new version parser */

    /*l_cstr_info->numcomps = p_j2k->m_image->numcomps;

    l_cstr_info->numlayers = (&p_j2k->m_cp.tcps[0])->numlayers;

    l_cstr_info->numdecompos = (OPJ_INT32*) opj_malloc(p_j2k->m_image->numcomps * sizeof(OPJ_INT32));

    for (compno=0; compno < p_j2k->m_image->numcomps; compno++) {
            l_cstr_info->numdecompos[compno] = (&p_j2k->m_cp.tcps[0])->tccps->numresolutions - 1;
    }

    l_cstr_info->D_max = 0.0;       */      /* ADD Marcela */

    /*l_cstr_info->main_head_start = opj_stream_tell(p_stream);*/ /* position of SOC */

    /*l_cstr_info->maxmarknum = 100;
    l_cstr_info->marker = (opj_marker_info_t *) opj_malloc(l_cstr_info->maxmarknum * sizeof(opj_marker_info_t));
    l_cstr_info->marknum = 0;
    }*/

    return opj_j2k_calculate_tp(p_j2k, &(p_j2k->m_cp),
                                &p_j2k->m_specific_param.m_encoder.m_total_tile_parts, p_j2k->m_private_image,
                                p_manager);
}

/**
 * Creates a tile-coder encoder.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
static OPJ_BOOL opj_j2k_create_tcd(opj_j2k_t *p_j2k,
                                   opj_stream_private_t *p_stream,
                                   opj_event_mgr_t * p_manager
                                  )
{
    /* preconditions */
    assert(p_j2k != 00);
    assert(p_manager != 00);
    assert(p_stream != 00);

    OPJ_UNUSED(p_stream);

    p_j2k->m_tcd = opj_tcd_create(OPJ_FALSE);

    if (! p_j2k->m_tcd) {
        opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to create Tile Coder\n");
        return OPJ_FALSE;
    }

    if (!opj_tcd_init(p_j2k->m_tcd, p_j2k->m_private_image, &p_j2k->m_cp,
                      p_j2k->m_tp)) {
        opj_tcd_destroy(p_j2k->m_tcd);
        p_j2k->m_tcd = 00;
        return OPJ_FALSE;
    }

    return OPJ_TRUE;
}

OPJ_BOOL opj_j2k_write_tile(opj_j2k_t * p_j2k,
                            OPJ_UINT32 p_tile_index,
                            OPJ_BYTE * p_data,
                            OPJ_UINT32 p_data_size,
                            opj_stream_private_t *p_stream,
                            opj_event_mgr_t * p_manager)
{
    if (! opj_j2k_pre_write_tile(p_j2k, p_tile_index, p_stream, p_manager)) {
        opj_event_msg(p_manager, EVT_ERROR,
                      "Error while opj_j2k_pre_write_tile with tile index = %d\n", p_tile_index);
        return OPJ_FALSE;
    } else {
        OPJ_UINT32 j;
        /* Allocate data */
        for (j = 0; j < p_j2k->m_tcd->image->numcomps; ++j) {
            opj_tcd_tilecomp_t* l_tilec = p_j2k->m_tcd->tcd_image->tiles->comps + j;

            if (! opj_alloc_tile_component_data(l_tilec)) {
                opj_event_msg(p_manager, EVT_ERROR, "Error allocating tile component data.");
                return OPJ_FALSE;
            }
        }

        /* now copy data into the tile component */
        if (! opj_tcd_copy_tile_data(p_j2k->m_tcd, p_data, p_data_size)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Size mismatch between tile data and sent data.");
            return OPJ_FALSE;
        }
        if (! opj_j2k_post_write_tile(p_j2k, p_stream, p_manager)) {
            opj_event_msg(p_manager, EVT_ERROR,
                          "Error while opj_j2k_post_write_tile with tile index = %d\n", p_tile_index);
            return OPJ_FALSE;
        }
    }

    return OPJ_TRUE;
}
