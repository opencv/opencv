/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2006-2007, Parvatha Elangovan
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

#include "j2k.h"
#include "opj_malloc.h"
#include "opj_includes.h"
#include "pi.h"
#include "event.h"
#include "cio.h"
#include "int.h"
#include "tcd.h"
#include "function_list.h"
#include "invert.h"
#include "dwt.h"
#include "mct.h"
#include "image.h"

/** @defgroup J2K J2K - JPEG-2000 codestream reader/writer */
/*@{*/


/***************************************************************************
 ********************** TYPEDEFS *******************************************
 ***************************************************************************/
/**
 * Correspondance prog order <-> string representation
 */
typedef struct j2k_prog_order
{
  OPJ_PROG_ORDER enum_prog;
  OPJ_CHAR str_prog[5];
}
j2k_prog_order_t;

typedef struct opj_dec_memory_marker_handler
{
  /** marker value */
  OPJ_UINT32 id;
  /** value of the state when the marker can appear */
  OPJ_UINT32 states;
  /** action linked to the marker */
  bool (*handler) (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
            );
}
opj_dec_memory_marker_handler_t;



/** @name Local static functions */
/*@{*/
/**
 * Writes a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 *
 * @param  p_comp_no  the component number to output.
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
 *
*/
static bool j2k_write_SPCod_SPCoc(
                opj_j2k_t *p_j2k,
              OPJ_UINT32 p_tile_no,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
          );

/**
 * Reads a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 * @param  p_header_data  the data contained in the COM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COM marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_SPCod_SPCoc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 compno,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
              );

/**
 * Gets the size taken by writting a SPCod or SPCoc for the given tile and component.
 *
 * @param  p_tile_no    the tile indix.
 * @param  p_comp_no    the component being outputted.
 * @param  p_j2k      the J2K codec.
 *
 * @return  the number of bytes taken by the SPCod element.
 */
static OPJ_UINT32 j2k_get_SPCod_SPCoc_size (
            opj_j2k_t *p_j2k,
            OPJ_UINT32 p_tile_no,
            OPJ_UINT32 p_comp_no
            );

/**
 * Writes a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param  p_tile_no    the tile to output.
 * @param  p_comp_no    the component number to output.
 * @param  p_data      the data buffer.
 * @param  p_header_size  pointer to the size of the data buffer, it is changed by the function.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
 *
*/
static bool j2k_write_SQcd_SQcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_tile_no,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
          );

/**
 * Reads a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param  p_tile_no    the tile to output.
 * @param  p_comp_no    the component number to output.
 * @param  p_data      the data buffer.
 * @param  p_header_size  pointer to the size of the data buffer, it is changed by the function.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
 *
*/
static bool j2k_read_SQcd_SQcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 compno,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
          );
/**
 * Updates the Tile Length Marker.
 */
static void j2k_update_tlm (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 p_tile_part_size);

/**
 * Gets the size taken by writting SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param  p_tile_no    the tile indix.
 * @param  p_comp_no    the component being outputted.
 * @param  p_j2k      the J2K codec.
 *
 * @return  the number of bytes taken by the SPCod element.
 */
static OPJ_UINT32 j2k_get_SQcd_SQcc_size (
                  opj_j2k_t *p_j2k,
                    OPJ_UINT32 p_tile_no,
                  OPJ_UINT32 p_comp_no

            );

/**
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param    p_j2k    the J2k codec.
 */
static void j2k_copy_tile_component_parameters(
              opj_j2k_t *p_j2k
              );

/**
 * Writes the SOC marker (Start Of Codestream)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/

static bool j2k_write_soc(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
                );
/**
 * Reads a SOC marker (Start of Codestream)
 * @param  p_header_data  the data contained in the SOC box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the SOC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_soc(
          opj_j2k_t *p_j2k,
          struct opj_stream_private *p_stream,
          struct opj_event_mgr * p_manager
         );
/**
 * Writes the SIZ marker (image and tile size)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
static bool j2k_write_siz(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
                );
/**
 * Writes the CBD-MCT-MCC-MCO markers (Multi components transform)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
static bool j2k_write_mct_data_group(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
                );

/**
 * Reads a SIZ marker (image and tile size)
 * @param  p_header_data  the data contained in the SIZ box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the SIZ marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_siz (
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 p_header_size,
              struct opj_event_mgr * p_manager
          );
/**
 * Writes the COM marker (comment)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
static bool j2k_write_com(
          opj_j2k_t *p_j2k,
          struct opj_stream_private *p_stream,
          struct opj_event_mgr * p_manager
          );
/**
 * Reads a COM marker (comments)
 * @param  p_header_data  the data contained in the COM box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the COM marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_com (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          );



/**
 * Writes the COD marker (Coding style default)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
static bool j2k_write_cod(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
            );
/**
 * Reads a COD marker (Coding Styke defaults)
 * @param  p_header_data  the data contained in the COD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COD marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_cod (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          );

/**
 * Writes the COC marker (Coding style component)
 *
 * @param  p_comp_number  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_coc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_number,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
              );

/**
 * Writes the COC marker (Coding style component)
 *
 * @param  p_comp_no    the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static void j2k_write_coc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
            );
/**
 * Gets the maximum size taken by a coc.
 *
 * @param  p_j2k  the jpeg2000 codec to use.
 */
static OPJ_UINT32 j2k_get_max_coc_size(opj_j2k_t *p_j2k);

/**
 * Reads a COC marker (Coding Style Component)
 * @param  p_header_data  the data contained in the COC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_coc (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          );

/**
 * Writes the QCD marker (quantization default)
 *
 * @param  p_comp_number  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_qcd(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
              );


/**
 * Reads a QCD marker (Quantization defaults)
 * @param  p_header_data  the data contained in the QCD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the QCD marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_qcd (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          );
/**
 * Writes the QCC marker (quantization component)
 *
 * @param  p_comp_no  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_qcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
              );
/**
 * Writes the QCC marker (quantization component)
 *
 * @param  p_comp_no  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static void j2k_write_qcc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
              );
/**
 * Gets the maximum size taken by a qcc.
 */
static OPJ_UINT32 j2k_get_max_qcc_size (opj_j2k_t *p_j2k);

/**
 * Reads a QCC marker (Quantization component)
 * @param  p_header_data  the data contained in the QCC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the QCC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_qcc(
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 p_header_size,
              struct opj_event_mgr * p_manager);
/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_poc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the updated tlm.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_updated_tlm(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
 */
static void j2k_write_poc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
          );

/**
 * Gets the maximum size taken by the writting of a POC.
 */
static OPJ_UINT32 j2k_get_max_poc_size(opj_j2k_t *p_j2k);

/**
 * Gets the maximum size taken by the toc headers of all the tile parts of any given tile.
 */
static OPJ_UINT32 j2k_get_max_toc_size (opj_j2k_t *p_j2k);

/**
 * Gets the maximum size taken by the headers of the SOT.
 *
 * @param  p_j2k  the jpeg2000 codec to use.
 */
static OPJ_UINT32 j2k_get_specific_header_sizes(opj_j2k_t *p_j2k);

/**
 * Reads a POC marker (Progression Order Change)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_poc (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a CRG marker (Component registration)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_crg (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_tlm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_plm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param  p_header_data  the data contained in the PLT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PLT marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_plt (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a PPM marker (Packed packet headers, main header)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_ppm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param  p_header_data  the data contained in the PPT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PPT marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_ppt (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes the TLM marker (Tile Length Marker)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_tlm(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes the SOT marker (Start of tile-part)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_sot(
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_data,
            OPJ_UINT32 * p_data_written,
            const struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param  p_header_data  the data contained in the PPT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PPT marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_sot (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes the SOD marker (Start of data)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_sod(
            opj_j2k_t *p_j2k,
            struct opj_tcd * p_tile_coder,
            OPJ_BYTE * p_data,
            OPJ_UINT32 * p_data_written,
            OPJ_UINT32 p_total_data_size,
            const struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a SOD marker (Start Of Data)
 *
 * @param  p_header_data  the data contained in the SOD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the SOD marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_sod (
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes the RGN marker (Region Of Interest)
 *
 * @param  p_tile_no    the tile to output
 * @param  p_comp_no    the component to output
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_rgn(
            opj_j2k_t *p_j2k,
            OPJ_UINT32 p_tile_no,
            OPJ_UINT32 p_comp_no,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads a RGN marker (Region Of Interest)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_rgn (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          ) ;
/**
 * Writes the EOC marker (End of Codestream)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_eoc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param    p_j2k    the J2k codec.
 */
static void j2k_copy_tile_quantization_parameters(
              opj_j2k_t *p_j2k
              );

/**
 * Reads a EOC marker (End Of Codestream)
 *
 * @param  p_header_data  the data contained in the SOD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the SOD marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_eoc (
              opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          ) ;

/**
 * Inits the Info
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_init_info(
              opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Reads an unknown marker
 *
 * @param  p_stream        the stream object to read from.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_manager    the user event manager.
 *
 * @return  true      if the marker could be deduced.
*/
static bool j2k_read_unk (
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Ends the encoding, i.e. frees memory.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_end_encoding(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the CBD marker (Component bit depth definition)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_cbd(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Reads a CBD marker (Component bit depth definition)
 * @param  p_header_data  the data contained in the CBD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the CBD marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_cbd (
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 p_header_size,
              struct opj_event_mgr * p_manager);

/**
 * Writes the MCT marker (Multiple Component Transform)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_mct_record(
            opj_j2k_t *p_j2k,
            opj_mct_data_t * p_mct_record,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param  p_header_data  the data contained in the MCT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCT marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_mct (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the MCC marker (Multiple Component Collection)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_mcc_record(
              opj_j2k_t *p_j2k,
            struct opj_simple_mcc_decorrelation_data * p_mcc_record,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Reads a MCC marker (Multiple Component Collection)
 *
 * @param  p_header_data  the data contained in the MCC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCC marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_mcc (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes the MCO marker (Multiple component transformation ordering)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_mco(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param  p_header_data  the data contained in the MCO box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCO marker.
 * @param  p_manager    the user event manager.
*/
static bool j2k_read_mco (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes the image components.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_image_components(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Writes regions of interests.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_regions(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );
/**
 * Writes EPC ????
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_write_epc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Checks the progression order changes values. Tells of the poc given as input are valid.
 * A nice message is outputted at errors.
 *
 * @param  p_pocs        the progression order changes.
 * @param  p_nb_pocs      the number of progression order changes.
 * @param  p_nb_resolutions  the number of resolutions.
 * @param  numcomps      the number of components
 * @param  numlayers      the number of layers.
 *
 * @return  true if the pocs are valid.
 */
static bool j2k_check_poc_val(
                const opj_poc_t *p_pocs,
                OPJ_UINT32 p_nb_pocs,
                OPJ_UINT32 p_nb_resolutions,
                OPJ_UINT32 numcomps,
                OPJ_UINT32 numlayers,
                opj_event_mgr_t * p_manager);

/**
 * Gets the number of tile parts used for the given change of progression (if any) and the given tile.
 *
 * @param    cp      the coding parameters.
 * @param    pino    the offset of the given poc (i.e. its position in the coding parameter).
 * @param    tileno    the given tile.
 *
 * @return    the number of tile parts.
 */
static OPJ_UINT32 j2k_get_num_tp(
              opj_cp_t *cp,
              OPJ_UINT32 pino,
              OPJ_UINT32 tileno);
/**
 * Calculates the total number of tile parts needed by the encoder to
 * encode such an image. If not enough memory is available, then the function return false.
 *
 * @param  p_nb_tiles  pointer that will hold the number of tile parts.
 * @param  cp      the coding parameters for the image.
 * @param  image    the image to encode.
 * @param  p_j2k      the p_j2k encoder.
 * @param  p_manager  the user event manager.
 *
 * @return true if the function was successful, false else.
 */
static bool j2k_calculate_tp(
            opj_j2k_t *p_j2k,
            opj_cp_t *cp,
            OPJ_UINT32 * p_nb_tiles,
            opj_image_t *image,
            opj_event_mgr_t * p_manager);

static bool j2k_write_first_tile_part (
                  opj_j2k_t *p_j2k,
                  OPJ_BYTE * p_data,
                  OPJ_UINT32 * p_data_written,
                  OPJ_UINT32 p_total_data_size,
                  opj_stream_private_t *p_stream,
                  struct opj_event_mgr * p_manager
                );
static bool j2k_write_all_tile_parts(
                  opj_j2k_t *p_j2k,
                  OPJ_BYTE * p_data,
                  OPJ_UINT32 * p_data_written,
                  OPJ_UINT32 p_total_data_size,
                  opj_stream_private_t *p_stream,
                  struct opj_event_mgr * p_manager
                );

/**
 * Reads the lookup table containing all the marker, status and action, and returns the handler associated
 * with the marker value.
 * @param  p_id    Marker value to look up
 *
 * @return  the handler associated with the id.
*/
static const struct opj_dec_memory_marker_handler * j2k_get_marker_handler (OPJ_UINT32 p_id);

/**
 * Destroys a tile coding parameter structure.
 *
 * @param  p_tcp    the tile coding parameter to destroy.
 */
static void j2k_tcp_destroy (opj_tcp_t *p_tcp);

static void j2k_get_tile_data (opj_tcd_t * p_tcd, OPJ_BYTE * p_data);

/**
 * Destroys a coding parameter structure.
 *
 * @param  p_cp    the coding parameter to destroy.
 */
static void j2k_cp_destroy (opj_cp_t *p_cp);

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static void j2k_setup_encoding_validation (opj_j2k_t *p_j2k);

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static void j2k_setup_decoding_validation (opj_j2k_t *p_j2k);

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
static void j2k_setup_end_compress (opj_j2k_t *p_j2k);

/**
 * Creates a tile-coder decoder.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_create_tcd(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * Excutes the given procedures on the given codec.
 *
 * @param  p_procedure_list  the list of procedures to execute
 * @param  p_j2k          the jpeg2000 codec to execute the procedures on.
 * @param  p_stream          the stream to execute the procedures on.
 * @param  p_manager      the user manager.
 *
 * @return  true        if all the procedures were successfully executed.
 */
static bool j2k_exec (
          opj_j2k_t * p_j2k,
          opj_procedure_list_t * p_procedure_list,
          opj_stream_private_t *p_stream,
          opj_event_mgr_t * p_manager
          );
/**
 * Updates the rates of the tcp.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_update_rates(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

/**
 * The default encoding validation procedure without any extension.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_encoding_validation (
                opj_j2k_t * p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
              );
/**
 * The read header procedure.
 */
bool j2k_read_header_procedure(
                  opj_j2k_t *p_j2k,
                struct opj_stream_private *p_stream,
                struct opj_event_mgr * p_manager);

/**
 * The default decoding validation procedure without any extension.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_decoding_validation (
                opj_j2k_t * p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
              );
/**
 * Reads the tiles.
 */
bool j2k_decode_tiles (
                opj_j2k_t *p_j2k,
                struct opj_stream_private *p_stream,
                struct opj_event_mgr * p_manager);

/**
 * The mct encoding validation procedure.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_mct_validation (
                opj_j2k_t * p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
              );
/**
 * Builds the tcd decoder to use to decode tile.
 */
bool j2k_build_decoder (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            );
/**
 * Builds the tcd encoder to use to encode tile.
 */
bool j2k_build_encoder (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            );
/**
 * Copies the decoding tile parameters onto all the tile parameters.
 * Creates also the tile decoder.
 */
bool j2k_copy_default_tcp_and_create_tcd(
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            );
/**
 * Destroys the memory associated with the decoding of headers.
 */
bool j2k_destroy_header_memory (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            );

/**
 * Sets up the procedures to do on writting header. Developpers wanting to extend the library can add their own writting procedures.
 */
void j2k_setup_header_writting (opj_j2k_t *p_j2k);

/**
 * Sets up the procedures to do on reading header. Developpers wanting to extend the library can add their own reading procedures.
 */
void j2k_setup_header_reading (opj_j2k_t *p_j2k);

/**
 * Writes a tile.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
static bool j2k_post_write_tile (
           opj_j2k_t * p_j2k,
           OPJ_BYTE * p_data,
           OPJ_UINT32 p_data_size,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          );

static bool j2k_pre_write_tile (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 p_tile_index,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          );
static bool j2k_update_image_data (opj_tcd_t * p_tcd, OPJ_BYTE * p_data);

static bool j2k_add_mct(opj_tcp_t * p_tcp,opj_image_t * p_image, OPJ_UINT32 p_index);
/**
 * Gets the offset of the header.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
static bool j2k_get_end_header(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          );

static void  j2k_read_int16_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_int32_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_float32_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_float64_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);

static void  j2k_read_int16_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_int32_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_float32_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_read_float64_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);

static void  j2k_write_float_to_int16 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_write_float_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_write_float_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);
static void  j2k_write_float_to_float64 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);



/*@}*/

/*@}*/

/* ----------------------------------------------------------------------- */



/****************************************************************************
 ********************* CONSTANTS ********************************************
 ****************************************************************************/




/**
 * List of progression orders.
 */
const j2k_prog_order_t j2k_prog_order_list [] =
{
  {CPRL, "CPRL"},
  {LRCP, "LRCP"},
  {PCRL, "PCRL"},
  {RLCP, "RLCP"},
  {RPCL, "RPCL"},
  {(OPJ_PROG_ORDER)-1, ""}
};

const OPJ_UINT32 MCT_ELEMENT_SIZE [] =
{
  2,
  4,
  4,
  8
};

typedef void (* j2k_mct_function) (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem);

const j2k_mct_function j2k_mct_read_functions_to_float [] =
{
  j2k_read_int16_to_float,
  j2k_read_int32_to_float,
  j2k_read_float32_to_float,
  j2k_read_float64_to_float
};

const j2k_mct_function j2k_mct_read_functions_to_int32 [] =
{
  j2k_read_int16_to_int32,
  j2k_read_int32_to_int32,
  j2k_read_float32_to_int32,
  j2k_read_float64_to_int32
};

const j2k_mct_function j2k_mct_write_functions_from_float [] =
{
  j2k_write_float_to_int16,
  j2k_write_float_to_int32,
  j2k_write_float_to_float,
  j2k_write_float_to_float64
};




/*const opj_dec_stream_marker_handler_t j2k_stream_marker_handler_tab[] =
{
  {J2K_MS_SOC, J2K_DEC_STATE_MHSOC, j2k_read_soc},
  {J2K_MS_SOD, J2K_DEC_STATE_TPH, j2k_read_sod},
  {J2K_MS_EOC, J2K_DEC_STATE_TPHSOT, j2k_read_eoc},
  {J2K_MS_SOP, 0, 0},
#ifdef USE_JPWL
  {J2K_MS_EPC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_epc},
  {J2K_MS_EPB, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_epb},
  {J2K_MS_ESD, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_esd},
  {J2K_MS_RED, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_red},
#endif
#ifdef USE_JPSEC
  {J2K_MS_SEC, J2K_DEC_STATE_MH, j2k_read_sec},
  {J2K_MS_INSEC, 0, j2k_read_insec},
#endif

  {0, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_unk}
};*/

const opj_dec_memory_marker_handler_t j2k_memory_marker_handler_tab [] =
{
  {J2K_MS_SOT, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPHSOT, j2k_read_sot},
  {J2K_MS_COD, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_cod},
  {J2K_MS_COC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_coc},
  {J2K_MS_RGN, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_rgn},
  {J2K_MS_QCD, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_qcd},
  {J2K_MS_QCC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_qcc},
  {J2K_MS_POC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_poc},
  {J2K_MS_SIZ, J2K_DEC_STATE_MHSIZ , j2k_read_siz},
  {J2K_MS_TLM, J2K_DEC_STATE_MH, j2k_read_tlm},
  {J2K_MS_PLM, J2K_DEC_STATE_MH, j2k_read_plm},
  {J2K_MS_PLT, J2K_DEC_STATE_TPH, j2k_read_plt},
  {J2K_MS_PPM, J2K_DEC_STATE_MH, j2k_read_ppm},
  {J2K_MS_PPT, J2K_DEC_STATE_TPH, j2k_read_ppt},
  {J2K_MS_SOP, 0, 0},
  {J2K_MS_CRG, J2K_DEC_STATE_MH, j2k_read_crg},
  {J2K_MS_COM, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_com},
  {J2K_MS_MCT, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mct},
  {J2K_MS_CBD, J2K_DEC_STATE_MH , j2k_read_cbd},
  {J2K_MS_MCC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mcc},
  {J2K_MS_MCO, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mco},
#ifdef USE_JPWL
  {J2K_MS_EPC, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_epc},
  {J2K_MS_EPB, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_epb},
  {J2K_MS_ESD, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_esd},
  {J2K_MS_RED, J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_red},
#endif /* USE_JPWL */
#ifdef USE_JPSEC
  {J2K_MS_SEC, J2K_DEC_STATE_MH, j2k_read_sec},
  {J2K_MS_INSEC, 0, j2k_read_insec}
#endif /* USE_JPSEC */
};

void  j2k_read_int16_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_bytes(l_src_data,&l_temp,2);
    l_src_data+=sizeof(OPJ_INT16);
    *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
  }
}

void  j2k_read_int32_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_bytes(l_src_data,&l_temp,4);
    l_src_data+=sizeof(OPJ_INT32);
    *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
  }
}
void  j2k_read_float32_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_FLOAT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_float(l_src_data,&l_temp);
    l_src_data+=sizeof(OPJ_FLOAT32);
    *(l_dest_data++) = l_temp;
  }
}

void  j2k_read_float64_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_FLOAT32 * l_dest_data = (OPJ_FLOAT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_FLOAT64 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_double(l_src_data,&l_temp);
    l_src_data+=sizeof(OPJ_FLOAT64);
    *(l_dest_data++) = (OPJ_FLOAT32) l_temp;
  }

}

void  j2k_read_int16_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_bytes(l_src_data,&l_temp,2);
    l_src_data+=sizeof(OPJ_INT16);
    *(l_dest_data++) = (OPJ_INT32) l_temp;
  }
}

void  j2k_read_int32_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_bytes(l_src_data,&l_temp,4);
    l_src_data+=sizeof(OPJ_INT32);
    *(l_dest_data++) = (OPJ_INT32) l_temp;
  }
}
void  j2k_read_float32_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_FLOAT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_float(l_src_data,&l_temp);
    l_src_data+=sizeof(OPJ_FLOAT32);
    *(l_dest_data++) = (OPJ_INT32) l_temp;
  }
}

void  j2k_read_float64_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_src_data = (OPJ_BYTE *) p_src_data;
  OPJ_INT32 * l_dest_data = (OPJ_INT32 *) p_dest_data;
  OPJ_UINT32 i;
  OPJ_FLOAT64 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    opj_read_double(l_src_data,&l_temp);
    l_src_data+=sizeof(OPJ_FLOAT64);
    *(l_dest_data++) = (OPJ_INT32) l_temp;
  }

}

void  j2k_write_float_to_int16 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
  OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    l_temp = (OPJ_UINT32) *(l_src_data++);
    opj_write_bytes(l_dest_data,l_temp,sizeof(OPJ_INT16));
    l_dest_data+=sizeof(OPJ_INT16);
  }
}

void  j2k_write_float_to_int32 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
  OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
  OPJ_UINT32 i;
  OPJ_UINT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    l_temp = (OPJ_UINT32) *(l_src_data++);
    opj_write_bytes(l_dest_data,l_temp,sizeof(OPJ_INT32));
    l_dest_data+=sizeof(OPJ_INT32);
  }
}

void  j2k_write_float_to_float (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
  OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
  OPJ_UINT32 i;
  OPJ_FLOAT32 l_temp;

  for
    (i=0;i<p_nb_elem;++i)
  {
    l_temp = (OPJ_FLOAT32) *(l_src_data++);
    opj_write_float(l_dest_data,l_temp);
    l_dest_data+=sizeof(OPJ_FLOAT32);
  }

}

void  j2k_write_float_to_float64 (const void * p_src_data, void * p_dest_data, OPJ_UINT32 p_nb_elem)
{
  OPJ_BYTE * l_dest_data = (OPJ_BYTE *) p_dest_data;
  OPJ_FLOAT32 * l_src_data = (OPJ_FLOAT32 *) p_src_data;
  OPJ_UINT32 i;
  OPJ_FLOAT64 l_temp;
  for
    (i=0;i<p_nb_elem;++i)
  {
    l_temp = (OPJ_FLOAT64) *(l_src_data++);
    opj_write_double(l_dest_data,l_temp);
    l_dest_data+=sizeof(OPJ_FLOAT64);
  }
}




/**
 * Converts an enum type progression order to string type.
 *
 * @param prg_order    the progression order to get.
 *
 * @return  the string representation of the gicen progression order.
 */
const OPJ_CHAR * j2k_convert_progression_order(OPJ_PROG_ORDER p_prg_order)
{
  const j2k_prog_order_t *po;
  for
    (po = j2k_prog_order_list; po->enum_prog != -1; ++po )
  {
    if
      (po->enum_prog == p_prg_order)
    {
      return po->str_prog;
    }
  }
  return po->str_prog;
}







/**
 * Checks the progression order changes values. Tells if the poc given as input are valid.
 *
 * @param  p_pocs        the progression order changes.
 * @param  p_nb_pocs      the number of progression order changes.
 * @param  p_nb_resolutions  the number of resolutions.
 * @param  numcomps      the number of components
 * @param  numlayers      the number of layers.
 * @param  p_manager      the user event manager.
 *
 * @return  true if the pocs are valid.
 */
bool j2k_check_poc_val(const opj_poc_t *p_pocs, OPJ_UINT32 p_nb_pocs, OPJ_UINT32 p_nb_resolutions, OPJ_UINT32 p_num_comps, OPJ_UINT32 p_num_layers, opj_event_mgr_t * p_manager)
{
  OPJ_UINT32* packet_array;
  OPJ_UINT32 index , resno, compno, layno;
  OPJ_UINT32 i;
  OPJ_UINT32 step_c = 1;
  OPJ_UINT32 step_r = p_num_comps * step_c;
  OPJ_UINT32 step_l = p_nb_resolutions * step_r;
  bool loss = false;
  OPJ_UINT32 layno0 = 0;

  packet_array = (OPJ_UINT32*) opj_calloc(step_l * p_num_layers, sizeof(OPJ_UINT32));
  if
    (packet_array == 00)
  {
    opj_event_msg(p_manager , EVT_ERROR, "Not enough memory for checking the poc values.\n");
    return false;
  }
  memset(packet_array,0,step_l * p_num_layers* sizeof(OPJ_UINT32));
  if
    (p_nb_pocs == 0)
  {
    return true;
  }

  index = step_r * p_pocs->resno0;
  // take each resolution for each poc
  for
    (resno = p_pocs->resno0 ; resno < p_pocs->resno1 ; ++resno)
  {
    OPJ_UINT32 res_index = index + p_pocs->compno0 * step_c;
    // take each comp of each resolution for each poc
    for
      (compno = p_pocs->compno0 ; compno < p_pocs->compno1 ; ++compno)
    {
      OPJ_UINT32 comp_index = res_index + layno0 * step_l;
      // and finally take each layer of each res of ...
      for
        (layno = layno0; layno < p_pocs->layno1 ; ++layno)
      {
        //index = step_r * resno + step_c * compno + step_l * layno;
        packet_array[comp_index] = 1;
        comp_index += step_l;
      }
      res_index += step_c;
    }
    index += step_r;
  }
  ++p_pocs;
  // iterate through all the pocs
  for
    (i = 1; i < p_nb_pocs ; ++i)
  {
    OPJ_UINT32 l_last_layno1 = (p_pocs-1)->layno1 ;
    layno0 = (p_pocs->layno1 > l_last_layno1)? l_last_layno1 : 0;
    index = step_r * p_pocs->resno0;
    // take each resolution for each poc
    for
      (resno = p_pocs->resno0 ; resno < p_pocs->resno1 ; ++resno)
    {
      OPJ_UINT32 res_index = index + p_pocs->compno0 * step_c;
      // take each comp of each resolution for each poc
      for
        (compno = p_pocs->compno0 ; compno < p_pocs->compno1 ; ++compno)
      {
        OPJ_UINT32 comp_index = res_index + layno0 * step_l;
        // and finally take each layer of each res of ...
        for
          (layno = layno0; layno < p_pocs->layno1 ; ++layno)
        {
          //index = step_r * resno + step_c * compno + step_l * layno;
          packet_array[comp_index] = 1;
          comp_index += step_l;
        }
        res_index += step_c;
      }
      index += step_r;
    }
    ++p_pocs;
  }

  index = 0;
  for
    (layno = 0; layno < p_num_layers ; ++layno)
  {
    for
      (resno = 0; resno < p_nb_resolutions; ++resno)
    {
      for
        (compno = 0; compno < p_num_comps; ++compno)
      {
        loss |= (packet_array[index]!=1);
        //index = step_r * resno + step_c * compno + step_l * layno;
        index += step_c;
      }
    }
  }
  if
    (loss)
  {
    opj_event_msg(p_manager , EVT_ERROR, "Missing packets possible loss of data\n");
  }
  opj_free(packet_array);
  return !loss;
}


/* ----------------------------------------------------------------------- */

/**
 * Gets the number of tile parts used for the given change of progression (if any) and the given tile.
 *
 * @param    cp      the coding parameters.
 * @param    pino    the offset of the given poc (i.e. its position in the coding parameter).
 * @param    tileno    the given tile.
 *
 * @return    the number of tile parts.
 */
OPJ_UINT32 j2k_get_num_tp(opj_cp_t *cp,OPJ_UINT32 pino,OPJ_UINT32 tileno)
{
  const OPJ_CHAR *prog = 00;
  OPJ_UINT32 i;
  OPJ_UINT32 tpnum = 1;
  opj_tcp_t *tcp = 00;
  opj_poc_t * l_current_poc = 00;

  // preconditions only in debug
  assert(tileno < (cp->tw * cp->th));
  assert(pino < (cp->tcps[tileno].numpocs + 1));

  // get the given tile coding parameter
  tcp = &cp->tcps[tileno];
  assert(tcp != 00);
  l_current_poc = &(tcp->pocs[pino]);
  assert(l_current_poc != 0);

  // get the progression order as a character string
  prog = j2k_convert_progression_order(tcp->prg);
  assert(strlen(prog) > 0);

  if
    (cp->m_specific_param.m_enc.m_tp_on == 1)
  {
    for
      (i=0;i<4;++i)
    {
      switch
        (prog[i])
      {
        // component wise
        case 'C':
          tpnum *= l_current_poc->compE;
          break;
        // resolution wise
        case 'R':
          tpnum *= l_current_poc->resE;
          break;
        // precinct wise
        case 'P':
          tpnum *= l_current_poc->prcE;
          break;
        // layer wise
        case 'L':
          tpnum *= l_current_poc->layE;
          break;
      }
      // whould we split here ?
      if
        ( cp->m_specific_param.m_enc.m_tp_flag == prog[i] )
      {
        cp->m_specific_param.m_enc.m_tp_pos=i;
        break;
      }
    }
  }
  else
  {
    tpnum=1;
  }
  return tpnum;
}

/**
 * Calculates the total number of tile parts needed by the encoder to
 * encode such an image. If not enough memory is available, then the function return false.
 *
 * @param  p_nb_tiles  pointer that will hold the number of tile parts.
 * @param  cp      the coding parameters for the image.
 * @param  image    the image to encode.
 * @param  p_j2k      the p_j2k encoder.
 * @param  p_manager  the user event manager.
 *
 * @return true if the function was successful, false else.
 */
bool j2k_calculate_tp(
            opj_j2k_t *p_j2k,
            opj_cp_t *cp,
            OPJ_UINT32 * p_nb_tiles,
            opj_image_t *image,
            opj_event_mgr_t * p_manager)
{
  OPJ_UINT32 pino,tileno;
  OPJ_UINT32 l_nb_tiles;
  opj_tcp_t *tcp;

  // preconditions
  assert(p_nb_tiles != 00);
  assert(cp != 00);
  assert(image != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_nb_tiles = cp->tw * cp->th;
  * p_nb_tiles = 0;
  tcp = cp->tcps;

  /* INDEX >> */
  if
    (p_j2k->cstr_info)
  {
    opj_tile_info_t * l_info_tile_ptr = p_j2k->cstr_info->tile;
    for
      (tileno = 0; tileno < l_nb_tiles; ++tileno)
    {
      OPJ_UINT32 cur_totnum_tp = 0;
      pi_update_encoding_parameters(image,cp,tileno);
      for
        (pino = 0; pino <= tcp->numpocs; ++pino)
      {
        OPJ_UINT32 tp_num = j2k_get_num_tp(cp,pino,tileno);
        *p_nb_tiles = *p_nb_tiles + tp_num;
        cur_totnum_tp += tp_num;
      }
      tcp->m_nb_tile_parts = cur_totnum_tp;
      l_info_tile_ptr->tp = (opj_tp_info_t *) opj_malloc(cur_totnum_tp * sizeof(opj_tp_info_t));
      if
        (l_info_tile_ptr->tp == 00)
      {
        return false;
      }
      memset(l_info_tile_ptr->tp,0,cur_totnum_tp * sizeof(opj_tp_info_t));
      l_info_tile_ptr->num_tps = cur_totnum_tp;
      ++l_info_tile_ptr;
      ++tcp;
    }
  }
  else
  {
    for
      (tileno = 0; tileno < l_nb_tiles; ++tileno)
    {
      OPJ_UINT32 cur_totnum_tp = 0;
      pi_update_encoding_parameters(image,cp,tileno);
      for
        (pino = 0; pino <= tcp->numpocs; ++pino)
      {
        OPJ_UINT32 tp_num=0;
        tp_num = j2k_get_num_tp(cp,pino,tileno);
        *p_nb_tiles = *p_nb_tiles + tp_num;
        cur_totnum_tp += tp_num;
      }
      tcp->m_nb_tile_parts = cur_totnum_tp;
      ++tcp;
    }
  }
  return true;
}

/**
 * Writes the SOC marker (Start Of Codestream)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/

bool j2k_write_soc(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
                )
{
  /* 2 bytes will be written */
  OPJ_BYTE * l_start_stream = 00;

  // preconditions
  assert(p_stream != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_start_stream = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  /* write SOC identifier */
  opj_write_bytes(l_start_stream,J2K_MS_SOC,2);
  if
    (opj_stream_write_data(p_stream,l_start_stream,2,p_manager) != 2)
  {
    return false;
  }
/* UniPG>> */
#ifdef USE_JPWL
  /* update markers struct */
  j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOC, p_stream_tell(p_stream) - 2, 2);
#endif /* USE_JPWL */
  return true;
/* <<UniPG */
}

/**
 * Reads a SOC marker (Start of Codestream)
 * @param  p_header_data  the data contained in the SOC box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the SOC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_soc(
          opj_j2k_t *p_j2k,
          struct opj_stream_private *p_stream,
          struct opj_event_mgr * p_manager
         )

{
  OPJ_BYTE l_data [2];
  OPJ_UINT32 l_marker;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);
  if
    (opj_stream_read_data(p_stream,l_data,2,p_manager) != 2)
  {
    return false;
  }
  opj_read_bytes(l_data,&l_marker,2);
  if
    (l_marker != J2K_MS_SOC)
  {
    return false;
  }
  /* assure length of data is correct (0) */
  p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_MHSIZ;
  /* Index */
  if
    (p_j2k->cstr_info)
  {
    //TODO p_j2k->cstr_info->main_head_start = opj_stream_tell(p_stream) - 2; // why - 2 ?
    p_j2k->cstr_info->codestream_size = 0;/*p_stream_numbytesleft(p_j2k->p_stream) + 2 - p_j2k->cstr_info->main_head_start*/;
  }
  return true;
}

/**
 * Writes the SIZ marker (image and tile size)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
bool j2k_write_siz(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_size_len;
  OPJ_BYTE * l_current_ptr;
  opj_image_t * l_image = 00;
  opj_cp_t *cp = 00;
  opj_image_comp_t * l_img_comp = 00;

  // preconditions
  assert(p_stream != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_image = p_j2k->m_image;
  cp = &(p_j2k->m_cp);
  l_size_len = 40 + 3 * l_image->numcomps;
  l_img_comp = l_image->comps;

  if
    (l_size_len > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_size_len);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_size_len;
  }

  l_current_ptr = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  /* write SOC identifier */
  opj_write_bytes(l_current_ptr,J2K_MS_SIZ,2);  /* SIZ */
  l_current_ptr+=2;
  opj_write_bytes(l_current_ptr,l_size_len-2,2); /* L_SIZ */
  l_current_ptr+=2;
  opj_write_bytes(l_current_ptr, cp->rsiz, 2);  /* Rsiz (capabilities) */
  l_current_ptr+=2;
  opj_write_bytes(l_current_ptr, l_image->x1, 4);  /* Xsiz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, l_image->y1, 4);  /* Ysiz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, l_image->x0, 4);  /* X0siz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, l_image->y0, 4);  /* Y0siz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, cp->tdx, 4);    /* XTsiz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, cp->tdy, 4);    /* YTsiz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, cp->tx0, 4);    /* XT0siz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, cp->ty0, 4);    /* YT0siz */
  l_current_ptr+=4;
  opj_write_bytes(l_current_ptr, l_image->numcomps, 2);  /* Csiz */
  l_current_ptr+=2;
  for
    (i = 0; i < l_image->numcomps; ++i)
  {
    // TODO here with MCT ?
    opj_write_bytes(l_current_ptr, l_img_comp->prec - 1 + (l_img_comp->sgnd << 7), 1);  /* Ssiz_i */
    ++l_current_ptr;
    opj_write_bytes(l_current_ptr, l_img_comp->dx, 1);  /* XRsiz_i */
    ++l_current_ptr;
    opj_write_bytes(l_current_ptr, l_img_comp->dy, 1);  /* YRsiz_i */
    ++l_current_ptr;
    ++l_img_comp;
  }
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_size_len,p_manager) != l_size_len)
  {
    return false;
  }
  return true;
}

/**
 * Reads a SIZ marker (image and tile size)
 * @param  p_header_data  the data contained in the SIZ box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the SIZ marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_siz (
            opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_size, i;
  OPJ_UINT32 l_nb_comp;
  OPJ_UINT32 l_nb_comp_remain;
  OPJ_UINT32 l_remaining_size;
  OPJ_UINT32 l_nb_tiles;
  OPJ_UINT32 l_tmp;
  opj_image_t *l_image = 00;
  opj_cp_t *l_cp = 00;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcp_t * l_current_tile_param = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_header_data != 00);

  l_image = p_j2k->m_image;
  l_cp = &(p_j2k->m_cp);
  if
    (p_header_size < 36)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with SIZ marker size\n");
    return false;
  }
  l_remaining_size = p_header_size - 36;

  l_nb_comp = l_remaining_size / 3;
  l_nb_comp_remain = l_remaining_size % 3;
  if
    (l_nb_comp_remain != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with SIZ marker size\n");
    return false;
  }
  l_size = p_header_size + 2;      /* Lsiz */

  opj_read_bytes(p_header_data,&l_tmp ,2);      /* Rsiz (capabilities) */
  p_header_data+=2;
  l_cp->rsiz = (OPJ_RSIZ_CAPABILITIES) l_tmp;
  opj_read_bytes(p_header_data,(OPJ_UINT32 *) (&l_image->x1) ,4);      /* Xsiz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(OPJ_UINT32*) (&l_image->y1),4);        /* Ysiz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(OPJ_UINT32*) &l_image->x0,4);        /* X0siz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(OPJ_UINT32*) &l_image->y0,4);        /* Y0siz */
  p_header_data+=4;
  opj_read_bytes(p_header_data, (&l_cp->tdx),4);        /* XTsiz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,&l_cp->tdy,4);        /* YTsiz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(OPJ_UINT32 *) (&l_cp->tx0),4);        /* XT0siz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(OPJ_UINT32 *) (&l_cp->ty0),4);        /* YT0siz */
  p_header_data+=4;
  opj_read_bytes(p_header_data,(&l_image->numcomps),2);        /* Csiz */
  p_header_data+=2;
  if
    (l_image->numcomps != l_nb_comp)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error with SIZ marker size\n");
    return false;
  }

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {
    /* if JPWL is on, we check whether TX errors have damaged
      too much the SIZ parameters */
    if (!(image->x1 * image->y1)) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "JPWL: bad image size (%d x %d)\n",
        image->x1, image->y1);
      if (!JPWL_ASSUME || JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
    }
    if (image->numcomps != ((len - 38) / 3)) {
      opj_event_msg(p_j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
        "JPWL: Csiz is %d => space in SIZ only for %d comps.!!!\n",
        image->numcomps, ((len - 38) / 3));
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust this\n");
      if (image->numcomps < ((len - 38) / 3)) {
        len = 38 + 3 * image->numcomps;
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting Lsiz to %d => HYPOTHESIS!!!\n",
          len);
      } else {
        image->numcomps = ((len - 38) / 3);
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting Csiz to %d => HYPOTHESIS!!!\n",
          image->numcomps);
      }
    }

    /* update components number in the jpwl_exp_comps filed */
    cp->exp_comps = image->numcomps;
  }
#endif /* USE_JPWL */

  l_image->comps = (opj_image_comp_t*) opj_calloc(l_image->numcomps, sizeof(opj_image_comp_t));
  if
    (l_image->comps == 00)
  {
    l_image->numcomps = 0;
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
    return false;
  }
  memset(l_image->comps,0,l_image->numcomps * sizeof(opj_image_comp_t));
  l_img_comp = l_image->comps;
  for
    (i = 0; i < l_image->numcomps; ++i)
  {
    OPJ_UINT32 tmp;
    opj_read_bytes(p_header_data,&tmp,1);        /* Ssiz_i */
    ++p_header_data;
    l_img_comp->prec = (tmp & 0x7f) + 1;
    l_img_comp->sgnd = tmp >> 7;
    opj_read_bytes(p_header_data,&l_img_comp->dx,1);        /* XRsiz_i */
    ++p_header_data;
    opj_read_bytes(p_header_data,&l_img_comp->dy,1);        /* YRsiz_i */
    ++p_header_data;
#ifdef USE_JPWL
    if (p_j2k->m_cp->correct) {
    /* if JPWL is on, we check whether TX errors have damaged
      too much the SIZ parameters, again */
      if (!(image->comps[i].dx * image->comps[i].dy)) {
        opj_event_msg(p_j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
          "JPWL: bad XRsiz_%d/YRsiz_%d (%d x %d)\n",
          i, i, image->comps[i].dx, image->comps[i].dy);
        if (!JPWL_ASSUME) {
          opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
          return;
        }
        /* we try to correct */
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust them\n");
        if (!image->comps[i].dx) {
          image->comps[i].dx = 1;
          opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting XRsiz_%d to %d => HYPOTHESIS!!!\n",
            i, image->comps[i].dx);
        }
        if (!image->comps[i].dy) {
          image->comps[i].dy = 1;
          opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting YRsiz_%d to %d => HYPOTHESIS!!!\n",
            i, image->comps[i].dy);
        }
      }

    }
#endif /* USE_JPWL */
    l_img_comp->resno_decoded = 0;  /* number of resolution decoded */
    l_img_comp->factor = l_cp->m_specific_param.m_dec.m_reduce; /* reducing factor per component */
    ++l_img_comp;
  }

  l_cp->tw = int_ceildiv(l_image->x1 - l_cp->tx0, l_cp->tdx);
  l_cp->th = int_ceildiv(l_image->y1 - l_cp->ty0, l_cp->tdy);
  l_nb_tiles = l_cp->tw * l_cp->th;
  if
    (p_j2k->m_specific_param.m_decoder.m_discard_tiles)
  {
    p_j2k->m_specific_param.m_decoder.m_start_tile_x = (p_j2k->m_specific_param.m_decoder.m_start_tile_x - l_cp->tx0) / l_cp->tdx;
    p_j2k->m_specific_param.m_decoder.m_start_tile_y = (p_j2k->m_specific_param.m_decoder.m_start_tile_y - l_cp->ty0) / l_cp->tdy;
    p_j2k->m_specific_param.m_decoder.m_end_tile_x = int_ceildiv((p_j2k->m_specific_param.m_decoder.m_end_tile_x - l_cp->tx0), l_cp->tdx);
    p_j2k->m_specific_param.m_decoder.m_end_tile_y = int_ceildiv((p_j2k->m_specific_param.m_decoder.m_end_tile_y - l_cp->ty0), l_cp->tdy);
  }
  else
  {
    p_j2k->m_specific_param.m_decoder.m_start_tile_x = 0;
    p_j2k->m_specific_param.m_decoder.m_start_tile_y = 0;
    p_j2k->m_specific_param.m_decoder.m_end_tile_x = l_cp->tw;
    p_j2k->m_specific_param.m_decoder.m_end_tile_y = l_cp->th;
  }

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {
    /* if JPWL is on, we check whether TX errors have damaged
      too much the SIZ parameters */
    if ((cp->tw < 1) || (cp->th < 1) || (cp->tw > cp->max_tiles) || (cp->th > cp->max_tiles)) {
      opj_event_msg(p_j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
        "JPWL: bad number of tiles (%d x %d)\n",
        cp->tw, cp->th);
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust them\n");
      if (cp->tw < 1) {
        cp->tw= 1;
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting %d tiles in x => HYPOTHESIS!!!\n",
          cp->tw);
      }
      if (cp->tw > cp->max_tiles) {
        cp->tw= 1;
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- too large x, increase expectance of %d\n"
          "- setting %d tiles in x => HYPOTHESIS!!!\n",
          cp->max_tiles, cp->tw);
      }
      if (cp->th < 1) {
        cp->th= 1;
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- setting %d tiles in y => HYPOTHESIS!!!\n",
          cp->th);
      }
      if (cp->th > cp->max_tiles) {
        cp->th= 1;
        opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- too large y, increase expectance of %d to continue\n",
          "- setting %d tiles in y => HYPOTHESIS!!!\n",
          cp->max_tiles, cp->th);
      }
    }
  }
#endif /* USE_JPWL */
  /* memory allocations */
  l_cp->tcps = (opj_tcp_t*) opj_calloc(l_nb_tiles, sizeof(opj_tcp_t));
  if
    (l_cp->tcps == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
    return false;
  }
  memset(l_cp->tcps,0,l_nb_tiles*sizeof(opj_tcp_t));

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {
    if (!cp->tcps) {
      opj_event_msg(p_j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
        "JPWL: could not alloc tcps field of cp\n");
      if (!JPWL_ASSUME || JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
    }
  }
#endif /* USE_JPWL */

  p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps = (opj_tccp_t*) opj_calloc(l_image->numcomps, sizeof(opj_tccp_t));
  if
    (p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps  == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
    return false;
  }
  memset(p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps ,0,l_image->numcomps*sizeof(opj_tccp_t));

  p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mct_records =
  (opj_mct_data_t*)opj_malloc(J2K_MCT_DEFAULT_NB_RECORDS * sizeof(opj_mct_data_t));
  if
    (! p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mct_records)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
    return false;
  }
  memset(p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mct_records,0,J2K_MCT_DEFAULT_NB_RECORDS * sizeof(opj_mct_data_t));
  p_j2k->m_specific_param.m_decoder.m_default_tcp->m_nb_max_mct_records = J2K_MCT_DEFAULT_NB_RECORDS;

  p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mcc_records =
  (opj_simple_mcc_decorrelation_data_t*)
  opj_malloc(J2K_MCC_DEFAULT_NB_RECORDS * sizeof(opj_simple_mcc_decorrelation_data_t));
  if
    (! p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mcc_records)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
    return false;
  }
  memset(p_j2k->m_specific_param.m_decoder.m_default_tcp->m_mcc_records,0,J2K_MCC_DEFAULT_NB_RECORDS * sizeof(opj_simple_mcc_decorrelation_data_t));
  p_j2k->m_specific_param.m_decoder.m_default_tcp->m_nb_max_mcc_records = J2K_MCC_DEFAULT_NB_RECORDS;

  /* set up default dc level shift */
  for
    (i=0;i<l_image->numcomps;++i)
  {
    if
      (! l_image->comps[i].sgnd)
    {
      p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps[i].m_dc_level_shift = 1 << (l_image->comps[i].prec - 1);
    }
  }

  l_current_tile_param = l_cp->tcps;
  for
    (i = 0; i < l_nb_tiles; ++i)
  {
    l_current_tile_param->tccps = (opj_tccp_t*) opj_malloc(l_image->numcomps * sizeof(opj_tccp_t));
    if
      (l_current_tile_param->tccps == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
      return false;
    }
    memset(l_current_tile_param->tccps,0,l_image->numcomps * sizeof(opj_tccp_t));

    ++l_current_tile_param;

  }
  p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_MH;
  opj_image_comp_update(l_image,l_cp);

  /* Index */
  if
    (p_j2k->cstr_info)
  {
    opj_codestream_info_t *cstr_info = p_j2k->cstr_info;
    cstr_info->image_w = l_image->x1 - l_image->x0;
    cstr_info->image_h = l_image->y1 - l_image->y0;
    cstr_info->numcomps = l_image->numcomps;
    cstr_info->tw = l_cp->tw;
    cstr_info->th = l_cp->th;
    cstr_info->tile_x = l_cp->tdx;
    cstr_info->tile_y = l_cp->tdy;
    cstr_info->tile_Ox = l_cp->tx0;
    cstr_info->tile_Oy = l_cp->ty0;
    cstr_info->tile = (opj_tile_info_t*) opj_calloc(l_nb_tiles, sizeof(opj_tile_info_t));
    if
      (cstr_info->tile == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to take in charge SIZ marker\n");
      return false;
    }
    memset(cstr_info->tile,0,l_nb_tiles * sizeof(opj_tile_info_t));
  }
  return true;
}

/**
 * Writes the COM marker (comment)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
bool j2k_write_com(
            opj_j2k_t *p_j2k,
          struct opj_stream_private *p_stream,
          struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_comment_size;
  OPJ_UINT32 l_total_com_size;
  const OPJ_CHAR *l_comment;
  OPJ_BYTE * l_current_ptr = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  l_comment = p_j2k->m_cp.comment;
  l_comment_size = strlen(l_comment);
  l_total_com_size = l_comment_size + 6;

  if
    (l_total_com_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_total_com_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_total_com_size;
  }
  l_current_ptr = p_j2k->m_specific_param.m_encoder.m_header_tile_data;
  opj_write_bytes(l_current_ptr,J2K_MS_COM , 2);  /* COM */
  l_current_ptr+=2;
  opj_write_bytes(l_current_ptr,l_total_com_size - 2 , 2);  /* L_COM */
  l_current_ptr+=2;
  opj_write_bytes(l_current_ptr,1 , 2);  /* General use (IS 8859-15:1999 (Latin) values) */
  l_current_ptr+=2,
  memcpy(  l_current_ptr,l_comment,l_comment_size);
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_total_com_size,p_manager) != l_total_com_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a COM marker (comments)
 * @param  p_header_data  the data contained in the COM box.
 * @param  jp2        the jpeg2000 file codec.
 * @param  p_header_size  the size of the data contained in the COM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_com (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_header_data != 00);
  return true;
}

/**
 * Gets the size taken by writting a SPCod or SPCoc for the given tile and component.
 *
 * @param  p_tile_no    the tile indix.
 * @param  p_comp_no    the component being outputted.
 * @param  p_j2k      the J2K codec.
 *
 * @return  the number of bytes taken by the SPCod element.
 */
OPJ_UINT32 j2k_get_SPCod_SPCoc_size (
            opj_j2k_t *p_j2k,
            OPJ_UINT32 p_tile_no,
            OPJ_UINT32 p_comp_no
            )
{
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;

  // preconditions
  assert(p_j2k != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_tile_no];
  l_tccp = &l_tcp->tccps[p_comp_no];

  // preconditions again
  assert(p_tile_no < (l_cp->tw * l_cp->th));
  assert(p_comp_no < p_j2k->m_image->numcomps);

  if
    (l_tccp->csty & J2K_CCP_CSTY_PRT)
  {
    return 5 + l_tccp->numresolutions;
  }
  else
  {
    return 5;
  }
}


/**
 * Writes a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 *
 * @param  p_comp_no  the component number to output.
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
 *
*/
bool j2k_write_SPCod_SPCoc(
                opj_j2k_t *p_j2k,
              OPJ_UINT32 p_tile_no,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_header_size != 00);
  assert(p_manager != 00);
  assert(p_data != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_tile_no];
  l_tccp = &l_tcp->tccps[p_comp_no];

  // preconditions again
  assert(p_tile_no < (l_cp->tw * l_cp->th));
  assert(p_comp_no <(p_j2k->m_image->numcomps));

  if
    (*p_header_size < 5)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error writting SPCod SPCoc element\n");
    return false;
  }

  opj_write_bytes(p_data,l_tccp->numresolutions - 1, 1);  /* SPcoc (D) */
  ++p_data;
  opj_write_bytes(p_data,l_tccp->cblkw - 2, 1);        /* SPcoc (E) */
  ++p_data;
  opj_write_bytes(p_data,l_tccp->cblkh - 2, 1);        /* SPcoc (F) */
  ++p_data;
  opj_write_bytes(p_data,l_tccp->cblksty, 1);        /* SPcoc (G) */
  ++p_data;
  opj_write_bytes(p_data,l_tccp->qmfbid, 1);        /* SPcoc (H) */
  ++p_data;

  *p_header_size = *p_header_size - 5;
  if
    (l_tccp->csty & J2K_CCP_CSTY_PRT)
  {
    if
      (*p_header_size < l_tccp->numresolutions)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error writting SPCod SPCoc element\n");
      return false;
    }
    for
      (i = 0; i < l_tccp->numresolutions; ++i)
    {
      opj_write_bytes(p_data,l_tccp->prcw[i] + (l_tccp->prch[i] << 4), 1);        /* SPcoc (I_i) */
      ++p_data;
    }
    *p_header_size = *p_header_size - l_tccp->numresolutions;

  }
  return true;
}


/**
 * Reads a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 * @param  p_header_data  the data contained in the COM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_SPCod_SPCoc(
                opj_j2k_t *p_j2k,
              OPJ_UINT32 compno,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
              )
{
  // loop
  OPJ_UINT32 i;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;
  OPJ_BYTE * l_current_ptr = 00;
  OPJ_UINT32 l_tmp;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_header_data != 00);


  l_cp = &(p_j2k->m_cp);
  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;

  // precondition again
  assert(compno < p_j2k->m_image->numcomps);
  l_tccp = &l_tcp->tccps[compno];
  l_current_ptr = p_header_data;


  // make sure room is sufficient
  if
    (* p_header_size < 5)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n");
    return false;
  }
  opj_read_bytes(l_current_ptr, &l_tccp->numresolutions ,1);    /* SPcox (D) */
  ++l_tccp->numresolutions;                    /* tccp->numresolutions = read() + 1 */
  ++l_current_ptr;

  // If user wants to remove more resolutions than the codestream contains, return error
  if
    (l_cp->m_specific_param.m_dec.m_reduce >= l_tccp->numresolutions)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error decoding component %d.\nThe number of resolutions to remove is higher than the number "
          "of resolutions of this component\nModify the cp_reduce parameter.\n\n", compno);
    p_j2k->m_specific_param.m_decoder.m_state |= J2K_DEC_STATE_ERR;
    return false;
  }

  opj_read_bytes(l_current_ptr,&l_tccp->cblkw ,1);    /* SPcoc (E) */
  ++l_current_ptr;
  l_tccp->cblkw += 2;

  opj_read_bytes(l_current_ptr,&l_tccp->cblkh ,1);    /* SPcoc (F) */
  ++l_current_ptr;
  l_tccp->cblkh += 2;

  opj_read_bytes(l_current_ptr,&l_tccp->cblksty ,1);    /* SPcoc (G) */
  ++l_current_ptr;

  opj_read_bytes(l_current_ptr,&l_tccp->qmfbid ,1);    /* SPcoc (H) */
  ++l_current_ptr;

  * p_header_size = * p_header_size - 5;

  // use custom precinct size ?
  if
    (l_tccp->csty & J2K_CCP_CSTY_PRT)
  {
    if
      (* p_header_size < l_tccp->numresolutions)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n");
      return false;
    }
    for
      (i = 0; i < l_tccp->numresolutions; ++i)
    {
      opj_read_bytes(l_current_ptr,&l_tmp ,1);    /* SPcoc (I_i) */
      ++l_current_ptr;
      l_tccp->prcw[i] = l_tmp & 0xf;
      l_tccp->prch[i] = l_tmp >> 4;
    }
    * p_header_size = * p_header_size - l_tccp->numresolutions;
  }
  else
  {
    /* set default size for the precinct width and height */
    for
      (i = 0; i < l_tccp->numresolutions; ++i)
    {
      l_tccp->prcw[i] = 15;
      l_tccp->prch[i] = 15;
    }
  }

  /* INDEX >> */
  if
    (p_j2k->cstr_info && compno == 0)
  {
    OPJ_UINT32 l_data_size = l_tccp->numresolutions * sizeof(OPJ_UINT32);
    memcpy(p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].pdx,l_tccp->prcw, l_data_size);
    memcpy(p_j2k->cstr_info->tile[p_j2k->m_current_tile_number].pdy,l_tccp->prch, l_data_size);
  }
  /* << INDEX */
  return true;
}

/**
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param    p_j2k    the J2k codec.
 */
void j2k_copy_tile_component_parameters(
              opj_j2k_t *p_j2k
              )
{
  // loop
  OPJ_UINT32 i;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_ref_tccp = 00;
  opj_tccp_t *l_copied_tccp = 00;
  OPJ_UINT32 l_prc_size;
  // preconditions
  assert(p_j2k != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;

  l_ref_tccp = &l_tcp->tccps[0];
  l_copied_tccp = l_ref_tccp + 1;
  l_prc_size = l_ref_tccp->numresolutions * sizeof(OPJ_UINT32);

  for
    (i=1;i<p_j2k->m_image->numcomps;++i)
  {
    l_copied_tccp->numresolutions = l_ref_tccp->numresolutions;
    l_copied_tccp->cblkw = l_ref_tccp->cblkw;
    l_copied_tccp->cblkh = l_ref_tccp->cblkh;
    l_copied_tccp->cblksty = l_ref_tccp->cblksty;
    l_copied_tccp->qmfbid = l_ref_tccp->qmfbid;
    memcpy(l_copied_tccp->prcw,l_ref_tccp->prcw,l_prc_size);
    memcpy(l_copied_tccp->prch,l_ref_tccp->prch,l_prc_size);
    ++l_copied_tccp;
  }
}



/**
 * Writes the COD marker (Coding style default)
 *
 * @param  p_stream      the stream to write data to.
 * @param  p_j2k      J2K codec.
 * @param  p_manager  the user event manager.
*/
bool j2k_write_cod(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
            )
{
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_code_size,l_remaining_size;
  OPJ_BYTE * l_current_data = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
  l_code_size = 9 + j2k_get_SPCod_SPCoc_size(p_j2k,p_j2k->m_current_tile_number,0);
  l_remaining_size = l_code_size;

  if
    (l_code_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_code_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_code_size;
  }

  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  opj_write_bytes(l_current_data,J2K_MS_COD,2);    /* COD */
  l_current_data += 2;

  opj_write_bytes(l_current_data,l_code_size-2,2);    /* L_COD */
  l_current_data += 2;

  opj_write_bytes(l_current_data,l_tcp->csty,1);    /* Scod */
  ++l_current_data;

  opj_write_bytes(l_current_data,l_tcp->prg,1);    /* SGcod (A) */
  ++l_current_data;

  opj_write_bytes(l_current_data,l_tcp->numlayers,2);    /* SGcod (B) */
  l_current_data+=2;

  opj_write_bytes(l_current_data,l_tcp->mct,1);    /* SGcod (C) */
  ++l_current_data;

  l_remaining_size -= 9;

  if
    (! j2k_write_SPCod_SPCoc(p_j2k,p_j2k->m_current_tile_number,0,l_current_data,&l_remaining_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error writting COD marker\n");
    return false;
  }
  if
    (l_remaining_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error writting COD marker\n");
    return false;
  }

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_code_size,p_manager) != l_code_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a COD marker (Coding Styke defaults)
 * @param  p_header_data  the data contained in the COD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COD marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_cod (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          )
{
  // loop
  OPJ_UINT32 i;
  OPJ_UINT32 l_tmp;
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_image_t *l_image = 00;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH) ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  l_image = p_j2k->m_image;

  // make sure room is sufficient
  if
    (p_header_size < 5)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
    return false;
  }

  opj_read_bytes(p_header_data,&l_tcp->csty,1);      /* Scod */
  ++p_header_data;
  opj_read_bytes(p_header_data,&l_tmp,1);      /* SGcod (A) */
  ++p_header_data;
  l_tcp->prg = (OPJ_PROG_ORDER) l_tmp;
  opj_read_bytes(p_header_data,&l_tcp->numlayers,2);  /* SGcod (B) */
  p_header_data+=2;
  if
    (l_cp->m_specific_param.m_dec.m_layer)
  {
    l_tcp->num_layers_to_decode = l_cp->m_specific_param.m_dec.m_layer;
  }
  else
  {
    l_tcp->num_layers_to_decode = l_tcp->numlayers;
  }

  opj_read_bytes(p_header_data,&l_tcp->mct,1);      /* SGcod (C) */
  ++p_header_data;

  p_header_size -= 5;
  for
    (i = 0; i < l_image->numcomps; ++i)
  {
    l_tcp->tccps[i].csty = l_tcp->csty & J2K_CCP_CSTY_PRT;
  }

  if
    (! j2k_read_SPCod_SPCoc(p_j2k,0,p_header_data,&p_header_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
    return false;
  }
  if
    (p_header_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COD marker\n");
    return false;
  }
  j2k_copy_tile_component_parameters(p_j2k);


  /* Index */
  if
    (p_j2k->cstr_info)
  {
    opj_codestream_info_t *l_cstr_info = p_j2k->cstr_info;
    l_cstr_info->prog = l_tcp->prg;
    l_cstr_info->numlayers = l_tcp->numlayers;
    l_cstr_info->numdecompos = (OPJ_INT32*) opj_malloc(l_image->numcomps * sizeof(OPJ_UINT32));
    for
      (i = 0; i < l_image->numcomps; ++i)
    {
      l_cstr_info->numdecompos[i] = l_tcp->tccps[i].numresolutions - 1;
    }
  }
  return true;
}

/**
 * Writes the COC marker (Coding style component)
 *
 * @param  p_comp_no    the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_coc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
            )
{
  OPJ_UINT32 l_coc_size,l_remaining_size;
  OPJ_UINT32 l_comp_room;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_comp_room = (p_j2k->m_image->numcomps <= 256) ? 1 : 2;

  l_coc_size = 5 + l_comp_room + j2k_get_SPCod_SPCoc_size(p_j2k,p_j2k->m_current_tile_number,p_comp_no);
  if
    (l_coc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_coc_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_coc_size;
  }

  j2k_write_coc_in_memory(p_j2k,p_comp_no,p_j2k->m_specific_param.m_encoder.m_header_tile_data,&l_remaining_size,p_manager);

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_coc_size,p_manager) != l_coc_size)
  {
    return false;
  }
  return true;
}

/**
 * Gets the maximum size taken by a coc.
 *
 * @param  p_j2k  the jpeg2000 codec to use.
 */
OPJ_UINT32 j2k_get_max_coc_size(opj_j2k_t *p_j2k)
{
  OPJ_UINT32 i,j;
  OPJ_UINT32 l_nb_comp;
  OPJ_UINT32 l_nb_tiles;
  OPJ_UINT32 l_max = 0;

  // preconditions

  l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th ;
  l_nb_comp = p_j2k->m_image->numcomps;

  for
    (i=0;i<l_nb_tiles;++i)
  {
    for
      (j=0;j<l_nb_comp;++j)
    {
      l_max = uint_max(l_max,j2k_get_SPCod_SPCoc_size(p_j2k,i,j));
    }
  }
  return 6 + l_max;
}

/**
 * Gets the maximum size taken by the toc headers of all the tile parts of any given tile.
 */
OPJ_UINT32 j2k_get_max_toc_size (opj_j2k_t *p_j2k)
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_nb_tiles;
  OPJ_UINT32 l_max = 0;
  opj_tcp_t * l_tcp = 00;
  // preconditions

  l_tcp = p_j2k->m_cp.tcps;
  l_nb_tiles = p_j2k->m_cp.tw * p_j2k->m_cp.th ;

  for
    (i=0;i<l_nb_tiles;++i)
  {
    l_max = uint_max(l_max,l_tcp->m_nb_tile_parts);
    ++l_tcp;
  }
  return 12 * l_max;
}


/**
 * Gets the maximum size taken by the headers of the SOT.
 *
 * @param  p_j2k  the jpeg2000 codec to use.
 */
OPJ_UINT32 j2k_get_specific_header_sizes(opj_j2k_t *p_j2k)
{
  OPJ_UINT32 l_nb_bytes = 0;
  OPJ_UINT32 l_nb_comps;
  OPJ_UINT32 l_coc_bytes,l_qcc_bytes;


  l_nb_comps = p_j2k->m_image->numcomps - 1;
  l_nb_bytes += j2k_get_max_toc_size(p_j2k);
  if
    (p_j2k->m_cp.m_specific_param.m_enc.m_cinema == 0)
  {
    l_coc_bytes = j2k_get_max_coc_size(p_j2k);
    l_nb_bytes += l_nb_comps * l_coc_bytes;
    l_qcc_bytes = j2k_get_max_qcc_size(p_j2k);
    l_nb_bytes += l_nb_comps * l_qcc_bytes;
  }
  l_nb_bytes += j2k_get_max_poc_size(p_j2k);
  /*** DEVELOPER CORNER, Add room for your headers ***/


  return l_nb_bytes;
}


/**
 * Writes the COC marker (Coding style component)
 *
 * @param  p_comp_no    the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
void j2k_write_coc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
            )
{
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_coc_size,l_remaining_size;
  OPJ_BYTE * l_current_data = 00;
  opj_image_t *l_image = 00;
  OPJ_UINT32 l_comp_room;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
  l_image = p_j2k->m_image;
  l_comp_room = (l_image->numcomps <= 256) ? 1 : 2;

  l_coc_size = 5 + l_comp_room + j2k_get_SPCod_SPCoc_size(p_j2k,p_j2k->m_current_tile_number,p_comp_no);
  l_remaining_size = l_coc_size;

  l_current_data = p_data;

  opj_write_bytes(l_current_data,J2K_MS_COC,2);        /* COC */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_coc_size-2,2);        /* L_COC */
  l_current_data += 2;
  opj_write_bytes(l_current_data,p_comp_no, l_comp_room);    /* Ccoc */
  l_current_data+=l_comp_room;
  opj_write_bytes(l_current_data, l_tcp->tccps[p_comp_no].csty, 1);    /* Scoc */
  ++l_current_data;
  l_remaining_size -= (5 + l_comp_room);
  j2k_write_SPCod_SPCoc(p_j2k,p_j2k->m_current_tile_number,0,l_current_data,&l_remaining_size,p_manager);
  * p_data_written = l_coc_size;
}


/**
 * Reads a COC marker (Coding Style Component)
 * @param  p_header_data  the data contained in the COC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_coc (
          opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          )
{
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_image_t *l_image = 00;
  OPJ_UINT32 l_comp_room;
  OPJ_UINT32 l_comp_no;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH) ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  l_image = p_j2k->m_image;

  l_comp_room = l_image->numcomps <= 256 ? 1 : 2;
  // make sure room is sufficient
  if
    (p_header_size < l_comp_room + 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
    return false;
  }
  p_header_size -= l_comp_room + 1;

  opj_read_bytes(p_header_data,&l_comp_no,l_comp_room);      /* Ccoc */
  p_header_data += l_comp_room;
  if
    (l_comp_no >= l_image->numcomps)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker (bad number of components)\n");
    return false;
  }
  opj_read_bytes(p_header_data,&l_tcp->tccps[l_comp_no].csty,1);      /* Scoc */
  ++p_header_data ;

  if
    (! j2k_read_SPCod_SPCoc(p_j2k,l_comp_no,p_header_data,&p_header_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
    return false;
  }
  if
    (p_header_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading COC marker\n");
    return false;
  }
  return true;
}

/**
 * Gets the size taken by writting SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param  p_tile_no    the tile indix.
 * @param  p_comp_no    the component being outputted.
 * @param  p_j2k      the J2K codec.
 *
 * @return  the number of bytes taken by the SPCod element.
 */
OPJ_UINT32 j2k_get_SQcd_SQcc_size (
            opj_j2k_t *p_j2k,
            OPJ_UINT32 p_tile_no,
            OPJ_UINT32 p_comp_no
            )
{
  OPJ_UINT32 l_num_bands;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;

  // preconditions
  assert(p_j2k != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_tile_no];
  l_tccp = &l_tcp->tccps[p_comp_no];

  // preconditions again
  assert(p_tile_no < l_cp->tw * l_cp->th);
  assert(p_comp_no < p_j2k->m_image->numcomps);

  l_num_bands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 : (l_tccp->numresolutions * 3 - 2);

  if
    (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT)
  {
    return 1 + l_num_bands;
  }
  else
  {
    return 1 + 2*l_num_bands;
  }
}

/**
 * Writes a SQcd or SQcc element, i.e. the quantization values of a band.
 *
 * @param  p_tile_no    the tile to output.
 * @param  p_comp_no    the component number to output.
 * @param  p_data      the data buffer.
 * @param  p_header_size  pointer to the size of the data buffer, it is changed by the function.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
 *
*/
bool j2k_write_SQcd_SQcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_tile_no,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_header_size;
  OPJ_UINT32 l_band_no, l_num_bands;
  OPJ_UINT32 l_expn,l_mant;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_header_size != 00);
  assert(p_manager != 00);
  assert(p_data != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_tile_no];
  l_tccp = &l_tcp->tccps[p_comp_no];

  // preconditions again
  assert(p_tile_no < l_cp->tw * l_cp->th);
  assert(p_comp_no <p_j2k->m_image->numcomps);

  l_num_bands = (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT) ? 1 : (l_tccp->numresolutions * 3 - 2);

  if
    (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT)
  {
    l_header_size = 1 + l_num_bands;
    if
      (*p_header_size < l_header_size)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error writting SQcd SQcc element\n");
      return false;
    }
    opj_write_bytes(p_data,l_tccp->qntsty + (l_tccp->numgbits << 5), 1);  /* Sqcx */
    ++p_data;
    for
      (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no)
    {
      l_expn = l_tccp->stepsizes[l_band_no].expn;
      opj_write_bytes(p_data, l_expn << 3, 1);  /* SPqcx_i */
      ++p_data;
    }
  }
  else
  {
    l_header_size = 1 + 2*l_num_bands;
    if
      (*p_header_size < l_header_size)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error writting SQcd SQcc element\n");
      return false;
    }
    opj_write_bytes(p_data,l_tccp->qntsty + (l_tccp->numgbits << 5), 1);  /* Sqcx */
    ++p_data;
    for
      (l_band_no = 0; l_band_no < l_num_bands; ++l_band_no)
    {
      l_expn = l_tccp->stepsizes[l_band_no].expn;
      l_mant = l_tccp->stepsizes[l_band_no].mant;
      opj_write_bytes(p_data, (l_expn << 11) + l_mant, 2);  /* SPqcx_i */
      p_data += 2;
    }
  }
  *p_header_size = *p_header_size - l_header_size;
  return true;
}

/**
 * Reads a SQcd or SQcc element, i.e. the quantization values of a band.
 *
 * @param  p_comp_no    the component being targeted.
 * @param  p_header_data  the data contained in the COM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the COM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_SQcd_SQcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE* p_header_data,
              OPJ_UINT32 * p_header_size,
              struct opj_event_mgr * p_manager
              )
{
  // loop
  OPJ_UINT32 l_band_no;
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;
  OPJ_BYTE * l_current_ptr = 00;
  OPJ_UINT32 l_tmp;
  OPJ_UINT32 l_num_band;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_header_data != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  // precondition again
  assert(p_comp_no <  p_j2k->m_image->numcomps);
  l_tccp = &l_tcp->tccps[p_comp_no];
  l_current_ptr = p_header_data;

  if
    (* p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading SQcd or SQcc element\n");
    return false;
  }
  * p_header_size -= 1;

  opj_read_bytes(l_current_ptr, &l_tmp ,1);      /* Sqcx */
  ++l_current_ptr;

  l_tccp->qntsty = l_tmp & 0x1f;
  l_tccp->numgbits = l_tmp >> 5;
  if
    (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT)
  {
        l_num_band = 1;
  }
  else
  {
    l_num_band = (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) ? (*p_header_size) : (*p_header_size) / 2;
    if( l_num_band > J2K_MAXBANDS )
      {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading CCP_QNTSTY element\n");
      return false;
      }
  }

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {

    /* if JPWL is on, we check whether there are too many subbands */
    if ((numbands < 0) || (numbands >= J2K_MAXBANDS)) {
      opj_event_msg(p_j2k->cinfo, JPWL_ASSUME ? EVT_WARNING : EVT_ERROR,
        "JPWL: bad number of subbands in Sqcx (%d)\n",
        numbands);
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      numbands = 1;
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust them\n"
        "- setting number of bands to %d => HYPOTHESIS!!!\n",
        numbands);
    };

  };
#endif /* USE_JPWL */
  if
    (l_tccp->qntsty == J2K_CCP_QNTSTY_NOQNT)
  {
    for
      (l_band_no = 0; l_band_no < l_num_band; l_band_no++)
    {
      opj_read_bytes(l_current_ptr, &l_tmp ,1);      /* SPqcx_i */
      ++l_current_ptr;
      l_tccp->stepsizes[l_band_no].expn = l_tmp>>3;
      l_tccp->stepsizes[l_band_no].mant = 0;
    }
    * p_header_size = * p_header_size - l_num_band;
  }
  else
  {
    for
      (l_band_no = 0; l_band_no < l_num_band; l_band_no++)
    {
      opj_read_bytes(l_current_ptr, &l_tmp ,2);      /* SPqcx_i */
      l_current_ptr+=2;
      l_tccp->stepsizes[l_band_no].expn = l_tmp >> 11;
      l_tccp->stepsizes[l_band_no].mant = l_tmp & 0x7ff;
    }
    * p_header_size = * p_header_size - 2*l_num_band;
  }

  /* Add Antonin : if scalar_derived -> compute other stepsizes */
  if
    (l_tccp->qntsty == J2K_CCP_QNTSTY_SIQNT)
  {
    for
      (l_band_no = 1; l_band_no < J2K_MAXBANDS; l_band_no++)
    {
      l_tccp->stepsizes[l_band_no].expn =
        ((l_tccp->stepsizes[0].expn) - ((l_band_no - 1) / 3) > 0) ?
          (l_tccp->stepsizes[0].expn) - ((l_band_no - 1) / 3) : 0;
      l_tccp->stepsizes[l_band_no].mant = l_tccp->stepsizes[0].mant;
    }

  }
  return true;
}



/**
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param    p_j2k    the J2k codec.
 */
void j2k_copy_tile_quantization_parameters(
              opj_j2k_t *p_j2k
              )
{
  // loop
  OPJ_UINT32 i;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_ref_tccp = 00;
  opj_tccp_t *l_copied_tccp = 00;
  OPJ_UINT32 l_size;
  // preconditions
  assert(p_j2k != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  // precondition again
  l_ref_tccp = &l_tcp->tccps[0];
  l_copied_tccp = l_ref_tccp + 1;
  l_size = J2K_MAXBANDS * sizeof(opj_stepsize_t);

  for
    (i=1;i<p_j2k->m_image->numcomps;++i)
  {
    l_copied_tccp->qntsty = l_ref_tccp->qntsty;
    l_copied_tccp->numgbits = l_ref_tccp->numgbits;
    memcpy(l_copied_tccp->stepsizes,l_ref_tccp->stepsizes,l_size);
    ++l_copied_tccp;
  }
}



/**
 * Writes the QCD marker (quantization default)
 *
 * @param  p_comp_number  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_qcd(
              opj_j2k_t *p_j2k,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
              )
{
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_qcd_size,l_remaining_size;
  OPJ_BYTE * l_current_data = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
  l_qcd_size = 4 + j2k_get_SQcd_SQcc_size(p_j2k,p_j2k->m_current_tile_number,0);
  l_remaining_size = l_qcd_size;

  if
    (l_qcd_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_qcd_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_qcd_size;
  }
  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  opj_write_bytes(l_current_data,J2K_MS_QCD,2);    /* QCD */
  l_current_data += 2;

  opj_write_bytes(l_current_data,l_qcd_size-2,2);    /* L_QCD */
  l_current_data += 2;

  l_remaining_size -= 4;

  if
    (! j2k_write_SQcd_SQcc(p_j2k,p_j2k->m_current_tile_number,0,l_current_data,&l_remaining_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error writting QCD marker\n");
    return false;
  }
  if
    (l_remaining_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error writting QCD marker\n");
    return false;
  }

  if
    (opj_stream_write_data(p_stream, p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_qcd_size,p_manager) != l_qcd_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a QCD marker (Quantization defaults)
 * @param  p_header_data  the data contained in the QCD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the QCD marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_qcd (
            opj_j2k_t *p_j2k,
          OPJ_BYTE * p_header_data,
          OPJ_UINT32 p_header_size,
          struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (! j2k_read_SQcd_SQcc(p_j2k,0,p_header_data,&p_header_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading QCD marker\n");
    return false;
  }
  if
    (p_header_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading QCD marker\n");
    return false;
  }
  j2k_copy_tile_quantization_parameters(p_j2k);
  return true;
}


/**
 * Writes the QCC marker (quantization component)
 *
 * @param  p_comp_no  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_qcc(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              struct opj_stream_private *p_stream,
              struct opj_event_mgr * p_manager
              )
{
  OPJ_UINT32 l_qcc_size,l_remaining_size;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_qcc_size = 6 + j2k_get_SQcd_SQcc_size(p_j2k,p_j2k->m_current_tile_number,p_comp_no);
  l_remaining_size = l_qcc_size;
  if
    (l_qcc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_qcc_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_qcc_size;
  }
  j2k_write_qcc_in_memory(p_j2k,p_comp_no,p_j2k->m_specific_param.m_encoder.m_header_tile_data,&l_remaining_size,p_manager);

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_qcc_size,p_manager) != l_qcc_size)
  {
    return false;
  }
  return true;
}


/**
 * Writes the QCC marker (quantization component)
 *
 * @param  p_comp_no  the index of the component to output.
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
void j2k_write_qcc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_UINT32 p_comp_no,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
              )
{
  OPJ_UINT32 l_qcc_size,l_remaining_size;
  OPJ_BYTE * l_current_data = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_qcc_size = 6 + j2k_get_SQcd_SQcc_size(p_j2k,p_j2k->m_current_tile_number,p_comp_no);
  l_remaining_size = l_qcc_size;

  l_current_data = p_data;

  opj_write_bytes(l_current_data,J2K_MS_QCC,2);    /* QCC */
  l_current_data += 2;

  if
    (p_j2k->m_image->numcomps <= 256)
  {
    --l_qcc_size;
    opj_write_bytes(l_current_data,l_qcc_size-2,2);    /* L_QCC */
    l_current_data += 2;
    opj_write_bytes(l_current_data, p_comp_no, 1);  /* Cqcc */
    ++l_current_data;
    // in the case only one byte is sufficient the last byte allocated is useless -> still do -6 for available
    l_remaining_size -= 6;
  }
  else
  {
    opj_write_bytes(l_current_data,l_qcc_size-2,2);    /* L_QCC */
    l_current_data += 2;
    opj_write_bytes(l_current_data, p_comp_no, 2);  /* Cqcc */
    l_current_data+=2;
    l_remaining_size -= 6;
  }
  j2k_write_SQcd_SQcc(p_j2k,p_j2k->m_current_tile_number,p_comp_no,l_current_data,&l_remaining_size,p_manager);
  * p_data_written = l_qcc_size;
}

/**
 * Gets the maximum size taken by a qcc.
 */
OPJ_UINT32 j2k_get_max_qcc_size (opj_j2k_t *p_j2k)
{
  return j2k_get_max_coc_size(p_j2k);
}

/**
 * Reads a QCC marker (Quantization component)
 * @param  p_header_data  the data contained in the QCC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the QCC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_qcc(
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 p_header_size,
              struct opj_event_mgr * p_manager)
{
  OPJ_UINT32 l_num_comp,l_comp_no;
  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_num_comp = p_j2k->m_image->numcomps;

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {

    static OPJ_UINT32 backup_compno = 0;

    /* compno is negative or larger than the number of components!!! */
    if ((compno < 0) || (compno >= numcomp)) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "JPWL: bad component number in QCC (%d out of a maximum of %d)\n",
        compno, numcomp);
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      compno = backup_compno % numcomp;
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
        "- setting component number to %d\n",
        compno);
    }

    /* keep your private count of tiles */
    backup_compno++;
  };
#endif /* USE_JPWL */
  if
    (l_num_comp <= 256)
  {
    if
      (p_header_size < 1)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
      return false;
    }
    opj_read_bytes(p_header_data,&l_comp_no,1);
    ++p_header_data;
    --p_header_size;
  }
  else
  {
    if
      (p_header_size < 2)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
      return false;
    }
    opj_read_bytes(p_header_data,&l_comp_no,2);
    p_header_data+=2;
    p_header_size-=2;
  }
  if
    (! j2k_read_SQcd_SQcc(p_j2k,l_comp_no,p_header_data,&p_header_size,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
    return false;
  }
  if
    (p_header_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading QCC marker\n");
    return false;
  }
  return true;

}


/**
 * Writes the CBD marker (Component bit depth definition)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_cbd(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_cbd_size;
  OPJ_BYTE * l_current_data = 00;
  opj_image_t *l_image = 00;
  opj_image_comp_t * l_comp = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_image = p_j2k->m_image;
  l_cbd_size = 6 + p_j2k->m_image->numcomps;

  if
    (l_cbd_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_cbd_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_cbd_size;
  }

  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;
  opj_write_bytes(l_current_data,J2K_MS_CBD,2);          /* CBD */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_cbd_size-2,2);          /* L_CBD */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_image->numcomps, 2);    /* Ncbd */
  l_current_data+=2;
  l_comp = l_image->comps;
  for
    (i=0;i<l_image->numcomps;++i)
  {
    opj_write_bytes(l_current_data, (l_comp->sgnd << 7) | (l_comp->prec - 1), 1);    /* Component bit depth */
    ++l_current_data;
    ++l_comp;
  }
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_cbd_size,p_manager) != l_cbd_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a CBD marker (Component bit depth definition)
 * @param  p_header_data  the data contained in the CBD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the CBD marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_cbd (
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_header_data,
              OPJ_UINT32 p_header_size,
              struct opj_event_mgr * p_manager)
{
  OPJ_UINT32 l_nb_comp,l_num_comp;
  OPJ_UINT32 l_comp_def;
  OPJ_UINT32 i;
  opj_image_comp_t * l_comp = 00;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_num_comp = p_j2k->m_image->numcomps;

  if
    (p_header_size != (p_j2k->m_image->numcomps + 2))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Crror reading CBD marker\n");
    return false;
  }
  opj_read_bytes(p_header_data,&l_nb_comp,2);        /* Ncbd */
  p_header_data+=2;
  if
    (l_nb_comp != l_num_comp)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Crror reading CBD marker\n");
    return false;
  }

  l_comp = p_j2k->m_image->comps;
  for
    (i=0;i<l_num_comp;++i)
  {
    opj_read_bytes(p_header_data,&l_comp_def,1);      /* Component bit depth */
    ++p_header_data;
        l_comp->sgnd = (l_comp_def>>7) & 1;
    l_comp->prec = (l_comp_def&0x7f) + 1;
    ++l_comp;
  }
  return true;
}

/**
 * Writes the MCC marker (Multiple Component Collection)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_mcc_record(
            opj_j2k_t *p_j2k,
            struct opj_simple_mcc_decorrelation_data * p_mcc_record,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_mcc_size;
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_nb_bytes_for_comp;
  OPJ_UINT32 l_mask;
  OPJ_UINT32 l_tmcc;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  if
    (p_mcc_record->m_nb_comps > 255 )
  {
        l_nb_bytes_for_comp = 2;
    l_mask = 0x8000;
  }
  else
  {
    l_nb_bytes_for_comp = 1;
    l_mask = 0;
  }

  l_mcc_size = p_mcc_record->m_nb_comps * 2 * l_nb_bytes_for_comp + 19;
  if
    (l_mcc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_mcc_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mcc_size;
  }
  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;
  opj_write_bytes(l_current_data,J2K_MS_MCC,2);          /* MCC */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_mcc_size-2,2);          /* Lmcc */
  l_current_data += 2;

  /* first marker */
  opj_write_bytes(l_current_data,0,2);          /* Zmcc */
  l_current_data += 2;
  opj_write_bytes(l_current_data,p_mcc_record->m_index,1);          /* Imcc -> no need for other values, take the first */
  ++l_current_data;
  /* only one marker atm */
  opj_write_bytes(l_current_data,0,2);          /* Ymcc */
  l_current_data+=2;
  opj_write_bytes(l_current_data,1,2);          /* Qmcc -> number of collections -> 1 */
  l_current_data+=2;
  opj_write_bytes(l_current_data,0x1,1);          /* Xmcci type of component transformation -> array based decorrelation */
  ++l_current_data;

  opj_write_bytes(l_current_data,p_mcc_record->m_nb_comps | l_mask,2);  /* Nmcci number of input components involved and size for each component offset = 8 bits */
  l_current_data+=2;

  for
    (i=0;i<p_mcc_record->m_nb_comps;++i)
  {
    opj_write_bytes(l_current_data,i,l_nb_bytes_for_comp);        /* Cmccij Component offset*/
    l_current_data+=l_nb_bytes_for_comp;
  }

  opj_write_bytes(l_current_data,p_mcc_record->m_nb_comps|l_mask,2);  /* Mmcci number of output components involved and size for each component offset = 8 bits */
  l_current_data+=2;
  for
    (i=0;i<p_mcc_record->m_nb_comps;++i)
  {
    opj_write_bytes(l_current_data,i,l_nb_bytes_for_comp);        /* Wmccij Component offset*/
    l_current_data+=l_nb_bytes_for_comp;
  }
  l_tmcc = ((!p_mcc_record->m_is_irreversible)&1)<<16;
  if
    (p_mcc_record->m_decorrelation_array)
  {
    l_tmcc |= p_mcc_record->m_decorrelation_array->m_index;
  }
  if
    (p_mcc_record->m_offset_array)
  {
    l_tmcc |= ((p_mcc_record->m_offset_array->m_index)<<8);
  }
  opj_write_bytes(l_current_data,l_tmcc,3);  /* Tmcci : use MCT defined as number 1 and irreversible array based. */
  l_current_data+=3;
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_mcc_size,p_manager) != l_mcc_size)
  {
    return false;
  }
  return true;
}


/**
 * Reads a MCC marker (Multiple Component Collection)
 *
 * @param  p_header_data  the data contained in the MCC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_mcc (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i,j;
  OPJ_UINT32 l_tmp;
  OPJ_UINT32 l_indix;
  opj_tcp_t * l_tcp;
  opj_simple_mcc_decorrelation_data_t * l_mcc_record;
  opj_mct_data_t * l_mct_data;
  OPJ_UINT32 l_nb_collections;
  OPJ_UINT32 l_nb_comps;
  OPJ_UINT32 l_nb_bytes_by_comp;


  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;

  if
    (p_header_size < 2)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
    return false;
  }

  /* first marker */
  opj_read_bytes(p_header_data,&l_tmp,2);        /* Zmcc */
  p_header_data += 2;
  if
    (l_tmp != 0)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge multiple data spanning\n");
    return true;
  }
  if
    (p_header_size < 7)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
    return false;
  }
  opj_read_bytes(p_header_data,&l_indix,1);        /* Imcc -> no need for other values, take the first */
  ++p_header_data;

  l_mcc_record = l_tcp->m_mcc_records;
  for
    (i=0;i<l_tcp->m_nb_mcc_records;++i)
  {
    if
      (l_mcc_record->m_index == l_indix)
    {
      break;
    }
    ++l_mcc_record;
  }
  /** NOT FOUND */
  if
    (i == l_tcp->m_nb_mcc_records)
  {
    if
      (l_tcp->m_nb_mcc_records == l_tcp->m_nb_max_mcc_records)
    {
      l_tcp->m_nb_max_mcc_records += J2K_MCC_DEFAULT_NB_RECORDS;
      l_tcp->m_mcc_records = (opj_simple_mcc_decorrelation_data_t*)
      opj_realloc(l_tcp->m_mcc_records,l_tcp->m_nb_max_mcc_records * sizeof(opj_simple_mcc_decorrelation_data_t));
      if
        (! l_tcp->m_mcc_records)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return false;
      }
      l_mcc_record = l_tcp->m_mcc_records + l_tcp->m_nb_mcc_records;
      memset(l_mcc_record,0,(l_tcp->m_nb_max_mcc_records-l_tcp->m_nb_mcc_records) * sizeof(opj_simple_mcc_decorrelation_data_t));
    }
    l_mcc_record = l_tcp->m_mcc_records + l_tcp->m_nb_mcc_records;
  }
  l_mcc_record->m_index = l_indix;

  /* only one marker atm */
  opj_read_bytes(p_header_data,&l_tmp,2);        /* Ymcc */
  p_header_data+=2;
  if
    (l_tmp != 0)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge multiple data spanning\n");
    return true;
  }
  opj_read_bytes(p_header_data,&l_nb_collections,2);        /* Qmcc -> number of collections -> 1 */
  p_header_data+=2;
  if
    (l_nb_collections > 1)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge multiple collections\n");
    return true;
  }
  p_header_size -= 7;
  for
    (i=0;i<l_nb_collections;++i)
  {
    if
      (p_header_size < 3)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
      return false;
    }
    opj_read_bytes(p_header_data,&l_tmp,1);  /* Xmcci type of component transformation -> array based decorrelation */
    ++p_header_data;
    if
      (l_tmp != 1)
    {
      opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge collections other than array decorrelation\n");
      return true;
    }
    opj_read_bytes(p_header_data,&l_nb_comps,2);
    p_header_data+=2;
    p_header_size-=3;
    l_nb_bytes_by_comp = 1 + (l_nb_comps>>15);
    l_mcc_record->m_nb_comps = l_nb_comps & 0x7fff;
    if
      (p_header_size < (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 2))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
      return false;
    }
    p_header_size -= (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 2);
    for
      (j=0;j<l_mcc_record->m_nb_comps;++j)
    {
      opj_read_bytes(p_header_data,&l_tmp,l_nb_bytes_by_comp);  /* Cmccij Component offset*/
      p_header_data+=l_nb_bytes_by_comp;
      if
        (l_tmp != j)
      {
        opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge collections with indix shuffle\n");
        return true;
      }
    }
    opj_read_bytes(p_header_data,&l_nb_comps,2);
    p_header_data+=2;
    l_nb_bytes_by_comp = 1 + (l_nb_comps>>15);
    l_nb_comps &= 0x7fff;
    if
      (l_nb_comps != l_mcc_record->m_nb_comps)
    {
      opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge collections without same number of indixes\n");
      return true;
    }
    if
      (p_header_size < (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 3))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
      return false;
    }
    p_header_size -= (l_nb_bytes_by_comp * l_mcc_record->m_nb_comps + 3);
    for
      (j=0;j<l_mcc_record->m_nb_comps;++j)
    {
      opj_read_bytes(p_header_data,&l_tmp,l_nb_bytes_by_comp);  /* Wmccij Component offset*/
      p_header_data+=l_nb_bytes_by_comp;
      if
        (l_tmp != j)
      {
        opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge collections with indix shuffle\n");
        return true;
      }
    }
    opj_read_bytes(p_header_data,&l_tmp,3);  /* Wmccij Component offset*/
    p_header_data += 3;
    l_mcc_record->m_is_irreversible = ! ((l_tmp>>16) & 1);
    l_mcc_record->m_decorrelation_array = 00;
    l_mcc_record->m_offset_array = 00;
    l_indix = l_tmp & 0xff;
    if
      (l_indix != 0)
    {
      l_mct_data = l_tcp->m_mct_records;
      for
        (j=0;j<l_tcp->m_nb_mct_records;++j)
      {
        if
          (l_mct_data->m_index == l_indix)
        {
          l_mcc_record->m_decorrelation_array = l_mct_data;
          break;
        }
        ++l_mct_data;
      }
      if
        (l_mcc_record->m_decorrelation_array == 00)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return false;
      }
    }
    l_indix = (l_tmp >> 8) & 0xff;
    if
      (l_indix != 0)
    {
      l_mct_data = l_tcp->m_mct_records;
      for
        (j=0;j<l_tcp->m_nb_mct_records;++j)
      {
        if
          (l_mct_data->m_index == l_indix)
        {
          l_mcc_record->m_offset_array = l_mct_data;
          break;
        }
        ++l_mct_data;
      }
      if
        (l_mcc_record->m_offset_array == 00)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
        return false;
      }
    }
  }
  if
    (p_header_size != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCC marker\n");
    return false;
  }
  ++l_tcp->m_nb_mcc_records;
  return true;
}

/**
 * Writes the MCT marker (Multiple Component Transform)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_mct_record(
            opj_j2k_t *p_j2k,
            opj_mct_data_t * p_mct_record,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_mct_size;
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_tmp;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_mct_size = 10 + p_mct_record->m_data_size;
  if
    (l_mct_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_mct_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mct_size;
  }

  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  opj_write_bytes(l_current_data,J2K_MS_MCT,2);          /* MCT */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_mct_size-2,2);          /* Lmct */
  l_current_data += 2;
  opj_write_bytes(l_current_data,0,2);              /* Zmct */
  l_current_data += 2;
  /* only one marker atm */
  l_tmp = (p_mct_record->m_index & 0xff) | (p_mct_record->m_array_type << 8) | (p_mct_record->m_element_type << 10);
  opj_write_bytes(l_current_data,l_tmp,2);
  l_current_data += 2;
  opj_write_bytes(l_current_data,0,2);              /* Ymct */
  l_current_data+=2;

  memcpy(l_current_data,p_mct_record->m_data,p_mct_record->m_data_size);
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_mct_size,p_manager) != l_mct_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param  p_header_data  the data contained in the MCT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCT marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_mct (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_tmp;
  OPJ_UINT32 l_indix;
  opj_mct_data_t * l_mct_data;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);

  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;

  if
    (p_header_size < 2)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
    return false;
  }
  /* first marker */
  opj_read_bytes(p_header_data,&l_tmp,2);        /* Zmct */
  p_header_data += 2;
  if
    (l_tmp != 0)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge mct data within multiple MCT records\n");
    return true;
  }
  if
    (p_header_size <= 6)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
    return false;
  }
  opj_read_bytes(p_header_data,&l_tmp,2);        /* Imct -> no need for other values, take the first, type is double with decorrelation x0000 1101 0000 0000*/
  p_header_data += 2;

  l_indix = l_tmp & 0xff;
  l_mct_data = l_tcp->m_mct_records;
  for
    (i=0;i<l_tcp->m_nb_mct_records;++i)
  {
    if
      (l_mct_data->m_index == l_indix)
    {
      break;
    }
    ++l_mct_data;
  }
  /* NOT FOUND */
  if
    (i == l_tcp->m_nb_mct_records)
  {
    if
      (l_tcp->m_nb_mct_records == l_tcp->m_nb_max_mct_records)
    {
      l_tcp->m_nb_max_mct_records += J2K_MCT_DEFAULT_NB_RECORDS;
      l_tcp->m_mct_records = (opj_mct_data_t*)opj_realloc(l_tcp->m_mct_records,l_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
      if
        (! l_tcp->m_mct_records)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
        return false;
      }
      l_mct_data = l_tcp->m_mct_records + l_tcp->m_nb_mct_records;
      memset(l_mct_data ,0,(l_tcp->m_nb_max_mct_records - l_tcp->m_nb_mct_records) * sizeof(opj_mct_data_t));
    }
    l_mct_data = l_tcp->m_mct_records + l_tcp->m_nb_mct_records;
  }
  if
    (l_mct_data->m_data)
  {
    opj_free(l_mct_data->m_data);
    l_mct_data->m_data = 00;
  }
  l_mct_data->m_index = l_indix;
  l_mct_data->m_array_type = (J2K_MCT_ARRAY_TYPE)((l_tmp  >> 8) & 3);
  l_mct_data->m_element_type = (J2K_MCT_ELEMENT_TYPE)((l_tmp  >> 10) & 3);

  opj_read_bytes(p_header_data,&l_tmp,2);        /* Ymct */
  p_header_data+=2;
  if
    (l_tmp != 0)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge multiple MCT markers\n");
    return true;
  }
  p_header_size -= 6;
  l_mct_data->m_data = (OPJ_BYTE*)opj_malloc(p_header_size);
  if
    (! l_mct_data->m_data)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCT marker\n");
    return false;
  }
  memcpy(l_mct_data->m_data,p_header_data,p_header_size);
  l_mct_data->m_data_size = p_header_size;
  ++l_tcp->m_nb_mct_records;
  return true;
}

bool   j2k_setup_mct_encoding (opj_tcp_t * p_tcp,opj_image_t * p_image)
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_indix = 1;
  opj_mct_data_t * l_mct_deco_data = 00,* l_mct_offset_data = 00;
  opj_simple_mcc_decorrelation_data_t * l_mcc_data;
  OPJ_UINT32 l_mct_size,l_nb_elem;
  OPJ_FLOAT32 * l_data, * l_current_data;
  opj_tccp_t * l_tccp;

  // preconditions
  assert(p_tcp != 00);

  if
    (p_tcp->mct != 2)
  {
    return true;
  }

  if
    (p_tcp->m_mct_decoding_matrix)
  {
    if
      (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records)
    {
      p_tcp->m_nb_max_mct_records += J2K_MCT_DEFAULT_NB_RECORDS;
      p_tcp->m_mct_records = (opj_mct_data_t*)opj_realloc(p_tcp->m_mct_records,p_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
      if
        (! p_tcp->m_mct_records)
      {
        return false;
      }
      l_mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;
      memset(l_mct_deco_data ,0,(p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records) * sizeof(opj_mct_data_t));
    }
    l_mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

    if
      (l_mct_deco_data->m_data)
    {
      opj_free(l_mct_deco_data->m_data);
      l_mct_deco_data->m_data = 00;
    }
    l_mct_deco_data->m_index = l_indix++;
    l_mct_deco_data->m_array_type = MCT_TYPE_DECORRELATION;
    l_mct_deco_data->m_element_type = MCT_TYPE_FLOAT;
    l_nb_elem = p_image->numcomps * p_image->numcomps;
    l_mct_size = l_nb_elem * MCT_ELEMENT_SIZE[l_mct_deco_data->m_element_type];
    l_mct_deco_data->m_data = (OPJ_BYTE*)opj_malloc(l_mct_size );
    if
      (! l_mct_deco_data->m_data)
    {
      return false;
    }
    j2k_mct_write_functions_from_float[l_mct_deco_data->m_element_type](p_tcp->m_mct_decoding_matrix,l_mct_deco_data->m_data,l_nb_elem);
    l_mct_deco_data->m_data_size = l_mct_size;
    ++p_tcp->m_nb_mct_records;
  }

  if
    (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records)
  {
    p_tcp->m_nb_max_mct_records += J2K_MCT_DEFAULT_NB_RECORDS;
    p_tcp->m_mct_records = (opj_mct_data_t*)opj_realloc(p_tcp->m_mct_records,p_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t));
    if
      (! p_tcp->m_mct_records)
    {
      return false;
    }
    l_mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;
    memset(l_mct_offset_data ,0,(p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records) * sizeof(opj_mct_data_t));
    if
      (l_mct_deco_data)
    {
      l_mct_deco_data = l_mct_offset_data - 1;
    }
  }
  l_mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;
  if
    (l_mct_offset_data->m_data)
  {
    opj_free(l_mct_offset_data->m_data);
    l_mct_offset_data->m_data = 00;
  }

  l_mct_offset_data->m_index = l_indix++;
  l_mct_offset_data->m_array_type = MCT_TYPE_OFFSET;
  l_mct_offset_data->m_element_type = MCT_TYPE_FLOAT;
  l_nb_elem = p_image->numcomps;
  l_mct_size = l_nb_elem * MCT_ELEMENT_SIZE[l_mct_offset_data->m_element_type];
  l_mct_offset_data->m_data = (OPJ_BYTE*)opj_malloc(l_mct_size );
  if
    (! l_mct_offset_data->m_data)
  {
    return false;
  }
  l_data = (OPJ_FLOAT32*)opj_malloc(l_nb_elem * sizeof(OPJ_FLOAT32));
  if
    (! l_data)
  {
    opj_free(l_mct_offset_data->m_data);
    l_mct_offset_data->m_data = 00;
    return false;
  }
  l_tccp = p_tcp->tccps;
  l_current_data = l_data;
  for
    (i=0;i<l_nb_elem;++i)
  {
    *(l_current_data++) = (OPJ_FLOAT32) (l_tccp->m_dc_level_shift);
    ++l_tccp;
  }
  j2k_mct_write_functions_from_float[l_mct_offset_data->m_element_type](l_data,l_mct_offset_data->m_data,l_nb_elem);
  opj_free(l_data);
  l_mct_offset_data->m_data_size = l_mct_size;
  ++p_tcp->m_nb_mct_records;

  if
    (p_tcp->m_nb_mcc_records == p_tcp->m_nb_max_mcc_records)
  {
    p_tcp->m_nb_max_mcc_records += J2K_MCT_DEFAULT_NB_RECORDS;
    p_tcp->m_mcc_records = (opj_simple_mcc_decorrelation_data_t*)
    opj_realloc(p_tcp->m_mcc_records,p_tcp->m_nb_max_mcc_records * sizeof(opj_simple_mcc_decorrelation_data_t));
    if
      (! p_tcp->m_mcc_records)
    {
      return false;
    }
    l_mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
    memset(l_mcc_data ,0,(p_tcp->m_nb_max_mcc_records - p_tcp->m_nb_mcc_records) * sizeof(opj_simple_mcc_decorrelation_data_t));

  }
  l_mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
  l_mcc_data->m_decorrelation_array = l_mct_deco_data;
  l_mcc_data->m_is_irreversible = 1;
  l_mcc_data->m_nb_comps = p_image->numcomps;
  l_mcc_data->m_index = l_indix++;
  l_mcc_data->m_offset_array = l_mct_offset_data;
  ++p_tcp->m_nb_mcc_records;
  return true;
}

/**
 * Writes the MCO marker (Multiple component transformation ordering)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_mco(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_mco_size;
  opj_tcp_t * l_tcp = 00;
  opj_simple_mcc_decorrelation_data_t * l_mcc_record;
  OPJ_UINT32 i;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tcp =&(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);
  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;
  l_mco_size = 5 + l_tcp->m_nb_mcc_records;
  if
    (l_mco_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_mco_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_mco_size;
  }

  opj_write_bytes(l_current_data,J2K_MS_MCO,2);      /* MCO */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_mco_size-2,2);          /* Lmco */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_tcp->m_nb_mcc_records,1);          /* Nmco : only one tranform stage*/
  ++l_current_data;

  l_mcc_record = l_tcp->m_mcc_records;
  for
    (i=0;i<l_tcp->m_nb_mcc_records;++i)
  {
    opj_write_bytes(l_current_data,l_mcc_record->m_index,1);          /* Imco -> use the mcc indicated by 1*/
    ++l_current_data;
    ++l_mcc_record;
  }

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_mco_size,p_manager) != l_mco_size)
  {
    return false;
  }
  return true;
}
/**
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param  p_header_data  the data contained in the MCO box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the MCO marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_mco (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_tmp, i;
  OPJ_UINT32 l_nb_stages;
  opj_tcp_t * l_tcp;
  opj_tccp_t * l_tccp;
  opj_image_t * l_image;
  opj_image_comp_t * l_img_comp;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_image = p_j2k->m_image;
  l_tcp = p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH ? &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  if
    (p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading MCO marker\n");
    return false;
  }

  opj_read_bytes(p_header_data,&l_nb_stages,1);        /* Nmco : only one tranform stage*/
  ++p_header_data;
  if
    (l_nb_stages > 1)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Cannot take in charge multiple transformation stages.\n");
    return true;
  }
  if
    (p_header_size != l_nb_stages + 1)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Error reading MCO marker\n");
    return false;
  }

  l_tccp = l_tcp->tccps;
  l_img_comp = l_image->comps;
  for
    (i=0;i<l_image->numcomps;++i)
  {
    l_tccp->m_dc_level_shift = 0;
    ++l_tccp;
  }
  if
    (l_tcp->m_mct_decoding_matrix)
  {
    opj_free(l_tcp->m_mct_decoding_matrix);
    l_tcp->m_mct_decoding_matrix = 00;
  }

  for
    (i=0;i<l_nb_stages;++i)
  {
    opj_read_bytes(p_header_data,&l_tmp,1);
    ++p_header_data;
    if
      (! j2k_add_mct(l_tcp,p_j2k->m_image,l_tmp))
    {
      return false;
    }
  }
  return true;
}

bool j2k_add_mct(opj_tcp_t * p_tcp,opj_image_t * p_image, OPJ_UINT32 p_index)
{
  OPJ_UINT32 i;
  opj_simple_mcc_decorrelation_data_t * l_mcc_record;
  opj_mct_data_t * l_deco_array, * l_offset_array;
  OPJ_UINT32 l_data_size,l_mct_size, l_offset_size;
  OPJ_UINT32 l_nb_elem;
  OPJ_UINT32 * l_offset_data, * l_current_offset_data;
  opj_tccp_t * l_tccp;


  // preconditions
  assert(p_tcp != 00);

  l_mcc_record = p_tcp->m_mcc_records;
  for
    (i=0;i<p_tcp->m_nb_mcc_records;++i)
  {
    if
      (l_mcc_record->m_index == p_index)
    {
      break;
    }
  }
  if
    (i==p_tcp->m_nb_mcc_records)
  {
    /** element discarded **/
    return true;
  }
  if
    (l_mcc_record->m_nb_comps != p_image->numcomps)
  {
    /** do not support number of comps != image */
    return true;
  }
  l_deco_array = l_mcc_record->m_decorrelation_array;
  if
    (l_deco_array)
  {
    l_data_size = MCT_ELEMENT_SIZE[l_deco_array->m_element_type] * p_image->numcomps * p_image->numcomps;
    if
      (l_deco_array->m_data_size != l_data_size)
    {
      return false;
    }
    l_nb_elem = p_image->numcomps * p_image->numcomps;
    l_mct_size = l_nb_elem * sizeof(OPJ_FLOAT32);
    p_tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(l_mct_size);
    if
      (! p_tcp->m_mct_decoding_matrix )
    {
      return false;
    }
    j2k_mct_read_functions_to_float[l_deco_array->m_element_type](l_deco_array->m_data,p_tcp->m_mct_decoding_matrix,l_nb_elem);
  }
  l_offset_array = l_mcc_record->m_offset_array;
  if
    (l_offset_array)
  {
    l_data_size = MCT_ELEMENT_SIZE[l_offset_array->m_element_type] * p_image->numcomps;
    if
      (l_offset_array->m_data_size != l_data_size)
    {
      return false;
    }
    l_nb_elem = p_image->numcomps;
    l_offset_size = l_nb_elem * sizeof(OPJ_UINT32);
    l_offset_data = (OPJ_UINT32*)opj_malloc(l_offset_size);
    if
      (! l_offset_data )
    {
      return false;
    }
    j2k_mct_read_functions_to_int32[l_offset_array->m_element_type](l_offset_array->m_data,l_offset_data,l_nb_elem);
    l_tccp = p_tcp->tccps;
    l_current_offset_data = l_offset_data;
    for
      (i=0;i<p_image->numcomps;++i)
    {
      l_tccp->m_dc_level_shift = *(l_current_offset_data++);
      ++l_tccp;
    }
    opj_free(l_offset_data);
  }
  return true;
}

/**
 * Writes the MCT marker (Multiple Component Transform)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_mct_data_group(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  opj_simple_mcc_decorrelation_data_t * l_mcc_record;
  opj_mct_data_t * l_mct_record;
  opj_tcp_t * l_tcp;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  if
    (! j2k_write_cbd(p_j2k,p_stream,p_manager))
  {
    return false;
  }
  l_tcp = &(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);
  l_mct_record = l_tcp->m_mct_records;
  for
    (i=0;i<l_tcp->m_nb_mct_records;++i)
  {
    if
      (! j2k_write_mct_record(p_j2k,l_mct_record,p_stream,p_manager))
    {
      return false;
    }
    ++l_mct_record;
  }
  l_mcc_record = l_tcp->m_mcc_records;
  for
    (i=0;i<l_tcp->m_nb_mcc_records;++i)
  {
    if
      (! j2k_write_mcc_record(p_j2k,l_mcc_record,p_stream,p_manager))
    {
      return false;
    }
    ++l_mcc_record;
  }
  if
    (! j2k_write_mco(p_j2k,p_stream,p_manager))
  {
    return false;
  }
  return true;
}


/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_poc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_nb_comp;
  OPJ_UINT32 l_nb_poc;
  OPJ_UINT32 l_poc_size;
  OPJ_UINT32 l_written_size = 0;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;
  OPJ_UINT32 l_poc_room;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tcp = &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number];
  l_tccp = &l_tcp->tccps[0];
  l_nb_comp = p_j2k->m_image->numcomps;
  l_nb_poc = 1 + l_tcp->numpocs;
  if
    (l_nb_comp <= 256)
  {
    l_poc_room = 1;
  }
  else
  {
    l_poc_room = 2;
  }
  l_poc_size = 4 + (5 + 2 * l_poc_room) * l_nb_poc;
  if
    (l_poc_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_poc_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_poc_size;
  }

  j2k_write_poc_in_memory(p_j2k,p_j2k->m_specific_param.m_encoder.m_header_tile_data,&l_written_size,p_manager);

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_poc_size,p_manager) != l_poc_size)
  {
    return false;
  }
  return true;
}


/**
 * Writes EPC ????
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_epc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  opj_codestream_info_t * l_info = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_info = p_j2k->cstr_info;
  if
    (l_info)
  {
    l_info->codestream_size = opj_stream_tell(p_stream);
    /* UniPG>> */
    /* The following adjustment is done to adjust the codestream size */
    /* if SOD is not at 0 in the buffer. Useful in case of JP2, where */
    /* the first bunch of bytes is not in the codestream              */
    l_info->codestream_size -= l_info->main_head_start;
    /* <<UniPG */
  }

#ifdef USE_JPWL
  /*
  preparation of JPWL marker segments
  */
  if(cp->epc_on) {

    /* encode according to JPWL */
    jpwl_encode(p_j2k, p_stream, image);

  }
#endif /* USE_JPWL */
  return true;
}


/**
 * Gets the maximum size taken by the writting of a POC.
 */
OPJ_UINT32 j2k_get_max_poc_size(opj_j2k_t *p_j2k)
{
  opj_tcp_t * l_tcp = 00;
  OPJ_UINT32 l_nb_tiles = 0;
  OPJ_UINT32 l_max_poc = 0;
  OPJ_UINT32 i;

  l_tcp = p_j2k->m_cp.tcps;
  l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;

  for
    (i=0;i<l_nb_tiles;++i)
  {
    l_max_poc = uint_max(l_max_poc,l_tcp->numpocs);
    ++l_tcp;
  }
  ++l_max_poc;
  return 4 + 9 * l_max_poc;
}


/**
 * Writes the POC marker (Progression Order Change)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
void j2k_write_poc_in_memory(
              opj_j2k_t *p_j2k,
              OPJ_BYTE * p_data,
              OPJ_UINT32 * p_data_written,
              struct opj_event_mgr * p_manager
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

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_tcp = &p_j2k->m_cp.tcps[p_j2k->m_current_tile_number];
  l_tccp = &l_tcp->tccps[0];
  l_image = p_j2k->m_image;
  l_nb_comp = l_image->numcomps;
  l_nb_poc = 1 + l_tcp->numpocs;
  if
    (l_nb_comp <= 256)
  {
    l_poc_room = 1;
  }
  else
  {
    l_poc_room = 2;
  }
  l_poc_size = 4 + (5 + 2 * l_poc_room) * l_nb_poc;

  l_current_data = p_data;

  opj_write_bytes(l_current_data,J2K_MS_POC,2);          /* POC  */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_poc_size-2,2);          /* Lpoc */
  l_current_data += 2;

  l_current_poc =  l_tcp->pocs;
  for
    (i = 0; i < l_nb_poc; ++i)
  {
    opj_write_bytes(l_current_data,l_current_poc->resno0,1);        /* RSpoc_i */
    ++l_current_data;
    opj_write_bytes(l_current_data,l_current_poc->compno0,l_poc_room);    /* CSpoc_i */
    l_current_data+=l_poc_room;
    opj_write_bytes(l_current_data,l_current_poc->layno1,2);        /* LYEpoc_i */
    l_current_data+=2;
    opj_write_bytes(l_current_data,l_current_poc->resno1,1);        /* REpoc_i */
    ++l_current_data;
    opj_write_bytes(l_current_data,l_current_poc->compno1,l_poc_room);    /* CEpoc_i */
    l_current_data+=l_poc_room;
    opj_write_bytes(l_current_data,l_current_poc->prg,1);          /* Ppoc_i */
    ++l_current_data;

    /* change the value of the max layer according to the actual number of layers in the file, components and resolutions*/
    l_current_poc->layno1 = int_min(l_current_poc->layno1, l_tcp->numlayers);
    l_current_poc->resno1 = int_min(l_current_poc->resno1, l_tccp->numresolutions);
    l_current_poc->compno1 = int_min(l_current_poc->compno1, l_nb_comp);
    ++l_current_poc;
  }
  * p_data_written = l_poc_size;
}


/**
 * Reads a POC marker (Progression Order Change)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_poc (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_nb_comp;
  opj_image_t * l_image = 00;
  OPJ_UINT32 l_old_poc_nb,l_current_poc_nb,l_current_poc_remaining;
  OPJ_UINT32 l_chunk_size;
  OPJ_UINT32 l_tmp;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_poc_t *l_current_poc = 00;
  OPJ_UINT32 l_comp_room;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_image = p_j2k->m_image;
  l_nb_comp = l_image->numcomps;
  if
    (l_nb_comp <= 256)
  {
    l_comp_room = 1;
  }
  else
  {
    l_comp_room = 2;
  }
  l_chunk_size = 5 + 2 * l_comp_room;
  l_current_poc_nb = p_header_size / l_chunk_size;
  l_current_poc_remaining = p_header_size % l_chunk_size;

  if
    ((l_current_poc_nb <= 0) || (l_current_poc_remaining != 0))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading POC marker\n");
    return false;
  }

  l_cp = &(p_j2k->m_cp);
  l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH) ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;
  l_old_poc_nb = l_tcp->POC ? l_tcp->numpocs + 1 : 0;
  l_current_poc_nb += l_old_poc_nb;
  assert(l_current_poc_nb < 32);

  /* now poc is in use.*/
  l_tcp->POC = 1;

  l_current_poc = &l_tcp->pocs[l_old_poc_nb];
  for
    (i = l_old_poc_nb; i < l_current_poc_nb; ++i)
  {
    opj_read_bytes(p_header_data,&(l_current_poc->resno0),1);          /* RSpoc_i */
    ++p_header_data;
    opj_read_bytes(p_header_data,&(l_current_poc->compno0),l_comp_room);    /* CSpoc_i */
    p_header_data+=l_comp_room;
    opj_read_bytes(p_header_data,&(l_current_poc->layno1),2);          /* LYEpoc_i */
    p_header_data+=2;
    opj_read_bytes(p_header_data,&(l_current_poc->resno1),1);           /* REpoc_i */
    ++p_header_data;
    opj_read_bytes(p_header_data,&(l_current_poc->compno1),l_comp_room);    /* CEpoc_i */
    p_header_data+=l_comp_room;
    opj_read_bytes(p_header_data,&l_tmp,1);            /* Ppoc_i */
    ++p_header_data;
    l_current_poc->prg = (OPJ_PROG_ORDER) l_tmp;
    /* make sure comp is in acceptable bounds */
    l_current_poc->compno1 = uint_min(l_current_poc->compno1, l_nb_comp);
    ++l_current_poc;
  }
  l_tcp->numpocs = l_current_poc_nb - 1;
  return true;
}

/**
 * Writes the RGN marker (Region Of Interest)
 *
 * @param  p_tile_no    the tile to output
 * @param  p_comp_no    the component to output
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_rgn(
            opj_j2k_t *p_j2k,
            OPJ_UINT32 p_tile_no,
            OPJ_UINT32 p_comp_no,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_nb_comp;
  OPJ_UINT32 l_rgn_size;
  opj_image_t *l_image = 00;
  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  opj_tccp_t *l_tccp = 00;
  OPJ_UINT32 l_comp_room;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_tile_no];
  l_tccp = &l_tcp->tccps[p_comp_no];

  l_nb_comp = l_image->numcomps;

  if
    (l_nb_comp <= 256)
  {
    l_comp_room = 1;
  }
  else
  {
    l_comp_room = 2;
  }
  l_rgn_size = 6 + l_comp_room;

  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  opj_write_bytes(l_current_data,J2K_MS_RGN,2);          /* RGN  */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_rgn_size-2,2);          /* Lrgn */
  l_current_data += 2;
  opj_write_bytes(l_current_data,p_comp_no,l_comp_room);      /* Crgn */
  l_current_data+=l_comp_room;
  opj_write_bytes(l_current_data, 0,1);              /* Srgn */
  ++l_current_data;
  opj_write_bytes(l_current_data, l_tccp->roishift,1);      /* SPrgn */
  ++l_current_data;

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_rgn_size,p_manager) != l_rgn_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a RGN marker (Region Of Interest)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_rgn (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_nb_comp;
  opj_image_t * l_image = 00;

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_comp_room;
  OPJ_UINT32 l_comp_no;
  OPJ_UINT32 l_roi_sty;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_image = p_j2k->m_image;
  l_nb_comp = l_image->numcomps;
  if
    (l_nb_comp <= 256)
  {
    l_comp_room = 1;
  }
  else
  {
    l_comp_room = 2;
  }
  if
    (p_header_size != 2 + l_comp_room)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading RGN marker\n");
    return false;
  }

  l_cp = &(p_j2k->m_cp);
  l_tcp = (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_TPH) ? &l_cp->tcps[p_j2k->m_current_tile_number] : p_j2k->m_specific_param.m_decoder.m_default_tcp;

  opj_read_bytes(p_header_data,&l_comp_no,l_comp_room);    /* Crgn */
  p_header_data+=l_comp_room;
  opj_read_bytes(p_header_data,&l_roi_sty,1);          /* Srgn */
  ++p_header_data;

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {
    /* totlen is negative or larger than the bytes left!!! */
    if (compno >= numcomps) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "JPWL: bad component number in RGN (%d when there are only %d)\n",
        compno, numcomps);
      if (!JPWL_ASSUME || JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
    }
  };
#endif /* USE_JPWL */

  opj_read_bytes(p_header_data,(OPJ_UINT32 *) (&(l_tcp->tccps[l_comp_no].roishift)),1);  /* SPrgn */
  ++p_header_data;
  return true;

}

/**
 * Writes the TLM marker (Tile Length Marker)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_tlm(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_tlm_size;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tlm_size = 6 + (5*p_j2k->m_specific_param.m_encoder.m_total_tile_parts);
  if
    (l_tlm_size > p_j2k->m_specific_param.m_encoder.m_header_tile_data_size)
  {
    p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
        p_j2k->m_specific_param.m_encoder.m_header_tile_data,
        l_tlm_size);
    if
      (! p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = l_tlm_size;
  }
  l_current_data = p_j2k->m_specific_param.m_encoder.m_header_tile_data;

  /* change the way data is written to avoid seeking if possible */
  // TODO
  p_j2k->m_specific_param.m_encoder.m_tlm_start = opj_stream_tell(p_stream);

  opj_write_bytes(l_current_data,J2K_MS_TLM,2);          /* TLM */
  l_current_data += 2;
  opj_write_bytes(l_current_data,l_tlm_size-2,2);          /* Lpoc */
  l_current_data += 2;
  opj_write_bytes(l_current_data,0,1);              /* Ztlm=0*/
  ++l_current_data;
  opj_write_bytes(l_current_data,0x50,1);              /* Stlm ST=1(8bits-255 tiles max),SP=1(Ptlm=32bits) */
  ++l_current_data;
  /* do nothing on the 5 * l_j2k->m_specific_param.m_encoder.m_total_tile_parts remaining data */

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,l_tlm_size,p_manager) != l_tlm_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_tlm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_Ztlm, l_Stlm, l_ST, l_SP, l_tot_num_tp, l_tot_num_tp_remaining, l_quotient, l_Ptlm_size;
  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size < 2)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading TLM marker\n");
    return false;
  }
  p_header_size -= 2;


  opj_read_bytes(p_header_data,&l_Ztlm,1);        /* Ztlm */
  ++p_header_data;
  opj_read_bytes(p_header_data,&l_Stlm,1);        /* Stlm */
  ++p_header_data;

  l_ST = ((l_Stlm >> 4) & 0x3);
  l_SP = (l_Stlm >> 6) & 0x1;

  l_Ptlm_size = (l_SP + 1) * 2;
  l_quotient = l_Ptlm_size + l_ST;

  l_tot_num_tp = p_header_size / l_quotient;
  l_tot_num_tp_remaining = p_header_size % l_quotient;
  if
    (l_tot_num_tp_remaining != 0)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading TLM marker\n");
    return false;
  }
  /* Do not care of this at the moment since only local variables are set here */
  /*
  for
    (i = 0; i < l_tot_num_tp; ++i)
  {
    opj_read_bytes(p_header_data,&l_Ttlm_i,l_ST);        // Ttlm_i
    p_header_data += l_ST;
    opj_read_bytes(p_header_data,&l_Ptlm_i,l_Ptlm_size);    // Ptlm_i
    p_header_data += l_Ptlm_size;
  }*/
  return true;
}

/**
 * Reads a CRG marker (Component registration)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_crg (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_nb_comp;
  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  l_nb_comp = p_j2k->m_image->numcomps;

  if
    (p_header_size != l_nb_comp *4)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading CRG marker\n");
    return false;
  }
  /* Do not care of this at the moment since only local variables are set here */
  /*
  for
    (i = 0; i < l_nb_comp; ++i)
  {
    opj_read_bytes(p_header_data,&l_Xcrg_i,2);        // Xcrg_i
    p_header_data+=2;
    opj_read_bytes(p_header_data,&l_Ycrg_i,2);        // Xcrg_i
    p_header_data+=2;
  }
  */
  return true;
}

/**
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param  p_header_data  the data contained in the TLM box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the TLM marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_plm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading PLM marker\n");
    return false;
  }
  /* Do not care of this at the moment since only local variables are set here */
  /*
  opj_read_bytes(p_header_data,&l_Zplm,1);          // Zplm
  ++p_header_data;
  --p_header_size;

  while
    (p_header_size > 0)
  {
    opj_read_bytes(p_header_data,&l_Nplm,1);        // Nplm
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
      opj_read_bytes(p_header_data,&l_tmp,1);        // Iplm_ij
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
  return true;
}

/**
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param  p_header_data  the data contained in the PLT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PLT marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_plt (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_Zplt, l_tmp, l_packet_len = 0, i;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading PLM marker\n");
    return false;
  }

  opj_read_bytes(p_header_data,&l_Zplt,1);          // Zplt
  ++p_header_data;
  --p_header_size;
  for
    (i = 0; i < p_header_size; ++i)
  {
    opj_read_bytes(p_header_data,&l_tmp,1);        // Iplm_ij
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
  return true;
}

/**
 * Reads a PPM marker (Packed packet headers, main header)
 *
 * @param  p_header_data  the data contained in the POC box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the POC marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_ppm (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{

  opj_cp_t *l_cp = 00;
  OPJ_UINT32 l_remaining_data, l_Z_ppm, l_N_ppm;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading RGN marker\n");
    return false;
  }
  l_cp = &(p_j2k->m_cp);
  l_cp->ppm = 1;

  opj_read_bytes(p_header_data,&l_Z_ppm,1);    /* Z_ppm */
  ++p_header_data;
  --p_header_size;

  // first PPM marker
  if
    (l_Z_ppm == 0)
  {
    if
      (p_header_size < 4)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Error reading PPM marker\n");
      return false;
    }
    // read a N_ppm
    opj_read_bytes(p_header_data,&l_N_ppm,4);    /* N_ppm */
    p_header_data+=4;
    p_header_size-=4;
    /* First PPM marker */
    l_cp->ppm_len = l_N_ppm;
    l_cp->ppm_data_size = 0;
    l_cp->ppm_buffer = (OPJ_BYTE *) opj_malloc(l_cp->ppm_len);
    l_cp->ppm_data = l_cp->ppm_buffer;
    if
      (l_cp->ppm_buffer == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory reading ppm marker\n");
      return false;
    }
    memset(l_cp->ppm_buffer,0,l_cp->ppm_len);
  }

  while
    (true)
  {
    if
      (l_cp->ppm_data_size == l_cp->ppm_len)
    {
      if
        (p_header_size >= 4)
      {
        // read a N_ppm
        opj_read_bytes(p_header_data,&l_N_ppm,4);    /* N_ppm */
        p_header_data+=4;
        p_header_size-=4;
        l_cp->ppm_len += l_N_ppm ;
        l_cp->ppm_buffer = (OPJ_BYTE *) opj_realloc(l_cp->ppm_buffer, l_cp->ppm_len);
        l_cp->ppm_data = l_cp->ppm_buffer;
        if
          (l_cp->ppm_buffer == 00)
        {
          opj_event_msg(p_manager, EVT_ERROR, "Not enough memory reading ppm marker\n");
          return false;
        }
        memset(l_cp->ppm_buffer+l_cp->ppm_data_size,0,l_N_ppm);
      }
      else
      {
        return false;
      }
    }
    l_remaining_data = l_cp->ppm_len - l_cp->ppm_data_size;
    if
      (l_remaining_data <= p_header_size)
    {
      /* we must store less information than available in the packet */
      memcpy(l_cp->ppm_buffer + l_cp->ppm_data_size , p_header_data , l_remaining_data);
      l_cp->ppm_data_size = l_cp->ppm_len;
      p_header_size -= l_remaining_data;
      p_header_data += l_remaining_data;
    }
    else
    {
      memcpy(l_cp->ppm_buffer + l_cp->ppm_data_size , p_header_data , p_header_size);
      l_cp->ppm_data_size += p_header_size;
      p_header_data += p_header_size;
      p_header_size = 0;
      break;
    }
  }
  return true;
}

/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param  p_header_data  the data contained in the PPT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PPT marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_ppt (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_Z_ppt;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size < 1)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading PPT marker\n");
    return false;
  }

  l_cp = &(p_j2k->m_cp);
  l_tcp = &(l_cp->tcps[p_j2k->m_current_tile_number]);
  l_tcp->ppt = 1;

  opj_read_bytes(p_header_data,&l_Z_ppt,1);    /* Z_ppt */
  ++p_header_data;
  --p_header_size;

  // first PPM marker
  if
    (l_Z_ppt == 0)
  {
    /* First PPM marker */
    l_tcp->ppt_len = p_header_size;
    l_tcp->ppt_data_size = 0;
    l_tcp->ppt_buffer = (OPJ_BYTE *) opj_malloc(l_tcp->ppt_len);
    l_tcp->ppt_data = l_tcp->ppt_buffer;
    if
      (l_tcp->ppt_buffer == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory reading PPT marker\n");
      return false;
    }
    memset(l_tcp->ppt_buffer,0,l_tcp->ppt_len);
  }
  else
  {
    l_tcp->ppt_len += p_header_size;
    l_tcp->ppt_buffer = (OPJ_BYTE *) opj_realloc(l_tcp->ppt_buffer,l_tcp->ppt_len);
    if
      (l_tcp->ppt_buffer == 00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Not enough memory reading PPT marker\n");
      return false;
    }
    l_tcp->ppt_data = l_tcp->ppt_buffer;
    memset(l_tcp->ppt_buffer+l_tcp->ppt_data_size,0,p_header_size);
  }
  memcpy(l_tcp->ppt_buffer+l_tcp->ppt_data_size,p_header_data,p_header_size);
  l_tcp->ppt_data_size += p_header_size;
  return true;
}

/**
 * Writes the SOT marker (Start of tile-part)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_sot(
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_data,
            OPJ_UINT32 * p_data_written,
            const struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  opj_write_bytes(p_data,J2K_MS_SOT,2);          /* SOT */
  p_data += 2;

  opj_write_bytes(p_data,10,2);              /* Lsot */
  p_data += 2;

  opj_write_bytes(p_data, p_j2k->m_current_tile_number,2);      /* Isot */
  p_data += 2;

  /* Psot  */
  p_data += 4;

  opj_write_bytes(p_data, p_j2k->m_specific_param.m_encoder.m_current_tile_part_number,1);      /* TPsot */
  ++p_data;

  opj_write_bytes(p_data, p_j2k->m_cp.tcps[p_j2k->m_current_tile_number].m_nb_tile_parts,1);      /* TNsot */
  ++p_data;
  /* UniPG>> */
#ifdef USE_JPWL
  /* update markers struct */
  j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOT, p_j2k->sot_start, len + 2);
#endif /* USE_JPWL */

  * p_data_written = 12;
  return true;
}

/**
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param  p_header_data  the data contained in the PPT box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the PPT marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_sot (
            opj_j2k_t *p_j2k,
            OPJ_BYTE * p_header_data,
            OPJ_UINT32 p_header_size,
            struct opj_event_mgr * p_manager
          )
{

  opj_cp_t *l_cp = 00;
  opj_tcp_t *l_tcp = 00;
  OPJ_UINT32 l_tot_len, l_num_parts = 0;
  OPJ_UINT32 l_current_part;
  OPJ_UINT32 l_tile_x,l_tile_y;

  // preconditions
  assert(p_header_data != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_header_size != 8)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Error reading SOT marker\n");
    return false;
  }
  l_cp = &(p_j2k->m_cp);
  opj_read_bytes(p_header_data,&(p_j2k->m_current_tile_number),2);    /* Isot */
  p_header_data+=2;


  l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
  l_tile_x = p_j2k->m_current_tile_number % l_cp->tw;
  l_tile_y = p_j2k->m_current_tile_number / l_cp->tw;

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {

    static int backup_tileno = 0;

    /* tileno is negative or larger than the number of tiles!!! */
    if ((tileno < 0) || (tileno > (cp->tw * cp->th))) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "JPWL: bad tile number (%d out of a maximum of %d)\n",
        tileno, (cp->tw * cp->th));
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      tileno = backup_tileno;
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
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

  opj_read_bytes(p_header_data,&l_tot_len,4);    /* Psot */
  p_header_data+=4;

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {

    /* totlen is negative or larger than the bytes left!!! */
    if ((totlen < 0) || (totlen > (p_stream_numbytesleft(p_stream) + 8))) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "JPWL: bad tile byte size (%d bytes against %d bytes left)\n",
        totlen, p_stream_numbytesleft(p_stream) + 8);
      if (!JPWL_ASSUME) {
        opj_event_msg(p_j2k->cinfo, EVT_ERROR, "JPWL: giving up\n");
        return;
      }
      /* we try to correct */
      totlen = 0;
      opj_event_msg(p_j2k->cinfo, EVT_WARNING, "- trying to adjust this\n"
        "- setting Psot to %d => assuming it is the last tile\n",
        totlen);
    }

  };
#endif /* USE_JPWL */

  if
    (!l_tot_len)
  {
    void* l_data = p_manager->m_error_data;
      assert( l_data );
    if( l_data )
      {
      OPJ_UINT32 **s = (OPJ_UINT32**)l_data;
      assert( s[1] == 0 );
      if( s[1] == 0 )
        {
        s[1] = &l_tot_len;
        }
      }
    opj_event_msg(p_manager, EVT_ERROR, "Cannot read data with no size known, giving up\n");
    assert( l_tot_len != 0 );
    if( !l_tot_len )
      return false;
  }

  opj_read_bytes(p_header_data,&l_current_part ,1);    /* Psot */
  ++p_header_data;

  opj_read_bytes(p_header_data,&l_num_parts ,1);    /* Psot */
  ++p_header_data;

  if
    (l_num_parts != 0)
  {
    l_tcp->m_nb_tile_parts = l_num_parts;
  }
  if
    (l_tcp->m_nb_tile_parts)
  {
    if
      (l_tcp->m_nb_tile_parts == (l_current_part + 1))
    {
      p_j2k->m_specific_param.m_decoder.m_can_decode = 1;
    }
  }
  p_j2k->m_specific_param.m_decoder.m_sot_length = l_tot_len - 12;
  p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_TPH;
  p_j2k->m_specific_param.m_decoder.m_skip_data =
      (l_tile_x < p_j2k->m_specific_param.m_decoder.m_start_tile_x)
    ||  (l_tile_x >= p_j2k->m_specific_param.m_decoder.m_end_tile_x)
    ||  (l_tile_y < p_j2k->m_specific_param.m_decoder.m_start_tile_y)
    ||  (l_tile_y >= p_j2k->m_specific_param.m_decoder.m_end_tile_y);
  /* Index */

  /* move this onto a separate method to call before reading any SOT */
  /*if
    TODO
    (p_j2k->cstr_info)
  {
    if
      (l_tcp->first)
    {
      if
        (tileno == 0)
      {
        p_j2k->cstr_info->main_head_end = p_stream_tell(p_stream) - 13;
      }
      p_j2k->cstr_info->tile[tileno].tileno = tileno;
      p_j2k->cstr_info->tile[tileno].start_pos = p_stream_tell(p_stream) - 12;
      p_j2k->cstr_info->tile[tileno].end_pos = p_j2k->cstr_info->tile[tileno].start_pos + totlen - 1;
      p_j2k->cstr_info->tile[tileno].num_tps = numparts;
      if
        (numparts)
      {
        p_j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(numparts * sizeof(opj_tp_info_t));
      }
      else
      {
        p_j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(10 * sizeof(opj_tp_info_t)); // Fixme (10)
      }
    }
    else
    {
      p_j2k->cstr_info->tile[tileno].end_pos += totlen;
    }
    p_j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos = p_stream_tell(p_stream) - 12;
    p_j2k->cstr_info->tile[tileno].tp[partno].tp_end_pos =
    p_j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos + totlen - 1;
  }*/
  return true;
}

/**
 * Writes the SOD marker (Start of data)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_sod(
            opj_j2k_t *p_j2k,
            struct opj_tcd * p_tile_coder,
            OPJ_BYTE * p_data,
            OPJ_UINT32 * p_data_written,
            OPJ_UINT32 p_total_data_size,
            const struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  opj_tcp_t *l_tcp = 00;
  opj_codestream_info_t *l_cstr_info = 00;
  opj_cp_t *l_cp = 00;

  OPJ_UINT32 l_size_tile;
  OPJ_UINT32 l_remaining_data;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  opj_write_bytes(p_data,J2K_MS_SOD,2);          /* SOD */
  p_data += 2;

  /* make room for the EOF marker */
  l_remaining_data =  p_total_data_size - 4;

  l_cp = &(p_j2k->m_cp);
  l_tcp = &l_cp->tcps[p_j2k->m_current_tile_number];
  l_cstr_info = p_j2k->cstr_info;

  /* update tile coder */
  p_tile_coder->tp_num = p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number ;
  p_tile_coder->cur_tp_num = p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
  l_size_tile = l_cp->th * l_cp->tw;

  /* INDEX >> */
  if
    (l_cstr_info)
  {
    if
      (!p_j2k->m_specific_param.m_encoder.m_current_tile_part_number )
    {
      //TODO cstr_info->tile[p_j2k->m_current_tile_number].end_header = p_stream_tell(p_stream) + p_j2k->pos_correction - 1;
      l_cstr_info->tile[p_j2k->m_current_tile_number].tileno = p_j2k->m_current_tile_number;
    }
    else
    {
      /*
      TODO
      if
        (cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno - 1].end_pos < p_stream_tell(p_stream))
      {
        cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno].start_pos = p_stream_tell(p_stream);
      }*/

    }
    /* UniPG>> */
#ifdef USE_JPWL
    /* update markers struct */
    j2k_add_marker(p_j2k->cstr_info, J2K_MS_SOD, p_j2k->sod_start, 2);
#endif /* USE_JPWL */
    /* <<UniPG */
  }
  /* << INDEX */

  if
    (p_j2k->m_specific_param.m_encoder.m_current_tile_part_number == 0)
  {
    p_tile_coder->tcd_image->tiles->packno = 0;
    if
      (l_cstr_info)
    {
      l_cstr_info->packno = 0;
    }
  }
  *p_data_written = 0;
  if
    (! tcd_encode_tile(p_tile_coder, p_j2k->m_current_tile_number, p_data, p_data_written, l_remaining_data , l_cstr_info))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot encode tile\n");
    return false;
  }
  *p_data_written += 2;
  return true;
}

/**
 * Updates the Tile Length Marker.
 */
void j2k_update_tlm (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 p_tile_part_size
           )
{
  opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current,p_j2k->m_current_tile_number,1);          /* PSOT */
  ++p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current;
  opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current,p_tile_part_size,4);          /* PSOT */
  p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current += 4;
}


/**
 * Reads a SOD marker (Start Of Data)
 *
 * @param  p_header_data  the data contained in the SOD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the SOD marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_sod (
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_current_read_size;
  opj_codestream_info_t * l_cstr_info = 00;
  OPJ_BYTE ** l_current_data = 00;
  opj_tcp_t * l_tcp = 00;
  OPJ_UINT32 * l_tile_len = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tcp = &(p_j2k->m_cp.tcps[p_j2k->m_current_tile_number]);
  p_j2k->m_specific_param.m_decoder.m_sot_length -= 2;
  l_cstr_info = p_j2k->cstr_info;

  l_current_data = &(l_tcp->m_data);
  l_tile_len = &l_tcp->m_data_size;

  if
    (! *l_current_data)
  {
    *l_current_data = (OPJ_BYTE*) my_opj_malloc(p_j2k->m_specific_param.m_decoder.m_sot_length);
  }
  else
  {
    *l_current_data = (OPJ_BYTE*) my_opj_realloc(*l_current_data, *l_tile_len + p_j2k->m_specific_param.m_decoder.m_sot_length);
  }
  if
    (*l_current_data == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile\n");
    return false;
  }

  /* Index */
  if
    (l_cstr_info)
  {
    OPJ_SIZE_T l_current_pos = opj_stream_tell(p_stream)-1;
    l_cstr_info->tile[p_j2k->m_current_tile_number].tp[p_j2k->m_specific_param.m_encoder.m_current_tile_part_number].tp_end_header = l_current_pos;
    if
      (p_j2k->m_specific_param.m_encoder.m_current_tile_part_number == 0)
    {
      l_cstr_info->tile[p_j2k->m_current_tile_number].end_header = l_current_pos;
    }
    l_cstr_info->packno = 0;
  }
  l_current_read_size = opj_stream_read_data(p_stream, *l_current_data + *l_tile_len , p_j2k->m_specific_param.m_decoder.m_sot_length,p_manager);
  if
    (l_current_read_size != p_j2k->m_specific_param.m_decoder.m_sot_length)
  {
    p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_NEOC;
  }
  else
  {
    p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_TPHSOT;
  }
  *l_tile_len +=  l_current_read_size;
  return true;
}

/**
 * Writes the EOC marker (End of Codestream)
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_eoc(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  opj_write_bytes(p_j2k->m_specific_param.m_encoder.m_header_tile_data,J2K_MS_EOC,2);          /* EOC */


/* UniPG>> */
#ifdef USE_JPWL
  /* update markers struct */
  j2k_add_marker(p_j2k->cstr_info, J2K_MS_EOC, p_stream_tell(p_stream) - 2, 2);
#endif /* USE_JPWL */

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_header_tile_data,2,p_manager) != 2)
  {
    return false;
  }
  if
    (! opj_stream_flush(p_stream,p_manager))
  {
    return false;
  }
  return true;
}


/**
 * Inits the Info
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_init_info(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  opj_codestream_info_t * l_cstr_info = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);
  l_cstr_info = p_j2k->cstr_info;

  if
    (l_cstr_info)
  {
    OPJ_UINT32 compno;
    l_cstr_info->tile = (opj_tile_info_t *) opj_malloc(p_j2k->m_cp.tw * p_j2k->m_cp.th * sizeof(opj_tile_info_t));
    l_cstr_info->image_w = p_j2k->m_image->x1 - p_j2k->m_image->x0;
    l_cstr_info->image_h = p_j2k->m_image->y1 - p_j2k->m_image->y0;
    l_cstr_info->prog = (&p_j2k->m_cp.tcps[0])->prg;
    l_cstr_info->tw = p_j2k->m_cp.tw;
    l_cstr_info->th = p_j2k->m_cp.th;
    l_cstr_info->tile_x = p_j2k->m_cp.tdx;  /* new version parser */
    l_cstr_info->tile_y = p_j2k->m_cp.tdy;  /* new version parser */
    l_cstr_info->tile_Ox = p_j2k->m_cp.tx0;  /* new version parser */
    l_cstr_info->tile_Oy = p_j2k->m_cp.ty0;  /* new version parser */
    l_cstr_info->numcomps = p_j2k->m_image->numcomps;
    l_cstr_info->numlayers = (&p_j2k->m_cp.tcps[0])->numlayers;
    l_cstr_info->numdecompos = (OPJ_INT32*) opj_malloc(p_j2k->m_image->numcomps * sizeof(OPJ_INT32));
    for (compno=0; compno < p_j2k->m_image->numcomps; compno++) {
      l_cstr_info->numdecompos[compno] = (&p_j2k->m_cp.tcps[0])->tccps->numresolutions - 1;
    }
    l_cstr_info->D_max = 0.0;    /* ADD Marcela */
    l_cstr_info->main_head_start = opj_stream_tell(p_stream); /* position of SOC */
    l_cstr_info->maxmarknum = 100;
    l_cstr_info->marker = (opj_marker_info_t *) opj_malloc(l_cstr_info->maxmarknum * sizeof(opj_marker_info_t));
    l_cstr_info->marknum = 0;
  }
  return j2k_calculate_tp(p_j2k,&(p_j2k->m_cp),&p_j2k->m_specific_param.m_encoder.m_total_tile_parts,p_j2k->m_image,p_manager);
}

/**
 * Creates a tile-coder decoder.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_create_tcd(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  p_j2k->m_tcd = tcd_create(false);
  if
    (! p_j2k->m_tcd)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Not enough memory to create Tile Coder\n");
    return false;
  }
  if
    (! tcd_init(p_j2k->m_tcd,p_j2k->m_image,&p_j2k->m_cp))
  {
    tcd_destroy(p_j2k->m_tcd);
    p_j2k->m_tcd = 00;
    return false;
  }
  return true;
}

OPJ_FLOAT32 get_tp_stride (opj_tcp_t * p_tcp)
{
  return (OPJ_FLOAT32) ((p_tcp->m_nb_tile_parts - 1) * 14);
}

OPJ_FLOAT32 get_default_stride (opj_tcp_t * p_tcp)
{
  return 0;
}

/**
 * Updates the rates of the tcp.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_update_rates(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  opj_cp_t * l_cp = 00;
  opj_image_t * l_image = 00;
  opj_tcp_t * l_tcp = 00;
  opj_image_comp_t * l_img_comp = 00;

  OPJ_UINT32 i,j,k;
  OPJ_INT32 l_x0,l_y0,l_x1,l_y1;
  OPJ_FLOAT32 * l_rates = 0;
  OPJ_FLOAT32 l_sot_remove;
  OPJ_UINT32 l_bits_empty, l_size_pixel;
  OPJ_UINT32 l_tile_size = 0;
  OPJ_UINT32 l_last_res;
  OPJ_FLOAT32 (* l_tp_stride_func)(opj_tcp_t *) = 00;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);


  l_cp = &(p_j2k->m_cp);
  l_image = p_j2k->m_image;
  l_tcp = l_cp->tcps;

  l_bits_empty = 8 * l_image->comps->dx * l_image->comps->dy;
  l_size_pixel = l_image->numcomps * l_image->comps->prec;
  l_sot_remove = ((OPJ_FLOAT32) opj_stream_tell(p_stream)) / (l_cp->th * l_cp->tw);

  if
    (l_cp->m_specific_param.m_enc.m_tp_on)
  {
    l_tp_stride_func = get_tp_stride;
  }
  else
  {
    l_tp_stride_func = get_default_stride;
  }

  for
    (i=0;i<l_cp->th;++i)
  {
    for
      (j=0;j<l_cp->tw;++j)
    {
      OPJ_FLOAT32 l_offset = ((*l_tp_stride_func)(l_tcp)) / l_tcp->numlayers;
      /* 4 borders of the tile rescale on the image if necessary */
      l_x0 = int_max(l_cp->tx0 + j * l_cp->tdx, l_image->x0);
      l_y0 = int_max(l_cp->ty0 + i * l_cp->tdy, l_image->y0);
      l_x1 = int_min(l_cp->tx0 + (j + 1) * l_cp->tdx, l_image->x1);
      l_y1 = int_min(l_cp->ty0 + (i + 1) * l_cp->tdy, l_image->y1);
      l_rates = l_tcp->rates;

      /* Modification of the RATE >> */
      if
        (*l_rates)
      {
        *l_rates =     (( (float) (l_size_pixel * (l_x1 - l_x0) * (l_y1 - l_y0)))
                /
                ((*l_rates) * l_bits_empty)
                )
                -
                l_offset;
      }
      ++l_rates;
      for
        (k = 1; k < l_tcp->numlayers; ++k)
      {
        if
          (*l_rates)
        {
          *l_rates =     (( (OPJ_FLOAT32) (l_size_pixel * (l_x1 - l_x0) * (l_y1 - l_y0)))
                  /
                    ((*l_rates) * l_bits_empty)
                  )
                  -
                  l_offset;
        }
        ++l_rates;
      }
      ++l_tcp;
    }
  }

  l_tcp = l_cp->tcps;
  for
    (i=0;i<l_cp->th;++i)
  {
    for
      (j=0;j<l_cp->tw;++j)
    {
      l_rates = l_tcp->rates;
      if
        (*l_rates)
      {
        *l_rates -= l_sot_remove;
        if
          (*l_rates < 30)
        {
          *l_rates = 30;
        }
      }
      ++l_rates;
      l_last_res = l_tcp->numlayers - 1;
      for
        (k = 1; k < l_last_res; ++k)
      {
        if
          (*l_rates)
        {
          *l_rates -= l_sot_remove;
          if
            (*l_rates < *(l_rates - 1) + 10)
          {
            *l_rates  = (*(l_rates - 1)) + 20;
          }
        }
        ++l_rates;
      }
      if
        (*l_rates)
      {
        *l_rates -= (l_sot_remove + 2.f);
        if
          (*l_rates < *(l_rates - 1) + 10)
        {
          *l_rates  = (*(l_rates - 1)) + 20;
        }
      }
      ++l_tcp;
    }
  }

  l_img_comp = l_image->comps;
  l_tile_size = 0;
  for
    (i=0;i<l_image->numcomps;++i)
  {
    l_tile_size += (    uint_ceildiv(l_cp->tdx,l_img_comp->dx)
             *
                uint_ceildiv(l_cp->tdy,l_img_comp->dy)
             *
                l_img_comp->prec
            );
    ++l_img_comp;
  }

  l_tile_size = (OPJ_UINT32) (l_tile_size * 0.1625); /* 1.3/8 = 0.1625 */
  l_tile_size += j2k_get_specific_header_sizes(p_j2k);

  p_j2k->m_specific_param.m_encoder.m_encoded_tile_size = l_tile_size;
  p_j2k->m_specific_param.m_encoder.m_encoded_tile_data = (OPJ_BYTE *) my_opj_malloc(p_j2k->m_specific_param.m_encoder.m_encoded_tile_size);
  if
    (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data == 00)
  {
    return false;
  }
  if
    (l_cp->m_specific_param.m_enc.m_cinema)
  {
    p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = (OPJ_BYTE *) opj_malloc(5*p_j2k->m_specific_param.m_encoder.m_total_tile_parts);
    if
      (! p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer)
    {
      return false;
    }
    p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current = p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer;
  }
  return true;
}

/**
 * Reads a EOC marker (End Of Codestream)
 *
 * @param  p_header_data  the data contained in the SOD box.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_header_size  the size of the data contained in the SOD marker.
 * @param  p_manager    the user event manager.
*/
bool j2k_read_eoc (
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 i;
  opj_tcd_t * l_tcd = 00;
  OPJ_UINT32 l_nb_tiles;
  opj_tcp_t * l_tcp = 00;
  bool l_success;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
  l_tcp = p_j2k->m_cp.tcps;

  l_tcd = tcd_create(true);
  if
    (l_tcd == 00)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
    return false;
  }



  for
    (i = 0; i < l_nb_tiles; ++i)
  {
    if
      (l_tcp->m_data)
    {
      if
        (! tcd_init_decode_tile(l_tcd, i))
      {
        tcd_destroy(l_tcd);
        opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
        return false;
      }
      l_success = tcd_decode_tile(l_tcd, l_tcp->m_data, l_tcp->m_data_size, i, p_j2k->cstr_info);
      /* cleanup */
      if
        (! l_success)
      {
        p_j2k->m_specific_param.m_decoder.m_state |= J2K_DEC_STATE_ERR;
        break;
      }
    }
    j2k_tcp_destroy(l_tcp);
    ++l_tcp;
  }
  tcd_destroy(l_tcd);
  return true;
}

/**
 * Writes the image components.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_image_components(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 compno;
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  for
    (compno = 1; compno < p_j2k->m_image->numcomps; ++compno)
  {
    if
      (! j2k_write_coc(p_j2k,compno,p_stream, p_manager))
    {
      return false;
    }
    if
      (! j2k_write_qcc(p_j2k,compno,p_stream, p_manager))
    {
      return false;
    }
  }
  return true;
}

/**
 * Writes regions of interests.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_regions(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 compno;
  const opj_tccp_t *l_tccp = 00;
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tccp = p_j2k->m_cp.tcps->tccps;
  for
    (compno = 0; compno < p_j2k->m_image->numcomps; ++compno)
  {
    if
      (l_tccp->roishift)
    {
      if
        (! j2k_write_rgn(p_j2k,0,compno,p_stream,p_manager))
      {
        return false;
      }
    }
    ++l_tccp;
  }
  return true;
}
/**
 * Writes the updated tlm.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_write_updated_tlm(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_UINT32 l_tlm_size;
  OPJ_SIZE_T l_tlm_position, l_current_position;

  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  l_tlm_size = 5 * p_j2k->m_specific_param.m_encoder.m_total_tile_parts;
  l_tlm_position = 6 + p_j2k->m_specific_param.m_encoder.m_tlm_start;
  l_current_position = opj_stream_tell(p_stream);

  if
    (! opj_stream_seek(p_stream,l_tlm_position,p_manager))
  {
    return false;
  }
  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer,l_tlm_size,p_manager) != l_tlm_size)
  {
    return false;
  }
  if
    (! opj_stream_seek(p_stream,l_current_position,p_manager))
  {
    return false;
  }
  return true;
}

/**
 * Ends the encoding, i.e. frees memory.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_end_encoding(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  tcd_destroy(p_j2k->m_tcd);
  p_j2k->m_tcd = 00;

  if
    (p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer)
  {
    opj_free(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer);
    p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = 0;
    p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current = 0;
  }
  if
    (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data)
  {
    opj_free(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);
    p_j2k->m_specific_param.m_encoder.m_encoded_tile_data = 0;
  }
  p_j2k->m_specific_param.m_encoder.m_encoded_tile_size = 0;

  return true;
}

/**
 * Gets the offset of the header.
 *
 * @param  p_stream        the stream to write data to.
 * @param  p_j2k        J2K codec.
 * @param  p_manager    the user event manager.
*/
bool j2k_get_end_header(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  p_j2k->cstr_info->main_head_end = opj_stream_tell(p_stream);
  return true;
}




/**
 * Reads an unknown marker
 *
 * @param  p_stream        the stream object to read from.
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_manager    the user event manager.
 *
 * @return  true      if the marker could be deduced.
*/
bool j2k_read_unk (
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager
          )
{
  OPJ_BYTE l_data [2];
  OPJ_UINT32 l_unknown_size;
  // preconditions
  assert(p_j2k != 00);
  assert(p_manager != 00);
  assert(p_stream != 00);

  opj_event_msg(p_manager, EVT_WARNING, "Unknown marker\n");

#ifdef USE_JPWL
  if (p_j2k->m_cp->correct) {
    OPJ_INT32 m = 0, id, i;
    OPJ_INT32 min_id = 0, min_dist = 17, cur_dist = 0, tmp_id;
    p_stream_seek(p_j2k->p_stream, p_stream_tell(p_j2k->p_stream) - 2);
    id = p_stream_read(p_j2k->p_stream, 2);
    opj_event_msg(p_j2k->cinfo, EVT_ERROR,
      "JPWL: really don't know this marker %x\n",
      id);
    if (!JPWL_ASSUME) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "- possible synch loss due to uncorrectable codestream errors => giving up\n");
      return;
    }
    /* OK, activate this at your own risk!!! */
    /* we look for the marker at the minimum hamming distance from this */
    while (j2k_dec_mstab[m].id) {

      /* 1's where they differ */
      tmp_id = j2k_dec_mstab[m].id ^ id;

      /* compute the hamming distance between our id and the current */
      cur_dist = 0;
      for (i = 0; i < 16; i++) {
        if ((tmp_id >> i) & 0x0001) {
          cur_dist++;
        }
      }

      /* if current distance is smaller, set the minimum */
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        min_id = j2k_dec_mstab[m].id;
      }

      /* jump to the next marker */
      m++;
    }

    /* do we substitute the marker? */
    if (min_dist < JPWL_MAXIMUM_HAMMING) {
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "- marker %x is at distance %d from the read %x\n",
        min_id, min_dist, id);
      opj_event_msg(p_j2k->cinfo, EVT_ERROR,
        "- trying to substitute in place and crossing fingers!\n");
      p_stream_seek(p_j2k->p_stream, p_stream_tell(p_j2k->p_stream) - 2);
      p_stream_write(p_j2k->p_stream, min_id, 2);

      /* rewind */
      p_stream_seek(p_j2k->p_stream, p_stream_tell(p_j2k->p_stream) - 2);

    }

  };
#endif /* USE_JPWL */
  if
    (opj_stream_read_data(p_stream,l_data,2,p_manager) != 2)
  {
    opj_event_msg(p_manager, EVT_WARNING, "Unknown marker\n");
    return false;
  }
  opj_read_bytes(l_data,&l_unknown_size,2);
  if
    (l_unknown_size < 2)
  {
    return false;
  }
  l_unknown_size-=2;

  if
    (opj_stream_skip(p_stream,l_unknown_size,p_manager) != l_unknown_size)
  {
    return false;
  }
  return true;
}

/**
 * Reads the lookup table containing all the marker, status and action, and returns the handler associated
 * with the marker value.
 * @param  p_id    Marker value to look up
 *
 * @return  the handler associated with the id.
*/
const opj_dec_memory_marker_handler_t * j2k_get_marker_handler (OPJ_UINT32 p_id)
{
  const opj_dec_memory_marker_handler_t *e;
  for
    (e = j2k_memory_marker_handler_tab; e->id != 0; ++e)
  {
    if
      (e->id == p_id)
    {
      break;
    }
  }
  return e;
}

/**
 * Destroys a tile coding parameter structure.
 *
 * @param  p_tcp    the tile coding parameter to destroy.
 */
void j2k_tcp_destroy (opj_tcp_t *p_tcp)
{
  if
    (p_tcp == 00)
  {
    return;
  }
  if
    (p_tcp->ppt_buffer != 00)
  {
    opj_free(p_tcp->ppt_buffer);
    p_tcp->ppt_buffer = 00;
  }
  if
    (p_tcp->tccps != 00)
  {
    opj_free(p_tcp->tccps);
    p_tcp->tccps = 00;
  }
  if
    (p_tcp->m_mct_coding_matrix != 00)
  {
    opj_free(p_tcp->m_mct_coding_matrix);
    p_tcp->m_mct_coding_matrix = 00;
  }
  if
    (p_tcp->m_mct_decoding_matrix != 00)
  {
    opj_free(p_tcp->m_mct_decoding_matrix);
    p_tcp->m_mct_decoding_matrix = 00;
  }
  if
    (p_tcp->m_mcc_records)
  {
    opj_free(p_tcp->m_mcc_records);
    p_tcp->m_mcc_records = 00;
    p_tcp->m_nb_max_mcc_records = 0;
    p_tcp->m_nb_mcc_records = 0;
  }
  if
    (p_tcp->m_mct_records)
  {
    opj_mct_data_t * l_mct_data = p_tcp->m_mct_records;
    OPJ_UINT32 i;
    for
      (i=0;i<p_tcp->m_nb_mct_records;++i)
    {
      if
        (l_mct_data->m_data)
      {
        opj_free(l_mct_data->m_data);
        l_mct_data->m_data = 00;
      }
      ++l_mct_data;
    }
    opj_free(p_tcp->m_mct_records);
    p_tcp->m_mct_records = 00;
  }

  if
    (p_tcp->mct_norms != 00)
  {
    opj_free(p_tcp->mct_norms);
    p_tcp->mct_norms = 00;
  }
  if
    (p_tcp->m_data)
  {
    opj_free(p_tcp->m_data);
    p_tcp->m_data = 00;
  }
}

/**
 * Destroys a coding parameter structure.
 *
 * @param  p_cp    the coding parameter to destroy.
 */
void j2k_cp_destroy (opj_cp_t *p_cp)
{
  OPJ_UINT32 l_nb_tiles;
  opj_tcp_t * l_current_tile = 00;
  OPJ_UINT32 i;

  if
    (p_cp == 00)
  {
    return;
  }
  if
    (p_cp->tcps != 00)
  {
    l_current_tile = p_cp->tcps;
    l_nb_tiles = p_cp->th * p_cp->tw;

    for
      (i = 0; i < l_nb_tiles; ++i)
    {
      j2k_tcp_destroy(l_current_tile);
      ++l_current_tile;
    }
    opj_free(p_cp->tcps);
    p_cp->tcps = 00;
  }
  if
    (p_cp->ppm_buffer != 00)
  {
    opj_free(p_cp->ppm_buffer);
    p_cp->ppm_buffer = 00;
  }
  if
    (p_cp->comment != 00)
  {
    opj_free(p_cp->comment);
    p_cp->comment = 00;
  }
  if
    (! p_cp->m_is_decoder)
  {
    if
      (p_cp->m_specific_param.m_enc.m_matrice)
    {
      opj_free(p_cp->m_specific_param.m_enc.m_matrice);
      p_cp->m_specific_param.m_enc.m_matrice = 00;
    }
  }
}

/* ----------------------------------------------------------------------- */
/* J2K / JPT decoder interface                                             */
/* ----------------------------------------------------------------------- */
/**
 * Creates a J2K decompression structure.
 *
 * @return a handle to a J2K decompressor if successful, NULL otherwise.
*/
opj_j2k_t* j2k_create_decompress()
{
  opj_j2k_t *l_j2k = (opj_j2k_t*) opj_malloc(sizeof(opj_j2k_t));
  if
    (!l_j2k)
  {
    return 00;
  }
  memset(l_j2k,0,sizeof(opj_j2k_t));
  l_j2k->m_is_decoder = 1;
  l_j2k->m_cp.m_is_decoder = 1;
  l_j2k->m_specific_param.m_decoder.m_default_tcp = (opj_tcp_t*) opj_malloc(sizeof(opj_tcp_t));
  if
    (!l_j2k->m_specific_param.m_decoder.m_default_tcp)
  {
    opj_free(l_j2k);
    return 00;
  }
  memset(l_j2k->m_specific_param.m_decoder.m_default_tcp,0,sizeof(opj_tcp_t));

  l_j2k->m_specific_param.m_decoder.m_header_data = (OPJ_BYTE *) opj_malloc(J2K_DEFAULT_HEADER_SIZE);
  if
    (! l_j2k->m_specific_param.m_decoder.m_header_data)
  {
    j2k_destroy(l_j2k);
    return 00;
  }
  l_j2k->m_specific_param.m_decoder.m_header_data_size = J2K_DEFAULT_HEADER_SIZE;

  // validation list creation
  l_j2k->m_validation_list = opj_procedure_list_create();
  if
    (! l_j2k->m_validation_list)
  {
    j2k_destroy(l_j2k);
    return 00;
  }

  // execution list creation
  l_j2k->m_procedure_list = opj_procedure_list_create();
  if
    (! l_j2k->m_procedure_list)
  {
    j2k_destroy(l_j2k);
    return 00;
  }
  return l_j2k;
}

opj_j2k_t* j2k_create_compress()
{
  opj_j2k_t *l_j2k = (opj_j2k_t*) opj_malloc(sizeof(opj_j2k_t));
  if
    (!l_j2k)
  {
    return 00;
  }
  memset(l_j2k,0,sizeof(opj_j2k_t));
  l_j2k->m_is_decoder = 0;
  l_j2k->m_cp.m_is_decoder = 0;

  l_j2k->m_specific_param.m_encoder.m_header_tile_data = (OPJ_BYTE *) opj_malloc(J2K_DEFAULT_HEADER_SIZE);
  if
    (! l_j2k->m_specific_param.m_encoder.m_header_tile_data)
  {
    j2k_destroy(l_j2k);
    return 00;
  }
  l_j2k->m_specific_param.m_encoder.m_header_tile_data_size = J2K_DEFAULT_HEADER_SIZE;

  // validation list creation
  l_j2k->m_validation_list = opj_procedure_list_create();
  if
    (! l_j2k->m_validation_list)
  {
    j2k_destroy(l_j2k);
    return 00;
  }

  // execution list creation
  l_j2k->m_procedure_list = opj_procedure_list_create();
  if
    (! l_j2k->m_procedure_list)
  {
    j2k_destroy(l_j2k);
    return 00;
  }
  return l_j2k;
}


/**
 * Destroys a jpeg2000 codec.
 *
 * @param  p_j2k  the jpeg20000 structure to destroy.
 */
void j2k_destroy (opj_j2k_t *p_j2k)
{
  if
    (p_j2k == 00)
  {
    return;
  }

  if
    (p_j2k->m_is_decoder)
  {
    if
      (p_j2k->m_specific_param.m_decoder.m_default_tcp != 00)
    {
      j2k_tcp_destroy(p_j2k->m_specific_param.m_decoder.m_default_tcp);
      opj_free(p_j2k->m_specific_param.m_decoder.m_default_tcp);
      p_j2k->m_specific_param.m_decoder.m_default_tcp = 00;
    }
    if
      (p_j2k->m_specific_param.m_decoder.m_header_data != 00)
    {
      opj_free(p_j2k->m_specific_param.m_decoder.m_header_data);
      p_j2k->m_specific_param.m_decoder.m_header_data = 00;
      p_j2k->m_specific_param.m_decoder.m_header_data_size = 0;
    }

  }
  else
  {
    if
      (p_j2k->m_specific_param.m_encoder.m_encoded_tile_data)
    {
      opj_free(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);
      p_j2k->m_specific_param.m_encoder.m_encoded_tile_data = 00;
    }
    if
      (p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer)
    {
      opj_free(p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer);
      p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = 00;
      p_j2k->m_specific_param.m_encoder.m_tlm_sot_offsets_current = 00;
    }
    if
      (p_j2k->m_specific_param.m_encoder.m_header_tile_data)
    {
      opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
      p_j2k->m_specific_param.m_encoder.m_header_tile_data = 00;
      p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
    }
  }
  tcd_destroy(p_j2k->m_tcd);

  j2k_cp_destroy(&(p_j2k->m_cp));
  memset(&(p_j2k->m_cp),0,sizeof(opj_cp_t));

  opj_procedure_list_destroy(p_j2k->m_procedure_list);
  p_j2k->m_procedure_list = 00;

  opj_procedure_list_destroy(p_j2k->m_validation_list);
  p_j2k->m_procedure_list = 00;

  opj_free(p_j2k);
}

/**
 * Starts a compression scheme, i.e. validates the codec parameters, writes the header.
 *
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream object.
 * @param  p_manager  the user event manager.
 *
 * @return true if the codec is valid.
 */
bool j2k_start_compress(
            opj_j2k_t *p_j2k,
            opj_stream_private_t *p_stream,
            opj_image_t * p_image,
            opj_event_mgr_t * p_manager)
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);
  p_j2k->m_image = p_image;


  /* customization of the validation */
  j2k_setup_encoding_validation (p_j2k);

  /* validation of the parameters codec */
  if
    (! j2k_exec(p_j2k,p_j2k->m_validation_list,p_stream,p_manager))
  {
    return false;
  }

  /* customization of the encoding */
  j2k_setup_header_writting(p_j2k);

  /* write header */
  if
    (! j2k_exec (p_j2k,p_j2k->m_procedure_list,p_stream,p_manager))
  {
    return false;
  }
  return true;
}
/**
 * Sets up the procedures to do on reading header. Developpers wanting to extend the library can add their own reading procedures.
 */
void j2k_setup_header_reading (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_read_header_procedure);

  /* DEVELOPER CORNER, add your custom procedures */
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_copy_default_tcp_and_create_tcd);

}

/**
 * Sets up the procedures to do on decoding data. Developpers wanting to extend the library can add their own reading procedures.
 */
void j2k_setup_decoding (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);

  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_decode_tiles);
  /* DEVELOPER CORNER, add your custom procedures */

}

/**
 * Sets up the procedures to do on writting header. Developpers wanting to extend the library can add their own writting procedures.
 */
void j2k_setup_header_writting (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_init_info );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_soc );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_siz );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_cod );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_qcd );


  if
    (p_j2k->m_cp.m_specific_param.m_enc.m_cinema)
  {
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_image_components );
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_tlm );
    if
      (p_j2k->m_cp.m_specific_param.m_enc.m_cinema == CINEMA4K_24)
    {
      opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_poc );
    }
  }
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_regions);

  if
    (p_j2k->m_cp.comment != 00)
  {
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_com);
  }

  /* DEVELOPER CORNER, insert your custom procedures */
  if
    (p_j2k->m_cp.rsiz & MCT)
  {
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_mct_data_group );
  }
  /* End of Developer Corner */

  if
    (p_j2k->cstr_info)
  {
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_get_end_header );
  }
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_create_tcd);
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_update_rates);
}

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
void j2k_setup_end_compress (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);

  /* DEVELOPER CORNER, insert your custom procedures */
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_eoc );
  if
    (p_j2k->m_cp.m_specific_param.m_enc.m_cinema)
  {
    opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_updated_tlm);
  }
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_write_epc );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_end_encoding );
  opj_procedure_list_add_procedure(p_j2k->m_procedure_list,(void*)j2k_destroy_header_memory);
}



/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
void j2k_setup_encoding_validation (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);
  opj_procedure_list_add_procedure(p_j2k->m_validation_list, (void*)j2k_build_encoder);
  opj_procedure_list_add_procedure(p_j2k->m_validation_list, (void*)j2k_encoding_validation);


  /* DEVELOPER CORNER, add your custom validation procedure */
  opj_procedure_list_add_procedure(p_j2k->m_validation_list, (void*)j2k_mct_validation);
}

/**
 * Sets up the validation ,i.e. adds the procedures to lauch to make sure the codec parameters
 * are valid. Developpers wanting to extend the library can add their own validation procedures.
 */
void j2k_setup_decoding_validation (opj_j2k_t *p_j2k)
{
  // preconditions
  assert(p_j2k != 00);
  opj_procedure_list_add_procedure(p_j2k->m_validation_list, (void*)j2k_build_decoder);
  opj_procedure_list_add_procedure(p_j2k->m_validation_list, (void*)j2k_decoding_validation);
  /* DEVELOPER CORNER, add your custom validation procedure */

}


/**
 * Excutes the given procedures on the given codec.
 *
 * @param  p_procedure_list  the list of procedures to execute
 * @param  p_j2k          the jpeg2000 codec to execute the procedures on.
 * @param  p_stream          the stream to execute the procedures on.
 * @param  p_manager      the user manager.
 *
 * @return  true        if all the procedures were successfully executed.
 */
bool j2k_exec (
          opj_j2k_t * p_j2k,
          opj_procedure_list_t * p_procedure_list,
          opj_stream_private_t *p_stream,
          opj_event_mgr_t * p_manager
          )
{
  bool (** l_procedure) (opj_j2k_t * ,opj_stream_private_t *,opj_event_mgr_t *) = 00;
  bool l_result = true;
  OPJ_UINT32 l_nb_proc, i;

  // preconditions
  assert(p_procedure_list != 00);
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  l_nb_proc = opj_procedure_list_get_nb_procedures(p_procedure_list);
  l_procedure = (bool (**) (opj_j2k_t * ,opj_stream_private_t *,opj_event_mgr_t *)) opj_procedure_list_get_first_procedure(p_procedure_list);
  for
    (i=0;i<l_nb_proc;++i)
  {
    l_result = l_result && ((*l_procedure) (p_j2k,p_stream,p_manager));
    ++l_procedure;
  }
  // and clear the procedure list at the end.
  opj_procedure_list_clear(p_procedure_list);
  return l_result;
}

/**
 * The default encoding validation procedure without any extension.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_encoding_validation (
                opj_j2k_t * p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
              )
{
  bool l_is_valid = true;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  /* STATE checking */
  /* make sure the state is at 0 */
  l_is_valid &= (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_NONE);

  /* POINTER validation */
  /* make sure a p_j2k codec is present */
  l_is_valid &= (p_j2k->m_procedure_list != 00);
  /* make sure a validation list is present */
  l_is_valid &= (p_j2k->m_validation_list != 00);

  if
    ((p_j2k->m_cp.tdx) < (OPJ_UINT32) (1 << p_j2k->m_cp.tcps->tccps->numresolutions))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Number of resolutions is too high in comparison to the size of tiles\n");
    return false;
  }
  if
    ((p_j2k->m_cp.tdy) < (OPJ_UINT32) (1 << p_j2k->m_cp.tcps->tccps->numresolutions))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Number of resolutions is too high in comparison to the size of tiles\n");
    return false;
  }

  /* PARAMETER VALIDATION */
  return l_is_valid;
}

/**
 * The default decoding validation procedure without any extension.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_decoding_validation (
                opj_j2k_t *p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
                )
{
  bool l_is_valid = true;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  /* STATE checking */
  /* make sure the state is at 0 */
  l_is_valid &= (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_NONE);

  /* POINTER validation */
  /* make sure a p_j2k codec is present */
  /* make sure a procedure list is present */
  l_is_valid &= (p_j2k->m_procedure_list != 00);
  /* make sure a validation list is present */
  l_is_valid &= (p_j2k->m_validation_list != 00);

  /* PARAMETER VALIDATION */
  return l_is_valid;
}

/**
 * The mct encoding validation procedure.
 *
 * @param  p_j2k      the jpeg2000 codec to validate.
 * @param  p_stream        the input stream to validate.
 * @param  p_manager    the user event manager.
 *
 * @return true if the parameters are correct.
 */
bool j2k_mct_validation (
                opj_j2k_t * p_j2k,
                opj_stream_private_t *p_stream,
                opj_event_mgr_t * p_manager
              )
{
  bool l_is_valid = true;
  OPJ_UINT32 i,j;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  if
    ((p_j2k->m_cp.rsiz & 0x8200) == 0x8200)
  {
    OPJ_UINT32 l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
    opj_tcp_t * l_tcp = p_j2k->m_cp.tcps;
    for
      (i=0;i<l_nb_tiles;++i)
    {
      if
        (l_tcp->mct == 2)
      {
        opj_tccp_t * l_tccp = l_tcp->tccps;
        l_is_valid &= (l_tcp->m_mct_coding_matrix != 00);
        for
          (j=0;j<p_j2k->m_image->numcomps;++j)
        {
          l_is_valid &= ! (l_tccp->qmfbid & 1);
          ++l_tccp;
        }
      }
      ++l_tcp;
    }
  }
  return l_is_valid;
}

/**
 * Builds the cp decoder parameters to use to decode tile.
 */
bool j2k_build_decoder (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            )
{
  // add here initialization of cp
  // copy paste of setup_decoder
  return true;
}

/**
 * Builds the cp encoder parameters to use to encode tile.
 */
bool j2k_build_encoder (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            )
{
  // add here initialization of cp
  // copy paste of setup_encoder
  return true;
}

bool j2k_copy_default_tcp_and_create_tcd
            (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            )
{
  opj_tcp_t * l_tcp = 00;
  opj_tcp_t * l_default_tcp = 00;
  OPJ_UINT32 l_nb_tiles;
  OPJ_UINT32 i,j;
  opj_tccp_t *l_current_tccp = 00;
  OPJ_UINT32 l_tccp_size;
  OPJ_UINT32 l_mct_size;
  opj_image_t * l_image;
  OPJ_UINT32 l_mcc_records_size,l_mct_records_size;
  opj_mct_data_t * l_src_mct_rec, *l_dest_mct_rec;
  opj_simple_mcc_decorrelation_data_t * l_src_mcc_rec, *l_dest_mcc_rec;
  OPJ_UINT32 l_offset;

  // preconditions in debug
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  l_image = p_j2k->m_image;
  l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
  l_tcp = p_j2k->m_cp.tcps;
  l_tccp_size = l_image->numcomps * sizeof(opj_tccp_t);
  l_default_tcp = p_j2k->m_specific_param.m_decoder.m_default_tcp;
  l_mct_size = l_image->numcomps * l_image->numcomps * sizeof(OPJ_FLOAT32);
  for
    (i=0;i<l_nb_tiles;++i)
  {
    l_current_tccp = l_tcp->tccps;
    memcpy(l_tcp,l_default_tcp, sizeof(opj_tcp_t));
    l_tcp->ppt = 0;
    l_tcp->ppt_data = 00;
    l_tcp->tccps = l_current_tccp;
    if
      (l_default_tcp->m_mct_decoding_matrix)
    {
      l_tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(l_mct_size);
      if
        (! l_tcp->m_mct_decoding_matrix )
      {
        return false;
      }
      memcpy(l_tcp->m_mct_decoding_matrix,l_default_tcp->m_mct_decoding_matrix,l_mct_size);
    }
    l_mct_records_size = l_default_tcp->m_nb_max_mct_records * sizeof(opj_mct_data_t);
    l_tcp->m_mct_records = (opj_mct_data_t*)opj_malloc(l_mct_records_size);
    if
      (! l_tcp->m_mct_records)
    {
      return false;
    }
    memcpy(l_tcp->m_mct_records, l_default_tcp->m_mct_records,l_mct_records_size);
    l_src_mct_rec = l_default_tcp->m_mct_records;
    l_dest_mct_rec = l_tcp->m_mct_records;
    for
      (j=0;j<l_default_tcp->m_nb_mct_records;++j)
    {
      if
        (l_src_mct_rec->m_data)
      {
        l_dest_mct_rec->m_data = (OPJ_BYTE*)
        opj_malloc(l_src_mct_rec->m_data_size);
        if
          (! l_dest_mct_rec->m_data)
        {
          return false;
        }
        memcpy(l_dest_mct_rec->m_data,l_src_mct_rec->m_data,l_src_mct_rec->m_data_size);
      }
      ++l_src_mct_rec;
      ++l_dest_mct_rec;
    }
    l_mcc_records_size = l_default_tcp->m_nb_max_mcc_records * sizeof(opj_simple_mcc_decorrelation_data_t);
    l_tcp->m_mcc_records = (opj_simple_mcc_decorrelation_data_t*)
    opj_malloc(l_mcc_records_size);
    if
      (! l_tcp->m_mcc_records)
    {
      return false;
    }
    memcpy(l_tcp->m_mcc_records,l_default_tcp->m_mcc_records,l_mcc_records_size);
    l_src_mcc_rec = l_default_tcp->m_mcc_records;
    l_dest_mcc_rec = l_tcp->m_mcc_records;
    for
      (j=0;j<l_default_tcp->m_nb_max_mcc_records;++j)
    {
      if
        (l_src_mcc_rec->m_decorrelation_array)
      {
        l_offset = l_src_mcc_rec->m_decorrelation_array - l_default_tcp->m_mct_records;
        l_dest_mcc_rec->m_decorrelation_array = l_tcp->m_mct_records + l_offset;
      }
      if
        (l_src_mcc_rec->m_offset_array)
      {
        l_offset = l_src_mcc_rec->m_offset_array - l_default_tcp->m_mct_records;
        l_dest_mcc_rec->m_offset_array = l_tcp->m_mct_records + l_offset;
      }
      ++l_src_mcc_rec;
      ++l_dest_mcc_rec;
    }
    memcpy(l_current_tccp,l_default_tcp->tccps,l_tccp_size);
    ++l_tcp;
  }
  p_j2k->m_tcd = tcd_create(true);
  if
    (! p_j2k->m_tcd )
  {
    return false;
  }
  if
    (! tcd_init(p_j2k->m_tcd, l_image, &(p_j2k->m_cp)))
  {
    tcd_destroy(p_j2k->m_tcd);
    p_j2k->m_tcd = 00;
    opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
    return false;
  }
  return true;
}

/**
 * Destroys the memory associated with the decoding of headers.
 */
bool j2k_destroy_header_memory (
            opj_j2k_t * p_j2k,
            opj_stream_private_t *p_stream,
            opj_event_mgr_t * p_manager
            )
{
  // preconditions in debug
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  if
    (p_j2k->m_specific_param.m_encoder.m_header_tile_data)
  {
    opj_free(p_j2k->m_specific_param.m_encoder.m_header_tile_data);
    p_j2k->m_specific_param.m_encoder.m_header_tile_data = 0;
  }
  p_j2k->m_specific_param.m_encoder.m_header_tile_data_size = 0;
  return true;
}

/**
 * Sets up the decoder decoding parameters using user parameters.
 * Decoding parameters are stored in p_j2k->m_cp.
 *
 * @param  p_j2k      J2K codec
 * @param  p_parameters  decompression parameters
 * @deprecated
*/
void j2k_setup_decoder(
             opj_j2k_t *p_j2k,
             opj_dparameters_t *p_parameters
             )
{
  if
    (p_j2k && p_parameters)
  {
    /* create and initialize the coding parameters structure */
    p_j2k->m_cp.m_specific_param.m_dec.m_reduce = p_parameters->cp_reduce;
    p_j2k->m_cp.m_specific_param.m_dec.m_layer = p_parameters->cp_layer;
    p_j2k->m_specific_param.m_decoder.m_discard_tiles = p_parameters->m_use_restrict_decode;
    if
      (p_parameters->m_use_restrict_decode)
    {
      p_j2k->m_specific_param.m_decoder.m_start_tile_x = p_parameters->m_decode_start_x;
      p_j2k->m_specific_param.m_decoder.m_start_tile_y = p_parameters->m_decode_start_y;
      p_j2k->m_specific_param.m_decoder.m_end_tile_x = p_parameters->m_decode_end_x;
      p_j2k->m_specific_param.m_decoder.m_end_tile_y = p_parameters->m_decode_end_y;
    }

#ifdef USE_JPWL
    cp->correct = parameters->jpwl_correct;
    cp->exp_comps = parameters->jpwl_exp_comps;
    cp->max_tiles = parameters->jpwl_max_tiles;
#endif /* USE_JPWL */
  }
}

void j2k_setup_encoder(opj_j2k_t *p_j2k, opj_cparameters_t *parameters, opj_image_t *image, struct opj_event_mgr * p_manager) {
  OPJ_UINT32 i, j, tileno, numpocs_tile;
  opj_cp_t *cp = 00;
  bool l_res;
  if(!p_j2k || !parameters || ! image) {
    return;
  }

  /* keep a link to cp so that we can destroy it later in j2k_destroy_compress */
  cp = &(p_j2k->m_cp);

  /* set default values for cp */
  cp->tw = 1;
  cp->th = 1;

  /*
  copy user encoding parameters
  */
  cp->m_specific_param.m_enc.m_cinema = parameters->cp_cinema;
  cp->m_specific_param.m_enc.m_max_comp_size =  parameters->max_comp_size;
  cp->rsiz   = parameters->cp_rsiz;
  cp->m_specific_param.m_enc.m_disto_alloc = parameters->cp_disto_alloc;
  cp->m_specific_param.m_enc.m_fixed_alloc = parameters->cp_fixed_alloc;
  cp->m_specific_param.m_enc.m_fixed_quality = parameters->cp_fixed_quality;

  /* mod fixed_quality */
  if
    (parameters->cp_matrice)
  {
    size_t array_size = parameters->tcp_numlayers * parameters->numresolution * 3 * sizeof(OPJ_INT32);
    cp->m_specific_param.m_enc.m_matrice = (OPJ_INT32 *) opj_malloc(array_size);
    memcpy(cp->m_specific_param.m_enc.m_matrice, parameters->cp_matrice, array_size);
  }

  /* tiles */
  cp->tdx = parameters->cp_tdx;
  cp->tdy = parameters->cp_tdy;

  /* tile offset */
  cp->tx0 = parameters->cp_tx0;
  cp->ty0 = parameters->cp_ty0;

  /* comment string */
  if(parameters->cp_comment) {
    cp->comment = (char*)opj_malloc(strlen(parameters->cp_comment) + 1);
    if(cp->comment) {
      strcpy(cp->comment, parameters->cp_comment);
    }
  }

  /*
  calculate other encoding parameters
  */

  if (parameters->tile_size_on) {
    cp->tw = int_ceildiv(image->x1 - cp->tx0, cp->tdx);
    cp->th = int_ceildiv(image->y1 - cp->ty0, cp->tdy);
  } else {
    cp->tdx = image->x1 - cp->tx0;
    cp->tdy = image->y1 - cp->ty0;
  }

  if
    (parameters->tp_on)
  {
    cp->m_specific_param.m_enc.m_tp_flag = parameters->tp_flag;
    cp->m_specific_param.m_enc.m_tp_on = 1;
  }

#ifdef USE_JPWL
  /*
  calculate JPWL encoding parameters
  */

  if (parameters->jpwl_epc_on) {
    OPJ_INT32 i;

    /* set JPWL on */
    cp->epc_on = true;
    cp->info_on = false; /* no informative technique */

    /* set EPB on */
    if ((parameters->jpwl_hprot_MH > 0) || (parameters->jpwl_hprot_TPH[0] > 0)) {
      cp->epb_on = true;

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
      cp->esd_on = true;

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
    cp->red_on = false;

  } else {
    cp->epc_on = false;
  }
#endif /* USE_JPWL */


  /* initialize the mutiple tiles */
  /* ---------------------------- */
  cp->tcps = (opj_tcp_t*) opj_calloc(cp->tw * cp->th, sizeof(opj_tcp_t));
  if
    (parameters->numpocs)
  {
    /* initialisation of POC */
    l_res = j2k_check_poc_val(parameters->POC,parameters->numpocs, parameters->numresolution, image->numcomps, parameters->tcp_numlayers, p_manager);
    // TODO
  }
  for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
    opj_tcp_t *tcp = &cp->tcps[tileno];
    tcp->numlayers = parameters->tcp_numlayers;
    for (j = 0; j < tcp->numlayers; j++) {
      if(cp->m_specific_param.m_enc.m_cinema){
        if (cp->m_specific_param.m_enc.m_fixed_quality) {
          tcp->distoratio[j] = parameters->tcp_distoratio[j];
        }
        tcp->rates[j] = parameters->tcp_rates[j];
      }else{
        if (cp->m_specific_param.m_enc.m_fixed_quality) {  /* add fixed_quality */
          tcp->distoratio[j] = parameters->tcp_distoratio[j];
        } else {
          tcp->rates[j] = parameters->tcp_rates[j];
        }
      }
    }
    tcp->csty = parameters->csty;
    tcp->prg = parameters->prog_order;
    tcp->mct = parameters->tcp_mct;



    numpocs_tile = 0;
    tcp->POC = 0;
    if
      (parameters->numpocs)
    {
      /* initialisation of POC */
      tcp->POC = 1;
      // TODO
      for (i = 0; i < (unsigned int) parameters->numpocs; i++) {
        if((tileno == parameters->POC[i].tile - 1) || (parameters->POC[i].tile == -1)) {
          opj_poc_t *tcp_poc = &tcp->pocs[numpocs_tile];
          tcp_poc->resno0    = parameters->POC[numpocs_tile].resno0;
          tcp_poc->compno0  = parameters->POC[numpocs_tile].compno0;
          tcp_poc->layno1    = parameters->POC[numpocs_tile].layno1;
          tcp_poc->resno1    = parameters->POC[numpocs_tile].resno1;
          tcp_poc->compno1  = parameters->POC[numpocs_tile].compno1;
          tcp_poc->prg1    = parameters->POC[numpocs_tile].prg1;
          tcp_poc->tile    = parameters->POC[numpocs_tile].tile;
          numpocs_tile++;
        }
      }
      tcp->numpocs = numpocs_tile -1 ;
    }else{
      tcp->numpocs = 0;
    }

    tcp->tccps = (opj_tccp_t*) opj_calloc(image->numcomps, sizeof(opj_tccp_t));
    if
      (parameters->mct_data)
    {
      OPJ_UINT32 lMctSize = image->numcomps * image->numcomps * sizeof(OPJ_FLOAT32);
      OPJ_FLOAT32 * lTmpBuf = (OPJ_FLOAT32*)opj_malloc(lMctSize);
      OPJ_INT32 * l_dc_shift = (OPJ_INT32 *) ((OPJ_BYTE *) parameters->mct_data + lMctSize);
      tcp->mct = 2;
      tcp->m_mct_coding_matrix = (OPJ_FLOAT32*)opj_malloc(lMctSize);
      memcpy(tcp->m_mct_coding_matrix,parameters->mct_data,lMctSize);
      memcpy(lTmpBuf,parameters->mct_data,lMctSize);
      tcp->m_mct_decoding_matrix = (OPJ_FLOAT32*)opj_malloc(lMctSize);
      assert(opj_matrix_inversion_f(lTmpBuf,(tcp->m_mct_decoding_matrix),image->numcomps));
      tcp->mct_norms = (OPJ_FLOAT64*)
      opj_malloc(image->numcomps * sizeof(OPJ_FLOAT64));
      opj_calculate_norms(tcp->mct_norms,image->numcomps,tcp->m_mct_decoding_matrix);
      opj_free(lTmpBuf);
      for
        (i = 0; i < image->numcomps; i++)
      {
        opj_tccp_t *tccp = &tcp->tccps[i];
        tccp->m_dc_level_shift = l_dc_shift[i];
      }
      j2k_setup_mct_encoding(tcp,image);
    }
    else
    {
      for
        (i = 0; i < image->numcomps; i++)
      {
        opj_tccp_t *tccp = &tcp->tccps[i];
        opj_image_comp_t * l_comp = &(image->comps[i]);
        if
          (! l_comp->sgnd)
        {
          tccp->m_dc_level_shift = 1 << (l_comp->prec - 1);
        }
      }
    }


    for (i = 0; i < image->numcomps; i++) {
      opj_tccp_t *tccp = &tcp->tccps[i];
      tccp->csty = parameters->csty & 0x01;  /* 0 => one precinct || 1 => custom precinct  */
      tccp->numresolutions = parameters->numresolution;
      tccp->cblkw = int_floorlog2(parameters->cblockw_init);
      tccp->cblkh = int_floorlog2(parameters->cblockh_init);
      tccp->cblksty = parameters->mode;
      tccp->qmfbid = parameters->irreversible ? 0 : 1;
      tccp->qntsty = parameters->irreversible ? J2K_CCP_QNTSTY_SEQNT : J2K_CCP_QNTSTY_NOQNT;
      tccp->numgbits = 2;
      if (i == parameters->roi_compno) {
        tccp->roishift = parameters->roi_shift;
      } else {
        tccp->roishift = 0;
      }

      if(parameters->cp_cinema)
      {
        //Precinct size for lowest frequency subband=128
        tccp->prcw[0] = 7;
        tccp->prch[0] = 7;
        //Precinct size at all other resolutions = 256
        for (j = 1; j < tccp->numresolutions; j++) {
          tccp->prcw[j] = 8;
          tccp->prch[j] = 8;
        }
      }else{
        if (parameters->csty & J2K_CCP_CSTY_PRT) {
          int p = 0;
          for (j = tccp->numresolutions - 1; j >= 0; j--) {
            if (p < parameters->res_spec) {

              if (parameters->prcw_init[p] < 1) {
                tccp->prcw[j] = 1;
              } else {
                tccp->prcw[j] = int_floorlog2(parameters->prcw_init[p]);
              }

              if (parameters->prch_init[p] < 1) {
                tccp->prch[j] = 1;
              }else {
                tccp->prch[j] = int_floorlog2(parameters->prch_init[p]);
              }

            } else {
              int res_spec = parameters->res_spec;
              int size_prcw = parameters->prcw_init[res_spec - 1] >> (p - (res_spec - 1));
              int size_prch = parameters->prch_init[res_spec - 1] >> (p - (res_spec - 1));

              if (size_prcw < 1) {
                tccp->prcw[j] = 1;
              } else {
                tccp->prcw[j] = int_floorlog2(size_prcw);
              }

              if (size_prch < 1) {
                tccp->prch[j] = 1;
              } else {
                tccp->prch[j] = int_floorlog2(size_prch);
              }
            }
            p++;
            /*printf("\nsize precinct for level %d : %d,%d\n", j,tccp->prcw[j], tccp->prch[j]); */
          }  //end for
        } else {
          for (j = 0; j < tccp->numresolutions; j++) {
            tccp->prcw[j] = 15;
            tccp->prch[j] = 15;
          }
        }
      }

      dwt_calc_explicit_stepsizes(tccp, image->comps[i].prec);
    }
  }
  if
    (parameters->mct_data)
  {
    opj_free(parameters->mct_data);
    parameters->mct_data = 00;
  }
}

bool j2k_write_first_tile_part (
                  opj_j2k_t *p_j2k,
                  OPJ_BYTE * p_data,
                  OPJ_UINT32 * p_data_written,
                  OPJ_UINT32 p_total_data_size,
                  opj_stream_private_t *p_stream,
                  struct opj_event_mgr * p_manager
                )
{
  OPJ_UINT32 compno;
  OPJ_UINT32 l_nb_bytes_written = 0;
  OPJ_UINT32 l_current_nb_bytes_written;
  OPJ_BYTE * l_begin_data = 00;

  opj_tcp_t *l_tcp = 00;
  opj_tcd_t * l_tcd = 00;
  opj_cp_t * l_cp = 00;

  l_tcd = p_j2k->m_tcd;
  l_cp = &(p_j2k->m_cp);
  l_tcp = l_cp->tcps + p_j2k->m_current_tile_number;

  l_tcd->cur_pino = 0;
  /*Get number of tile parts*/

  p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = 0;
  /* INDEX >> */

  /* << INDEX */
  l_current_nb_bytes_written = 0;
  l_begin_data = p_data;
  if
    (! j2k_write_sot(p_j2k,p_data,&l_current_nb_bytes_written,p_stream,p_manager))
  {
    return false;
  }
  l_nb_bytes_written += l_current_nb_bytes_written;
  p_data += l_current_nb_bytes_written;
  p_total_data_size -= l_current_nb_bytes_written;

  if
    (l_cp->m_specific_param.m_enc.m_cinema == 0)
  {
    for
      (compno = 1; compno < p_j2k->m_image->numcomps; compno++)
    {
      l_current_nb_bytes_written = 0;
      j2k_write_coc_in_memory(p_j2k,compno,p_data,&l_current_nb_bytes_written,p_manager);
      l_nb_bytes_written += l_current_nb_bytes_written;
      p_data += l_current_nb_bytes_written;
      p_total_data_size -= l_current_nb_bytes_written;

      l_current_nb_bytes_written = 0;
      j2k_write_qcc_in_memory(p_j2k,compno,p_data,&l_current_nb_bytes_written,p_manager);
      l_nb_bytes_written += l_current_nb_bytes_written;
      p_data += l_current_nb_bytes_written;
      p_total_data_size -= l_current_nb_bytes_written;
    }
    if
      (l_cp->tcps[p_j2k->m_current_tile_number].numpocs)
    {
      l_current_nb_bytes_written = 0;
      j2k_write_poc_in_memory(p_j2k,p_data,&l_current_nb_bytes_written,p_manager);
      l_nb_bytes_written += l_current_nb_bytes_written;
      p_data += l_current_nb_bytes_written;
      p_total_data_size -= l_current_nb_bytes_written;
    }
  }
  l_current_nb_bytes_written = 0;
  if
    (! j2k_write_sod(p_j2k,l_tcd,p_data,&l_current_nb_bytes_written,p_total_data_size,p_stream,p_manager))
  {
    return false;
  }
  l_nb_bytes_written += l_current_nb_bytes_written;
  * p_data_written = l_nb_bytes_written;

  /* Writing Psot in SOT marker */
  opj_write_bytes(l_begin_data + 6,l_nb_bytes_written,4);          /* PSOT */
  if
    (l_cp->m_specific_param.m_enc.m_cinema)
  {
    j2k_update_tlm(p_j2k,l_nb_bytes_written);
  }
  return true;
}

bool j2k_write_all_tile_parts(
                  opj_j2k_t *p_j2k,
                  OPJ_BYTE * p_data,
                  OPJ_UINT32 * p_data_written,
                  OPJ_UINT32 p_total_data_size,
                  opj_stream_private_t *p_stream,
                  struct opj_event_mgr * p_manager
                )
{
  OPJ_UINT32 tilepartno=0;
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
  tot_num_tp = j2k_get_num_tp(l_cp,0,p_j2k->m_current_tile_number);
  for
    (tilepartno = 1; tilepartno < tot_num_tp ; ++tilepartno)
  {
    p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = tilepartno;
    l_current_nb_bytes_written = 0;
    l_part_tile_size = 0;
    l_begin_data = p_data;
    if
      (! j2k_write_sot(p_j2k,p_data,&l_current_nb_bytes_written,p_stream,p_manager))
    {
      return false;
    }
    l_nb_bytes_written += l_current_nb_bytes_written;
    p_data += l_current_nb_bytes_written;
    p_total_data_size -= l_current_nb_bytes_written;
    l_part_tile_size += l_nb_bytes_written;

    l_current_nb_bytes_written = 0;
    if
      (! j2k_write_sod(p_j2k,l_tcd,p_data,&l_current_nb_bytes_written,p_total_data_size,p_stream,p_manager))
    {
      return false;
    }
    p_data += l_current_nb_bytes_written;
    l_nb_bytes_written += l_current_nb_bytes_written;
    p_total_data_size -= l_current_nb_bytes_written;
    l_part_tile_size += l_nb_bytes_written;

    /* Writing Psot in SOT marker */
    opj_write_bytes(l_begin_data + 6,l_part_tile_size,4);          /* PSOT */

    if
      (l_cp->m_specific_param.m_enc.m_cinema)
    {
      j2k_update_tlm(p_j2k,l_part_tile_size);
    }
    ++p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
  }
  for
    (pino = 1; pino <= l_tcp->numpocs; ++pino)
  {
    l_tcd->cur_pino = pino;
    /*Get number of tile parts*/
    tot_num_tp = j2k_get_num_tp(l_cp,pino,p_j2k->m_current_tile_number);
    for
      (tilepartno = 0; tilepartno < tot_num_tp ; ++tilepartno)
    {
      p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = tilepartno;
      l_current_nb_bytes_written = 0;
      l_part_tile_size = 0;
      l_begin_data = p_data;
      if
        (! j2k_write_sot(p_j2k,p_data,&l_current_nb_bytes_written,p_stream,p_manager))
      {
        return false;
      }
      l_nb_bytes_written += l_current_nb_bytes_written;
      p_data += l_current_nb_bytes_written;
      p_total_data_size -= l_current_nb_bytes_written;
      l_part_tile_size += l_current_nb_bytes_written;

      l_current_nb_bytes_written = 0;
      if
        (! j2k_write_sod(p_j2k,l_tcd,p_data,&l_current_nb_bytes_written,p_total_data_size,p_stream,p_manager))
      {
        return false;
      }
      l_nb_bytes_written += l_current_nb_bytes_written;
      p_data += l_current_nb_bytes_written;
      p_total_data_size -= l_current_nb_bytes_written;
      l_part_tile_size += l_current_nb_bytes_written;

      /* Writing Psot in SOT marker */
      opj_write_bytes(l_begin_data + 6,l_part_tile_size,4);          /* PSOT */

      if
        (l_cp->m_specific_param.m_enc.m_cinema)
      {
        j2k_update_tlm(p_j2k,l_part_tile_size);
      }
      ++p_j2k->m_specific_param.m_encoder.m_current_tile_part_number;
    }
  }
  *p_data_written = l_nb_bytes_written;
  return true;
}


bool j2k_pre_write_tile (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 p_tile_index,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          )
{
  if
    (p_tile_index != p_j2k->m_current_tile_number)
  {
    opj_event_msg(p_manager, EVT_ERROR, "The given tile index does not match." );
    return false;
  }

  opj_event_msg(p_manager, EVT_INFO, "tile number %d / %d\n", p_j2k->m_current_tile_number + 1, p_j2k->m_cp.tw * p_j2k->m_cp.th);

  p_j2k->m_specific_param.m_encoder.m_current_tile_part_number = 0;
  p_j2k->m_tcd->cur_totnum_tp = p_j2k->m_cp.tcps[p_tile_index].m_nb_tile_parts;
  p_j2k->m_specific_param.m_encoder.m_current_poc_tile_part_number = 0;
  /* initialisation before tile encoding  */
  if
    (! tcd_init_encode_tile(p_j2k->m_tcd, p_j2k->m_current_tile_number))
  {
    return false;
  }
  return true;
}

/**
 * Writes a tile.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool j2k_write_tile (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 p_tile_index,
           OPJ_BYTE * p_data,
           OPJ_UINT32 p_data_size,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          )
{
  if
    (! j2k_pre_write_tile(p_j2k,p_tile_index,p_stream,p_manager))
  {
    return false;
  }
  return j2k_post_write_tile(p_j2k,p_data,p_data_size,p_stream,p_manager);
}

/**
 * Writes a tile.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool j2k_post_write_tile (
           opj_j2k_t * p_j2k,
           OPJ_BYTE * p_data,
           OPJ_UINT32 p_data_size,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          )
{
  opj_tcd_t * l_tcd = 00;
  opj_cp_t * l_cp = 00;
  opj_tcp_t * l_tcp = 00;
  OPJ_UINT32 l_nb_bytes_written;
  OPJ_BYTE * l_current_data = 00;
  OPJ_UINT32 l_tile_size = 0;
  OPJ_UINT32 l_available_data;

  assert(p_j2k->m_specific_param.m_encoder.m_encoded_tile_data);

  l_tcd = p_j2k->m_tcd;
  l_cp = &(p_j2k->m_cp);
  l_tcp = l_cp->tcps + p_j2k->m_current_tile_number;

  l_tile_size = p_j2k->m_specific_param.m_encoder.m_encoded_tile_size;
  l_available_data = l_tile_size;
  l_current_data = p_j2k->m_specific_param.m_encoder.m_encoded_tile_data;
  if
    (! tcd_copy_tile_data(l_tcd,p_data,p_data_size))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Size mismtach between tile data and sent data." );
    return false;
  }

  l_nb_bytes_written = 0;
  if
    (! j2k_write_first_tile_part(p_j2k,l_current_data,&l_nb_bytes_written,l_available_data,p_stream,p_manager))
  {
    return false;
  }
  l_current_data += l_nb_bytes_written;
  l_available_data -= l_nb_bytes_written;

  l_nb_bytes_written = 0;
  if
    (! j2k_write_all_tile_parts(p_j2k,l_current_data,&l_nb_bytes_written,l_available_data,p_stream,p_manager))
  {
    return false;
  }

  l_available_data -= l_nb_bytes_written;
  l_nb_bytes_written = l_tile_size - l_available_data;

  if
    (opj_stream_write_data(p_stream,p_j2k->m_specific_param.m_encoder.m_encoded_tile_data,l_nb_bytes_written,p_manager) != l_nb_bytes_written)
  {
    return false;
  }
  ++p_j2k->m_current_tile_number;
  return true;
}

/**
 * Reads a tile header.
 * @param  p_j2k    the jpeg2000 codec.
 * @param  p_stream      the stream to write data to.
 * @param  p_manager  the user event manager.
 */
bool j2k_read_tile_header (
           opj_j2k_t * p_j2k,
           OPJ_UINT32 * p_tile_index,
           OPJ_UINT32 * p_data_size,
           OPJ_INT32 * p_tile_x0,
           OPJ_INT32 * p_tile_y0,
           OPJ_INT32 * p_tile_x1,
           OPJ_INT32 * p_tile_y1,
           OPJ_UINT32 * p_nb_comps,
           bool * p_go_on,
           opj_stream_private_t *p_stream,
           opj_event_mgr_t * p_manager
          )
{
  OPJ_UINT32 l_current_marker = J2K_MS_SOT;
  OPJ_UINT32 l_marker_size;
  const opj_dec_memory_marker_handler_t * l_marker_handler = 00;
  opj_tcp_t * l_tcp = 00;
  OPJ_UINT32 l_nb_tiles;

  // preconditions
  assert(p_stream != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (p_j2k->m_specific_param.m_decoder.m_state == J2K_DEC_STATE_EOC)
  {
    l_current_marker = J2K_MS_EOC;
  }
  else if
    (p_j2k->m_specific_param.m_decoder.m_state != J2K_DEC_STATE_TPHSOT)
  {
    return false;
  }

  while
    (! p_j2k->m_specific_param.m_decoder.m_can_decode && l_current_marker != J2K_MS_EOC)
  {
    while
      (l_current_marker != J2K_MS_SOD)
    {
      if
        (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
        return false;
      }
      opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_marker_size,2);
      if
        (p_j2k->m_specific_param.m_decoder.m_state & J2K_DEC_STATE_TPH)
      {
        p_j2k->m_specific_param.m_decoder.m_sot_length -= (l_marker_size + 2);
      }
      l_marker_size -= 2;

      l_marker_handler = j2k_get_marker_handler(l_current_marker);
      // Check if the marker is known
      if
        (! (p_j2k->m_specific_param.m_decoder.m_state & l_marker_handler->states) )
      {
        opj_event_msg(p_manager, EVT_ERROR, "Marker is not compliant with its position\n");
        return false;
      }
      if
        (l_marker_size > p_j2k->m_specific_param.m_decoder.m_header_data_size)
      {
        p_j2k->m_specific_param.m_decoder.m_header_data = (OPJ_BYTE*)
        opj_realloc(p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size);
        if
          (p_j2k->m_specific_param.m_decoder.m_header_data == 00)
        {
          return false;
        }
        p_j2k->m_specific_param.m_decoder.m_header_data_size = l_marker_size;

      }
      if
        (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size,p_manager) != l_marker_size)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
        return false;
      }
      if
        (! (*(l_marker_handler->handler))(p_j2k,p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size,p_manager))
      {
        opj_event_msg(p_manager, EVT_ERROR, "Marker is not compliant with its position\n");
        return false;
      }
      if
        (p_j2k->m_specific_param.m_decoder.m_skip_data)
      {
        if
          (opj_stream_skip(p_stream,p_j2k->m_specific_param.m_decoder.m_sot_length,p_manager) != p_j2k->m_specific_param.m_decoder.m_sot_length)
        {
          opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
          return false;
        }
        l_current_marker = J2K_MS_SOD;
      }
      else
      {
        if
          (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
        {
          opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
          return false;
        }
        opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_current_marker,2);
      }
    }

    if
      (! p_j2k->m_specific_param.m_decoder.m_skip_data)
    {
      if
        (! j2k_read_sod(p_j2k,p_stream,p_manager))
      {
        return false;
      }
    }
    else
    {
      p_j2k->m_specific_param.m_decoder.m_skip_data = 0;
      p_j2k->m_specific_param.m_decoder.m_can_decode = 0;
      p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_TPHSOT;
      if
        (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
      {
        opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
        return false;
      }
      opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_current_marker,2);
    }
  }

  if
    (l_current_marker == J2K_MS_EOC)
  {
    if
      (p_j2k->m_specific_param.m_decoder.m_state != J2K_DEC_STATE_EOC)
    {
      p_j2k->m_current_tile_number = 0;
      p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_EOC;
    }
  }
  if
    ( ! p_j2k->m_specific_param.m_decoder.m_can_decode)
  {
    l_tcp = p_j2k->m_cp.tcps + p_j2k->m_current_tile_number;
    l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
    while
      (
        (p_j2k->m_current_tile_number < l_nb_tiles)
        &&  (l_tcp->m_data == 00)
      )
    {
      ++p_j2k->m_current_tile_number;
      ++l_tcp;
    }
    if
      (p_j2k->m_current_tile_number == l_nb_tiles)
    {
      *p_go_on = false;
      return true;
    }
  }
  if
    (! tcd_init_decode_tile(p_j2k->m_tcd, p_j2k->m_current_tile_number))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n");
    return false;
  }
  *p_tile_index = p_j2k->m_current_tile_number;
  *p_go_on = true;
  *p_data_size = tcd_get_decoded_tile_size(p_j2k->m_tcd);
  * p_tile_x0 = p_j2k->m_tcd->tcd_image->tiles->x0;
  * p_tile_y0 = p_j2k->m_tcd->tcd_image->tiles->y0;
  * p_tile_x1 = p_j2k->m_tcd->tcd_image->tiles->x1;
  * p_tile_y1 = p_j2k->m_tcd->tcd_image->tiles->y1;
  * p_nb_comps = p_j2k->m_tcd->tcd_image->tiles->numcomps;
  p_j2k->m_specific_param.m_decoder.m_state |= J2K_DEC_STATE_DATA;
  return true;
}

bool j2k_decode_tile (
          opj_j2k_t * p_j2k,
          OPJ_UINT32 p_tile_index,
          OPJ_BYTE * p_data,
          OPJ_UINT32 p_data_size,
          opj_stream_private_t *p_stream,
          opj_event_mgr_t * p_manager
          )
{
  OPJ_UINT32 l_current_marker;
  OPJ_BYTE l_data [2];
  opj_tcp_t * l_tcp;

  // preconditions
  assert(p_stream != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  if
    (! (p_j2k->m_specific_param.m_decoder.m_state & J2K_DEC_STATE_DATA) || p_tile_index != p_j2k->m_current_tile_number)
  {
    return false;
  }
  l_tcp = &(p_j2k->m_cp.tcps[p_tile_index]);
  if
    (! l_tcp->m_data)
  {
    j2k_tcp_destroy(&(p_j2k->m_cp.tcps[p_tile_index]));
    return false;
  }
  if
    (! tcd_decode_tile(p_j2k->m_tcd, l_tcp->m_data, l_tcp->m_data_size, p_tile_index, p_j2k->cstr_info))
  {
    j2k_tcp_destroy(l_tcp);
    p_j2k->m_specific_param.m_decoder.m_state |= J2K_DEC_STATE_ERR;
    return false;
  }
  if
    (! tcd_update_tile_data(p_j2k->m_tcd,p_data,p_data_size))
  {
    return false;
  }
  j2k_tcp_destroy(l_tcp);
  p_j2k->m_tcd->tcp = 0;

  p_j2k->m_specific_param.m_decoder.m_can_decode = 0;
  p_j2k->m_specific_param.m_decoder.m_state &= (~J2K_DEC_STATE_DATA);
  if
    (p_j2k->m_specific_param.m_decoder.m_state != J2K_DEC_STATE_EOC)
  {
    if
      (opj_stream_read_data(p_stream,l_data,2,p_manager) != 2)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
      return false;
    }
    opj_read_bytes(l_data,&l_current_marker,2);
    if
      (l_current_marker == J2K_MS_EOC)
    {
      p_j2k->m_current_tile_number = 0;
      p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_EOC;
    }
    else if
      (l_current_marker != J2K_MS_SOT)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream too short, expected SOT\n");
      return false;
    }
  }
  return true;
}


/**
 * Ends the compression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool j2k_end_compress(opj_j2k_t *p_j2k, struct opj_stream_private *p_stream, struct opj_event_mgr * p_manager)
{
  /* customization of the encoding */
  j2k_setup_end_compress(p_j2k);

  if
    (! j2k_exec (p_j2k,p_j2k->m_procedure_list,p_stream,p_manager))
  {
    return false;
  }
  return true;
}

/**
 * Reads a jpeg2000 codestream header structure.
 *
 * @param p_stream the stream to read data from.
 * @param p_j2k the jpeg2000 codec.
 * @param p_manager the user event manager.
 *
 * @return true if the box is valid.
 */
bool j2k_read_header(
                opj_j2k_t *p_j2k,
                struct opj_image ** p_image,
                OPJ_INT32 * p_tile_x0,
                OPJ_INT32 * p_tile_y0,
                OPJ_UINT32 * p_tile_width,
                OPJ_UINT32 * p_tile_height,
                OPJ_UINT32 * p_nb_tiles_x,
                OPJ_UINT32 * p_nb_tiles_y,
                struct opj_stream_private *p_stream,
                struct opj_event_mgr * p_manager
              )
{
  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  *p_image = 00;
  /* create an empty image */
  p_j2k->m_image = opj_image_create0();
  if
    (! p_j2k->m_image)
  {
    return false;
  }

  /* customization of the validation */
  j2k_setup_decoding_validation (p_j2k);

  /* validation of the parameters codec */
  if
    (! j2k_exec(p_j2k,p_j2k->m_validation_list,p_stream,p_manager))
  {
    opj_image_destroy(p_j2k->m_image);
    p_j2k->m_image = 00;
    return false;
  }

  /* customization of the encoding */
  j2k_setup_header_reading(p_j2k);

  /* read header */
  if
    (! j2k_exec (p_j2k,p_j2k->m_procedure_list,p_stream,p_manager))
  {
    opj_image_destroy(p_j2k->m_image);
    p_j2k->m_image = 00;
    return false;
  }
  *p_image = p_j2k->m_image;
  * p_tile_x0 = p_j2k->m_cp.tx0;
    * p_tile_y0 = p_j2k->m_cp.ty0;
  * p_tile_width = p_j2k->m_cp.tdx;
    * p_tile_height = p_j2k->m_cp.tdy;
  * p_nb_tiles_x = p_j2k->m_cp.tw;
  * p_nb_tiles_y = p_j2k->m_cp.th;
  return true;
}

/**
 * The read header procedure.
 */
bool j2k_read_header_procedure(
                opj_j2k_t *p_j2k,
                struct opj_stream_private *p_stream,
                struct opj_event_mgr * p_manager)
{
  OPJ_UINT32 l_current_marker;
  OPJ_UINT32 l_marker_size;
  const opj_dec_memory_marker_handler_t * l_marker_handler = 00;

  // preconditions
  assert(p_stream != 00);
  assert(p_j2k != 00);
  assert(p_manager != 00);

  p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_MHSOC;

  if
    (! j2k_read_soc(p_j2k,p_stream,p_manager))
  {
    opj_event_msg(p_manager, EVT_ERROR, "Expected a SOC marker \n");
    return false;
  }
  if
    (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
  {
    opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
    return false;
  }
  opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_current_marker,2);

  while
    (l_current_marker != J2K_MS_SOT)
  {
    if
      (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
      return false;
    }
    opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_marker_size,2);
    l_marker_size -= 2;
    /*if
      (l_current_marker < 0xff00)
    {
      opj_event_msg(p_manager, EVT_ERROR, "%.8x: expected a marker instead of %x\n", opj_stream_tell(p_stream) - 2, l_current_marker);
      return 0;
    }
    */
    l_marker_handler = j2k_get_marker_handler(l_current_marker);
    // Check if the marker is known
    if
      (! (p_j2k->m_specific_param.m_decoder.m_state & l_marker_handler->states) )
    {
      opj_event_msg(p_manager, EVT_ERROR, "Marker is not compliant with its position\n");
      return false;
    }
    if
      (l_marker_size > p_j2k->m_specific_param.m_decoder.m_header_data_size)
    {
      p_j2k->m_specific_param.m_decoder.m_header_data = (OPJ_BYTE*)
      opj_realloc(p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size);
      if
        (p_j2k->m_specific_param.m_decoder.m_header_data == 00)
      {
        return false;
      }
      p_j2k->m_specific_param.m_decoder.m_header_data_size = l_marker_size;
    }
    if
      (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size,p_manager) != l_marker_size)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
      return false;
    }
    if
      (! (*(l_marker_handler->handler))(p_j2k,p_j2k->m_specific_param.m_decoder.m_header_data,l_marker_size,p_manager))
    {
      opj_event_msg(p_manager, EVT_ERROR, "Marker is not compliant with its position\n");
      return false;
    }
    if
      (opj_stream_read_data(p_stream,p_j2k->m_specific_param.m_decoder.m_header_data,2,p_manager) != 2)
    {
      opj_event_msg(p_manager, EVT_ERROR, "Stream too short\n");
      return false;
    }
    opj_read_bytes(p_j2k->m_specific_param.m_decoder.m_header_data,&l_current_marker,2);
  }
  p_j2k->m_specific_param.m_decoder.m_state = J2K_DEC_STATE_TPHSOT;
  return true;
}



/**
 * Reads the tiles.
 */
bool j2k_decode_tiles (
                opj_j2k_t *p_j2k,
                struct opj_stream_private *p_stream,
                struct opj_event_mgr * p_manager)
{
  bool l_go_on = true;
  OPJ_UINT32 l_current_tile_no;
  OPJ_UINT32 l_data_size,l_max_data_size;
  OPJ_INT32 l_tile_x0,l_tile_y0,l_tile_x1,l_tile_y1;
  OPJ_UINT32 l_nb_comps;
  OPJ_BYTE * l_current_data;

  l_current_data = (OPJ_BYTE*)opj_malloc(1000);
  if
    (! l_current_data)
  {
    return false;
  }
  l_max_data_size = 1000;

  while
    (true)
  {
    if
      (! j2k_read_tile_header(
        p_j2k,&l_current_tile_no,
        &l_data_size,
        &l_tile_x0,
        &l_tile_y0,
        &l_tile_x1,
        &l_tile_y1,
        &l_nb_comps,
        &l_go_on,
        p_stream,
        p_manager))
    {
      return false;
    }
    if
      (! l_go_on)
    {
      break;
    }
    if
      (l_data_size > l_max_data_size)
    {
      l_current_data = (OPJ_BYTE*)opj_realloc(l_current_data,l_data_size);
      if
        (! l_current_data)
      {
        return false;
      }
      l_max_data_size = l_data_size;
    }
    if
      (! j2k_decode_tile(p_j2k,l_current_tile_no,l_current_data,l_data_size,p_stream,p_manager))
    {
      opj_free(l_current_data);
      return false;
    }
    if
      (! j2k_update_image_data(p_j2k->m_tcd,l_current_data))
    {
      opj_free(l_current_data);
      return false;
    }

  }
  opj_free(l_current_data);
  return true;
}






/**
 * Decodes the tiles of the stream.
 */
opj_image_t * j2k_decode(
              opj_j2k_t * p_j2k,
             opj_stream_private_t * p_stream,
             opj_event_mgr_t * p_manager)
{
  /* customization of the encoding */
  j2k_setup_decoding(p_j2k);

  /* write header */
  if
    (! j2k_exec (p_j2k,p_j2k->m_procedure_list,p_stream,p_manager))
  {
    opj_image_destroy(p_j2k->m_image);
    p_j2k->m_image = 00;
  }
  return p_j2k->m_image;
}

/**
 * Encodes all the tiles in a row.
 */
bool j2k_encode(
          opj_j2k_t * p_j2k,
          opj_stream_private_t *p_stream,
          opj_event_mgr_t * p_manager
        )
{
  OPJ_UINT32 i;
  OPJ_UINT32 l_nb_tiles;
  OPJ_UINT32 l_max_tile_size, l_current_tile_size;
  OPJ_BYTE * l_current_data;

  // preconditions
  assert(p_j2k != 00);
  assert(p_stream != 00);
  assert(p_manager != 00);

  l_current_data = (OPJ_BYTE*)opj_malloc(1000);
  if
    (! l_current_data)
  {
    return false;
  }
  l_max_tile_size = 1000;

  l_nb_tiles = p_j2k->m_cp.th * p_j2k->m_cp.tw;
  for
    (i=0;i<l_nb_tiles;++i)
  {
    if
      (! j2k_pre_write_tile(p_j2k,i,p_stream,p_manager))
    {
      opj_free(l_current_data);
      return false;
    }
    l_current_tile_size = tcd_get_encoded_tile_size(p_j2k->m_tcd);
    if
      (l_current_tile_size > l_max_tile_size)
    {
      l_current_data = (OPJ_BYTE*)opj_realloc(l_current_data,l_current_tile_size);
      if
        (! l_current_data)
      {
        return false;
      }
      l_max_tile_size = l_current_tile_size;
    }
    j2k_get_tile_data(p_j2k->m_tcd,l_current_data);
    if
      (! j2k_post_write_tile (p_j2k,l_current_data,l_current_tile_size,p_stream,p_manager))
    {
      return false;
    }
  }
  opj_free(l_current_data);
  return true;
}



/**
 * Ends the decompression procedures and possibiliy add data to be read after the
 * codestream.
 */
bool j2k_end_decompress(
            opj_j2k_t *p_j2k,
            struct opj_stream_private *p_stream,
            struct opj_event_mgr * p_manager)
{
  return true;
}



void j2k_get_tile_data (opj_tcd_t * p_tcd, OPJ_BYTE * p_data)
{
  OPJ_UINT32 i,j,k = 0;
  OPJ_UINT32 l_width,l_height,l_stride, l_offset_x,l_offset_y, l_image_width;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tilec = 00;
  opj_image_t * l_image = 00;
  OPJ_UINT32 l_size_comp, l_remaining;
  OPJ_INT32 * l_src_ptr;
  l_tilec = p_tcd->tcd_image->tiles->comps;
  l_image = p_tcd->image;
  l_img_comp = l_image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    l_width = (l_tilec->x1 - l_tilec->x0);
    l_height = (l_tilec->y1 - l_tilec->y0);
    l_offset_x = int_ceildiv(l_image->x0, l_img_comp->dx);
    l_offset_y = int_ceildiv(l_image->y0, l_img_comp->dy);
    l_image_width = int_ceildiv(l_image->x1 - l_image->x0, l_img_comp->dx);
    l_stride = l_image_width - l_width;
    l_src_ptr = l_img_comp->data + (l_tilec->x0 - l_offset_x) + (l_tilec->y0 - l_offset_y) * l_image_width;

    switch
      (l_size_comp)
    {
      case 1:
        {
          OPJ_CHAR * l_dest_ptr = (OPJ_CHAR*) p_data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr) = (OPJ_CHAR) (*l_src_ptr);
                ++l_dest_ptr;
                ++l_src_ptr;
              }
              l_src_ptr += l_stride;
            }
          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr) = (*l_src_ptr)&0xff;
                ++l_dest_ptr;
                ++l_src_ptr;
              }
              l_src_ptr += l_stride;
            }
          }
          p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
      case 2:
        {
          OPJ_INT16 * l_dest_ptr = (OPJ_INT16 *) p_data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_INT16) (*(l_src_ptr++));
              }
              l_src_ptr += l_stride;
            }
          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (*(l_src_ptr++))&0xffff;
              }
              l_src_ptr += l_stride;
            }
          }
          p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
      case 4:
        {
          OPJ_INT32 * l_dest_ptr = (OPJ_INT32 *) p_data;
          for
            (j=0;j<l_height;++j)
          {
            for
              (k=0;k<l_width;++k)
            {
              *(l_dest_ptr++) = *(l_src_ptr++);
            }
            l_src_ptr += l_stride;
          }
          p_data = (OPJ_BYTE*) l_dest_ptr;
        }
        break;
    }
    ++l_img_comp;
    ++l_tilec;
  }
}

bool j2k_update_image_data (opj_tcd_t * p_tcd, OPJ_BYTE * p_data)
{
  OPJ_UINT32 i,j,k = 0;
  OPJ_UINT32 l_width,l_height,l_offset_x,l_offset_y;
  opj_image_comp_t * l_img_comp = 00;
  opj_tcd_tilecomp_t * l_tilec = 00;
  opj_image_t * l_image = 00;
  OPJ_UINT32 l_size_comp, l_remaining;
  OPJ_UINT32 l_dest_stride;
  OPJ_INT32 * l_dest_ptr;
  opj_tcd_resolution_t* l_res= 00;


  l_tilec = p_tcd->tcd_image->tiles->comps;
  l_image = p_tcd->image;
  l_img_comp = l_image->comps;
  for
    (i=0;i<p_tcd->image->numcomps;++i)
  {
    if
      (!l_img_comp->data)
    {
      l_img_comp->data = (OPJ_INT32*) opj_malloc(l_img_comp->w * l_img_comp->h * sizeof(OPJ_INT32));
      if
        (! l_img_comp->data)
      {
        return false;
      }
      memset(l_img_comp->data,0,l_img_comp->w * l_img_comp->h * sizeof(OPJ_INT32));
    }

    l_size_comp = l_img_comp->prec >> 3; /*(/ 8)*/
    l_remaining = l_img_comp->prec & 7;  /* (%8) */
    l_res = l_tilec->resolutions + l_img_comp->resno_decoded;

    if
      (l_remaining)
    {
      ++l_size_comp;
    }
    if
      (l_size_comp == 3)
    {
      l_size_comp = 4;
    }
    l_width = (l_res->x1 - l_res->x0);
    l_height = (l_res->y1 - l_res->y0);
    l_dest_stride = (l_img_comp->w) - l_width;
    l_offset_x = int_ceildivpow2(l_img_comp->x0, l_img_comp->factor);
    l_offset_y = int_ceildivpow2(l_img_comp->y0, l_img_comp->factor);
    l_dest_ptr = l_img_comp->data + (l_res->x0 - l_offset_x) + (l_res->y0 - l_offset_y) * l_img_comp->w;

    switch
      (l_size_comp)
    {
      case 1:
        {
          OPJ_CHAR * l_src_ptr = (OPJ_CHAR*) p_data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_INT32) (*(l_src_ptr++));
              }
              l_dest_ptr += l_dest_stride;
            }

          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (OPJ_INT32) ((*(l_src_ptr++))&0xff);
              }
              l_dest_ptr += l_dest_stride;
            }
          }
          p_data = (OPJ_BYTE*) l_src_ptr;
        }
        break;
      case 2:
        {
          OPJ_INT16 * l_src_ptr = (OPJ_INT16 *) p_data;
          if
            (l_img_comp->sgnd)
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = *(l_src_ptr++);
              }
              l_dest_ptr += l_dest_stride;
            }
          }
          else
          {
            for
              (j=0;j<l_height;++j)
            {
              for
                (k=0;k<l_width;++k)
              {
                *(l_dest_ptr++) = (*(l_src_ptr++))&0xffff;
              }
              l_dest_ptr += l_dest_stride;
            }
          }
          p_data = (OPJ_BYTE*) l_src_ptr;
        }
        break;
      case 4:
        {
          OPJ_INT32 * l_src_ptr = (OPJ_INT32 *) p_data;
          for
            (j=0;j<l_height;++j)
          {
            for
              (k=0;k<l_width;++k)
            {
              *(l_dest_ptr++) = (*(l_src_ptr++));
            }
            l_dest_ptr += l_dest_stride;
          }
          p_data = (OPJ_BYTE*) l_src_ptr;
        }
        break;
    }
    ++l_img_comp;
    ++l_tilec;
  }
  return true;
}

/**
 * Sets the given area to be decoded. This function should be called right after opj_read_header and before any tile header reading.
 *
 * @param  p_j2k      the jpeg2000 codec.
 * @param  p_start_x    the left position of the rectangle to decode (in image coordinates).
 * @param  p_end_x      the right position of the rectangle to decode (in image coordinates).
 * @param  p_start_y    the up position of the rectangle to decode (in image coordinates).
 * @param  p_end_y      the bottom position of the rectangle to decode (in image coordinates).
 * @param  p_manager    the user event manager
 *
 * @return  true      if the area could be set.
 */
bool j2k_set_decode_area(
      opj_j2k_t *p_j2k,
      OPJ_INT32 p_start_x,
      OPJ_INT32 p_start_y,
      OPJ_INT32 p_end_x,
      OPJ_INT32 p_end_y,
      struct opj_event_mgr * p_manager
      )
{
  opj_cp_t * l_cp = &(p_j2k->m_cp);

  if
    (p_j2k->m_specific_param.m_decoder.m_state != J2K_DEC_STATE_TPHSOT)
  {
    return false;
  }
  p_j2k->m_specific_param.m_decoder.m_start_tile_x = (p_start_x - l_cp->tx0) / l_cp->tdx;
  p_j2k->m_specific_param.m_decoder.m_start_tile_y = (p_start_y - l_cp->ty0) / l_cp->tdy;
  p_j2k->m_specific_param.m_decoder.m_end_tile_x = int_ceildiv((p_end_x - l_cp->tx0), l_cp->tdx);
  p_j2k->m_specific_param.m_decoder.m_end_tile_y = int_ceildiv((p_end_y - l_cp->ty0), l_cp->tdy);
  p_j2k->m_specific_param.m_decoder.m_discard_tiles = 1;
  return true;
}

void j2k_dump_image(FILE *fd, opj_image_t * img) {
  OPJ_UINT32 compno; // to avoid signed/unsigned mismatch
  fprintf(fd, "image {\n");
  fprintf(fd, "  x0=%d, y0=%d, x1=%d, y1=%d\n", img->x0, img->y0, img->x1, img->y1);
  fprintf(fd, "  numcomps=%d\n", img->numcomps);
  for (compno = 0; compno < img->numcomps; compno++) {
    opj_image_comp_t *comp = &img->comps[compno];
    fprintf(fd, "  comp %d {\n", compno);
    fprintf(fd, "    dx=%d, dy=%d\n", comp->dx, comp->dy);
    fprintf(fd, "    prec=%d\n", comp->prec);
    //fprintf(fd, "    bpp=%d\n", comp->bpp);
    fprintf(fd, "    sgnd=%d\n", comp->sgnd);
    fprintf(fd, "  }\n");
  }
  fprintf(fd, "}\n");
}

/*
void j2k_dump_cp(FILE *fd, opj_image_t * img, opj_cp_t * cp) {
  int tileno, compno, layno, bandno, resno, numbands;
  fprintf(fd, "coding parameters {\n");
  fprintf(fd, "  tx0=%d, ty0=%d\n", cp->tx0, cp->ty0);
  fprintf(fd, "  tdx=%d, tdy=%d\n", cp->tdx, cp->tdy);
  fprintf(fd, "  tw=%d, th=%d\n", cp->tw, cp->th);
  for (tileno = 0; tileno < cp->tw * cp->th; tileno++) {
    opj_tcp_t *tcp = &cp->tcps[tileno];
    fprintf(fd, "  tile %d {\n", tileno);
    fprintf(fd, "    csty=%x\n", tcp->csty);
    fprintf(fd, "    prg=%d\n", tcp->prg);
    fprintf(fd, "    numlayers=%d\n", tcp->numlayers);
    fprintf(fd, "    mct=%d\n", tcp->mct);
    fprintf(fd, "    rates=");
    for (layno = 0; layno < tcp->numlayers; layno++) {
      fprintf(fd, "%.1f ", tcp->rates[layno]);
    }
    fprintf(fd, "\n");
    for (compno = 0; compno < img->numcomps; compno++) {
      opj_tccp_t *tccp = &tcp->tccps[compno];
      fprintf(fd, "    comp %d {\n", compno);
      fprintf(fd, "      csty=%x\n", tccp->csty);
      fprintf(fd, "      numresolutions=%d\n", tccp->numresolutions);
      fprintf(fd, "      cblkw=%d\n", tccp->cblkw);
      fprintf(fd, "      cblkh=%d\n", tccp->cblkh);
      fprintf(fd, "      cblksty=%x\n", tccp->cblksty);
      fprintf(fd, "      qmfbid=%d\n", tccp->qmfbid);
      fprintf(fd, "      qntsty=%d\n", tccp->qntsty);
      fprintf(fd, "      numgbits=%d\n", tccp->numgbits);
      fprintf(fd, "      roishift=%d\n", tccp->roishift);
      fprintf(fd, "      stepsizes=");
      numbands = tccp->qntsty == J2K_CCP_QNTSTY_SIQNT ? 1 : tccp->numresolutions * 3 - 2;
      for (bandno = 0; bandno < numbands; bandno++) {
        fprintf(fd, "(%d,%d) ", tccp->stepsizes[bandno].mant,
          tccp->stepsizes[bandno].expn);
      }
      fprintf(fd, "\n");

      if (tccp->csty & J2K_CCP_CSTY_PRT) {
        fprintf(fd, "      prcw=");
        for (resno = 0; resno < tccp->numresolutions; resno++) {
          fprintf(fd, "%d ", tccp->prcw[resno]);
        }
        fprintf(fd, "\n");
        fprintf(fd, "      prch=");
        for (resno = 0; resno < tccp->numresolutions; resno++) {
          fprintf(fd, "%d ", tccp->prch[resno]);
        }
        fprintf(fd, "\n");
      }
      fprintf(fd, "    }\n");
    }
    fprintf(fd, "  }\n");
  }
  fprintf(fd, "}\n");
}
*/
