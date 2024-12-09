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
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
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
#ifndef OPJ_TCD_H
#define OPJ_TCD_H
/**
@file tcd.h
@brief Implementation of a tile coder/decoder (TCD)

The functions in TCD.C encode or decode each tile independently from
each other. The functions in TCD.C are used by other functions in J2K.C.
*/

/** @defgroup TCD TCD - Implementation of a tile coder/decoder */
/*@{*/


/**
FIXME DOC
*/
typedef struct opj_tcd_pass {
    OPJ_UINT32 rate;
    OPJ_FLOAT64 distortiondec;
    OPJ_UINT32 len;
    OPJ_BITFIELD term : 1;
} opj_tcd_pass_t;

/**
FIXME DOC
*/
typedef struct opj_tcd_layer {
    OPJ_UINT32 numpasses;       /* Number of passes in the layer */
    OPJ_UINT32 len;             /* len of information */
    OPJ_FLOAT64 disto;          /* add for index (Cfr. Marcela) */
    OPJ_BYTE *data;             /* data */
} opj_tcd_layer_t;

/**
FIXME DOC
*/
typedef struct opj_tcd_cblk_enc {
    OPJ_BYTE* data;               /* Data */
    opj_tcd_layer_t* layers;      /* layer information */
    opj_tcd_pass_t* passes;       /* information about the passes */
    OPJ_INT32 x0, y0, x1,
              y1;     /* dimension of the code-blocks : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_UINT32 numbps;
    OPJ_UINT32 numlenbits;
    OPJ_UINT32 data_size;         /* Size of allocated data buffer */
    OPJ_UINT32
    numpasses;         /* number of pass already done for the code-blocks */
    OPJ_UINT32 numpassesinlayers; /* number of passes in the layer */
    OPJ_UINT32 totalpasses;       /* total number of passes */
} opj_tcd_cblk_enc_t;


/** Chunk of codestream data that is part of a code block */
typedef struct opj_tcd_seg_data_chunk {
    /* Point to tilepart buffer. We don't make a copy !
       So the tilepart buffer must be kept alive
       as long as we need to decode the codeblocks */
    OPJ_BYTE * data;
    OPJ_UINT32 len;                 /* Usable length of data */
} opj_tcd_seg_data_chunk_t;

/** Segment of a code-block.
 * A segment represent a number of consecutive coding passes, without termination
 * of MQC or RAW between them. */
typedef struct opj_tcd_seg {
    OPJ_UINT32 len;      /* Size of data related to this segment */
    /* Number of passes decoded. Including those that we skip */
    OPJ_UINT32 numpasses;
    /* Number of passes actually to be decoded. To be used for code-block decoding */
    OPJ_UINT32 real_num_passes;
    /* Maximum number of passes for this segment */
    OPJ_UINT32 maxpasses;
    /* Number of new passes for current packed. Transitory value */
    OPJ_UINT32 numnewpasses;
    /* Codestream length for this segment for current packed. Transitory value */
    OPJ_UINT32 newlen;
} opj_tcd_seg_t;

/** Code-block for decoding */
typedef struct opj_tcd_cblk_dec {
    opj_tcd_seg_t* segs;            /* segments information */
    opj_tcd_seg_data_chunk_t* chunks; /* Array of chunks */
    /* position of the code-blocks : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    /* Mb is The maximum number of bit-planes available for the representation of
       coefficients in any sub-band, b, as defined in Equation (E-2). See
       Section B.10.5 of the standard */
    OPJ_UINT32 Mb;  /* currently used only to check if HT decoding is correct */
    /* numbps is Mb - P as defined in Section B.10.5 of the standard */
    OPJ_UINT32 numbps;
    /* number of bits for len, for the current packet. Transitory value */
    OPJ_UINT32 numlenbits;
    /* number of pass added to the code-blocks, for the current packet. Transitory value */
    OPJ_UINT32 numnewpasses;
    /* number of segments, including those of packet we skip */
    OPJ_UINT32 numsegs;
    /* number of segments, to be used for code block decoding */
    OPJ_UINT32 real_num_segs;
    OPJ_UINT32 m_current_max_segs;  /* allocated number of segs[] items */
    OPJ_UINT32 numchunks;           /* Number of valid chunks items */
    OPJ_UINT32 numchunksalloc;      /* Number of chunks item allocated */
    /* Decoded code-block. Only used for subtile decoding. Otherwise tilec->data is directly updated */
    OPJ_INT32* decoded_data;
} opj_tcd_cblk_dec_t;

/** Precinct structure */
typedef struct opj_tcd_precinct {
    /* dimension of the precinct : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    OPJ_UINT32 cw, ch;              /* number of code-blocks, in width and height */
    union {                         /* code-blocks information */
        opj_tcd_cblk_enc_t* enc;
        opj_tcd_cblk_dec_t* dec;
        void*               blocks;
    } cblks;
    OPJ_UINT32 block_size;          /* size taken by cblks (in bytes) */
    opj_tgt_tree_t *incltree;       /* inclusion tree */
    opj_tgt_tree_t *imsbtree;       /* IMSB tree */
} opj_tcd_precinct_t;

/** Sub-band structure */
typedef struct opj_tcd_band {
    /* dimension of the subband : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    /* band number: for lowest resolution level (0=LL), otherwise (1=HL, 2=LH, 3=HH) */
    OPJ_UINT32 bandno;
    /* precinct information */
    opj_tcd_precinct_t *precincts;
    /* size of data taken by precincts */
    OPJ_UINT32 precincts_data_size;
    OPJ_INT32 numbps;
    OPJ_FLOAT32 stepsize;
} opj_tcd_band_t;

/** Tile-component resolution structure */
typedef struct opj_tcd_resolution {
    /* dimension of the resolution level : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    /* number of precincts, in width and height, for this resolution level */
    OPJ_UINT32 pw, ph;
    /* number of sub-bands for the resolution level (1 for lowest resolution level, 3 otherwise) */
    OPJ_UINT32 numbands;
    /* subband information */
    opj_tcd_band_t bands[3];

    /* dimension of the resolution limited to window of interest. Only valid if tcd->whole_tile_decoding is set */
    OPJ_UINT32 win_x0;
    OPJ_UINT32 win_y0;
    OPJ_UINT32 win_x1;
    OPJ_UINT32 win_y1;
} opj_tcd_resolution_t;

/** Tile-component structure */
typedef struct opj_tcd_tilecomp {
    /* dimension of component : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    /* component number */
    OPJ_UINT32 compno;
    /* number of resolutions level */
    OPJ_UINT32 numresolutions;
    /* number of resolutions level to decode (at max)*/
    OPJ_UINT32 minimum_num_resolutions;
    /* resolutions information */
    opj_tcd_resolution_t *resolutions;
    /* size of data for resolutions (in bytes) */
    OPJ_UINT32 resolutions_size;

    /* data of the component. For decoding, only valid if tcd->whole_tile_decoding is set (so exclusive of data_win member) */
    OPJ_INT32 *data;
    /* if true, then need to free after usage, otherwise do not free */
    OPJ_BOOL  ownsData;
    /* we may either need to allocate this amount of data, or re-use image data and ignore this value */
    size_t data_size_needed;
    /* size of the data of the component */
    size_t data_size;

    /** data of the component limited to window of interest. Only valid for decoding and if tcd->whole_tile_decoding is NOT set (so exclusive of data member) */
    OPJ_INT32 *data_win;
    /* dimension of the component limited to window of interest. Only valid for decoding and  if tcd->whole_tile_decoding is NOT set */
    OPJ_UINT32 win_x0;
    OPJ_UINT32 win_y0;
    OPJ_UINT32 win_x1;
    OPJ_UINT32 win_y1;

    /* number of pixels */
    OPJ_SIZE_T numpix;
} opj_tcd_tilecomp_t;


/**
FIXME DOC
*/
typedef struct opj_tcd_tile {
    /* dimension of the tile : left upper corner (x0, y0) right low corner (x1,y1) */
    OPJ_INT32 x0, y0, x1, y1;
    OPJ_UINT32 numcomps;            /* number of components in tile */
    opj_tcd_tilecomp_t *comps;  /* Components information */
    OPJ_SIZE_T numpix;               /* number of pixels */
    OPJ_FLOAT64 distotile;          /* distortion of the tile */
    OPJ_FLOAT64 distolayer[100];    /* distortion per layer */
    OPJ_UINT32 packno;              /* packet number */
} opj_tcd_tile_t;

/**
FIXME DOC
*/
typedef struct opj_tcd_image {
    opj_tcd_tile_t *tiles;      /* Tiles information */
}
opj_tcd_image_t;


/**
Tile coder/decoder
*/
typedef struct opj_tcd {
    /** Position of the tilepart flag in Progression order*/
    OPJ_INT32 tp_pos;
    /** Tile part number*/
    OPJ_UINT32 tp_num;
    /** Current tile part number*/
    OPJ_UINT32 cur_tp_num;
    /** Total number of tileparts of the current tile*/
    OPJ_UINT32 cur_totnum_tp;
    /** Current Packet iterator number */
    OPJ_UINT32 cur_pino;
    /** info on each image tile */
    opj_tcd_image_t *tcd_image;
    /** image header */
    opj_image_t *image;
    /** coding parameters */
    opj_cp_t *cp;
    /** coding/decoding parameters common to all tiles */
    opj_tcp_t *tcp;
    /** current encoded/decoded tile */
    OPJ_UINT32 tcd_tileno;
    /** tell if the tcd is a decoder. */
    OPJ_BITFIELD m_is_decoder : 1;
    /** Thread pool */
    opj_thread_pool_t* thread_pool;
    /** Coordinates of the window of interest, in grid reference space */
    OPJ_UINT32 win_x0;
    OPJ_UINT32 win_y0;
    OPJ_UINT32 win_x1;
    OPJ_UINT32 win_y1;
    /** Only valid for decoding. Whether the whole tile is decoded, or just the region in win_x0/win_y0/win_x1/win_y1 */
    OPJ_BOOL   whole_tile_decoding;
    /* Array of size image->numcomps indicating if a component must be decoded. NULL if all components must be decoded */
    OPJ_BOOL* used_component;
} opj_tcd_t;

/**
 * Structure to hold information needed to generate some markers.
 * Used by encoder.
 */
typedef struct opj_tcd_marker_info {
    /** In: Whether information to generate PLT markers in needed */
    OPJ_BOOL    need_PLT;

    /** OUT: Number of elements in p_packet_size[] array */
    OPJ_UINT32  packet_count;

    /** OUT: Array of size packet_count, such that p_packet_size[i] is
     *       the size in bytes of the ith packet */
    OPJ_UINT32* p_packet_size;
} opj_tcd_marker_info_t;

/** @name Exported functions */
/*@{*/
/* ----------------------------------------------------------------------- */

/**
Dump the content of a tcd structure
*/
/*void tcd_dump(FILE *fd, opj_tcd_t *tcd, opj_tcd_image_t *img);*/ /* TODO MSD shoul use the new v2 structures */

/**
Create a new TCD handle
@param p_is_decoder FIXME DOC
@return Returns a new TCD handle if successful returns NULL otherwise
*/
opj_tcd_t* opj_tcd_create(OPJ_BOOL p_is_decoder);

/**
Destroy a previously created TCD handle
@param tcd TCD handle to destroy
*/
void opj_tcd_destroy(opj_tcd_t *tcd);


/**
 * Create a new opj_tcd_marker_info_t* structure
 * @param need_PLT Whether information is needed to generate PLT markers.
 */
opj_tcd_marker_info_t* opj_tcd_marker_info_create(OPJ_BOOL need_PLT);


/**
Destroy a previously created opj_tcd_marker_info_t* structure
@param p_tcd_marker_info Structure to destroy
*/
void opj_tcd_marker_info_destroy(opj_tcd_marker_info_t *p_tcd_marker_info);


/**
 * Initialize the tile coder and may reuse some memory.
 * @param   p_tcd       TCD handle.
 * @param   p_image     raw image.
 * @param   p_cp        coding parameters.
 * @param   p_tp        thread pool
 *
 * @return true if the encoding values could be set (false otherwise).
*/
OPJ_BOOL opj_tcd_init(opj_tcd_t *p_tcd,
                      opj_image_t * p_image,
                      opj_cp_t * p_cp,
                      opj_thread_pool_t* p_tp);

/**
 * Allocates memory for decoding a specific tile.
 *
 * @param   p_tcd       the tile decoder.
 * @param   p_tile_no   the index of the tile received in sequence. This not necessarily lead to the
 * tile at index p_tile_no.
 * @param p_manager the event manager.
 *
 * @return  true if the remaining data is sufficient.
 */
OPJ_BOOL opj_tcd_init_decode_tile(opj_tcd_t *p_tcd, OPJ_UINT32 p_tile_no,
                                  opj_event_mgr_t* p_manager);

/**
 * Gets the maximum tile size that will be taken by the tile once decoded.
 */
OPJ_UINT32 opj_tcd_get_decoded_tile_size(opj_tcd_t *p_tcd,
        OPJ_BOOL take_into_account_partial_decoding);

/**
 * Encodes a tile from the raw image into the given buffer.
 * @param   p_tcd           Tile Coder handle
 * @param   p_tile_no       Index of the tile to encode.
 * @param   p_dest          Destination buffer
 * @param   p_data_written  pointer to an int that is incremented by the number of bytes really written on p_dest
 * @param   p_len           Maximum length of the destination buffer
 * @param   p_cstr_info     Codestream information structure
 * @param   p_marker_info   Marker information structure
 * @param   p_manager       the user event manager
 * @return  true if the coding is successful.
*/
OPJ_BOOL opj_tcd_encode_tile(opj_tcd_t *p_tcd,
                             OPJ_UINT32 p_tile_no,
                             OPJ_BYTE *p_dest,
                             OPJ_UINT32 * p_data_written,
                             OPJ_UINT32 p_len,
                             struct opj_codestream_info *p_cstr_info,
                             opj_tcd_marker_info_t* p_marker_info,
                             opj_event_mgr_t *p_manager);


/**
Decode a tile from a buffer into a raw image
@param tcd TCD handle
@param win_x0 Upper left x of region to decode (in grid coordinates)
@param win_y0 Upper left y of region to decode (in grid coordinates)
@param win_x1 Lower right x of region to decode (in grid coordinates)
@param win_y1 Lower right y of region to decode (in grid coordinates)
@param numcomps_to_decode  Size of the comps_indices array, or 0 if decoding all components.
@param comps_indices   Array of numcomps values representing the indices
                       of the components to decode (relative to the
                       codestream, starting at 0). Or NULL if decoding all components.
@param src Source buffer
@param len Length of source buffer
@param tileno Number that identifies one of the tiles to be decoded
@param cstr_info  FIXME DOC
@param manager the event manager.
*/
OPJ_BOOL opj_tcd_decode_tile(opj_tcd_t *tcd,
                             OPJ_UINT32 win_x0,
                             OPJ_UINT32 win_y0,
                             OPJ_UINT32 win_x1,
                             OPJ_UINT32 win_y1,
                             OPJ_UINT32 numcomps_to_decode,
                             const OPJ_UINT32 *comps_indices,
                             OPJ_BYTE *src,
                             OPJ_UINT32 len,
                             OPJ_UINT32 tileno,
                             opj_codestream_index_t *cstr_info,
                             opj_event_mgr_t *manager);


/**
 * Copies tile data from the system onto the given memory block.
 */
OPJ_BOOL opj_tcd_update_tile_data(opj_tcd_t *p_tcd,
                                  OPJ_BYTE * p_dest,
                                  OPJ_UINT32 p_dest_length);

/**
 * Get the size in bytes of the input buffer provided before encoded.
 * This must be the size provided to the p_src_length argument of
 * opj_tcd_copy_tile_data()
 */
OPJ_SIZE_T opj_tcd_get_encoder_input_buffer_size(opj_tcd_t *p_tcd);

/**
 * Initialize the tile coder and may reuse some meory.
 *
 * @param   p_tcd       TCD handle.
 * @param   p_tile_no   current tile index to encode.
 * @param p_manager the event manager.
 *
 * @return true if the encoding values could be set (false otherwise).
*/
OPJ_BOOL opj_tcd_init_encode_tile(opj_tcd_t *p_tcd,
                                  OPJ_UINT32 p_tile_no, opj_event_mgr_t* p_manager);

/**
 * Copies tile data from the given memory block onto the system.
 *
 * p_src_length must be equal to opj_tcd_get_encoder_input_buffer_size()
 */
OPJ_BOOL opj_tcd_copy_tile_data(opj_tcd_t *p_tcd,
                                OPJ_BYTE * p_src,
                                OPJ_SIZE_T p_src_length);

/**
 * Allocates tile component data
 *
 *
 */
OPJ_BOOL opj_alloc_tile_component_data(opj_tcd_tilecomp_t *l_tilec);

/** Returns whether a sub-band is empty (i.e. whether it has a null area)
 * @param band Sub-band handle.
 * @return OPJ_TRUE whether the sub-band is empty.
 */
OPJ_BOOL opj_tcd_is_band_empty(opj_tcd_band_t* band);

/** Reinitialize a segment */
void opj_tcd_reinit_segment(opj_tcd_seg_t* seg);


/** Returns whether a sub-band region contributes to the area of interest
 * tcd->win_x0,tcd->win_y0,tcd->win_x1,tcd->win_y1.
 *
 * @param tcd    TCD handle.
 * @param compno Component number
 * @param resno  Resolution number
 * @param bandno Band number (*not* band index, ie 0, 1, 2 or 3)
 * @param x0     Upper left x in subband coordinates
 * @param y0     Upper left y in subband coordinates
 * @param x1     Lower right x in subband coordinates
 * @param y1     Lower right y in subband coordinates
 * @return OPJ_TRUE whether the sub-band region contributs to the area of
 *                  interest.
 */
OPJ_BOOL opj_tcd_is_subband_area_of_interest(opj_tcd_t *tcd,
        OPJ_UINT32 compno,
        OPJ_UINT32 resno,
        OPJ_UINT32 bandno,
        OPJ_UINT32 x0,
        OPJ_UINT32 y0,
        OPJ_UINT32 x1,
        OPJ_UINT32 y1);

/* ----------------------------------------------------------------------- */
/*@}*/

/*@}*/

#endif /* OPJ_TCD_H */
