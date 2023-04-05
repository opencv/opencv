/* -*- C++ -*-
 * File: libraw_internal_funcs.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  14, 2008

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#ifndef _LIBRAW_INTERNAL_FUNCS_H
#define _LIBRAW_INTERNAL_FUNCS_H

#ifndef LIBRAW_LIBRARY_BUILD
#error "This file should be used only for libraw library build"
#else

/* inline functions */
	static int stread(char *buf, size_t len, LibRaw_abstract_datastream *fp);
	static int getwords(char *line, char *words[], int maxwords, int maxlen);
	static void remove_trailing_spaces(char *string, size_t len);
	static void remove_caseSubstr(char *string, char *remove);
	static void removeExcessiveSpaces(char *string);
	static void trimSpaces(char *s);
/* static tables/variables */
	static libraw_static_table_t tagtype_dataunit_bytes;
	static libraw_static_table_t Canon_wbi2std;
	static libraw_static_table_t Canon_KeyIsZero_Len2048_linenums_2_StdWBi;
	static libraw_static_table_t Canon_KeyIs0x0410_Len3072_linenums_2_StdWBi;
	static libraw_static_table_t Canon_KeyIs0x0410_Len2048_linenums_2_StdWBi;
	static libraw_static_table_t Canon_D30_linenums_2_StdWBi;
	static libraw_static_table_t Canon_G9_linenums_2_StdWBi;

	static libraw_static_table_t Fuji_wb_list1;
	static libraw_static_table_t FujiCCT_K;
	static libraw_static_table_t Fuji_wb_list2;

	static libraw_static_table_t Pentax_wb_list1;
	static libraw_static_table_t Pentax_wb_list2;

	static libraw_static_table_t Oly_wb_list1;
	static libraw_static_table_t Oly_wb_list2;

	static libraw_static_table_t Sony_SRF_wb_list;
	static libraw_static_table_t Sony_SR2_wb_list;
	static libraw_static_table_t Sony_SR2_wb_list1;
/*  */
	int	find_ifd_by_offset(int );
	void 	libraw_swab(void *arr, size_t len);
	ushort	sget2 (uchar *s);
	ushort	sget2Rev(uchar *s);
	libraw_area_t	get_CanonArea();
	int	parseCR3(INT64 oAtomList, INT64 szAtomList, short &nesting, char *AtomNameStack, short& nTrack, short &TrackType);
	void 	selectCRXTrack();
	void    parseCR3_Free();
	int     parseCR3_CTMD(short trackNum);
	int     selectCRXFrame(short trackNum, unsigned frameIndex);
	void	setCanonBodyFeatures (unsigned long long id);
	void	processCanonCameraInfo (unsigned long long id, uchar *CameraInfo, unsigned maxlen, unsigned type, unsigned dng_writer);
	static float _CanonConvertAperture(ushort in);
	void	Canon_CameraSettings(unsigned len);
	void	Canon_WBpresets (int skip1, int skip2);
	void	Canon_WBCTpresets (short WBCTversion);
	void	parseCanonMakernotes (unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	processNikonLensData (uchar *LensData, unsigned len);
	void	Nikon_NRW_WBtag (int wb, int skip);
	void	parseNikonMakernote (int base, int uptag, unsigned dng_writer);
	void	parseEpsonMakernote (int base, int uptag, unsigned dng_writer);
	void	parseSigmaMakernote (int base, int uptag, unsigned dng_writer);
	void	setOlympusBodyFeatures (unsigned long long id);
	void	getOlympus_CameraType2 ();
	void	getOlympus_SensorTemperature (unsigned len);
	void	parseOlympus_Equipment (unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	parseOlympus_CameraSettings (int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	parseOlympus_ImageProcessing (unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	parseOlympus_RawInfo (unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void parseOlympusMakernotes (int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	setPhaseOneFeatures (unsigned long long id);
	void	setPentaxBodyFeatures (unsigned long long id);
	void	PentaxISO (ushort c);
	void	PentaxLensInfo (unsigned long long id, unsigned len);
	void	parsePentaxMakernotes(int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	parseRicohMakernotes(int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	void	parseSamsungMakernotes(int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
	void	fixupArri();
#endif
	void	setSonyBodyFeatures (unsigned long long id);
	void	parseSonyLensType2 (uchar a, uchar b);
	void	parseSonyLensFeatures (uchar a, uchar b);
	void	process_Sony_0x0116 (uchar * buf, ushort, unsigned long long id);
	void	process_Sony_0x2010 (uchar * buf, ushort);
	void	process_Sony_0x9050 (uchar * buf, ushort, unsigned long long id);
	void	process_Sony_0x9400 (uchar * buf, ushort, unsigned long long id);
	void	process_Sony_0x9402 (uchar * buf, ushort);
	void	process_Sony_0x9403 (uchar * buf, ushort);
	void	process_Sony_0x9406 (uchar * buf, ushort);
	void	process_Sony_0x940c (uchar * buf, ushort);
	void	process_Sony_0x940e (uchar * buf, ushort, unsigned long long id);
	void	parseSonyMakernotes (int base, unsigned tag, unsigned type, unsigned len, unsigned dng_writer,
                               uchar *&table_buf_0x0116, ushort &table_buf_0x0116_len,
                               uchar *&table_buf_0x2010, ushort &table_buf_0x2010_len,
                               uchar *&table_buf_0x9050, ushort &table_buf_0x9050_len,
                               uchar *&table_buf_0x9400, ushort &table_buf_0x9400_len,
                               uchar *&table_buf_0x9402, ushort &table_buf_0x9402_len,
                               uchar *&table_buf_0x9403, ushort &table_buf_0x9403_len,
                               uchar *&table_buf_0x9406, ushort &table_buf_0x9406_len,
                               uchar *&table_buf_0x940c, ushort &table_buf_0x940c_len,
                               uchar *&table_buf_0x940e, ushort &table_buf_0x940e_len);
	void	parseSonySR2 (uchar *cbuf_SR2, unsigned SR2SubIFDOffset, unsigned SR2SubIFDLength, unsigned dng_writer);
	void	parseSonySRF (unsigned len);
	void	parseFujiMakernotes (unsigned tag, unsigned type, unsigned len, unsigned dng_writer);
	const char* HassyRawFormat_idx2HR(unsigned idx);
	void	process_Hassy_Lens (int LensMount);
	void parseHassyModel ();

	void	setLeicaBodyFeatures(int LeicaMakernoteSignature);
	void	parseLeicaLensID();
	int 	parseLeicaLensName(unsigned len);
	int 	parseLeicaInternalBodySerial(unsigned len);
	void	parseLeicaMakernote(int base, int uptag, unsigned MakernoteTagType);
	void	parseAdobePanoMakernote ();
	void	parseAdobeRAFMakernote ();
	void	GetNormalizedModel ();
	void	SetStandardIlluminants (unsigned, const char* );

	ushort      get2();
	unsigned    sget4 (uchar *s);
	unsigned    getint(int type);
	float       int_to_float (int i);
	double      getreal (int type);
	double      sgetreal(int type, uchar *s);
	void        read_shorts (ushort *pixel, unsigned count);

/* Canon P&S cameras */
	void        canon_600_fixed_wb (int temp);
	int         canon_600_color (int ratio[2], int mar);
	void        canon_600_auto_wb();
	void        canon_600_coeff();
	void        canon_600_load_raw();
	void        canon_600_correct();
	int         canon_s2is();
	void        parse_ciff (int offset, int length, int);
	void        ciff_block_1030();


// LJPEG decoder
	unsigned    getbithuff (int nbits, ushort *huff);
	ushort*     make_decoder_ref (const uchar **source);
	ushort*     make_decoder (const uchar *source);
	int         ljpeg_start (struct jhead *jh, int info_only);
	void        ljpeg_end(struct jhead *jh);
	int         ljpeg_diff (ushort *huff);
	ushort *    ljpeg_row (int jrow, struct jhead *jh);
	ushort *    ljpeg_row_unrolled (int jrow, struct jhead *jh);
	void	    ljpeg_idct (struct jhead *jh);
	unsigned    ph1_bithuff (int nbits, ushort *huff);

// Canon DSLRs
	void        crw_init_tables (unsigned table, ushort *huff[2]);
	int         canon_has_lowbits();
	void        canon_load_raw();
	void        lossless_jpeg_load_raw();
	void        canon_sraw_load_raw();
// Adobe DNG
	void        adobe_copy_pixel (unsigned int row, unsigned int col, ushort **rp);
	void        lossless_dng_load_raw();
	void        deflate_dng_load_raw();
	void        packed_dng_load_raw();
    void        packed_tiled_dng_load_raw();
    void        uncompressed_fp_dng_load_raw();
	void        lossy_dng_load_raw();
//void        adobe_dng_load_raw_nc();

// Pentax
	void        pentax_load_raw();
	void	pentax_4shot_load_raw();

	void        pentax_tree();

// Nikon (and Minolta Z2)
	void        nikon_load_raw();
    void        nikon_he_load_raw_placeholder();
	void        nikon_read_curve();
	void        nikon_load_striped_packed_raw();
	void        nikon_load_padded_packed_raw();
	void        nikon_load_sraw();
	void        nikon_yuv_load_raw();
	void        nikon_coolscan_load_raw();
	int         nikon_e995();
	int         nikon_e2100();
	void        nikon_3700();
	int         minolta_z2();
//	void        nikon_e2100_load_raw();

// Fuji
//	void        fuji_load_raw();
	int         guess_RAFDataGeneration (uchar *RAFData_start);
	void        parse_fuji (int offset);
    void        parse_fuji_thumbnail(int offset);
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
// RedCine
	void        parse_redcine();
	void        redcine_load_raw();
#endif

// Rollei
	void        rollei_load_raw();
	void        parse_rollei();

// Contax
	void        parse_kyocera ();

// MF backs
//int         bayer (unsigned row, unsigned col);
	int         p1raw(unsigned,unsigned);
	void        phase_one_flat_field (int is_float, int nc);
	int 	    p1rawc(unsigned row, unsigned col, unsigned& count);
	void 	    phase_one_fix_col_pixel_avg(unsigned row, unsigned col);
	void 	    phase_one_fix_pixel_grad(unsigned row, unsigned col);
	void        phase_one_load_raw();
	unsigned    ph1_bits (int nbits);
	void        phase_one_load_raw_c();
    void		phase_one_load_raw_s();
	void        hasselblad_load_raw();
	void        leaf_hdr_load_raw();
	void        sinar_4shot_load_raw();
	void        imacon_full_load_raw();
	void        hasselblad_full_load_raw();
	void        packed_load_raw();
	float       find_green(int,int,int,int);
	void        unpacked_load_raw();
	void        unpacked_load_raw_FujiDBP();
	void        unpacked_load_raw_reversed();
	void        unpacked_load_raw_fuji_f700s20();
	void        parse_sinar_ia();
	void        parse_phase_one (int base);

// Misc P&S cameras
	void        parse_broadcom();
	void        broadcom_load_raw();
	void        nokia_load_raw();
	void        android_loose_load_raw();
	void        android_tight_load_raw();
#ifdef LIBRAW_OLD_VIDEO_SUPPORT
    void        canon_rmf_load_raw();
#endif
	unsigned    pana_data (int nb, unsigned *bytes);
	void        panasonic_load_raw();
//	void        panasonic_16x10_load_raw();
	void        olympus_load_raw();
//	void        olympus_cseries_load_raw();
	void        minolta_rd175_load_raw();
	void        quicktake_100_load_raw();
	const int*  make_decoder_int (const int *source, int level);
	int         radc_token (int tree);
	void        kodak_radc_load_raw();
	void        kodak_jpeg_load_raw();
	void        kodak_dc120_load_raw();
	void        eight_bit_load_raw();
	void        smal_decode_segment (unsigned seg[2][2], int holes);
	void        smal_v6_load_raw();
	int         median4 (int *p);
	void        fill_holes (int holes);
	void        smal_v9_load_raw();
	void        parse_riff(int maxdepth);
	void        parse_cine();
	void        parse_smal (int offset, int fsize);
	int         parse_jpeg (int offset);

// Kodak
	void        kodak_262_load_raw();
	int         kodak_65000_decode (short *out, int bsize);
	void        kodak_65000_load_raw();
	void        kodak_rgb_load_raw();
	void        kodak_ycbcr_load_raw();
//	void        kodak_yrgb_load_raw();
	void        kodak_c330_load_raw();
	void        kodak_c603_load_raw();
	void        kodak_rgb_load_thumb();
	void        kodak_ycbcr_load_thumb();
	void        vc5_dng_load_raw_placeholder();
// It's a Sony (and K&M)
	void        sony_decrypt (unsigned *data, int len, int start, int key);
	void        sony_load_raw();
	void        sony_arw_load_raw();
	void        sony_arw2_load_raw();
	void        sony_arq_load_raw();
	void        sony_ljpeg_load_raw();
	void        samsung_load_raw();
	void        samsung2_load_raw();
	void        samsung3_load_raw();
	void        parse_minolta (int base);

#ifdef USE_X3FTOOLS
// Foveon/Sigma
// We always have x3f code compiled in!
	void        parse_x3f();
	void        x3f_load_raw();
	void        x3f_dpq_interpolate_rg();
	void        x3f_dpq_interpolate_af(int xstep, int ystep, int scale); // 1x1 af pixels
	void        x3f_dpq_interpolate_af_sd(int xstart,int ystart, int xend, int yend, int xstep, int ystep, int scale); // sd Quattro interpolation
#else
	void        parse_x3f() {}
	void        x3f_load_raw(){}
#endif
#ifdef USE_6BY9RPI
	void		rpi_load_raw8();
	void		rpi_load_raw12();
	void		rpi_load_raw14();
	void		rpi_load_raw16();
	void		parse_raspberrypi();
#endif

// CAM/RGB
	void        pseudoinverse (double (*in)[3], double (*out)[3], int size);
	void        simple_coeff (int index);

// Openp
	char** malloc_omp_buffers(int buffer_count, size_t buffer_size);
	void free_omp_buffers(char** buffers, int buffer_count);


// Tiff/Exif parsers
	void        tiff_get (unsigned base,unsigned *tag, unsigned *type, unsigned *len, unsigned *save);
	short       tiff_sget(unsigned save, uchar *buf, unsigned buf_len, INT64 *tag_offset,
                          unsigned *tag_id, unsigned *tag_type, INT64 *tag_dataoffset,
                          unsigned *tag_datalen, int *tag_dataunit_len);
	void        parse_thumb_note (int base, unsigned toff, unsigned tlen);
	void        parse_makernote (int base, int uptag);
	void        parse_makernote_0xc634(int base, int uptag, unsigned dng_writer);
	void        parse_exif (int base);
	void        parse_exif_interop(int base);
	void        linear_table(unsigned len);
	void        Kodak_DCR_WBtags(int wb, unsigned type, int wbi);
	void        Kodak_KDC_WBtags(int wb, int wbi);
	short       KodakIllumMatrix (unsigned type, float *romm_camIllum);
	void        parse_kodak_ifd (int base);
	int         parse_tiff_ifd (int base);
	int         parse_tiff (int base);
	void        apply_tiff(void);
	void        parse_gps (int base);
	void        parse_gps_libraw(int base);
	void        aRGB_coeff(double aRGB_cam[3][3]);
	void        romm_coeff(float romm_cam[3][3]);
	void        parse_mos (INT64 offset);
	void        parse_qt (int end);
	void        get_timestamp (int reversed);

// The identify
    short       guess_byte_order (int words);
	void		identify_process_dng_fields();
	void		identify_finetune_pentax();
	void		identify_finetune_by_filesize(int);
	void		identify_finetune_dcr(char head[64],int,int);
// Tiff writer
	void        tiff_set(struct tiff_hdr *th, ushort *ntag,ushort tag, ushort type, int count, int val);
	void        tiff_head (struct tiff_hdr *th, int full);

// split AHD code
	void ahd_interpolate_green_h_and_v(int top, int left, ushort (*out_rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3]);
	void ahd_interpolate_r_and_b_in_rgb_and_convert_to_cielab(int top, int left, ushort (*inout_rgb)[LIBRAW_AHD_TILE][3], short (*out_lab)[LIBRAW_AHD_TILE][3]);
	void ahd_interpolate_r_and_b_and_convert_to_cielab(int top, int left, ushort (*inout_rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3], short (*out_lab)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3]);
	void ahd_interpolate_build_homogeneity_map(int top, int left, short (*lab)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3], char (*out_homogeneity_map)[LIBRAW_AHD_TILE][2]);
	void ahd_interpolate_combine_homogeneous_pixels(int top, int left, ushort (*rgb)[LIBRAW_AHD_TILE][LIBRAW_AHD_TILE][3], char (*homogeneity_map)[LIBRAW_AHD_TILE][2]);

	void init_fuji_compr(struct fuji_compressed_params* info);
	void init_fuji_block(struct fuji_compressed_block* info, const struct fuji_compressed_params *params, INT64 raw_offset, unsigned dsize);
	void copy_line_to_xtrans(struct fuji_compressed_block* info, int cur_line, int cur_block, int cur_block_width);
	void copy_line_to_bayer(struct fuji_compressed_block* info, int cur_line, int cur_block, int cur_block_width);
	void xtrans_decode_block(struct fuji_compressed_block* info, const struct fuji_compressed_params *params, int cur_line);
	void fuji_bayer_decode_block(struct fuji_compressed_block* info, const struct fuji_compressed_params *params, int cur_line);
	void fuji_compressed_load_raw();
	void fuji_14bit_load_raw();
	void parse_fuji_compressed_header();
	void crxLoadRaw();
	int  crxParseImageHeader(uchar *cmp1TagData, int nTrack, int size);
	void panasonicC6_load_raw();
	void panasonicC7_load_raw();

	void nikon_14bit_load_raw();

// DCB
	void  	dcb_pp();
	void  	dcb_copy_to_buffer(float (*image2)[3]);
	void  	dcb_restore_from_buffer(float (*image2)[3]);
	void  	dcb_color();
	void  	dcb_color_full();
	void  	dcb_map();
	void  	dcb_correction();
	void  	dcb_correction2();
	void  	dcb_refinement();
	void  	rgb_to_lch(double (*image3)[3]);
	void  	lch_to_rgb(double (*image3)[3]);
	void  	fbdd_correction();
	void  	fbdd_correction2(double (*image3)[3]);
	void  	fbdd_green();
	void  	dcb_ver(float (*image3)[3]);
	void 	dcb_hor(float (*image2)[3]);
	void 	dcb_color2(float (*image2)[3]);
	void 	dcb_color3(float (*image3)[3]);
	void 	dcb_decide(float (*image2)[3], float (*image3)[3]);
	void 	dcb_nyquist();
#endif

#endif
