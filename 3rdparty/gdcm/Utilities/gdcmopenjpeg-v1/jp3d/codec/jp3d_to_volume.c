/*
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2005, Francois Devaux and Antonin Descampe
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2002-2005, Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
 * Copyright (c) 2006, Mónica Díez García, Image Processing Laboratory, University of Valladolid, Spain
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "../libjp3dvm/openjpeg3d.h"
#include "getopt.h"
#include "convert.h"

#ifdef _WIN32
#include <windows.h>
#else
#define stricmp strcasecmp
#define strnicmp strncasecmp
#endif /* _WIN32 */

/* ----------------------------------------------------------------------- */
static double calc_PSNR(opj_volume_t *original, opj_volume_t *decoded)
{
	int max, i, k, compno = 0, size;
	double sum, total = 0;
	int global = 1;
    
	max = (original->comps[compno].prec <= 8) ? 255 : (1 << original->comps[compno].prec) - 1;
	if (global) {
		size = (original->x1 - original->x0) * (original->y1 - original->y0) * (original->z1 - original->z0);

		for (compno = 0; compno < original->numcomps; compno++) {
			for(sum = 0, i = 0; i < size; ++i) {
				if ((decoded->comps[compno].data[i] < 0) || (decoded->comps[compno].data[i] > max))
					fprintf(stdout,"[WARNING] Data out of range during PSNR computing...\n");
				else
					sum += (original->comps[compno].data[i] - decoded->comps[compno].data[i]) * (original->comps[compno].data[i] - decoded->comps[compno].data[i]);        
			}
		}
		sum /= size;
		total = ((sum==0.0) ? 0.0 : 10 * log10(max * max / sum));
	} else {
		size = (original->x1 - original->x0) * (original->y1 - original->y0);

		for (k = 0; k < original->z1 - original->z0; k++) {
			int offset = k * size;
			for (sum = 0, compno = 0; compno < original->numcomps; compno++) {
				for(i = 0; i < size; ++i) {
					if ((decoded->comps[compno].data[i + offset] < 0) || (decoded->comps[compno].data[i + offset] > max))
						fprintf(stdout,"[WARNING] Data out of range during PSNR computing...\n");
					else
						sum += (original->comps[compno].data[i + offset] - decoded->comps[compno].data[i + offset]) * (original->comps[compno].data[i + offset] - decoded->comps[compno].data[i + offset]);        
				}
			}
			sum /= size;
			total = total + ((sum==0.0) ? 0.0 : 10 * log10(max * max / sum));
		}

	}
	if(total == 0) /* perfect reconstruction, PSNR should return infinity */
		return -1.0;
	
	return total;
	//return 20 * log10((max - 1) / sqrt(sum));
}

static double calc_SSIM(opj_volume_t *original, opj_volume_t *decoded)
{
	int max, i, compno = 0, size, sizeM;
	double sum;
	double mux = 0.0, muy = 0.0, sigmax = 0.0, sigmay = 0.0,
		sigmaxy = 0.0, structx = 0.0, structy = 0.0;
	double lcomp,ccomp,scomp;
	double C1,C2,C3;

	max = (original->comps[compno].prec <= 8) ? 255 : (1 << original->comps[compno].prec) - 1;
	size = (original->x1 - original->x0) * (original->y1 - original->y0) * (original->z1 - original->z0);

	//MSSIM

//	sizeM = size / (original->z1 - original->z0);

	sizeM = size;	
	for(sum = 0, i = 0; i < sizeM; ++i) {
		// First, the luminance of each signal is compared.
		mux += original->comps[compno].data[i];
		muy += decoded->comps[compno].data[i];
	}
	mux /= sizeM;
	muy /= sizeM;
	
	//We use the standard deviation (the square root of variance) as an estimate of the signal contrast.
    for(sum = 0, i = 0; i < sizeM; ++i) {
		// First, the luminance of each signal is compared.
		sigmax += (original->comps[compno].data[i] - mux) * (original->comps[compno].data[i] - mux);
		sigmay += (decoded->comps[compno].data[i] - muy) * (decoded->comps[compno].data[i] - muy);
		sigmaxy += (original->comps[compno].data[i] - mux) * (decoded->comps[compno].data[i] - muy);
	}
	sigmax /= sizeM - 1;
	sigmay /= sizeM - 1;
	sigmaxy /= sizeM - 1;
	
	sigmax = sqrt(sigmax);
	sigmay = sqrt(sigmay);
	sigmaxy = sqrt(sigmaxy);

	//Third, the signal is normalized (divided) by its own standard deviation, 
	//so that the two signals being compared have unit standard deviation.

	//Luminance comparison
	C1 = (0.01 * max) * (0.01 * max);
	lcomp = ((2 * mux * muy) + C1)/((mux*mux) + (muy*mux) + C1);
	//Constrast comparison
	C2 = (0.03 * max) * (0.03 * max);
	ccomp = ((2 * sigmax * sigmay) + C2)/((sigmax*sigmax) + (sigmay*sigmay) + C2);
	//Structure comparison
	C3 = C2 / 2;
	scomp = (sigmaxy + C3) / (sigmax * sigmay + C3);
	//Similarity measure

	sum = lcomp * ccomp * scomp;
	return sum;
}

void decode_help_display() {
	fprintf(stdout,"HELP\n----\n\n");
	fprintf(stdout,"- the -h option displays this help information on screen\n\n");

	fprintf(stdout,"List of parameters for the JPEG 2000 encoder:\n");
	fprintf(stdout,"\n");
	fprintf(stdout," Required arguments \n");
	fprintf(stdout," ---------------------------- \n");
	fprintf(stdout,"  -i <compressed file> ( *.jp3d, *.j3d )\n");
	fprintf(stdout,"    Currently accepts J3D-files. The file type is identified based on its suffix.\n");
	fprintf(stdout,"  -o <decompressed file> ( *.pgx, *.bin )\n");
	fprintf(stdout,"    Currently accepts PGX-files and BIN-files. Binary data is written to the file (not ascii). \n");
	fprintf(stdout,"    If a PGX filename is given, there will be as many output files as slices; \n");
	fprintf(stdout,"    an indice starting from 0 will then be appended to the output filename,\n");
	fprintf(stdout,"    just before the \"pgx\" extension.\n");
	fprintf(stdout,"  -m <characteristics file> ( *.img ) \n");
	fprintf(stdout,"    Required only for BIN-files. Ascii data of volume characteristics is written. \n");
	fprintf(stdout,"\n");
	fprintf(stdout," Optional  \n");
	fprintf(stdout," ---------------------------- \n");
	fprintf(stdout,"  -h \n ");
	fprintf(stdout,"    Display the help information\n");
	fprintf(stdout,"  -r <RFx,RFy,RFz>\n");
	fprintf(stdout,"    Set the number of highest resolution levels to be discarded on each dimension. \n");
	fprintf(stdout,"    The volume resolution is effectively divided by 2 to the power of the\n");
	fprintf(stdout,"    number of discarded levels. The reduce factor is limited by the\n");
	fprintf(stdout,"    smallest total number of decomposition levels among tiles.\n");
	fprintf(stdout,"  -l <number of quality layers to decode>\n");
	fprintf(stdout,"    Set the maximum number of quality layers to decode. If there are\n");
	fprintf(stdout,"    less quality layers than the specified number, all the quality layers\n");
	fprintf(stdout,"    are decoded. \n");
	fprintf(stdout,"  -O original-file \n");
    fprintf(stdout,"    This option offers the possibility to compute some quality results  \n");
	fprintf(stdout,"    for the decompressed volume, like the PSNR value achieved or the global SSIM value.  \n");
	fprintf(stdout,"    Needs the original file in order to compare with the new one.\n");
    fprintf(stdout,"    NOTE: Only valid when -r option is 0,0,0 (both original and decompressed volumes have same resolutions) \n");
    fprintf(stdout,"    NOTE: If original file is .BIN file, the volume characteristics file shall be defined with the -m option. \n");
	fprintf(stdout,"    (i.e. -O original-BIN-file -m original-IMG-file) \n");
	fprintf(stdout,"  -BE \n");
	fprintf(stdout,"    Define that the recovered volume data will be saved with big endian byte order.\n");
	fprintf(stdout,"    By default, little endian byte order is used.\n");
	fprintf(stdout,"\n");
}

/* -------------------------------------------------------------------------- */

int get_file_format(char *filename) {
	int i;
	static const char *extension[] = {"pgx", "bin", "j3d", "jp3d", "j2k", "img"};
	static const int format[] = { PGX_DFMT, BIN_DFMT, J3D_CFMT, J3D_CFMT, J2K_CFMT, IMG_DFMT};
	char * ext = strrchr(filename, '.');
	if(ext) {
		ext++;
		for(i = 0; i < sizeof(format) / sizeof(format[0]); i++) {
			if(strnicmp(ext, extension[i], 3) == 0) {
				return format[i];
			}
		}
	}

	return -1;
}

/* -------------------------------------------------------------------------- */

int parse_cmdline_decoder(int argc, char **argv, opj_dparameters_t *parameters) {
	/* parse the command line */

	while (1) {
		int c = getopt(argc, argv, "i:o:O:r:l:B:m:h");
		if (c == -1)			  
			break;
		switch (c) {
			case 'i':			/* input file */
			{
				char *infile = optarg;
				parameters->decod_format = get_file_format(infile);
				switch(parameters->decod_format) {
					case J3D_CFMT:
					case J2K_CFMT:
						break;
					default:
						fprintf(stdout, "[ERROR] Unknown format for infile %s [only *.j3d]!! \n", infile);
						return 1;
						break;
				}
				strncpy(parameters->infile, infile, MAX_PATH);
				fprintf(stdout,	"[INFO] Infile: %s \n", parameters->infile);

			}
			break;

			case 'm':			/* img file */
			{
				char *imgfile = optarg;
				int imgformat = get_file_format(imgfile);
				switch(imgformat) {
					case IMG_DFMT:
						break;
					default:
						fprintf(stdout,	"[ERROR] Unrecognized format for imgfile : %s [accept only *.img] !!\n\n", imgfile);
						return 1;
						break;
				}
				strncpy(parameters->imgfile, imgfile, MAX_PATH);
				fprintf(stdout,	"[INFO] Imgfile: %s Format: %d\n", parameters->imgfile, imgformat);
			}
			break;
				
				/* ----------------------------------------------------- */

			case 'o':			/* output file */
			{
				char *outfile = optarg;
				parameters->cod_format = get_file_format(outfile);
				switch(parameters->cod_format) {
					case PGX_DFMT:
					case BIN_DFMT:
						break;
					default:
						fprintf(stdout,	"[ERROR] Unrecognized format for outfile : %s [accept only *.pgx or *.bin] !!\n\n", outfile);
						return 1;
						break;
				}
				strncpy(parameters->outfile, outfile, MAX_PATH);
				fprintf(stdout,	"[INFO] Outfile: %s \n", parameters->outfile);

			}
			break;
			
				/* ----------------------------------------------------- */

			case 'O':		/* Original image for PSNR computing */
			{
				char *original = optarg;
				parameters->orig_format = get_file_format(original);
				switch(parameters->orig_format) {
					case PGX_DFMT:
					case BIN_DFMT:
						break;
					default:
						fprintf(stdout,	"[ERROR] Unrecognized format for original file : %s [accept only *.pgx or *.bin] !!\n\n", original);
						return 1;
						break;
				}
				strncpy(parameters->original, original, MAX_PATH);
				fprintf(stdout,	"[INFO] Original file: %s \n", parameters->original);
			}
			break;

				/* ----------------------------------------------------- */
	    
			case 'r':		/* reduce option */
			{
				//sscanf(optarg, "%d, %d, %d", &parameters->cp_reduce[0], &parameters->cp_reduce[1], &parameters->cp_reduce[2]);
				int aux;
				aux = sscanf(optarg, "%d,%d,%d", &parameters->cp_reduce[0], &parameters->cp_reduce[1], &parameters->cp_reduce[2]);
				if (aux == 2) 
					parameters->cp_reduce[2] = 0;
				else if (aux == 1) {
					parameters->cp_reduce[1] = parameters->cp_reduce[0];
					parameters->cp_reduce[2] = 0;
				}else if (aux == 0){
					parameters->cp_reduce[0] = 0;
					parameters->cp_reduce[1] = 0;
					parameters->cp_reduce[2] = 0;
				}
			}
			break;
			
				/* ----------------------------------------------------- */

			case 'l':		/* layering option */
			{
				sscanf(optarg, "%d", &parameters->cp_layer);
			}
			break;

				/* ----------------------------------------------------- */

			case 'B':		/* BIGENDIAN vs. LITTLEENDIAN */
			{
				parameters->bigendian = 1;
			}
			break;
			
				/* ----------------------------------------------------- */

			case 'L':		/* BIGENDIAN vs. LITTLEENDIAN */
			{
				parameters->decod_format = LSE_CFMT;
			}
			break;
			
			/* ----------------------------------------------------- */
			
			case 'h': 			/* display an help description */
			{
				decode_help_display();
				return 1;
			}
			break;
            
				/* ----------------------------------------------------- */
			
			default:
				fprintf(stdout,"[WARNING] This option is not valid \"-%c %s\"\n",c, optarg);
				break;
		}
	}

	/* check for possible errors */

	if((parameters->infile[0] == 0) || (parameters->outfile[0] == 0)) {
		fprintf(stdout,"[ERROR] At least one required argument is missing\n Check jp3d_to_volume -help for usage information\n");
		return 1;
	}

	return 0;
}

/* -------------------------------------------------------------------------- */

/**
sample error callback expecting a FILE* client object
*/
void error_callback(const char *msg, void *client_data) {
	FILE *stream = (FILE*)client_data;
	fprintf(stream, "[ERROR] %s", msg);
}
/**
sample warning callback expecting a FILE* client object
*/
void warning_callback(const char *msg, void *client_data) {
	FILE *stream = (FILE*)client_data;
	fprintf(stream, "[WARNING] %s", msg);
}
/**
sample debug callback expecting no client object
*/
void info_callback(const char *msg, void *client_data) {
	fprintf(stdout, "[INFO] %s", msg);
}

/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {

	opj_dparameters_t parameters;	/* decompression parameters */
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_volume_t *volume = NULL;

	opj_volume_t *original = NULL;
	opj_cparameters_t cparameters;	/* original parameters */

	FILE *fsrc = NULL;
	unsigned char *src = NULL; 
	int file_length;
	int decodeok;
	double psnr, ssim;

	opj_dinfo_t* dinfo = NULL;	/* handle to a decompressor */
	opj_cio_t *cio = NULL;

	/* configure the event callbacks (not required) */
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = error_callback;
	event_mgr.warning_handler = warning_callback;
	event_mgr.info_handler = info_callback;

	/* set decoding parameters to default values */
	opj_set_default_decoder_parameters(&parameters);

    /* parse input and get user decoding parameters */
	strcpy(parameters.original,"NULL");
	strcpy(parameters.imgfile,"NULL");
	if(parse_cmdline_decoder(argc, argv, &parameters) == 1) {
		return 0;
	}
	
	/* read the input file and put it in memory */
	/* ---------------------------------------- */
	fprintf(stdout, "[INFO] Loading %s file \n",parameters.decod_format==J3D_CFMT ? ".jp3d" : ".j2k");
	fsrc = fopen(parameters.infile, "rb");
	if (!fsrc) {
		fprintf(stdout, "[ERROR] Failed to open %s for reading\n", parameters.infile);
		return 1;
	}  
	fseek(fsrc, 0, SEEK_END);
	file_length = ftell(fsrc);
	fseek(fsrc, 0, SEEK_SET);
	src = (unsigned char *) malloc(file_length);
	fread(src, 1, file_length, fsrc);
	fclose(fsrc);
	
	/* decode the code-stream */
	/* ---------------------- */
	if (parameters.decod_format == J3D_CFMT || parameters.decod_format == J2K_CFMT) {		
		/* get a JP3D or J2K decoder handle */
		if (parameters.decod_format == J3D_CFMT) 
			dinfo = opj_create_decompress(CODEC_J3D);
		else if (parameters.decod_format == J2K_CFMT) 
			dinfo = opj_create_decompress(CODEC_J2K);

		/* catch events using our callbacks and give a local context */
		opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);			

		/* setup the decoder decoding parameters using user parameters */
		opj_setup_decoder(dinfo, &parameters);

		/* open a byte stream */
		cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

		/* decode the stream and fill the volume structure */
		volume = opj_decode(dinfo, cio);
		if(!volume) {
			fprintf(stdout, "[ERROR] jp3d_to_volume: failed to decode volume!\n");				
			opj_destroy_decompress(dinfo);
			opj_cio_close(cio);
			return 1;
		}	

		/* close the byte stream */
		opj_cio_close(cio);
	}
  
	/* free the memory containing the code-stream */
	free(src);
	src = NULL;

	/* create output volume */
	/* ------------------- */

	switch (parameters.cod_format) {
		case PGX_DFMT:			/* PGX */
			decodeok = volumetopgx(volume, parameters.outfile);
			if (decodeok)
				fprintf(stdout,"[ERROR] Unable to write decoded volume into pgx files\n");
			break;
		
		case BIN_DFMT:			/* BMP */
			decodeok = volumetobin(volume, parameters.outfile);
			if (decodeok)
				fprintf(stdout,"[ERROR] Unable to write decoded volume into pgx files\n");
			break;
	}
	switch (parameters.orig_format) {
		case PGX_DFMT:			/* PGX */
			if (strcmp("NULL",parameters.original) != 0){
 				fprintf(stdout,"Loading original file %s \n",parameters.original);
				cparameters.subsampling_dx = 1;	cparameters.subsampling_dy = 1;	cparameters.subsampling_dz = 1;
				cparameters.volume_offset_x0 = 0;cparameters.volume_offset_y0 = 0;cparameters.volume_offset_z0 = 0;
				original = pgxtovolume(parameters.original,&cparameters);
			}
			break;
		
		case BIN_DFMT:			/* BMP */
			if (strcmp("NULL",parameters.original) != 0 && strcmp("NULL",parameters.imgfile) != 0){
				fprintf(stdout,"Loading original file %s %s\n",parameters.original,parameters.imgfile);
				cparameters.subsampling_dx = 1;	cparameters.subsampling_dy = 1;	cparameters.subsampling_dz = 1;
				cparameters.volume_offset_x0 = 0;cparameters.volume_offset_y0 = 0;cparameters.volume_offset_z0 = 0;
				original = bintovolume(parameters.original,parameters.imgfile,&cparameters);
			}
			break;
	}

	fprintf(stdout, "[RESULT] Volume: %d x %d x %d (x %d bpv)\n ", 
			 (volume->comps[0].w >> volume->comps[0].factor[0]),
			 (volume->comps[0].h >> volume->comps[0].factor[1]),
			 (volume->comps[0].l >> volume->comps[0].factor[2]),volume->comps[0].prec);

	if(original){
		psnr = calc_PSNR(original,volume);
		ssim = calc_SSIM(original,volume);
		if (psnr < 0.0)
			fprintf(stdout, "  PSNR: Inf , SSMI %f -- Perfect reconstruction!\n",ssim);
		else
			fprintf(stdout, "  PSNR: %f , SSIM %f \n",psnr,ssim);
	}
	/* free remaining structures */
	if(dinfo) {
		opj_destroy_decompress(dinfo);
	}

	/* free volume data structure */
	opj_volume_destroy(volume);
   
	return 0;
}

