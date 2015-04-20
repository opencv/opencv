/*
 * Copyright (c) 20010, Mathieu Malaterre, GDCM
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

#ifdef _WIN32
#include "windirent.h"
#else
#include <dirent.h>
#endif /* _WIN32 */

#ifdef _WIN32
#include <windows.h>
#else
#include <strings.h>
#define _stricmp strcasecmp
#define _strnicmp strncasecmp
#endif /* _WIN32 */

#include "opj_config.h"
#include "openjpeg.h"
#include "../libopenjpeg/j2k.h"
#include "../libopenjpeg/jp2.h"
#include "getopt.h"
#include "convert.h"
#include "index.h"

#include "format_defs.h"

typedef struct dircnt{
	/** Buffer for holding images read from Directory*/
	char *filename_buf;
	/** Pointer to the buffer*/
	char **filename;
}dircnt_t;


typedef struct img_folder{
	/** The directory path of the folder containing input images*/
	char *imgdirpath;
	/** Output format*/
	const char *out_format;
	/** Enable option*/
	char set_imgdir;
	/** Enable Cod Format for output*/
	char set_out_format;

}img_fol_t;

void decode_help_display() {
	fprintf(stdout,"HELP for j2k_dump\n----\n\n");
	fprintf(stdout,"- the -h option displays this help information on screen\n\n");

/* UniPG>> */
	fprintf(stdout,"List of parameters for the JPEG 2000 "
#ifdef USE_JPWL
		"+ JPWL "
#endif /* USE_JPWL */
		"decoder:\n");
/* <<UniPG */
	fprintf(stdout,"\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"  -ImgDir \n");
	fprintf(stdout,"	Image file Directory path \n");
	fprintf(stdout,"  -i <compressed file>\n");
	fprintf(stdout,"    REQUIRED only if an Input image directory not specified\n");
	fprintf(stdout,"    Currently accepts J2K-files, JP2-files and JPT-files. The file type\n");
	fprintf(stdout,"    is identified based on its suffix.\n");
	fprintf(stdout,"\n");
}

/* -------------------------------------------------------------------------- */
static void j2k_dump_image(FILE *fd, opj_image_t * img);
static void j2k_dump_cp(FILE *fd, opj_image_t * img, opj_cp_t * cp);

int get_num_images(char *imgdirpath){
	DIR *dir;
	struct dirent* content;	
	int num_images = 0;

	/*Reading the input images from given input directory*/

	dir= opendir(imgdirpath);
	if(!dir){
		fprintf(stderr,"Could not open Folder %s\n",imgdirpath);
		return 0;
	}
	
	while((content=readdir(dir))!=NULL){
		if(strcmp(".",content->d_name)==0 || strcmp("..",content->d_name)==0 )
			continue;
		num_images++;
	}
	return num_images;
}

int load_images(dircnt_t *dirptr, char *imgdirpath){
	DIR *dir;
	struct dirent* content;	
	int i = 0;

	/*Reading the input images from given input directory*/

	dir= opendir(imgdirpath);
	if(!dir){
		fprintf(stderr,"Could not open Folder %s\n",imgdirpath);
		return 1;
	}else	{
		fprintf(stderr,"Folder opened successfully\n");
	}
	
	while((content=readdir(dir))!=NULL){
		if(strcmp(".",content->d_name)==0 || strcmp("..",content->d_name)==0 )
			continue;

		strcpy(dirptr->filename[i],content->d_name);
		i++;
	}
	return 0;	
}

int get_file_format(char *filename) {
	unsigned int i;
	static const char *extension[] = {"pgx", "pnm", "pgm", "ppm", "bmp","tif", "raw", "tga", "png", "j2k", "jp2", "jpt", "j2c", "jpc"  };
	static const int format[] = { PGX_DFMT, PXM_DFMT, PXM_DFMT, PXM_DFMT, BMP_DFMT, TIF_DFMT, RAW_DFMT, TGA_DFMT, PNG_DFMT, J2K_CFMT, JP2_CFMT, JPT_CFMT, J2K_CFMT, J2K_CFMT };
	char * ext = strrchr(filename, '.');
	if (ext == NULL)
		return -1;
	ext++;
	if(ext) {
		for(i = 0; i < sizeof(format)/sizeof(*format); i++) {
			if(_strnicmp(ext, extension[i], 3) == 0) {
				return format[i];
			}
		}
	}

	return -1;
}

char get_next_file(int imageno,dircnt_t *dirptr,img_fol_t *img_fol, opj_dparameters_t *parameters){
	char image_filename[OPJ_PATH_LEN], infilename[OPJ_PATH_LEN],outfilename[OPJ_PATH_LEN],temp_ofname[OPJ_PATH_LEN];
	char *temp_p, temp1[OPJ_PATH_LEN]="";

	strcpy(image_filename,dirptr->filename[imageno]);
	fprintf(stderr,"File Number %d \"%s\"\n",imageno,image_filename);
	parameters->decod_format = get_file_format(image_filename);
	if (parameters->decod_format == -1)
		return 1;
	sprintf(infilename,"%s/%s",img_fol->imgdirpath,image_filename);
	strncpy(parameters->infile, infilename, sizeof(infilename));

	//Set output file
	strcpy(temp_ofname,strtok(image_filename,"."));
	while((temp_p = strtok(NULL,".")) != NULL){
		strcat(temp_ofname,temp1);
		sprintf(temp1,".%s",temp_p);
	}
	if(img_fol->set_out_format==1){
		sprintf(outfilename,"%s/%s.%s",img_fol->imgdirpath,temp_ofname,img_fol->out_format);
		strncpy(parameters->outfile, outfilename, sizeof(outfilename));
	}
	return 0;
}

/* -------------------------------------------------------------------------- */
int parse_cmdline_decoder(int argc, char **argv, opj_dparameters_t *parameters,img_fol_t *img_fol, char *indexfilename) {
	/* parse the command line */
	int totlen;
	option_t long_option[]={
		{"ImgDir",REQ_ARG, NULL ,'y'},
	};

	const char optlist[] = "i:h";
	totlen=sizeof(long_option);
	img_fol->set_out_format = 0;
	while (1) {
		int c = getopt_long(argc, argv,optlist,long_option,totlen);
		if (c == -1)
			break;
		switch (c) {
			case 'i':			/* input file */
			{
				char *infile = optarg;
				parameters->decod_format = get_file_format(infile);
				switch(parameters->decod_format) {
					case J2K_CFMT:
					case JP2_CFMT:
					case JPT_CFMT:
						break;
					default:
						fprintf(stderr, 
							"!! Unrecognized format for infile : %s [accept only *.j2k, *.jp2, *.jpc or *.jpt] !!\n\n", 
							infile);
						return 1;
				}
				strncpy(parameters->infile, infile, sizeof(parameters->infile)-1);
			}
			break;
				
				/* ----------------------------------------------------- */

			case 'h': 			/* display an help description */
				decode_help_display();
				return 1;				

				/* ------------------------------------------------------ */

			case 'y':			/* Image Directory path */
				{
					img_fol->imgdirpath = (char*)malloc(strlen(optarg) + 1);
					strcpy(img_fol->imgdirpath,optarg);
					img_fol->set_imgdir=1;
				}
				break;

				/* ----------------------------------------------------- */
			
			default:
				fprintf(stderr,"WARNING -> this option is not valid \"-%c %s\"\n",c, optarg);
				break;
		}
	}

	/* check for possible errors */
	if(img_fol->set_imgdir==1){
		if(!(parameters->infile[0]==0)){
			fprintf(stderr, "Error: options -ImgDir and -i cannot be used together !!\n");
			return 1;
		}
		if(img_fol->set_out_format == 0){
			fprintf(stderr, "Error: When -ImgDir is used, -OutFor <FORMAT> must be used !!\n");
			fprintf(stderr, "Only one format allowed! Valid format PGM, PPM, PNM, PGX, BMP, TIF, RAW and TGA!!\n");
			return 1;
		}
		if(!((parameters->outfile[0] == 0))){
			fprintf(stderr, "Error: options -ImgDir and -o cannot be used together !!\n");
			return 1;
		}
	}else{
		if((parameters->infile[0] == 0) ) {
			fprintf(stderr, "Example: %s -i image.j2k\n",argv[0]);
			fprintf(stderr, "    Try: %s -h\n",argv[0]);
			return 1;
		}
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
	(void)client_data;
	fprintf(stdout, "[INFO] %s", msg);
}

/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
	opj_dparameters_t parameters;	/* decompression parameters */
	img_fol_t img_fol;
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_image_t *image = NULL;
	FILE *fsrc = NULL;
	unsigned char *src = NULL;
	int file_length;
	int num_images;
	int i,imageno;
	dircnt_t *dirptr = NULL;
	opj_dinfo_t* dinfo = NULL;	/* handle to a decompressor */
	opj_cio_t *cio = NULL;
	opj_codestream_info_t cstr_info;  /* Codestream information structure */
	char indexfilename[OPJ_PATH_LEN];	/* index file name */

	/* configure the event callbacks (not required) */
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = error_callback;
	event_mgr.warning_handler = warning_callback;
	event_mgr.info_handler = info_callback;

	/* set decoding parameters to default values */
	opj_set_default_decoder_parameters(&parameters);

	/* Initialize indexfilename and img_fol */
	*indexfilename = 0;
	memset(&img_fol,0,sizeof(img_fol_t));

	/* parse input and get user encoding parameters */
	if(parse_cmdline_decoder(argc, argv, &parameters,&img_fol, indexfilename) == 1) {
		return 1;
	}

	/* Initialize reading of directory */
	if(img_fol.set_imgdir==1){	
		num_images=get_num_images(img_fol.imgdirpath);

		dirptr=(dircnt_t*)malloc(sizeof(dircnt_t));
		if(dirptr){
			dirptr->filename_buf = (char*)malloc(num_images*OPJ_PATH_LEN*sizeof(char));	// Stores at max 10 image file names
			dirptr->filename = (char**) malloc(num_images*sizeof(char*));

			if(!dirptr->filename_buf){
				return 1;
			}
			for(i=0;i<num_images;i++){
				dirptr->filename[i] = dirptr->filename_buf + i*OPJ_PATH_LEN;
			}
		}
		if(load_images(dirptr,img_fol.imgdirpath)==1){
			return 1;
		}
		if (num_images==0){
			fprintf(stdout,"Folder is empty\n");
			return 1;
		}
	}else{
		num_images=1;
	}

	/*Encoding image one by one*/
	for(imageno = 0; imageno < num_images ; imageno++)
  {
		image = NULL;
		fprintf(stderr,"\n");

		if(img_fol.set_imgdir==1){
			if (get_next_file(imageno, dirptr,&img_fol, &parameters)) {
				fprintf(stderr,"skipping file...\n");
				continue;
			}
		}

		/* read the input file and put it in memory */
		/* ---------------------------------------- */
		fsrc = fopen(parameters.infile, "rb");
		if (!fsrc) {
			fprintf(stderr, "ERROR -> failed to open %s for reading\n", parameters.infile);
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

		switch(parameters.decod_format) {
		case J2K_CFMT:
		{
			/* JPEG-2000 codestream */

			/* get a decoder handle */
			dinfo = opj_create_decompress(CODEC_J2K);

			/* catch events using our callbacks and give a local context */
			opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

			/* setup the decoder decoding parameters using user parameters */
			opj_setup_decoder(dinfo, &parameters);

			/* open a byte stream */
			cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

			/* decode the stream and fill the image structure */
			if (*indexfilename)				// If need to extract codestream information
				image = opj_decode_with_info(dinfo, cio, &cstr_info);
			else
				image = opj_decode(dinfo, cio);
			if(!image) {
				fprintf(stderr, "ERROR -> j2k_to_image: failed to decode image!\n");
				opj_destroy_decompress(dinfo);
				opj_cio_close(cio);
				return 1;
			}
			/* dump image */
      j2k_dump_image(stdout, image);

			/* dump cp */
      j2k_dump_cp(stdout, image, ((opj_j2k_t*)dinfo->j2k_handle)->cp);

			/* close the byte stream */
			opj_cio_close(cio);

			/* Write the index to disk */
			if (*indexfilename) {
				char bSuccess;
				bSuccess = write_index_file(&cstr_info, indexfilename);
				if (bSuccess) {
					fprintf(stderr, "Failed to output index file\n");
				}
			}
		}
		break;

		case JP2_CFMT:
		{
			/* JPEG 2000 compressed image data */

			/* get a decoder handle */
			dinfo = opj_create_decompress(CODEC_JP2);

			/* catch events using our callbacks and give a local context */
			opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

			/* setup the decoder decoding parameters using the current image and user parameters */
			opj_setup_decoder(dinfo, &parameters);

			/* open a byte stream */
			cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

			/* decode the stream and fill the image structure */
			if (*indexfilename)				// If need to extract codestream information
				image = opj_decode_with_info(dinfo, cio, &cstr_info);
			else
				image = opj_decode(dinfo, cio);			
			if(!image) {
				fprintf(stderr, "ERROR -> j2k_to_image: failed to decode image!\n");
				opj_destroy_decompress(dinfo);
				opj_cio_close(cio);
				return 1;
			}
			/* dump image */
	  if(image->icc_profile_buf)
	 {
	  free(image->icc_profile_buf); image->icc_profile_buf = NULL;
	 }	
      j2k_dump_image(stdout, image);

			/* dump cp */
      j2k_dump_cp(stdout, image, ((opj_jp2_t*)dinfo->jp2_handle)->j2k->cp);

			/* close the byte stream */
			opj_cio_close(cio);

			/* Write the index to disk */
			if (*indexfilename) {
				char bSuccess;
				bSuccess = write_index_file(&cstr_info, indexfilename);
				if (bSuccess) {
					fprintf(stderr, "Failed to output index file\n");
				}
			}
		}
		break;

		case JPT_CFMT:
		{
			/* JPEG 2000, JPIP */

			/* get a decoder handle */
			dinfo = opj_create_decompress(CODEC_JPT);

			/* catch events using our callbacks and give a local context */
			opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

			/* setup the decoder decoding parameters using user parameters */
			opj_setup_decoder(dinfo, &parameters);

			/* open a byte stream */
			cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

			/* decode the stream and fill the image structure */
			if (*indexfilename)				// If need to extract codestream information
				image = opj_decode_with_info(dinfo, cio, &cstr_info);
			else
				image = opj_decode(dinfo, cio);
			if(!image) {
				fprintf(stderr, "ERROR -> j2k_to_image: failed to decode image!\n");
				opj_destroy_decompress(dinfo);
				opj_cio_close(cio);
				return 1;
			}

			/* close the byte stream */
			opj_cio_close(cio);

			/* Write the index to disk */
			if (*indexfilename) {
				char bSuccess;
				bSuccess = write_index_file(&cstr_info, indexfilename);
				if (bSuccess) {
					fprintf(stderr, "Failed to output index file\n");
				}
			}
		}
		break;

		default:
			fprintf(stderr, "skipping file..\n");
			continue;
	}

		/* free the memory containing the code-stream */
		free(src);
		src = NULL;

		/* free remaining structures */
		if(dinfo) {
			opj_destroy_decompress(dinfo);
		}
		/* free codestream information structure */
		if (*indexfilename)	
			opj_destroy_cstr_info(&cstr_info);
		/* free image data structure */
		opj_image_destroy(image);

	}

  return EXIT_SUCCESS;
}


static void j2k_dump_image(FILE *fd, opj_image_t * img) {
	int compno;
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

static void j2k_dump_cp(FILE *fd, opj_image_t * img, opj_cp_t * cp) {
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

