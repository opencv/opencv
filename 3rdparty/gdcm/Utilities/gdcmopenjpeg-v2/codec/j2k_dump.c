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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#define USE_OPJ_DEPRECATED
#include "openjpeg.h"
#include "j2k.h"
#include "jp2.h"
#include "compat/getopt.h"
#include "convert.h"
#include "dirent.h"
#include "index.h"

#ifndef WIN32
#include <strings.h>
#define _stricmp strcasecmp
#define _strnicmp strncasecmp
#endif

/* ----------------------------------------------------------------------- */

#define J2K_CFMT 0
#define JP2_CFMT 1
#define JPT_CFMT 2

#define PXM_DFMT 10
#define PGX_DFMT 11
#define BMP_DFMT 12
#define YUV_DFMT 13
#define TIF_DFMT 14
#define RAW_DFMT 15
#define TGA_DFMT 16

/* ----------------------------------------------------------------------- */

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
  fprintf(stdout,"HELP\n----\n\n");
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
  fprintf(stdout,"  Image file Directory path \n");
  fprintf(stdout,"  -i <compressed file>\n");
  fprintf(stdout,"    REQUIRED only if an Input image directory not specified\n");
  fprintf(stdout,"    Currently accepts J2K-files, JP2-files and JPT-files. The file type\n");
  fprintf(stdout,"    is identified based on its suffix.\n");
  fprintf(stdout,"\n");
}

/* -------------------------------------------------------------------------- */

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
  }else  {
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
  static const char *extension[] = {"pgx", "pnm", "pgm", "ppm", "bmp","tif", "raw", "tga", "j2k", "jp2", "jpt", "j2c", "jpc" };
  static const int format[] = { PGX_DFMT, PXM_DFMT, PXM_DFMT, PXM_DFMT, BMP_DFMT, TIF_DFMT, RAW_DFMT, TGA_DFMT, J2K_CFMT, JP2_CFMT, JPT_CFMT, J2K_CFMT, J2K_CFMT };
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
      case 'i':      /* input file */
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

      case 'h':       /* display an help description */
        decode_help_display();
        return 1;

        /* ------------------------------------------------------ */

      case 'y':      /* Image Directory path */
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
      fprintf(stderr, "Error: One of the options -i or -ImgDir must be specified\n");
      fprintf(stderr, "usage: image_to_j2k -i *.j2k/jp2/j2c (+ options)\n");
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
  int ret;
  opj_dparameters_t parameters;  /* decompression parameters */
  img_fol_t img_fol;
  opj_image_t *image = NULL;
  FILE *fsrc = NULL;
  bool bResult;
  int num_images;
  int i,imageno;
  dircnt_t *dirptr;
  opj_codec_t* dinfo = NULL;  /* handle to a decompressor */
  opj_stream_t *cio = NULL;
  opj_codestream_info_t cstr_info;  /* Codestream information structure */
  char indexfilename[OPJ_PATH_LEN];  /* index file name */
  OPJ_INT32 l_tile_x0,l_tile_y0;
  OPJ_UINT32 l_tile_width,l_tile_height,l_nb_tiles_x,l_nb_tiles_y;

  /* configure the event callbacks (not required) */

  /* set decoding parameters to default values */
  opj_set_default_decoder_parameters(&parameters);

  /* Initialize indexfilename and img_fol */
  *indexfilename = 0;
  memset(&img_fol,0,sizeof(img_fol_t));

  /* parse input and get user encoding parameters */
  if(parse_cmdline_decoder(argc, argv, &parameters,&img_fol, indexfilename) == 1) {
    return EXIT_FAILURE;
  }

  /* Initialize reading of directory */
  if(img_fol.set_imgdir==1){
    num_images=get_num_images(img_fol.imgdirpath);

    dirptr=(dircnt_t*)malloc(sizeof(dircnt_t));
    if(dirptr){
      dirptr->filename_buf = (char*)malloc(num_images*OPJ_PATH_LEN*sizeof(char));  // Stores at max 10 image file names
      dirptr->filename = (char**) malloc(num_images*sizeof(char*));

      if(!dirptr->filename_buf){
        return EXIT_FAILURE;
      }
      for(i=0;i<num_images;i++){
        dirptr->filename[i] = dirptr->filename_buf + i*OPJ_PATH_LEN;
      }
    }
    if(load_images(dirptr,img_fol.imgdirpath)==1){
      return EXIT_FAILURE;
    }
    if (num_images==0){
      fprintf(stdout,"Folder is empty\n");
      return EXIT_FAILURE;
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
      return EXIT_FAILURE;
    }
    cio = opj_stream_create_default_file_stream(fsrc,true);
    /* decode the code-stream */
    /* ---------------------- */

    switch (parameters.decod_format)
    {
      case J2K_CFMT:
      {
        /* JPEG-2000 codestream */

        /* get a decoder handle */
        dinfo = opj_create_decompress(CODEC_J2K);
        break;
      }
      case JP2_CFMT:
      {
        /* JPEG 2000 compressed image data */
        /* get a decoder handle */
        dinfo = opj_create_decompress(CODEC_JP2);
        break;
      }
      case JPT_CFMT:
      {
        /* JPEG 2000, JPIP */
        /* get a decoder handle */
        dinfo = opj_create_decompress(CODEC_JPT);
        break;
      }
      default:
        fprintf(stderr, "skipping file..\n");
        opj_stream_destroy(cio);
        continue;
    }
    /* catch events using our callbacks and give a local context */

    /* setup the decoder decoding parameters using user parameters */
    opj_setup_decoder(dinfo, &parameters);

    /* decode the stream and fill the image structure */
    /*    if (*indexfilename)        // If need to extract codestream information
        image = opj_decode_with_info(dinfo, cio, &cstr_info);
      else
      */
    bResult = opj_read_header(
      dinfo,
      &image,
      &l_tile_x0,
      &l_tile_y0,
      &l_tile_width,
      &l_tile_height,
      &l_nb_tiles_x,
      &l_nb_tiles_y,
      cio);
    //image = opj_decode(dinfo, cio);
    //bResult = bResult && (image != 00);
    //bResult = bResult && opj_end_decompress(dinfo,cio);
    //if
    //  (!image)
    //{
    //  fprintf(stderr, "ERROR -> j2k_to_image: failed to decode image!\n");
    //  opj_destroy_codec(dinfo);
    //  opj_stream_destroy(cio);
    //  fclose(fsrc);
    //  return EXIT_FAILURE;
    //}
    /* dump image */
    if(!image)
      {
      fprintf(stderr, "ERROR -> j2k_to_image: failed to read header\n");
      return EXIT_FAILURE;
      }
    j2k_dump_image(stdout, image);

    /* dump cp */
    //j2k_dump_cp(stdout, image, dinfo->m_codec);

    /* close the byte stream */
    opj_stream_destroy(cio);
    fclose(fsrc);
    /* Write the index to disk */
    if (*indexfilename) {
      char bSuccess;
      bSuccess = write_index_file(&cstr_info, indexfilename);
      if (bSuccess) {
        fprintf(stderr, "Failed to output index file\n");
        ret = EXIT_FAILURE;
      }
    }

    /* free remaining structures */
    if (dinfo) {
      opj_destroy_codec(dinfo);
    }
    /* free codestream information structure */
    if (*indexfilename)
      opj_destroy_cstr_info(&cstr_info);
    /* free image data structure */
    opj_image_destroy(image);

  }

  return ret;
}
//end main
