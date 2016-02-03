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
#define PNG_DFMT 17
/* ----------------------------------------------------------------------- */
#define CINEMA_24_CS 1302083  /*Codestream length for 24fps*/
#define CINEMA_48_CS 651041    /*Codestream length for 48fps*/
#define COMP_24_CS 1041666    /*Maximum size per color component for 2K & 4K @ 24fps*/
#define COMP_48_CS 520833    /*Maximum size per color component for 2K @ 48fps*/

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
  char *out_format;
  /** Enable option*/
  char set_imgdir;
  /** Enable Cod Format for output*/
  char set_out_format;
  /** User specified rate stored in case of cinema option*/
  float *rates;
}img_fol_t;

void encode_help_display() {
  fprintf(stdout,"HELP\n----\n\n");
  fprintf(stdout,"- the -h option displays this help information on screen\n\n");

/* UniPG>> */
  fprintf(stdout,"List of parameters for the JPEG 2000 "
#ifdef USE_JPWL
    "+ JPWL "
#endif /* USE_JPWL */
    "encoder:\n");
/* <<UniPG */
  fprintf(stdout,"\n");
  fprintf(stdout,"REMARKS:\n");
  fprintf(stdout,"---------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"The markers written to the main_header are : SOC SIZ COD QCD COM.\n");
  fprintf(stdout,"COD and QCD never appear in the tile_header.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"By default:\n");
  fprintf(stdout,"------------\n");
  fprintf(stdout,"\n");
  fprintf(stdout," * Lossless\n");
  fprintf(stdout," * 1 tile\n");
  fprintf(stdout," * Size of precinct : 2^15 x 2^15 (means 1 precinct)\n");
  fprintf(stdout," * Size of code-block : 64 x 64\n");
  fprintf(stdout," * Number of resolutions: 6\n");
  fprintf(stdout," * No SOP marker in the codestream\n");
  fprintf(stdout," * No EPH marker in the codestream\n");
  fprintf(stdout," * No sub-sampling in x or y direction\n");
  fprintf(stdout," * No mode switch activated\n");
  fprintf(stdout," * Progression order: LRCP\n");
  fprintf(stdout," * No index file\n");
  fprintf(stdout," * No ROI upshifted\n");
  fprintf(stdout," * No offset of the origin of the image\n");
  fprintf(stdout," * No offset of the origin of the tiles\n");
  fprintf(stdout," * Reversible DWT 5-3\n");
/* UniPG>> */
#ifdef USE_JPWL
  fprintf(stdout," * No JPWL protection\n");
#endif /* USE_JPWL */
/* <<UniPG */
  fprintf(stdout,"\n");
  fprintf(stdout,"Parameters:\n");
  fprintf(stdout,"------------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Required Parameters (except with -h):\n");
  fprintf(stdout,"One of the two options -ImgDir or -i must be used\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-ImgDir      : Image file Directory path (example ../Images) \n");
  fprintf(stdout,"    When using this option -OutFor must be used\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-OutFor \n");
  fprintf(stdout,"    REQUIRED only if -ImgDir is used\n");
  fprintf(stdout,"    Need to specify only format without filename <BMP>  \n");
  fprintf(stdout,"    Currently accepts PGM, PPM, PNM, PGX, BMP, TIF, RAW and TGA formats\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-i           : source file  (-i source.pnm also *.pgm, *.ppm, *.bmp, *.tif, *.raw, *.tga) \n");
  fprintf(stdout,"    When using this option -o must be used\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-o           : destination file (-o dest.j2k or .jp2) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Optional Parameters:\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-h           : display the help information \n ");
  fprintf(stdout,"\n");
  fprintf(stdout,"-cinema2K    : Digital Cinema 2K profile compliant codestream for 2K resolution.(-cinema2k 24 or 48) \n");
  fprintf(stdout,"    Need to specify the frames per second for a 2K resolution. Only 24 or 48 fps is allowed\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-cinema4K    : Digital Cinema 4K profile compliant codestream for 4K resolution \n");
  fprintf(stdout,"    Frames per second not required. Default value is 24fps\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-r           : different compression ratios for successive layers (-r 20,10,5)\n ");
  fprintf(stdout,"           - The rate specified for each quality level is the desired \n");
  fprintf(stdout,"             compression factor.\n");
  fprintf(stdout,"       Example: -r 20,10,1 means quality 1: compress 20x, \n");
  fprintf(stdout,"         quality 2: compress 10x and quality 3: compress lossless\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"               (options -r and -q cannot be used together)\n ");
  fprintf(stdout,"\n");

  fprintf(stdout,"-q           : different psnr for successive layers (-q 30,40,50) \n ");

  fprintf(stdout,"               (options -r and -q cannot be used together)\n ");

  fprintf(stdout,"\n");
  fprintf(stdout,"-n           : number of resolutions (-n 3) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-b           : size of code block (-b 32,32) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-c           : size of precinct (-c 128,128) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-t           : size of tile (-t 512,512) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-p           : progression order (-p LRCP) [LRCP, RLCP, RPCL, PCRL, CPRL] \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-s           : subsampling factor (-s 2,2) [-s X,Y] \n");
  fprintf(stdout,"       Remark: subsampling bigger than 2 can produce error\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-POC         : Progression order change (-POC T1=0,0,1,5,3,CPRL/T1=5,0,1,6,3,CPRL) \n");
  fprintf(stdout,"      Example: T1=0,0,1,5,3,CPRL \n");
  fprintf(stdout,"       : Ttilenumber=Resolution num start,Component num start,Layer num end,Resolution num end,Component num end,Progression order\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-SOP         : write SOP marker before each packet \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-EPH         : write EPH marker after each header packet \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-M           : mode switch (-M 3) [1=BYPASS(LAZY) 2=RESET 4=RESTART(TERMALL)\n");
  fprintf(stdout,"                 8=VSC 16=ERTERM(SEGTERM) 32=SEGMARK(SEGSYM)] \n");
  fprintf(stdout,"                 Indicate multiple modes by adding their values. \n");
  fprintf(stdout,"                 ex: RESTART(4) + RESET(2) + SEGMARK(32) = -M 38\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-x           : create an index file *.Idx (-x index_name.Idx) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-ROI         : c=%%d,U=%%d : quantization indices upshifted \n");
  fprintf(stdout,"               for component c=%%d [%%d = 0,1,2]\n");
  fprintf(stdout,"               with a value of U=%%d [0 <= %%d <= 37] (i.e. -ROI c=0,U=25) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-d           : offset of the origin of the image (-d 150,300) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-T           : offset of the origin of the tiles (-T 100,75) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-I           : use the irreversible DWT 9-7 (-I) \n");
  fprintf(stdout,"\n");
  fprintf(stdout,"-F           : characteristics of the raw input image\n");
  fprintf(stdout,"               -F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
  fprintf(stdout,"               Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
  fprintf(stdout,"-m           : use array-based MCT, values are coma separated, line by line\n");
  fprintf(stdout,"         no specific separators between lines, no space allowed between values\n");
  fprintf(stdout,"\n");
/* UniPG>> */
#ifdef USE_JPWL
  fprintf(stdout,"-W           : adoption of JPWL (Part 11) capabilities (-W params)\n");
  fprintf(stdout,"               The parameters can be written and repeated in any order:\n");
  fprintf(stdout,"               [h<tilepart><=type>,s<tilepart><=method>,a=<addr>,...\n");
  fprintf(stdout,"                ...,z=<size>,g=<range>,p<tilepart:pack><=type>]\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 h selects the header error protection (EPB): 'type' can be\n");
  fprintf(stdout,"                   [0=none 1,absent=predefined 16=CRC-16 32=CRC-32 37-128=RS]\n");
  fprintf(stdout,"                   if 'tilepart' is absent, it is for main and tile headers\n");
  fprintf(stdout,"                   if 'tilepart' is present, it applies from that tile\n");
  fprintf(stdout,"                     onwards, up to the next h<> spec, or to the last tilepart\n");
  fprintf(stdout,"                     in the codestream (max. %d specs)\n", JPWL_MAX_NO_TILESPECS);
  fprintf(stdout,"\n");
  fprintf(stdout,"                 p selects the packet error protection (EEP/UEP with EPBs)\n");
  fprintf(stdout,"                  to be applied to raw data: 'type' can be\n");
  fprintf(stdout,"                   [0=none 1,absent=predefined 16=CRC-16 32=CRC-32 37-128=RS]\n");
  fprintf(stdout,"                   if 'tilepart:pack' is absent, it is from tile 0, packet 0\n");
  fprintf(stdout,"                   if 'tilepart:pack' is present, it applies from that tile\n");
  fprintf(stdout,"                     and that packet onwards, up to the next packet spec\n");
  fprintf(stdout,"                     or to the last packet in the last tilepart in the stream\n");
  fprintf(stdout,"                     (max. %d specs)\n", JPWL_MAX_NO_PACKSPECS);
  fprintf(stdout,"\n");
  fprintf(stdout,"                 s enables sensitivity data insertion (ESD): 'method' can be\n");
  fprintf(stdout,"                   [-1=NO ESD 0=RELATIVE ERROR 1=MSE 2=MSE REDUCTION 3=PSNR\n");
  fprintf(stdout,"                    4=PSNR INCREMENT 5=MAXERR 6=TSE 7=RESERVED]\n");
  fprintf(stdout,"                   if 'tilepart' is absent, it is for main header only\n");
  fprintf(stdout,"                   if 'tilepart' is present, it applies from that tile\n");
  fprintf(stdout,"                     onwards, up to the next s<> spec, or to the last tilepart\n");
  fprintf(stdout,"                     in the codestream (max. %d specs)\n", JPWL_MAX_NO_TILESPECS);
  fprintf(stdout,"\n");
  fprintf(stdout,"                 g determines the addressing mode: <range> can be\n");
  fprintf(stdout,"                   [0=PACKET 1=BYTE RANGE 2=PACKET RANGE]\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 a determines the size of data addressing: <addr> can be\n");
  fprintf(stdout,"                   2/4 bytes (small/large codestreams). If not set, auto-mode\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 z determines the size of sensitivity values: <size> can be\n");
  fprintf(stdout,"                   1/2 bytes, for the transformed pseudo-floating point value\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 ex.:\n");
  fprintf(stdout,"                   h,h0=64,h3=16,h5=32,p0=78,p0:24=56,p1,p3:0=0,p3:20=32,s=0,\n");
  fprintf(stdout,"                     s0=6,s3=-1,a=0,g=1,z=1\n");
  fprintf(stdout,"                 means\n");
  fprintf(stdout,"                   predefined EPB in MH, rs(64,32) from TPH 0 to TPH 2,\n");
  fprintf(stdout,"                   CRC-16 in TPH 3 and TPH 4, CRC-32 in remaining TPHs,\n");
  fprintf(stdout,"                   UEP rs(78,32) for packets 0 to 23 of tile 0,\n");
  fprintf(stdout,"                   UEP rs(56,32) for packs. 24 to the last of tilepart 0,\n");
  fprintf(stdout,"                   UEP rs default for packets of tilepart 1,\n");
  fprintf(stdout,"                   no UEP for packets 0 to 19 of tilepart 3,\n");
  fprintf(stdout,"                   UEP CRC-32 for packs. 20 of tilepart 3 to last tilepart,\n");
  fprintf(stdout,"                   relative sensitivity ESD for MH,\n");
  fprintf(stdout,"                   TSE ESD from TPH 0 to TPH 2, byte range with automatic\n");
  fprintf(stdout,"                   size of addresses and 1 byte for each sensitivity value\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 ex.:\n");
  fprintf(stdout,"                       h,s,p\n");
  fprintf(stdout,"                 means\n");
  fprintf(stdout,"                   default protection to headers (MH and TPHs) as well as\n");
  fprintf(stdout,"                   data packets, one ESD in MH\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"                 N.B.: use the following recommendations when specifying\n");
  fprintf(stdout,"                       the JPWL parameters list\n");
  fprintf(stdout,"                   - when you use UEP, always pair the 'p' option with 'h'\n");
  fprintf(stdout,"                 \n");
#endif /* USE_JPWL */
/* <<UniPG */
  fprintf(stdout,"IMPORTANT:\n");
  fprintf(stdout,"-----------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"The index file has the structure below:\n");
  fprintf(stdout,"---------------------------------------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Image_height Image_width\n");
  fprintf(stdout,"progression order\n");
  fprintf(stdout,"Tiles_size_X Tiles_size_Y\n");
  fprintf(stdout,"Tiles_nb_X Tiles_nb_Y\n");
  fprintf(stdout,"Components_nb\n");
  fprintf(stdout,"Layers_nb\n");
  fprintf(stdout,"decomposition_levels\n");
  fprintf(stdout,"[Precincts_size_X_res_Nr Precincts_size_Y_res_Nr]...\n");
  fprintf(stdout,"   [Precincts_size_X_res_0 Precincts_size_Y_res_0]\n");
  fprintf(stdout,"Main_header_start_position\n");
  fprintf(stdout,"Main_header_end_position\n");
  fprintf(stdout,"Codestream_size\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"INFO ON TILES\n");
  fprintf(stdout,"tileno start_pos end_hd end_tile nbparts disto nbpix disto/nbpix\n");
  fprintf(stdout,"Tile_0 start_pos end_Theader end_pos NumParts TotalDisto NumPix MaxMSE\n");
  fprintf(stdout,"Tile_1   ''           ''        ''        ''       ''    ''      ''\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"Tile_Nt   ''           ''        ''        ''       ''    ''     ''\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"TILE 0 DETAILS\n");
  fprintf(stdout,"part_nb tileno num_packs start_pos end_tph_pos end_pos\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"Progression_string\n");
  fprintf(stdout,"pack_nb tileno layno resno compno precno start_pos end_ph_pos end_pos disto\n");
  fprintf(stdout,"Tpacket_0 Tile layer res. comp. prec. start_pos end_pos disto\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"Tpacket_Np ''   ''    ''   ''    ''       ''       ''     ''\n");

  fprintf(stdout,"MaxDisto\n");

  fprintf(stdout,"TotalDisto\n\n");
}

OPJ_PROG_ORDER give_progression(char progression[4]) {
  if(strncmp(progression, "LRCP", 4) == 0) {
    return LRCP;
  }
  if(strncmp(progression, "RLCP", 4) == 0) {
    return RLCP;
  }
  if(strncmp(progression, "RPCL", 4) == 0) {
    return RPCL;
  }
  if(strncmp(progression, "PCRL", 4) == 0) {
    return PCRL;
  }
  if(strncmp(progression, "CPRL", 4) == 0) {
    return CPRL;
  }

  return PROG_UNKNOWN;
}

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

  num_images=0;
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
  static const char *extension[] = {
    "pgx", "pnm", "pgm", "ppm", "bmp", "tif", "raw", "tga", "png", "j2k", "jp2", "j2c", "jpc"
    };
  static const int format[] = {
    PGX_DFMT, PXM_DFMT, PXM_DFMT, PXM_DFMT, BMP_DFMT, TIF_DFMT, RAW_DFMT, TGA_DFMT, PNG_DFMT, J2K_CFMT, JP2_CFMT, J2K_CFMT, J2K_CFMT
    };
  char * ext = strrchr(filename, '.');
  if (ext == NULL)
    return -1;
  ext++;
  for(i = 0; i < sizeof(format)/sizeof(*format); i++) {
    if(_strnicmp(ext, extension[i], 3) == 0) {
      return format[i];
    }
  }
  return -1;
}

char * get_file_name(char *name){
  char *fname;
  fname= (char*)malloc(OPJ_PATH_LEN*sizeof(char));
  fname= strtok(name,".");
  return fname;
}

char get_next_file(int imageno,dircnt_t *dirptr,img_fol_t *img_fol, opj_cparameters_t *parameters){
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
  strcpy(temp_ofname,get_file_name(image_filename));
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

static int initialise_4K_poc(opj_poc_t *POC, int numres){
  POC[0].tile  = 1;
  POC[0].resno0  = 0;
  POC[0].compno0 = 0;
  POC[0].layno1  = 1;
  POC[0].resno1  = numres-1;
  POC[0].compno1 = 3;
  POC[0].prg1 = CPRL;
  POC[1].tile  = 1;
  POC[1].resno0  = numres-1;
  POC[1].compno0 = 0;
  POC[1].layno1  = 1;
  POC[1].resno1  = numres;
  POC[1].compno1 = 3;
  POC[1].prg1 = CPRL;
  return 2;
}

void cinema_parameters(opj_cparameters_t *parameters){
  parameters->tile_size_on = false;
  parameters->cp_tdx=1;
  parameters->cp_tdy=1;

  /*Tile part*/
  parameters->tp_flag = 'C';
  parameters->tp_on = 1;

  /*Tile and Image shall be at (0,0)*/
  parameters->cp_tx0 = 0;
  parameters->cp_ty0 = 0;
  parameters->image_offset_x0 = 0;
  parameters->image_offset_y0 = 0;

  /*Codeblock size= 32*32*/
  parameters->cblockw_init = 32;
  parameters->cblockh_init = 32;
  parameters->csty |= 0x01;

  /*The progression order shall be CPRL*/
  parameters->prog_order = CPRL;

  /* No ROI */
  parameters->roi_compno = -1;

  parameters->subsampling_dx = 1;    parameters->subsampling_dy = 1;

  /* 9-7 transform */
  parameters->irreversible = 1;

}

void cinema_setup_encoder(opj_cparameters_t *parameters,opj_image_t *image, img_fol_t *img_fol){
  int i;
  float temp_rate;

  switch (parameters->cp_cinema){
  case CINEMA2K_24:
  case CINEMA2K_48:
    if(parameters->numresolution > 6){
      parameters->numresolution = 6;
    }
    if (!((image->comps[0].w == 2048) | (image->comps[0].h == 1080))){
      fprintf(stdout,"Image coordinates %d x %d is not 2K compliant.\nJPEG Digital Cinema Profile-3 "
        "(2K profile) compliance requires that at least one of coordinates match 2048 x 1080\n",
        image->comps[0].w,image->comps[0].h);
      parameters->cp_rsiz = STD_RSIZ;
    }
  break;

  case CINEMA4K_24:
    if(parameters->numresolution < 1){
        parameters->numresolution = 1;
      }else if(parameters->numresolution > 7){
        parameters->numresolution = 7;
      }
    if (!((image->comps[0].w == 4096) | (image->comps[0].h == 2160))){
      fprintf(stdout,"Image coordinates %d x %d is not 4K compliant.\nJPEG Digital Cinema Profile-4"
        "(4K profile) compliance requires that at least one of coordinates match 4096 x 2160\n",
        image->comps[0].w,image->comps[0].h);
      parameters->cp_rsiz = STD_RSIZ;
    }
    parameters->numpocs = initialise_4K_poc(parameters->POC,parameters->numresolution);
    break;
  default :
    break;
  }

  switch (parameters->cp_cinema){
    case CINEMA2K_24:
    case CINEMA4K_24:
      for(i=0 ; i<parameters->tcp_numlayers ; i++){
        temp_rate = 0 ;
        if (img_fol->rates[i]== 0){
          parameters->tcp_rates[0]= ((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
          (CINEMA_24_CS * 8 * image->comps[0].dx * image->comps[0].dy);
        }else{
          temp_rate =((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
            (img_fol->rates[i] * 8 * image->comps[0].dx * image->comps[0].dy);
          if (temp_rate > CINEMA_24_CS ){
            parameters->tcp_rates[i]= ((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
            (CINEMA_24_CS * 8 * image->comps[0].dx * image->comps[0].dy);
          }else{
            parameters->tcp_rates[i]= img_fol->rates[i];
          }
        }
      }
      parameters->max_comp_size = COMP_24_CS;
      break;

    case CINEMA2K_48:
      for(i=0 ; i<parameters->tcp_numlayers ; i++){
        temp_rate = 0 ;
        if (img_fol->rates[i]== 0){
          parameters->tcp_rates[0]= ((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
          (CINEMA_48_CS * 8 * image->comps[0].dx * image->comps[0].dy);
        }else{
          temp_rate =((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
            (img_fol->rates[i] * 8 * image->comps[0].dx * image->comps[0].dy);
          if (temp_rate > CINEMA_48_CS ){
            parameters->tcp_rates[0]= ((float) (image->numcomps * image->comps[0].w * image->comps[0].h * image->comps[0].prec))/
            (CINEMA_48_CS * 8 * image->comps[0].dx * image->comps[0].dy);
          }else{
            parameters->tcp_rates[i]= img_fol->rates[i];
          }
        }
      }
      parameters->max_comp_size = COMP_48_CS;
      break;
    default:
      break;
  }
  parameters->cp_disto_alloc = 1;
}

/* ------------------------------------------------------------------------------------ */

int parse_cmdline_encoder(int argc, char **argv, opj_cparameters_t *parameters,
                          img_fol_t *img_fol, raw_cparameters_t *raw_cp, char *indexfilename) {
  int i, j,totlen;
  option_t long_option[]={
    {"cinema2K",REQ_ARG, NULL ,'w'},
    {"cinema4K",NO_ARG, NULL ,'y'},
    {"ImgDir",REQ_ARG, NULL ,'z'},
    {"TP",REQ_ARG, NULL ,'v'},
    {"SOP",NO_ARG, NULL ,'S'},
    {"EPH",NO_ARG, NULL ,'E'},
    {"OutFor",REQ_ARG, NULL ,'O'},
    {"POC",REQ_ARG, NULL ,'P'},
    {"ROI",REQ_ARG, NULL ,'R'},
  };

  /* parse the command line */
  const char optlist[] = "i:o:hr:q:n:b:c:t:p:s:SEM:x:R:d:T:If:P:C:F:m:"
#ifdef USE_JPWL
    "W:"
#endif /* USE_JPWL */
    ;

  totlen=sizeof(long_option);
  img_fol->set_out_format=0;
  raw_cp->rawWidth = 0;

  while (1) {
    int c = getopt_long(argc, argv, optlist,long_option,totlen);
    if (c == -1)
      break;
    switch (c) {
      case 'i':      /* input file */
      {
        char *infile = optarg;
        parameters->decod_format = get_file_format(infile);
        switch(parameters->decod_format) {
          case PGX_DFMT:
          case PXM_DFMT:
          case BMP_DFMT:
          case TIF_DFMT:
          case RAW_DFMT:
          case TGA_DFMT:
          case PNG_DFMT:
            break;
          default:
            fprintf(stderr,
              "!! Unrecognized format for infile : %s "
              "[accept only *.pnm, *.pgm, *.ppm, *.pgx, *.bmp, *.tif, *.raw or *.tga] !!\n\n",
              infile);
            return 1;
        }
        strncpy(parameters->infile, infile, sizeof(parameters->infile)-1);
      }
      break;

        /* ----------------------------------------------------- */

      case 'o':      /* output file */
      {
        char *outfile = optarg;
        parameters->cod_format = get_file_format(outfile);
        switch(parameters->cod_format) {
          case J2K_CFMT:
          case JP2_CFMT:
            break;
          default:
            fprintf(stderr, "Unknown output format image %s [only *.j2k, *.j2c or *.jp2]!! \n", outfile);
            return 1;
        }
        strncpy(parameters->outfile, outfile, sizeof(parameters->outfile)-1);
      }
      break;

        /* ----------------------------------------------------- */
      case 'O':      /* output format */
        {
          char outformat[50];
          char *of = optarg;
          sprintf(outformat,".%s",of);
          img_fol->set_out_format = 1;
          parameters->cod_format = get_file_format(outformat);
          switch(parameters->cod_format) {
            case J2K_CFMT:
            case JP2_CFMT:
              img_fol->out_format = optarg;
              break;
            default:
              fprintf(stderr, "Unknown output format image [only j2k, j2c, jp2]!! \n");
              return 1;
          }
        }
        break;


        /* ----------------------------------------------------- */


      case 'r':      /* rates rates/distorsion */
      {
        char *s = optarg;
        parameters->tcp_numlayers = 0;
        while (sscanf(s, "%f", &parameters->tcp_rates[parameters->tcp_numlayers]) == 1) {
          parameters->tcp_numlayers++;
          while (*s && *s != ',') {
            s++;
          }
          if (!*s)
            break;
          s++;
        }
        parameters->cp_disto_alloc = 1;
      }
      break;

        /* ----------------------------------------------------- */


      case 'F':      /* Raw image format parameters */
      {
        char signo;
        char *s = optarg;
        if (sscanf(s, "%d,%d,%d,%d,%c", &raw_cp->rawWidth, &raw_cp->rawHeight, &raw_cp->rawComp, &raw_cp->rawBitDepth, &signo) == 5) {
          if (signo == 's') {
            raw_cp->rawSigned = true;
            fprintf(stdout,"\nRaw file parameters: %d,%d,%d,%d Signed\n", raw_cp->rawWidth, raw_cp->rawHeight, raw_cp->rawComp, raw_cp->rawBitDepth);
          }
          else if (signo == 'u') {
            raw_cp->rawSigned = false;
            fprintf(stdout,"\nRaw file parameters: %d,%d,%d,%d Unsigned\n", raw_cp->rawWidth, raw_cp->rawHeight, raw_cp->rawComp, raw_cp->rawBitDepth);
          }
          else {
            fprintf(stderr,"\nError: invalid raw image parameters: Unknown sign of raw file\n");
            fprintf(stderr,"Please use the Format option -F:\n");
            fprintf(stderr,"-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
            fprintf(stderr,"Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
            fprintf(stderr,"Aborting\n");
          }
        }
        else {
          fprintf(stderr,"\nError: invalid raw image parameters\n");
          fprintf(stderr,"Please use the Format option -F:\n");
          fprintf(stderr,"-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
            fprintf(stderr,"Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
          fprintf(stderr,"Aborting\n");
          return 1;
        }
      }
      break;

        /* ----------------------------------------------------- */

      case 'q':      /* add fixed_quality */
      {
        char *s = optarg;
        while (sscanf(s, "%f", &parameters->tcp_distoratio[parameters->tcp_numlayers]) == 1) {
          parameters->tcp_numlayers++;
          while (*s && *s != ',') {
            s++;
          }
          if (!*s)
            break;
          s++;
        }
        parameters->cp_fixed_quality = 1;
      }
      break;

        /* dda */
        /* ----------------------------------------------------- */

      case 'f':      /* mod fixed_quality (before : -q) */
      {
        int *row = NULL, *col = NULL;
        int numlayers = 0, numresolution = 0, matrix_width = 0;

        char *s = optarg;
        sscanf(s, "%d", &numlayers);
        s++;
        if (numlayers > 9)
          s++;

        parameters->tcp_numlayers = numlayers;
        numresolution = parameters->numresolution;
        matrix_width = numresolution * 3;
        parameters->cp_matrice = (int *) malloc(numlayers * matrix_width * sizeof(int));
        s = s + 2;

        for (i = 0; i < numlayers; i++) {
          row = &parameters->cp_matrice[i * matrix_width];
          col = row;
          parameters->tcp_rates[i] = 1;
          sscanf(s, "%d,", &col[0]);
          s += 2;
          if (col[0] > 9)
            s++;
          col[1] = 0;
          col[2] = 0;
          for (j = 1; j < numresolution; j++) {
            col += 3;
            sscanf(s, "%d,%d,%d", &col[0], &col[1], &col[2]);
            s += 6;
            if (col[0] > 9)
              s++;
            if (col[1] > 9)
              s++;
            if (col[2] > 9)
              s++;
          }
          if (i < numlayers - 1)
            s++;
        }
        parameters->cp_fixed_alloc = 1;
      }
      break;

        /* ----------------------------------------------------- */

      case 't':      /* tiles */
      {
        sscanf(optarg, "%d,%d", &parameters->cp_tdx, &parameters->cp_tdy);
        parameters->tile_size_on = true;
      }
      break;

        /* ----------------------------------------------------- */

      case 'n':      /* resolution */
      {
        sscanf(optarg, "%d", &parameters->numresolution);
      }
      break;

        /* ----------------------------------------------------- */
      case 'c':      /* precinct dimension */
      {
        char sep;
        int res_spec = 0;

        char *s = optarg;
        do {
          sep = 0;
          sscanf(s, "[%d,%d]%c", &parameters->prcw_init[res_spec],
                                 &parameters->prch_init[res_spec], &sep);
          parameters->csty |= 0x01;
          res_spec++;
          s = strpbrk(s, "]") + 2;
        }
        while (sep == ',');
        parameters->res_spec = res_spec;
      }
      break;

        /* ----------------------------------------------------- */

      case 'b':      /* code-block dimension */
      {
        int cblockw_init = 0, cblockh_init = 0;
        sscanf(optarg, "%d,%d", &cblockw_init, &cblockh_init);
        if (cblockw_init * cblockh_init > 4096 || cblockw_init > 1024
          || cblockw_init < 4 || cblockh_init > 1024 || cblockh_init < 4) {
          fprintf(stderr,
            "!! Size of code_block error (option -b) !!\n\nRestriction :\n"
            "    * width*height<=4096\n    * 4<=width,height<= 1024\n\n");
          return 1;
        }
        parameters->cblockw_init = cblockw_init;
        parameters->cblockh_init = cblockh_init;
      }
      break;

        /* ----------------------------------------------------- */

      case 'x':      /* creation of index file */
      {
        char *index = optarg;
        strncpy(indexfilename, index, OPJ_PATH_LEN);
      }
      break;

        /* ----------------------------------------------------- */

      case 'p':      /* progression order */
      {
        char progression[4];

        strncpy(progression, optarg, 4);
        parameters->prog_order = give_progression(progression);
        if (parameters->prog_order == -1) {
          fprintf(stderr, "Unrecognized progression order "
            "[LRCP, RLCP, RPCL, PCRL, CPRL] !!\n");
          return 1;
        }
      }
      break;

        /* ----------------------------------------------------- */

      case 's':      /* subsampling factor */
      {
        if (sscanf(optarg, "%d,%d", &parameters->subsampling_dx,
                                    &parameters->subsampling_dy) != 2) {
          fprintf(stderr,  "'-s' sub-sampling argument error !  [-s dx,dy]\n");
          return 1;
        }
      }
      break;

        /* ----------------------------------------------------- */

      case 'd':      /* coordonnate of the reference grid */
      {
        if (sscanf(optarg, "%d,%d", &parameters->image_offset_x0,
                                    &parameters->image_offset_y0) != 2) {
          fprintf(stderr,  "-d 'coordonnate of the reference grid' argument "
            "error !! [-d x0,y0]\n");
          return 1;
        }
      }
      break;

        /* ----------------------------------------------------- */

      case 'h':      /* display an help description */
        encode_help_display();
        return 1;

        /* ----------------------------------------------------- */

      case 'P':      /* POC */
      {
        int numpocs = 0;    /* number of progression order change (POC) default 0 */
        opj_poc_t *POC = NULL;  /* POC : used in case of Progression order change */

        char *s = optarg;
        POC = parameters->POC;

        while (sscanf(s, "T%d=%d,%d,%d,%d,%d,%4s", &POC[numpocs].tile,
          &POC[numpocs].resno0, &POC[numpocs].compno0,
          &POC[numpocs].layno1, &POC[numpocs].resno1,
          &POC[numpocs].compno1, POC[numpocs].progorder) == 7) {
          POC[numpocs].prg1 = give_progression(POC[numpocs].progorder);
          numpocs++;
          while (*s && *s != '/') {
            s++;
          }
          if (!*s) {
            break;
          }
          s++;
        }
        parameters->numpocs = numpocs;
      }
      break;

        /* ------------------------------------------------------ */

      case 'S':      /* SOP marker */
      {
        parameters->csty |= 0x02;
      }
      break;

        /* ------------------------------------------------------ */

      case 'E':      /* EPH marker */
      {
        parameters->csty |= 0x04;
      }
      break;

        /* ------------------------------------------------------ */

      case 'M':      /* Mode switch pas tous au point !! */
      {
        int value = 0;
        if (sscanf(optarg, "%d", &value) == 1) {
          for (i = 0; i <= 5; i++) {
            int cache = value & (1 << i);
            if (cache)
              parameters->mode |= (1 << i);
          }
        }
      }
      break;

        /* ------------------------------------------------------ */

      case 'R':      /* ROI */
      {
        if (sscanf(optarg, "c=%d,U=%d", &parameters->roi_compno,
                                           &parameters->roi_shift) != 2) {
          fprintf(stderr, "ROI error !! [-ROI c='compno',U='shift']\n");
          return 1;
        }
      }
      break;

        /* ------------------------------------------------------ */

      case 'T':      /* Tile offset */
      {
        if (sscanf(optarg, "%d,%d", &parameters->cp_tx0, &parameters->cp_ty0) != 2) {
          fprintf(stderr, "-T 'tile offset' argument error !! [-T X0,Y0]");
          return 1;
        }
      }
      break;

        /* ------------------------------------------------------ */

      case 'C':      /* add a comment */
      {
        parameters->cp_comment = (char*)malloc(strlen(optarg) + 1);
        if(parameters->cp_comment) {
          strcpy(parameters->cp_comment, optarg);
        }
      }
      break;


        /* ------------------------------------------------------ */

      case 'I':      /* reversible or not */
      {
        parameters->irreversible = 1;
      }
      break;

      /* ------------------------------------------------------ */

      case 'v':      /* Tile part generation*/
      {
        parameters->tp_flag = optarg[0];
        parameters->tp_on = 1;
      }
      break;

        /* ------------------------------------------------------ */

      case 'z':      /* Image Directory path */
      {
        img_fol->imgdirpath = (char*)malloc(strlen(optarg) + 1);
        strcpy(img_fol->imgdirpath,optarg);
        img_fol->set_imgdir=1;
      }
      break;

        /* ------------------------------------------------------ */

      case 'w':      /* Digital Cinema 2K profile compliance*/
      {
        int fps=0;
        sscanf(optarg,"%d",&fps);
        if(fps == 24){
          parameters->cp_cinema = CINEMA2K_24;
        }else if(fps == 48 ){
          parameters->cp_cinema = CINEMA2K_48;
        }else {
          fprintf(stderr,"Incorrect value!! must be 24 or 48\n");
          return 1;
        }
        fprintf(stdout,"CINEMA 2K compliant codestream\n");
        parameters->cp_rsiz = CINEMA2K;

      }
      break;

        /* ------------------------------------------------------ */

      case 'y':      /* Digital Cinema 4K profile compliance*/
      {
        parameters->cp_cinema = CINEMA4K_24;
        fprintf(stdout,"CINEMA 4K compliant codestream\n");
        parameters->cp_rsiz = CINEMA4K;
      }
      break;

      case 'm':      /* output file */
      {
        char *lFilename = optarg;
        char * lMatrix;
        char *lCurrentPtr ;
        int lNbComp = 0;
        int lTotalComp;
        int lMctComp;
        float * lCurrentDoublePtr;
        float * lSpace;
        int * l_int_ptr;
        int i;
        int lStrLen;

        FILE * lFile = fopen(lFilename,"r");
        if
          (lFile == NULL)
        {
          return 1;
        }
        fseek(lFile,0,SEEK_END);
        lStrLen = ftell(lFile);
        fseek(lFile,0,SEEK_SET);
        lMatrix = (char *) malloc(lStrLen + 1);
        fread(lMatrix,lStrLen,1,lFile);
        fclose(lFile);
        lMatrix[lStrLen] = 0;
        lCurrentPtr = lMatrix;

        // replace ',' by 0
        while
          (*lCurrentPtr != 0 )
        {
          if
            (*lCurrentPtr == ' ')
          {
            *lCurrentPtr = 0;
            ++lNbComp;
          }
          ++lCurrentPtr;
        }
        ++lNbComp;
        lCurrentPtr = lMatrix;

        lNbComp = (int) (sqrt(4*lNbComp + 1)/2. - 0.5);
        lMctComp = lNbComp * lNbComp;
        lTotalComp = lMctComp + lNbComp;
        lSpace = (float *) malloc(lTotalComp * sizeof(float));
        lCurrentDoublePtr = lSpace;
        for
          (i=0;i<lMctComp;++i)
        {
          lStrLen = strlen(lCurrentPtr) + 1;
          *lCurrentDoublePtr++ = (float) atof(lCurrentPtr);
          lCurrentPtr += lStrLen;
        }
        l_int_ptr = (int*) lCurrentDoublePtr;
        for
          (i=0;i<lNbComp;++i)
        {
          lStrLen = strlen(lCurrentPtr) + 1;
          *l_int_ptr++ = atoi(lCurrentPtr);
          lCurrentPtr += lStrLen;
        }
        opj_set_MCT(parameters,lSpace,(int *)(lSpace + lMctComp), lNbComp);
        free(lSpace);
        free(lMatrix);
      }
      break;

        /* ------------------------------------------------------ */

/* UniPG>> */
#ifdef USE_JPWL
        /* ------------------------------------------------------ */

      case 'W':      /* JPWL capabilities switched on */
      {
        char *token = NULL;
        int hprot, pprot, sens, addr, size, range;

        /* we need to enable indexing */
        if (!indexfilename || !*indexfilename) {
          strncpy(indexfilename, JPWL_PRIVATEINDEX_NAME, OPJ_PATH_LEN);
        }

        /* search for different protection methods */

        /* break the option in comma points and parse the result */
        token = strtok(optarg, ",");
        while(token != NULL) {

          /* search header error protection method */
          if (*token == 'h') {

            static int tile = 0, tilespec = 0, lasttileno = 0;

            hprot = 1; /* predefined method */

            if(sscanf(token, "h=%d", &hprot) == 1) {
              /* Main header, specified */
              if (!((hprot == 0) || (hprot == 1) || (hprot == 16) || (hprot == 32) ||
                ((hprot >= 37) && (hprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid main header protection method h = %d\n", hprot);
                return 1;
              }
              parameters->jpwl_hprot_MH = hprot;

            } else if(sscanf(token, "h%d=%d", &tile, &hprot) == 2) {
              /* Tile part header, specified */
              if (!((hprot == 0) || (hprot == 1) || (hprot == 16) || (hprot == 32) ||
                ((hprot >= 37) && (hprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid tile part header protection method h = %d\n", hprot);
                return 1;
              }
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method t = %d\n", tile);
                return 1;
              }
              if (tilespec < JPWL_MAX_NO_TILESPECS) {
                parameters->jpwl_hprot_TPH_tileno[tilespec] = lasttileno = tile;
                parameters->jpwl_hprot_TPH[tilespec++] = hprot;
              }

            } else if(sscanf(token, "h%d", &tile) == 1) {
              /* Tile part header, unspecified */
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method t = %d\n", tile);
                return 1;
              }
              if (tilespec < JPWL_MAX_NO_TILESPECS) {
                parameters->jpwl_hprot_TPH_tileno[tilespec] = lasttileno = tile;
                parameters->jpwl_hprot_TPH[tilespec++] = hprot;
              }


            } else if (!strcmp(token, "h")) {
              /* Main header, unspecified */
              parameters->jpwl_hprot_MH = hprot;

            } else {
              fprintf(stderr, "ERROR -> invalid protection method selection = %s\n", token);
              return 1;
            };

          }

          /* search packet error protection method */
          if (*token == 'p') {

            static int pack = 0, tile = 0, packspec = 0;

            pprot = 1; /* predefined method */

            if (sscanf(token, "p=%d", &pprot) == 1) {
              /* Method for all tiles and all packets */
              if (!((pprot == 0) || (pprot == 1) || (pprot == 16) || (pprot == 32) ||
                ((pprot >= 37) && (pprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid default packet protection method p = %d\n", pprot);
                return 1;
              }
              parameters->jpwl_pprot_tileno[0] = 0;
              parameters->jpwl_pprot_packno[0] = 0;
              parameters->jpwl_pprot[0] = pprot;

            } else if (sscanf(token, "p%d=%d", &tile, &pprot) == 2) {
              /* method specified from that tile on */
              if (!((pprot == 0) || (pprot == 1) || (pprot == 16) || (pprot == 32) ||
                ((pprot >= 37) && (pprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid packet protection method p = %d\n", pprot);
                return 1;
              }
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method p = %d\n", tile);
                return 1;
              }
              if (packspec < JPWL_MAX_NO_PACKSPECS) {
                parameters->jpwl_pprot_tileno[packspec] = tile;
                parameters->jpwl_pprot_packno[packspec] = 0;
                parameters->jpwl_pprot[packspec++] = pprot;
              }

            } else if (sscanf(token, "p%d:%d=%d", &tile, &pack, &pprot) == 3) {
              /* method fully specified from that tile and that packet on */
              if (!((pprot == 0) || (pprot == 1) || (pprot == 16) || (pprot == 32) ||
                ((pprot >= 37) && (pprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid packet protection method p = %d\n", pprot);
                return 1;
              }
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method p = %d\n", tile);
                return 1;
              }
              if (pack < 0) {
                fprintf(stderr, "ERROR -> invalid packet number on protection method p = %d\n", pack);
                return 1;
              }
              if (packspec < JPWL_MAX_NO_PACKSPECS) {
                parameters->jpwl_pprot_tileno[packspec] = tile;
                parameters->jpwl_pprot_packno[packspec] = pack;
                parameters->jpwl_pprot[packspec++] = pprot;
              }

            } else if (sscanf(token, "p%d:%d", &tile, &pack) == 2) {
              /* default method from that tile and that packet on */
              if (!((pprot == 0) || (pprot == 1) || (pprot == 16) || (pprot == 32) ||
                ((pprot >= 37) && (pprot <= 128)))) {
                fprintf(stderr, "ERROR -> invalid packet protection method p = %d\n", pprot);
                return 1;
              }
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method p = %d\n", tile);
                return 1;
              }
              if (pack < 0) {
                fprintf(stderr, "ERROR -> invalid packet number on protection method p = %d\n", pack);
                return 1;
              }
              if (packspec < JPWL_MAX_NO_PACKSPECS) {
                parameters->jpwl_pprot_tileno[packspec] = tile;
                parameters->jpwl_pprot_packno[packspec] = pack;
                parameters->jpwl_pprot[packspec++] = pprot;
              }

            } else if (sscanf(token, "p%d", &tile) == 1) {
              /* default from a tile on */
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on protection method p = %d\n", tile);
                return 1;
              }
              if (packspec < JPWL_MAX_NO_PACKSPECS) {
                parameters->jpwl_pprot_tileno[packspec] = tile;
                parameters->jpwl_pprot_packno[packspec] = 0;
                parameters->jpwl_pprot[packspec++] = pprot;
              }


            } else if (!strcmp(token, "p")) {
              /* all default */
              parameters->jpwl_pprot_tileno[0] = 0;
              parameters->jpwl_pprot_packno[0] = 0;
              parameters->jpwl_pprot[0] = pprot;

            } else {
              fprintf(stderr, "ERROR -> invalid protection method selection = %s\n", token);
              return 1;
            };

          }

          /* search sensitivity method */
          if (*token == 's') {

            static int tile = 0, tilespec = 0, lasttileno = 0;

            sens = 0; /* predefined: relative error */

            if(sscanf(token, "s=%d", &sens) == 1) {
              /* Main header, specified */
              if ((sens < -1) || (sens > 7)) {
                fprintf(stderr, "ERROR -> invalid main header sensitivity method s = %d\n", sens);
                return 1;
              }
              parameters->jpwl_sens_MH = sens;

            } else if(sscanf(token, "s%d=%d", &tile, &sens) == 2) {
              /* Tile part header, specified */
              if ((sens < -1) || (sens > 7)) {
                fprintf(stderr, "ERROR -> invalid tile part header sensitivity method s = %d\n", sens);
                return 1;
              }
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on sensitivity method t = %d\n", tile);
                return 1;
              }
              if (tilespec < JPWL_MAX_NO_TILESPECS) {
                parameters->jpwl_sens_TPH_tileno[tilespec] = lasttileno = tile;
                parameters->jpwl_sens_TPH[tilespec++] = sens;
              }

            } else if(sscanf(token, "s%d", &tile) == 1) {
              /* Tile part header, unspecified */
              if (tile < 0) {
                fprintf(stderr, "ERROR -> invalid tile part number on sensitivity method t = %d\n", tile);
                return 1;
              }
              if (tilespec < JPWL_MAX_NO_TILESPECS) {
                parameters->jpwl_sens_TPH_tileno[tilespec] = lasttileno = tile;
                parameters->jpwl_sens_TPH[tilespec++] = hprot;
              }

            } else if (!strcmp(token, "s")) {
              /* Main header, unspecified */
              parameters->jpwl_sens_MH = sens;

            } else {
              fprintf(stderr, "ERROR -> invalid sensitivity method selection = %s\n", token);
              return 1;
            };

            parameters->jpwl_sens_size = 2; /* 2 bytes for default size */
          }

          /* search addressing size */
          if (*token == 'a') {


            addr = 0; /* predefined: auto */

            if(sscanf(token, "a=%d", &addr) == 1) {
              /* Specified */
              if ((addr != 0) && (addr != 2) && (addr != 4)) {
                fprintf(stderr, "ERROR -> invalid addressing size a = %d\n", addr);
                return 1;
              }
              parameters->jpwl_sens_addr = addr;

            } else if (!strcmp(token, "a")) {
              /* default */
              parameters->jpwl_sens_addr = addr; /* auto for default size */

            } else {
              fprintf(stderr, "ERROR -> invalid addressing selection = %s\n", token);
              return 1;
            };

          }

          /* search sensitivity size */
          if (*token == 'z') {


            size = 1; /* predefined: 1 byte */

            if(sscanf(token, "z=%d", &size) == 1) {
              /* Specified */
              if ((size != 0) && (size != 1) && (size != 2)) {
                fprintf(stderr, "ERROR -> invalid sensitivity size z = %d\n", size);
                return 1;
              }
              parameters->jpwl_sens_size = size;

            } else if (!strcmp(token, "a")) {
              /* default */
              parameters->jpwl_sens_size = size; /* 1 for default size */

            } else {
              fprintf(stderr, "ERROR -> invalid size selection = %s\n", token);
              return 1;
            };

          }

          /* search range method */
          if (*token == 'g') {


            range = 0; /* predefined: 0 (packet) */

            if(sscanf(token, "g=%d", &range) == 1) {
              /* Specified */
              if ((range < 0) || (range > 3)) {
                fprintf(stderr, "ERROR -> invalid sensitivity range method g = %d\n", range);
                return 1;
              }
              parameters->jpwl_sens_range = range;

            } else if (!strcmp(token, "g")) {
              /* default */
              parameters->jpwl_sens_range = range;

            } else {
              fprintf(stderr, "ERROR -> invalid range selection = %s\n", token);
              return 1;
            };

          }

          /* next token or bust */
          token = strtok(NULL, ",");
        };


        /* some info */
        fprintf(stdout, "Info: JPWL capabilities enabled\n");
        parameters->jpwl_epc_on = true;

      }
      break;
#endif /* USE_JPWL */
/* <<UniPG */

        /* ------------------------------------------------------ */

      default:
        fprintf(stderr, "ERROR -> Command line not valid\n");
        return 1;
    }
  }

  /* check for possible errors */
  if (parameters->cp_cinema){
    if(parameters->tcp_numlayers > 1){
      parameters->cp_rsiz = STD_RSIZ;
       fprintf(stdout,"Warning: DC profiles do not allow more than one quality layer. The codestream created will not be compliant with the DC profile\n");
    }
  }
  if(img_fol->set_imgdir == 1){
    if(!(parameters->infile[0] == 0)){
      fprintf(stderr, "Error: options -ImgDir and -i cannot be used together !!\n");
      return 1;
    }
    if(img_fol->set_out_format == 0){
      fprintf(stderr, "Error: When -ImgDir is used, -OutFor <FORMAT> must be used !!\n");
      fprintf(stderr, "Only one format allowed! Valid formats are j2k and jp2!!\n");
      return 1;
    }
    if(!((parameters->outfile[0] == 0))){
      fprintf(stderr, "Error: options -ImgDir and -o cannot be used together !!\n");
      fprintf(stderr, "Specify OutputFormat using -OutFor<FORMAT> !!\n");
      return 1;
    }
  }else{
    if((parameters->infile[0] == 0) || (parameters->outfile[0] == 0)) {
      fprintf(stderr, "Error: One of the options; -i or -ImgDir must be specified\n");
      fprintf(stderr, "Error: When using -i; -o must be used\n");
      fprintf(stderr, "usage: image_to_j2k -i image-file -o j2k/jp2-file (+ options)\n");
      return 1;
    }
  }

  if (parameters->decod_format == RAW_DFMT && raw_cp->rawWidth == 0) {
      fprintf(stderr,"\nError: invalid raw image parameters\n");
      fprintf(stderr,"Please use the Format option -F:\n");
      fprintf(stderr,"-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
            fprintf(stderr,"Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
      fprintf(stderr,"Aborting\n");
      return 1;
  }

  if ((parameters->cp_disto_alloc || parameters->cp_fixed_alloc || parameters->cp_fixed_quality)
    && (!(parameters->cp_disto_alloc ^ parameters->cp_fixed_alloc ^ parameters->cp_fixed_quality))) {
    fprintf(stderr, "Error: options -r -q and -f cannot be used together !!\n");
    return 1;
  }        /* mod fixed_quality */

  /* if no rate entered, lossless by default */
  if (parameters->tcp_numlayers == 0) {
    parameters->tcp_rates[0] = 0;  /* MOD antonin : losslessbug */
    parameters->tcp_numlayers++;
    parameters->cp_disto_alloc = 1;
  }

  if((parameters->cp_tx0 > parameters->image_offset_x0) || (parameters->cp_ty0 > parameters->image_offset_y0)) {
    fprintf(stderr,
      "Error: Tile offset dimension is unnappropriate --> TX0(%d)<=IMG_X0(%d) TYO(%d)<=IMG_Y0(%d) \n",
      parameters->cp_tx0, parameters->image_offset_x0, parameters->cp_ty0, parameters->image_offset_y0);
    return 1;
  }

  for (i = 0; i < parameters->numpocs; i++) {
    if (parameters->POC[i].prg == -1) {
      fprintf(stderr,
        "Unrecognized progression order in option -P (POC n %d) [LRCP, RLCP, RPCL, PCRL, CPRL] !!\n",
        i + 1);
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
sample debug callback expecting a FILE* client object
*/
void info_callback(const char *msg, void *client_data) {
  FILE *stream = (FILE*)client_data;
  fprintf(stream, "[INFO] %s", msg);
}

/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
  bool bSuccess;
  opj_cparameters_t parameters;  /* compression parameters */
  img_fol_t img_fol;
  opj_image_t *image = NULL;
  int i,num_images;
  int imageno;
  dircnt_t *dirptr;
  raw_cparameters_t raw_cp;
  opj_codestream_info_t cstr_info;    /* Codestream information structure */
  char indexfilename[OPJ_PATH_LEN];  /* index file name */
  opj_stream_t *cio = 00;
  opj_codec_t* cinfo = 00;
  FILE *f = NULL;

  /*
  configure the event callbacks (not required)
  setting of each callback is optionnal
  */
  /* set encoding parameters to default values */
  opj_set_default_encoder_parameters(&parameters);

  /* Initialize indexfilename and img_fol */
  *indexfilename = 0;
  memset(&img_fol,0,sizeof(img_fol_t));

  /* parse input and get user encoding parameters */
  if(parse_cmdline_encoder(argc, argv, &parameters,&img_fol, &raw_cp, indexfilename) == 1) {
    return 1;
  }

  if (parameters.cp_cinema){
    img_fol.rates = (float*)malloc(parameters.tcp_numlayers * sizeof(float));
    for(i=0; i< parameters.tcp_numlayers; i++){
      img_fol.rates[i] = parameters.tcp_rates[i];
    }
    cinema_parameters(&parameters);
  }

  /* Create comment for codestream */
  if(parameters.cp_comment == NULL) {
    const char comment[] = "Created by OpenJPEG version ";
    const size_t clen = strlen(comment);
    const char *version = opj_version();
/* UniPG>> */
#ifdef USE_JPWL
    parameters.cp_comment = (char*)malloc(clen+strlen(version)+11);
    sprintf(parameters.cp_comment,"%s%s with JPWL", comment, version);
#else
    parameters.cp_comment = (char*)malloc(clen+strlen(version)+1);
    sprintf(parameters.cp_comment,"%s%s", comment, version);
#endif
/* <<UniPG */
  }

  /* Read directory if necessary */
  if(img_fol.set_imgdir==1){
    num_images=get_num_images(img_fol.imgdirpath);
    dirptr=(dircnt_t*)malloc(sizeof(dircnt_t));
    if(dirptr){
      dirptr->filename_buf = (char*)malloc(num_images*OPJ_PATH_LEN*sizeof(char));  // Stores at max 10 image file names
      dirptr->filename = (char**) malloc(num_images*sizeof(char*));
      if(!dirptr->filename_buf){
        return 0;
      }
      for(i=0;i<num_images;i++){
        dirptr->filename[i] = dirptr->filename_buf + i*OPJ_PATH_LEN;
      }
    }
    if(load_images(dirptr,img_fol.imgdirpath)==1){
      return 0;
    }
    if (num_images==0){
      fprintf(stdout,"Folder is empty\n");
      return 0;
    }
  }else{
    num_images=1;
  }
  /*Encoding image one by one*/
  for(imageno=0;imageno<num_images;imageno++)  {
    image = NULL;
    fprintf(stderr,"\n");

    if(img_fol.set_imgdir==1){
      if (get_next_file(imageno, dirptr,&img_fol, &parameters)) {
        fprintf(stderr,"skipping file...\n");
        continue;
      }
    }
    switch(parameters.decod_format) {
      case PGX_DFMT:
        break;
      case PXM_DFMT:
        break;
      case BMP_DFMT:
        break;
      case TIF_DFMT:
        break;
      case RAW_DFMT:
        break;
      case TGA_DFMT:
        break;
      case PNG_DFMT:
        break;
      default:
        fprintf(stderr,"skipping file...\n");
        continue;
    }

      /* decode the source image */
      /* ----------------------- */

      switch (parameters.decod_format) {
        case PGX_DFMT:
          image = pgxtoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load pgx file\n");
            return 1;
          }
          break;

        case PXM_DFMT:
          image = pnmtoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load pnm file\n");
            return 1;
          }
          break;

        case BMP_DFMT:
          image = bmptoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load bmp file\n");
            return 1;
          }
          break;

        case TIF_DFMT:
          image = tiftoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load tiff file\n");
            return 1;
          }
        break;

        case RAW_DFMT:
          image = rawtoimage(parameters.infile, &parameters, &raw_cp);
          if (!image) {
            fprintf(stderr, "Unable to load raw file\n");
            return 1;
          }
        break;

        case TGA_DFMT:
          image = tgatoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load tga file\n");
            return 1;
          }
        break;

        case PNG_DFMT:
          image = pngtoimage(parameters.infile, &parameters);
          if (!image) {
            fprintf(stderr, "Unable to load png file\n");
            return 1;
          }
          break;
    }
      /* Decide if MCT should be used */
      parameters.tcp_mct = image->numcomps == 3 ? 1 : 0;

      if(parameters.cp_cinema){
        cinema_setup_encoder(&parameters,image,&img_fol);
      }

      /* encode the destination image */
      /* ---------------------------- */


      cinfo = parameters.cod_format == J2K_CFMT ? opj_create_compress(CODEC_J2K) : opj_create_compress(CODEC_JP2);
      opj_setup_encoder(cinfo, &parameters, image);
      f = fopen(parameters.outfile, "wb");
      if
        (! f)
      {
        fprintf(stderr, "failed to encode image\n");
        return 1;
      }
      /* open a byte stream for writing */
      /* allocate memory for all tiles */
      cio = opj_stream_create_default_file_stream(f,false);
      if
        (! cio)
      {
        return 1;
      }
      /* encode the image */
      /*if (*indexfilename)          // If need to extract codestream information
        bSuccess = opj_encode_with_info(cinfo, cio, image, &cstr_info);
        else*/
      bSuccess = opj_start_compress(cinfo,image,cio);
      bSuccess = bSuccess && opj_encode(cinfo, cio);
      bSuccess = bSuccess && opj_end_compress(cinfo, cio);

      if
        (!bSuccess)
      {
        opj_stream_destroy(cio);
        fclose(f);
        fprintf(stderr, "failed to encode image\n");
        return 1;
      }

      fprintf(stderr,"Generated outfile %s\n",parameters.outfile);
      /* close and free the byte stream */
      opj_stream_destroy(cio);
      fclose(f);

      /* Write the index to disk */
      if (*indexfilename) {
        bSuccess = write_index_file(&cstr_info, indexfilename);
        if (bSuccess) {
          fprintf(stderr, "Failed to output index file\n");
        }
      }

      /* free remaining compression structures */
      opj_destroy_codec(cinfo);
      if (*indexfilename)
        opj_destroy_cstr_info(&cstr_info);

      /* free image data */
      opj_image_destroy(image);
  }

  /* free user parameters structure */
  if(parameters.cp_comment) free(parameters.cp_comment);
  if(parameters.cp_matrice) free(parameters.cp_matrice);

  return 0;
}
