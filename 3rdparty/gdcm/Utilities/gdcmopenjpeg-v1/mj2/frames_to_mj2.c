/*
* Copyright (c) 2003-2004, François-Olivier Devaux
* Copyright (c) 2002-2004,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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
#include <stdlib.h>
#include <string.h>

#include "openjpeg.h"
#include "../libopenjpeg/j2k_lib.h"
#include "../libopenjpeg/j2k.h"
#include "../libopenjpeg/jp2.h"
#include "../libopenjpeg/cio.h"
#include "mj2.h"
#include "mj2_convert.h"
#include "getopt.h"

/**
Size of memory first allocated for MOOV box
*/
#define TEMP_BUF 10000 

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


void help_display()
{
  fprintf(stdout,"HELP for frames_to_mj2\n----\n\n");
  fprintf(stdout,"- the -h option displays this help information on screen\n\n");
  
  
  fprintf(stdout,"List of parameters for the MJ2 encoder:\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"REMARKS:\n");
  fprintf(stdout,"---------\n");
  fprintf(stdout,"\n");
  fprintf
    (stdout,"The markers written to the main_header are : SOC SIZ COD QCD COM.\n");
  fprintf
    (stdout,"COD and QCD never appear in the tile_header.\n");
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
  fprintf(stdout,"\n");
  fprintf(stdout,"Parameters:\n");
  fprintf(stdout,"------------\n");
  fprintf(stdout,"\n");
  fprintf
    (stdout,"Required Parameters (except with -h):\n");
  fprintf
    (stdout,"-i           : source file  (-i source.yuv) \n");
  fprintf
    (stdout,"-o           : destination file (-o dest.mj2) \n");
  fprintf
    (stdout,"Optional Parameters:\n");
  fprintf(stdout,"-h           : display the help information \n");
  fprintf(stdout,"-r           : different compression ratios for successive layers (-r 20,10,5)\n ");
  fprintf(stdout,"	         - The rate specified for each quality level is the desired \n");
  fprintf(stdout,"	           compression factor.\n");
  fprintf(stdout,"		   Example: -r 20,10,1 means quality 1: compress 20x, \n");
  fprintf(stdout,"		     quality 2: compress 10x and quality 3: compress lossless\n");
  fprintf(stdout,"               (options -r and -q cannot be used together)\n ");
  
  fprintf(stdout,"-q           : different psnr for successive layers (-q 30,40,50) \n ");
  
  fprintf(stdout,"               (options -r and -q cannot be used together)\n ");
  
  fprintf(stdout,"-n           : number of resolutions (-n 3) \n");
  fprintf(stdout,"-b           : size of code block (-b 32,32) \n");
  fprintf(stdout,"-c           : size of precinct (-c 128,128) \n");
  fprintf(stdout,"-t           : size of tile (-t 512,512) \n");
  fprintf
    (stdout,"-p           : progression order (-p LRCP) [LRCP, RLCP, RPCL, PCRL, CPRL] \n");
  fprintf
    (stdout,"-s           : subsampling factor (-s 2,2) [-s X,Y] \n");
  fprintf(stdout,"	     Remark: subsampling bigger than 2 can produce error\n");
  fprintf
    (stdout,"-SOP         : write SOP marker before each packet \n");
  fprintf
    (stdout,"-EPH         : write EPH marker after each header packet \n");
  fprintf
    (stdout,"-M           : mode switch (-M 3) [1=BYPASS(LAZY) 2=RESET 4=RESTART(TERMALL)\n");
  fprintf
    (stdout,"                 8=VSC 16=ERTERM(SEGTERM) 32=SEGMARK(SEGSYM)] \n");
  fprintf
    (stdout,"                 Indicate multiple modes by adding their values. \n");
  fprintf
    (stdout,"                 ex: RESTART(4) + RESET(2) + SEGMARK(32) = -M 38\n");
  fprintf
    (stdout,"-ROI         : c=%%d,U=%%d : quantization indices upshifted \n");
  fprintf
    (stdout,"               for component c=%%d [%%d = 0,1,2]\n");
  fprintf
    (stdout,"               with a value of U=%%d [0 <= %%d <= 37] (i.e. -ROI:c=0,U=25) \n");
  fprintf
    (stdout,"-d           : offset of the origin of the image (-d 150,300) \n");
  fprintf
    (stdout,"-T           : offset of the origin of the tiles (-T 100,75) \n");
  fprintf(stdout,"-I           : use the irreversible DWT 9-7 (-I) \n");
  fprintf(stdout,"-W           : image width, height and the dx and dy subsampling \n");
  fprintf(stdout,"               of the Cb and Cr components for YUV files \n");
  fprintf(stdout,"               (default is '352,288,2,2' for CIF format's 352x288 and 4:2:0)\n");
  fprintf(stdout,"-F           : video frame rate (set to 25 by default)\n");
  
  fprintf(stdout,"\n");
  fprintf(stdout,"IMPORTANT:\n");
  fprintf(stdout,"-----------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"The index file has the structure below:\n");
  fprintf(stdout,"---------------------------------------\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Image_height Image_width\n");
  fprintf(stdout,"progression order\n");
  fprintf(stdout,"Tiles_size_X Tiles_size_Y\n");
  fprintf(stdout,"Components_nb\n");
  fprintf(stdout,"Layers_nb\n");
  fprintf(stdout,"decomposition_levels\n");
  fprintf(stdout,"[Precincts_size_X_res_Nr Precincts_size_Y_res_Nr]...\n");
  fprintf(stdout,"   [Precincts_size_X_res_0 Precincts_size_Y_res_0]\n");
  fprintf(stdout,"Main_header_end_position\n");
  fprintf(stdout,"Codestream_size\n");
  fprintf(stdout,"Tile_0 start_pos end_Theader end_pos TotalDisto NumPix MaxMSE\n");
  fprintf(stdout,"Tile_1   ''           ''        ''        ''       ''    ''\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"Tile_Nt   ''           ''        ''        ''       ''    ''\n");
  fprintf(stdout,"Tpacket_0 Tile layer res. comp. prec. start_pos end_pos disto\n");
  fprintf(stdout,"...\n");
  fprintf(stdout,"Tpacket_Np ''   ''    ''   ''    ''       ''       ''     ''\n");
  
  fprintf(stdout,"MaxDisto\n");
  
  fprintf(stdout,"TotalDisto\n\n");
}

int give_progression(char progression[4])
{
  if (progression[0] == 'L' && progression[1] == 'R'
    && progression[2] == 'C' && progression[3] == 'P') {
    return 0;
  } else {
    if (progression[0] == 'R' && progression[1] == 'L'
      && progression[2] == 'C' && progression[3] == 'P') {
      return 1;
    } else {
      if (progression[0] == 'R' && progression[1] == 'P'
				&& progression[2] == 'C' && progression[3] == 'L') {
				return 2;
      } else {
				if (progression[0] == 'P' && progression[1] == 'C'
					&& progression[2] == 'R' && progression[3] == 'L') {
					return 3;
				} else {
					if (progression[0] == 'C' && progression[1] == 'P'
						&& progression[2] == 'R' && progression[3] == 'L') {
						return 4;
					} else {
						return -1;
					}
				}
      }
    }
  }
}




int main(int argc, char **argv)
{
	mj2_cparameters_t mj2_parameters;	/* MJ2 compression parameters */
	opj_cparameters_t *j2k_parameters;	/* J2K compression parameters */
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_cio_t *cio;
	int value;
  opj_mj2_t *movie;
	opj_image_t *img;
  int i, j;
  char *s, S1, S2, S3;
  unsigned char *buf;
  int x1, y1,  len;
  long mdat_initpos, offset;
  FILE *mj2file;
  int sampleno;  
	opj_cinfo_t* cinfo;
  bool bSuccess;
	int numframes;
	double total_time = 0;	

  /* default value */
  /* ------------- */
  mj2_parameters.Dim[0] = 0;
  mj2_parameters.Dim[1] = 0;
  mj2_parameters.w = 352;			// CIF default value
  mj2_parameters.h = 288;			// CIF default value
  mj2_parameters.CbCr_subsampling_dx = 2;	// CIF default value
  mj2_parameters.CbCr_subsampling_dy = 2;	// CIF default value
  mj2_parameters.frame_rate = 25;	  
	/*
	configure the event callbacks (not required)
	setting of each callback is optionnal
	*/
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = error_callback;
	event_mgr.warning_handler = warning_callback;
	event_mgr.info_handler = NULL;
    
	/* set J2K encoding parameters to default values */
	opj_set_default_encoder_parameters(&mj2_parameters.j2k_parameters);
	j2k_parameters = &mj2_parameters.j2k_parameters;

	/* Create comment for codestream */
	if(j2k_parameters->cp_comment == NULL) {
    const char comment[] = "Created by OpenJPEG version ";
		const size_t clen = strlen(comment);
    const char *version = opj_version();
		j2k_parameters->cp_comment = (char*)malloc(clen+strlen(version)+1);
		sprintf(j2k_parameters->cp_comment,"%s%s", comment, version);
	}

	mj2_parameters.decod_format = 0;
	mj2_parameters.cod_format = 0;

  while (1) {
    int c = getopt(argc, argv,
      "i:o:r:q:f:t:n:c:b:p:s:d:P:S:E:M:R:T:C:I:W:F:h");
    if (c == -1)
      break;
    switch (c) {
    case 'i':			/* IN fill */
			{
				char *infile = optarg;
				s = optarg;
				while (*s) {
					s++;
				}
				s--;
				S3 = *s;
				s--;
				S2 = *s;
				s--;
				S1 = *s;
				
				if ((S1 == 'y' && S2 == 'u' && S3 == 'v')
					|| (S1 == 'Y' && S2 == 'U' && S3 == 'V')) {
					mj2_parameters.decod_format = YUV_DFMT;				
				}
				else {
					fprintf(stderr,
						"!! Unrecognized format for infile : %c%c%c [accept only *.yuv] !!\n\n",
						S1, S2, S3);
					return 1;
				}
				strncpy(mj2_parameters.infile, infile, sizeof(mj2_parameters.infile)-1);
			}
      break;
      /* ----------------------------------------------------- */
    case 'o':			/* OUT fill */
			{
				char *outfile = optarg;
				while (*outfile) {
					outfile++;
				}
				outfile--;
				S3 = *outfile;
				outfile--;
				S2 = *outfile;
				outfile--;
				S1 = *outfile;
				
				outfile = optarg;
				
				if ((S1 == 'm' && S2 == 'j' && S3 == '2')
					|| (S1 == 'M' && S2 == 'J' && S3 == '2'))
					mj2_parameters.cod_format = MJ2_CFMT;
				else {
					fprintf(stderr,
						"Unknown output format image *.%c%c%c [only *.mj2]!! \n",
						S1, S2, S3);
					return 1;
				}
				strncpy(mj2_parameters.outfile, outfile, sizeof(mj2_parameters.outfile)-1);      
      }
      break;
      /* ----------------------------------------------------- */
    case 'r':			/* rates rates/distorsion */
			{
				float rate;
				s = optarg;
				while (sscanf(s, "%f", &rate) == 1) {
					j2k_parameters->tcp_rates[j2k_parameters->tcp_numlayers] = rate * 2;
					j2k_parameters->tcp_numlayers++;
					while (*s && *s != ',') {
						s++;
					}
					if (!*s)
						break;
					s++;
				}
				j2k_parameters->cp_disto_alloc = 1;
			}
      break;
      /* ----------------------------------------------------- */
    case 'q':			/* add fixed_quality */
      s = optarg;
			while (sscanf(s, "%f", &j2k_parameters->tcp_distoratio[j2k_parameters->tcp_numlayers]) == 1) {
				j2k_parameters->tcp_numlayers++;
				while (*s && *s != ',') {
					s++;
				}
				if (!*s)
					break;
				s++;
			}
			j2k_parameters->cp_fixed_quality = 1;
      break;
      /* dda */
      /* ----------------------------------------------------- */
    case 'f':			/* mod fixed_quality (before : -q) */
			{
				int *row = NULL, *col = NULL;
				int numlayers = 0, numresolution = 0, matrix_width = 0;
				
				s = optarg;
				sscanf(s, "%d", &numlayers);
				s++;
				if (numlayers > 9)
					s++;
				
				j2k_parameters->tcp_numlayers = numlayers;
				numresolution = j2k_parameters->numresolution;
				matrix_width = numresolution * 3;
				j2k_parameters->cp_matrice = (int *) malloc(numlayers * matrix_width * sizeof(int));
				s = s + 2;
				
				for (i = 0; i < numlayers; i++) {
					row = &j2k_parameters->cp_matrice[i * matrix_width];
					col = row;
					j2k_parameters->tcp_rates[i] = 1;
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
				j2k_parameters->cp_fixed_alloc = 1;
			}
			break;
      /* ----------------------------------------------------- */
    case 't':			/* tiles */
      sscanf(optarg, "%d,%d", &j2k_parameters->cp_tdx, &j2k_parameters->cp_tdy);
			j2k_parameters->tile_size_on = true;
      break;
      /* ----------------------------------------------------- */
    case 'n':			/* resolution */
      sscanf(optarg, "%d", &j2k_parameters->numresolution);
      break;
      /* ----------------------------------------------------- */
    case 'c':			/* precinct dimension */
			{
				char sep;
				int res_spec = 0;

				char *s = optarg;
				do {
					sep = 0;
					sscanf(s, "[%d,%d]%c", &j2k_parameters->prcw_init[res_spec],
                                 &j2k_parameters->prch_init[res_spec], &sep);
					j2k_parameters->csty |= 0x01;
					res_spec++;
					s = strpbrk(s, "]") + 2;
				}
				while (sep == ',');
				j2k_parameters->res_spec = res_spec;
			}
			break;

      /* ----------------------------------------------------- */
    case 'b':			/* code-block dimension */
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
				j2k_parameters->cblockw_init = cblockw_init;
				j2k_parameters->cblockh_init = cblockh_init;
			}
			break;
      /* ----------------------------------------------------- */
    case 'p':			/* progression order */
			{
				char progression[4];
				
				strncpy(progression, optarg, 4);
				j2k_parameters->prog_order = give_progression(progression);
				if (j2k_parameters->prog_order == -1) {
					fprintf(stderr, "Unrecognized progression order "
            "[LRCP, RLCP, RPCL, PCRL, CPRL] !!\n");
					return 1;
				}
			}
			break;
      /* ----------------------------------------------------- */
    case 's':			/* subsampling factor */
      {
				if (sscanf(optarg, "%d,%d", &j2k_parameters->subsampling_dx,
                                    &j2k_parameters->subsampling_dy) != 2) {
					fprintf(stderr,	"'-s' sub-sampling argument error !  [-s dx,dy]\n");
					return 1;
				}
			}
			break;
      /* ----------------------------------------------------- */
    case 'd':			/* coordonnate of the reference grid */
      {
				if (sscanf(optarg, "%d,%d", &j2k_parameters->image_offset_x0,
                                    &j2k_parameters->image_offset_y0) != 2) {
					fprintf(stderr,	"-d 'coordonnate of the reference grid' argument "
            "error !! [-d x0,y0]\n");
					return 1;
				}
			}
			break;
      /* ----------------------------------------------------- */
    case 'h':			/* Display an help description */
      help_display();
      return 0;
      break;
      /* ----------------------------------------------------- */
    case 'P':			/* POC */
      {
				int numpocs = 0;		/* number of progression order change (POC) default 0 */
				opj_poc_t *POC = NULL;	/* POC : used in case of Progression order change */

				char *s = optarg;
				POC = j2k_parameters->POC;

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
				j2k_parameters->numpocs = numpocs;
			}
			break;
      /* ------------------------------------------------------ */
    case 'S':			/* SOP marker */
      j2k_parameters->csty |= 0x02;
      break;
      /* ------------------------------------------------------ */
    case 'E':			/* EPH marker */
      j2k_parameters->csty |= 0x04;
      break;
      /* ------------------------------------------------------ */
    case 'M':			/* Mode switch pas tous au point !! */
      if (sscanf(optarg, "%d", &value) == 1) {
				for (i = 0; i <= 5; i++) {
					int cache = value & (1 << i);
					if (cache)
						j2k_parameters->mode |= (1 << i);
				}
      }
      break;
      /* ------------------------------------------------------ */
    case 'R':			/* ROI */
      {
				if (sscanf(optarg, "OI:c=%d,U=%d", &j2k_parameters->roi_compno,
                                           &j2k_parameters->roi_shift) != 2) {
					fprintf(stderr, "ROI error !! [-ROI:c='compno',U='shift']\n");
					return 1;
				}
			}
			break;
      /* ------------------------------------------------------ */
    case 'T':			/* Tile offset */
			{
				if (sscanf(optarg, "%d,%d", &j2k_parameters->cp_tx0, &j2k_parameters->cp_ty0) != 2) {
					fprintf(stderr, "-T 'tile offset' argument error !! [-T X0,Y0]");
					return 1;
				}
			}
			break;
      /* ------------------------------------------------------ */
    case 'C':			/* Add a comment */
			{
				j2k_parameters->cp_comment = (char*)malloc(strlen(optarg) + 1);
				if(j2k_parameters->cp_comment) {
					strcpy(j2k_parameters->cp_comment, optarg);
				}
			}
			break;
      /* ------------------------------------------------------ */
    case 'I':			/* reversible or not */
			{
				j2k_parameters->irreversible = 1;
			}
			break;
      /* ------------------------------------------------------ */
    case 'W':			/* Width and Height and Cb and Cr subsampling in case of YUV format files */
      if (sscanf
				(optarg, "%d,%d,%d,%d", &mj2_parameters.w, &mj2_parameters.h, &mj2_parameters.CbCr_subsampling_dx,
				&mj2_parameters.CbCr_subsampling_dy) != 4) {
				fprintf(stderr, "-W argument error");
				return 1;
      }
      break;
      /* ------------------------------------------------------ */
    case 'F':			/* Video frame rate */
      if (sscanf(optarg, "%d", &mj2_parameters.frame_rate) != 1) {
				fprintf(stderr, "-F argument error");
				return 1;
      }
      break;
      /* ------------------------------------------------------ */
    default:
      return 1;
    }
  }
    
  /* Error messages */
  /* -------------- */
	if (!mj2_parameters.cod_format || !mj2_parameters.decod_format) {
    fprintf(stderr,
      "Usage: %s -i yuv-file -o mj2-file (+ options)\n",argv[0]);
    return 1;
  }
  
	if ((j2k_parameters->cp_disto_alloc || j2k_parameters->cp_fixed_alloc || j2k_parameters->cp_fixed_quality)
		&& (!(j2k_parameters->cp_disto_alloc ^ j2k_parameters->cp_fixed_alloc ^ j2k_parameters->cp_fixed_quality))) {
		fprintf(stderr, "Error: options -r -q and -f cannot be used together !!\n");
		return 1;
	}				/* mod fixed_quality */

	/* if no rate entered, lossless by default */
	if (j2k_parameters->tcp_numlayers == 0) {
		j2k_parameters->tcp_rates[0] = 0;	/* MOD antonin : losslessbug */
		j2k_parameters->tcp_numlayers++;
		j2k_parameters->cp_disto_alloc = 1;
	}

	if((j2k_parameters->cp_tx0 > j2k_parameters->image_offset_x0) || (j2k_parameters->cp_ty0 > j2k_parameters->image_offset_y0)) {
		fprintf(stderr,
			"Error: Tile offset dimension is unnappropriate --> TX0(%d)<=IMG_X0(%d) TYO(%d)<=IMG_Y0(%d) \n",
			j2k_parameters->cp_tx0, j2k_parameters->image_offset_x0, j2k_parameters->cp_ty0, j2k_parameters->image_offset_y0);
		return 1;
	}

	for (i = 0; i < j2k_parameters->numpocs; i++) {
		if (j2k_parameters->POC[i].prg == -1) {
			fprintf(stderr,
				"Unrecognized progression order in option -P (POC n %d) [LRCP, RLCP, RPCL, PCRL, CPRL] !!\n",
				i + 1);
		}
	}
  
  if (j2k_parameters->cp_tdx > mj2_parameters.Dim[0] || j2k_parameters->cp_tdy > mj2_parameters.Dim[1]) {
    fprintf(stderr,
      "Error: Tile offset dimension is unnappropriate --> TX0(%d)<=IMG_X0(%d) TYO(%d)<=IMG_Y0(%d) \n",
      j2k_parameters->cp_tdx, mj2_parameters.Dim[0], j2k_parameters->cp_tdy, mj2_parameters.Dim[1]);
    return 1;
  }
    
  /* to respect profile - 0 */
  /* ---------------------- */
  
  x1 = !mj2_parameters.Dim[0] ? (mj2_parameters.w - 1) * j2k_parameters->subsampling_dx 
		+ 1 : mj2_parameters.Dim[0] + (mj2_parameters.w - 1) * j2k_parameters->subsampling_dx + 1;
  y1 = !mj2_parameters.Dim[1] ? (mj2_parameters.h - 1) * j2k_parameters->subsampling_dy 
		+ 1 : mj2_parameters.Dim[1] + (mj2_parameters.h - 1) * j2k_parameters->subsampling_dy + 1;   
  mj2_parameters.numcomps = 3;			/* Because YUV files only have 3 components */ 
  mj2_parameters.prec = 8;			/* Because in YUV files, components have 8-bit depth */

	j2k_parameters->tcp_mct = 0;
    
  mj2file = fopen(mj2_parameters.outfile, "wb");
  
  if (!mj2file) {
    fprintf(stderr, "failed to open %s for writing\n", argv[2]);
    return 1;
  }
    
	/* get a MJ2 decompressor handle */
	cinfo = mj2_create_compress();
	movie = (opj_mj2_t*)cinfo->mj2_handle;
	
	/* catch events using our callbacks and give a local context */
	opj_set_event_mgr((opj_common_ptr)cinfo, &event_mgr, stderr);

	/* setup encoder parameters */
	mj2_setup_encoder(movie, &mj2_parameters);   
  
  movie->tk[0].num_samples = yuv_num_frames(&movie->tk[0],mj2_parameters.infile); 
  if (movie->tk[0].num_samples == -1) {
		return 1;
  }
  
  // One sample per chunk
  movie->tk[0].chunk = (mj2_chunk_t*) malloc(movie->tk[0].num_samples * sizeof(mj2_chunk_t));     
  movie->tk[0].sample = (mj2_sample_t*) malloc(movie->tk[0].num_samples * sizeof(mj2_sample_t));
  
  if (mj2_init_stdmovie(movie)) {
    fprintf(stderr, "Error with movie initialization");
    return 1;
  };    
  
  // Writing JP, FTYP and MDAT boxes 
  buf = (unsigned char*) malloc (300 * sizeof(unsigned char)); // Assuming that the JP and FTYP
  // boxes won't be longer than 300 bytes
	cio = opj_cio_open((opj_common_ptr)movie->cinfo, buf, 300);
  mj2_write_jp(cio);
  mj2_write_ftyp(movie, cio);
  mdat_initpos = cio_tell(cio);
  cio_skip(cio, 4);
  cio_write(cio, MJ2_MDAT, 4);	
  fwrite(buf,cio_tell(cio),1,mj2file);
  offset = cio_tell(cio);
  opj_cio_close(cio);
  free(buf);

  for (i = 0; i < movie->num_stk + movie->num_htk + movie->num_vtk; i++) {
    if (movie->tk[i].track_type != 0) {
      fprintf(stderr, "Unable to write sound or hint tracks\n");
    } else {
      mj2_tk_t *tk;
			int buflen = 0;
      
      tk = &movie->tk[i];     
      tk->num_chunks = tk->num_samples;
			numframes = tk->num_samples;

      fprintf(stderr, "Video Track number %d\n", i + 1);
			
			img = mj2_image_create(tk, j2k_parameters);          
			buflen = 2 * (tk->w * tk->h * 8);
			buf = (unsigned char *) malloc(buflen*sizeof(unsigned char));	

      for (sampleno = 0; sampleno < numframes; sampleno++) {		
				double init_time = opj_clock();
				double elapsed_time;
				if (yuvtoimage(tk, img, sampleno, j2k_parameters, mj2_parameters.infile)) {
					fprintf(stderr, "Error with frame number %d in YUV file\n", sampleno);
					return 1;
				}

				/* setup the encoder parameters using the current image and user parameters */
				opj_setup_encoder(cinfo, j2k_parameters, img);

				cio = opj_cio_open((opj_common_ptr)movie->cinfo, buf, buflen);
								
				cio_skip(cio, 4);
				cio_write(cio, JP2_JP2C, 4);	// JP2C

				/* encode the image */
				bSuccess = opj_encode(cinfo, cio, img, NULL);
				if (!bSuccess) {
					opj_cio_close(cio);
					fprintf(stderr, "failed to encode image\n");
					return 1;
				}

				len = cio_tell(cio) - 8;
				cio_seek(cio, 0);
				cio_write(cio, len+8,4);
				opj_cio_close(cio);
				tk->sample[sampleno].sample_size = len+8;				
				tk->sample[sampleno].offset = offset;
				tk->chunk[sampleno].offset = offset;	// There is one sample per chunk 
				fwrite(buf, 1, len+8, mj2file);				
				offset += len+8;				
				elapsed_time = opj_clock()-init_time;
				fprintf(stderr, "Frame number %d/%d encoded in %.2f mseconds\n", sampleno + 1, numframes, elapsed_time*1000);
				total_time += elapsed_time;

      }
			/* free buffer data */
			free(buf);
			/* free image data */
			opj_image_destroy(img);
    }
  }
  
  fseek(mj2file, mdat_initpos, SEEK_SET);
	
  buf = (unsigned char*) malloc(4*sizeof(unsigned char));

	// Init a cio to write box length variable in a little endian way 
	cio = opj_cio_open(NULL, buf, 4);
  cio_write(cio, offset - mdat_initpos, 4);
  fwrite(buf, 4, 1, mj2file);
  fseek(mj2file,0,SEEK_END);
  free(buf);

  // Writing MOOV box 
	buf = (unsigned char*) malloc ((TEMP_BUF+numframes*20) * sizeof(unsigned char));
	cio = opj_cio_open(movie->cinfo, buf, (TEMP_BUF+numframes*20));
	mj2_write_moov(movie, cio);
  fwrite(buf,cio_tell(cio),1,mj2file);
  free(buf);

	fprintf(stdout,"Total encoding time: %.2f s for %d frames (%.1f fps)\n", total_time, numframes, (float)numframes/total_time);
  
  // Ending program 
  
  fclose(mj2file);
	/* free remaining compression structures */
	mj2_destroy_compress(movie);
	free(cinfo);
	/* free user parameters structure */
  if(j2k_parameters->cp_comment) free(j2k_parameters->cp_comment);
	if(j2k_parameters->cp_matrice) free(j2k_parameters->cp_matrice);
	opj_cio_close(cio);

  return 0;
}


