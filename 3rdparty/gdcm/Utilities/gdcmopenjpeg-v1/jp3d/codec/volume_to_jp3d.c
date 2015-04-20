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

void encode_help_display() {
	fprintf(stdout,"List of parameters for the JPEG2000 Part 10 encoder:\n");
	fprintf(stdout,"------------\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"Required Parameters (except with -h):\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-i           : source file  (-i source.bin or source*.pgx) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-m           : source characteristics file (-m imgfile.img) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-o           : destination file (-o dest.jp3d) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"Optional Parameters:\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-h           : display the help information \n ");
	fprintf(stdout,"\n");
	fprintf(stdout,"-n           : number of resolutions (-n 3,3,3) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-I           : use the irreversible transforms: ICT + DWT 9-7 (-I) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-C           : coding algorithm (-C 2EB) [2EB, 3EB] \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-r           : different compression ratios for successive layers (-r 20,10,5)\n ");
	fprintf(stdout,"	         - The rate specified for each quality level is the desired compression factor.\n");
	fprintf(stdout,"	         - Rate 1 means lossless compression\n");
	fprintf(stdout,"               (options -r and -q cannot be used together)\n ");
	fprintf(stdout,"\n");
	fprintf(stdout,"-q           : different psnr for successive layers (-q 30,40,50) \n ");
	fprintf(stdout,"               (options -r and -q cannot be used together)\n ");
	fprintf(stdout,"\n");
	fprintf(stdout,"-b           : size of code block (-b 32,32,32) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-c           : size of precinct (-c 128,128,128) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-t           : size of tile (-t 512,512,512) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-p           : progression order (-p LRCP) [LRCP, RLCP, RPCL, PCRL, CPRL] \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-s           : subsampling factor (-s 2,2,2) [-s X,Y,Z] \n");
	fprintf(stdout,"			  - Remark: subsampling bigger than 2 can produce error\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-SOP         : write SOP marker before each packet \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-EPH         : write EPH marker after each header packet \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-M           : code-block style (-M 0) [1=BYPASS(LAZY) 2=RESET 4=RESTART(TERMALL)\n");
	fprintf(stdout,"                 8=VSC 16=PTERM 32=SEGSYM 64=3DCTXT] \n");
	fprintf(stdout,"                 Indicate multiple modes by adding their values. \n");
	fprintf(stdout,"                 ex: RESTART(4) + RESET(2) + SEGMARK(32) = -M 38\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-D           : define DC offset (-D 12) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-x           : create an index file *.Idx (-x index_name.Idx) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-ROI         : c=%%d,U=%%d : quantization indices upshifted \n");
	fprintf(stdout,"               for component c=%%d [%%d = 0,1,2]\n");
	fprintf(stdout,"               with a value of U=%%d [0 <= %%d <= 37] (i.e. -ROI:c=0,U=25) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-d           : offset of the origin of the volume (-d 150,300,100) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"-l           : offset of the origin of the tiles (-l 100,75,25) \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"DEFAULT CODING:\n");
	fprintf(stdout,"------------\n");
	fprintf(stdout,"\n");
	fprintf(stdout," * Lossless\n");
	fprintf(stdout," * 1 tile\n");
	fprintf(stdout," * Size of precinct : 2^15 x 2^15 x 2^15 (means 1 precinct)\n");
	fprintf(stdout," * Size of code-block : 64 x 64 x 64\n");
	fprintf(stdout," * Number of resolutions in x, y and z axis: 3\n");
	fprintf(stdout," * No SOP marker in the codestream\n");
	fprintf(stdout," * No EPH marker in the codestream\n");
	fprintf(stdout," * No sub-sampling in x, y or z direction\n");
	fprintf(stdout," * No mode switch activated\n");
	fprintf(stdout," * Progression order: LRCP\n");
	fprintf(stdout," * No index file\n");
	fprintf(stdout," * No ROI upshifted\n");
	fprintf(stdout," * No offset of the origin of the volume\n");
	fprintf(stdout," * No offset of the origin of the tiles\n");
	fprintf(stdout," * Reversible DWT 5-3 on each 2D slice\n");
	fprintf(stdout," * Coding algorithm: 2D-EBCOT \n");
	fprintf(stdout,"\n");
	fprintf(stdout,"REMARKS:\n");
	fprintf(stdout,"---------\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"- The markers written to the main_header are : SOC SIZ COD QCD COM.\n");
	fprintf(stdout,"- COD and QCD markers will never appear in the tile_header.\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"- You need enough disk space memory (twice the original) to encode \n");
	fprintf(stdout,"the volume,i.e. for a 1.5 GB volume you need a minimum of 3GB of disk memory)\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"- When loading *.pgx files, a relative path to directory is needed for input argument \n");
	fprintf(stdout," followed by the common prefix of the slices and a '*' character representing sequential numeration.\n");
	fprintf(stdout,"( -i relativepath/slices*.pgx )\n");
	fprintf(stdout,"\n");
	fprintf(stdout," - The index file has the structure below:\n");
	fprintf(stdout,"\n");
	fprintf(stdout,"\t	Image_height Image_width Image_depth\n");
	fprintf(stdout,"\t	Progression order: 0 (LRCP)\n");
	fprintf(stdout,"\t	Tiles_size_X Tiles_size_Y Tiles_size_Z\n");
	fprintf(stdout,"\t	Components_nb\n");
	fprintf(stdout,"\t	Layers_nb\n");
	fprintf(stdout,"\t	Decomposition_levels\n");
	fprintf(stdout,"\t	[Precincts_size_X_res_Nr Precincts_size_Y_res_Nr Precincts_size_Z_res_Nr]\n\t  ...\n");
	fprintf(stdout,"\t	[Precincts_size_X_res_0 Precincts_size_Y_res_0 Precincts_size_Z_res_0]\n");
	fprintf(stdout,"\t	Main_header_end_position\n");
	fprintf(stdout,"\t	Codestream_size\n");
	fprintf(stdout,"\t	Tile_0 [start_pos end_header end_pos TotalDisto NumPix MaxMSE]\n");
	fprintf(stdout,"\t	...\n");
	fprintf(stdout,"\t	Tile_Nt [  ''         ''        ''        ''       ''    ''  ]\n");
	fprintf(stdout,"\t  Tpacket_0 [Tile layer res. comp. prec. start_pos end_pos disto]\n");
	fprintf(stdout,"\t  ...\n");
	fprintf(stdout,"\t  Tpacket_Np [''   ''    ''   ''    ''       ''       ''     '' ]\n");
	fprintf(stdout,"\t  MaxDisto\n");
	fprintf(stdout,"\t  TotalDisto\n\n");
	fprintf(stdout,"\n");

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

OPJ_TRANSFORM give_transform(char transform[4]) {
	if(strncmp(transform, "2DWT", 4) == 0) {
		return TRF_2D_DWT;
	}
	if(strncmp(transform, "3DWT", 4) == 0) {
		return TRF_3D_DWT;
	}
	return TRF_UNKNOWN;
}

OPJ_ENTROPY_CODING give_coding(char coding[3]) {

	if(strncmp(coding, "2EB", 3) == 0) {
		return ENCOD_2EB;
	}
	if(strncmp(coding, "3EB", 3) == 0) {
		return ENCOD_3EB;
	}
	/*if(strncmp(coding, "2GR", 3) == 0) {
		return ENCOD_2GR;
	}
	if(strncmp(coding, "3GR", 3) == 0) {
		return ENCOD_3GR;
	}*/

	return ENCOD_UNKNOWN;
}

int get_file_format(char *filename) {
	int i;
	static const char *extension[] = {"pgx", "bin", "img", "j3d", "jp3d", "j2k"};
	static const int format[] = { PGX_DFMT, BIN_DFMT, IMG_DFMT, J3D_CFMT, J3D_CFMT, J2K_CFMT};
	char * ext = strrchr(filename, '.');
	if (ext) {
		ext++;
        for(i = 0; i < sizeof(format)/sizeof(*format); i++) {
			if(strnicmp(ext, extension[i], 3) == 0) {
                return format[i];
			}
		}
	}

	return -1;
}

/* ------------------------------------------------------------------------------------ */

int parse_cmdline_encoder(int argc, char **argv, opj_cparameters_t *parameters) {
	int i, value;

	/* parse the command line */

	while (1) {
		int c = getopt(argc, argv, "i:m:o:r:q:f:t:n:c:b:x:p:s:d:hP:S:E:M:D:R:l:T:C:A:I");
		if (c == -1)
			break;
		switch (c) {
			case 'i':			/* input file */
			{
				char *infile = optarg;
				parameters->decod_format = get_file_format(infile);
				switch(parameters->decod_format) {
					case PGX_DFMT:
					case BIN_DFMT:
					case IMG_DFMT:
						break;
					default:
						fprintf(stdout,	"[ERROR] Unrecognized format for infile : %s [accept only *.pgx or *.bin] !!\n\n", infile);
						return 1;
						break;
				}
				strncpy(parameters->infile, infile, MAX_PATH);
				fprintf(stdout,	"[INFO] Infile: %s \n", parameters->infile);

			}
			break;
				
				/* ----------------------------------------------------- */
			case 'm':			/* input IMG file */
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
					case J3D_CFMT:
					case J2K_CFMT:
					case LSE_CFMT:
						break;
					default:
						fprintf(stdout, "[ERROR] Unknown output format volume %s [only *.j2k, *.lse3d or *.jp3d]!! \n", outfile);
						return 1;
						break;
				}
				strncpy(parameters->outfile, outfile, MAX_PATH);
				fprintf(stdout,	"[INFO] Outfile: %s \n", parameters->outfile);
			}
			break;

				/* ----------------------------------------------------- */
			
			case 'r':			/* define compression rates for each layer */
			{
				char *s = optarg;
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
			
			case 'q':			/* define distorsion (PSNR) for each layer */
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
				
				/* ----------------------------------------------------- */

			case 'f':	
			{
				fprintf(stdout, "/---------------------------------------------------\\\n");
				fprintf(stdout, "|  Fixed layer allocation option not implemented !!  |\n");
				fprintf(stdout, "\\---------------------------------------------------/\n");
				/*int *row = NULL, *col = NULL;
				int numlayers = 0, matrix_width = 0;

				char *s = optarg;
				sscanf(s, "%d", &numlayers);
				s++;
				if (numlayers > 9)
					s++;

				parameters->tcp_numlayers = numlayers;
				matrix_width = parameters->numresolution[0] + parameters->numresolution[1] + parameters->numresolution[2];
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
					for (j = 1; j < matrix_width; j++) {
						col += 3; j+=2;
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
				parameters->cp_fixed_alloc = 1; */
			}
			break;
				
				/* ----------------------------------------------------- */

			case 't':			/* tiles */
			{
				if (sscanf(optarg, "%d,%d,%d", &parameters->cp_tdx, &parameters->cp_tdy, &parameters->cp_tdz) !=3) {
					fprintf(stdout,	"[ERROR] '-t' 'dimensions of tiles' argument error !  [-t tdx,tdy,tdz]\n");
					return 1;
				}
				parameters->tile_size_on = true;
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'n':			/* resolution */
			{
				int aux;
				aux = sscanf(optarg, "%d,%d,%d", &parameters->numresolution[0], &parameters->numresolution[1], &parameters->numresolution[2]);
				if (aux == 2) 
					parameters->numresolution[2] = 1;
				else if (aux == 1) {
					parameters->numresolution[1] = parameters->numresolution[0];
					parameters->numresolution[2] = 1;
				}else if (aux == 0){
					parameters->numresolution[0] = 1;
					parameters->numresolution[1] = 1;
					parameters->numresolution[2] = 1;
				}
			}
			break;
				
				/* ----------------------------------------------------- */
			case 'c':			/* precinct dimension */
			{
				char sep;
				int res_spec = 0;
				int aux;
				char *s = optarg;
				do {
					sep = 0;
					aux = sscanf(s, "[%d,%d,%d]%c", &parameters->prct_init[0][res_spec], &parameters->prct_init[1][res_spec], &parameters->prct_init[2][res_spec], &sep);
					if (sep == ',' && aux != 4) {
						fprintf(stdout,	"[ERROR] '-c' 'dimensions of precincts' argument error !  [-c [prcx_res0,prcy_res0,prcz_res0],...,[prcx_resN,prcy_resN,prcz_resN]]\n");
						return 1;
					}
					parameters->csty |= 0x01;
					res_spec++;
					s = strpbrk(s, "]") + 2;
				}
				while (sep == ',');
				parameters->res_spec = res_spec; /* number of precinct size specifications */
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'b':			/* code-block dimension */
			{
				int cblockw_init = 0, cblockh_init = 0, cblockl_init = 0;
				if (sscanf(optarg, "%d,%d,%d", &cblockw_init, &cblockh_init, &cblockl_init) != 3) {
					fprintf(stdout,	"[ERROR] '-b' 'dimensions of codeblocks' argument error !  [-b cblkx,cblky,cblkz]\n");
					return 1;
				}
				if (cblockw_init * cblockh_init * cblockl_init > (1<<18) || cblockw_init > 1024 || cblockw_init < 4 || cblockh_init > 1024 || cblockh_init < 4 || cblockl_init > 1024 || cblockl_init < 4) {
					fprintf(stdout,"[ERROR] Size of code_block error (option -b) !!\n\nRestriction :\n * width*height*length<=4096\n * 4<=width,height,length<= 1024\n\n");
					return 1;
				}
				parameters->cblock_init[0] = cblockw_init;
				parameters->cblock_init[1] = cblockh_init;
				parameters->cblock_init[2] = cblockl_init;
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'x':			/* creation of index file */
			{
				char *index = optarg;
				strncpy(parameters->index, index, MAX_PATH);
				parameters->index_on = 1;
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'p':			/* progression order */
			{
				char progression[4];

				strncpy(progression, optarg, 4);
				parameters->prog_order = give_progression(progression);
				if (parameters->prog_order == -1) {
					fprintf(stdout, "[ERROR] Unrecognized progression order [LRCP, RLCP, RPCL, PCRL, CPRL] !!\n");
					return 1;
				}
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 's':			/* subsampling factor */
			{
				if (sscanf(optarg, "%d,%d,%d", &parameters->subsampling_dx, &parameters->subsampling_dy, &parameters->subsampling_dz) != 2) {
					fprintf(stdout,	"[ERROR] '-s' sub-sampling argument error !  [-s dx,dy,dz]\n");
					return 1;
				}
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'd':			/* coordonnate of the reference grid */
			{
				if (sscanf(optarg, "%d,%d,%d", &parameters->volume_offset_x0, &parameters->volume_offset_y0, &parameters->volume_offset_z0) != 3) {
					fprintf(stdout,	"[ERROR] -d 'coordonnate of the reference grid' argument error !! [-d x0,y0,z0]\n");
					return 1;
				}
			}
			break;
				
				/* ----------------------------------------------------- */
			
			case 'h':			/* display an help description */
			{
				encode_help_display();
				return 1;
			}
			break;
				
				/* ----------------------------------------------------- */

			case 'P':			/* POC */
			{
				int numpocs = 0;		/* number of progression order change (POC) default 0 */
				opj_poc_t *POC = NULL;	/* POC : used in case of Progression order change */

				char *s = optarg;
				POC = parameters->POC;

				fprintf(stdout, "/----------------------------------\\\n");
				fprintf(stdout, "|  POC option not fully tested !!  |\n");
				fprintf(stdout, "\\----------------------------------/\n");
				
				while (sscanf(s, "T%d=%d,%d,%d,%d,%d,%s", &POC[numpocs].tile,
														&POC[numpocs].resno0, &POC[numpocs].compno0,
														&POC[numpocs].layno1, &POC[numpocs].resno1,
														&POC[numpocs].compno1, POC[numpocs].progorder) == 7) {
					POC[numpocs].prg = give_progression(POC[numpocs].progorder);
					/* POC[numpocs].tile; */
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
				
			case 'S':			/* SOP marker */
			{
				parameters->csty |= 0x02;
			}
			break;
				
				/* ------------------------------------------------------ */
			
			case 'E':			/* EPH marker */
			{
				parameters->csty |= 0x04;
			}
			break;
				
				/* ------------------------------------------------------ */
			
			case 'M':			/* Codification mode switch */
			{
				fprintf(stdout, "[INFO] Mode switch option not fully tested !!\n");
				value = 0;
				if (sscanf(optarg, "%d", &value) == 1) {
					for (i = 0; i <= 6; i++) {
						int cache = value & (1 << i);
						if (cache)
							parameters->mode |= (1 << i);
					}
				}
			}
			break;
				
				/* ------------------------------------------------------ */
			
			case 'D':			/* DCO */
			{
				if (sscanf(optarg, "%d", &parameters->dcoffset) != 1) {
					fprintf(stdout, "[ERROR] DC offset error !! [-D %d]\n",parameters->dcoffset);
					return 1;
				}
			}
			break;
				
				/* ------------------------------------------------------ */
			
			case 'R':			/* ROI */
			{
				if (sscanf(optarg, "OI:c=%d,U=%d", &parameters->roi_compno, &parameters->roi_shift) != 2) {
					fprintf(stdout, "[ERROR] ROI error !! [-ROI:c='compno',U='shift']\n");
					return 1;
				}
			}
			break;
				
				/* ------------------------------------------------------ */
			
			case 'l':			/* Tile offset */
			{
				if (sscanf(optarg, "%d,%d,%d", &parameters->cp_tx0, &parameters->cp_ty0, &parameters->cp_tz0) != 3) {
					fprintf(stdout, "[ERROR] -l 'tile offset' argument error !! [-l X0,Y0,Z0]");
					return 1;
				}
			}
			break;
				
				/* ------------------------------------------------------ 
				
			case 'T':			// Tranformation of original data (2D-DWT/3D-DWT/3D-RLS/2D-DWT+1D-RLS) 
			{
				char transform[4];

				strncpy(transform, optarg, 4);
				parameters->transform_format = give_transform(transform);
				if (parameters->transform_format == -1) {
					fprintf(stdout, "[ERROR] -T 'Transform domain' argument error !! [-T 2DWT, 3DWT, 3RLS or 3LSE only]");
                    return 1;
				}
			}
			break;
				
				 ------------------------------------------------------ */
			
			case 'C':			/* Coding of transformed data */
			{
				char coding[3];

				strncpy(coding, optarg, 3);
				parameters->encoding_format = give_coding(coding);
				if (parameters->encoding_format == -1) {
					fprintf(stdout, "[ERROR] -C 'Coding algorithm' argument error !! [-C 2EB, 3EB, 2GR, 3GR or GRI only]");
                    return 1;
				}
			}
			break;
			
			/* ------------------------------------------------------ */
			
			case 'I':			/* reversible or not */
			{
				parameters->irreversible = 1;
			}
			break;
				
			default:
				fprintf(stdout, "[ERROR] This option is not valid \"-%c %s\"\n", c, optarg);
				return 1;
		}
	}

	/* check for possible errors */

	if((parameters->infile[0] == 0) || (parameters->outfile[0] == 0)) {
		fprintf(stdout, "usage: jp3d_vm_enc -i volume-file -o jp3d-file (+ options)\n");
		return 1;
	}

	if((parameters->decod_format == BIN_DFMT) && (parameters->imgfile[0] == 0)) {
		fprintf(stdout, "usage: jp3d_vm_enc -i bin-volume-file -m img-file -o jp3d-file (+ options)\n");
		return 1;
	}

	if((parameters->decod_format != BIN_DFMT) && (parameters->decod_format != PGX_DFMT) && (parameters->decod_format != IMG_DFMT)) {
		fprintf(stdout, "usage: jp3d_vm_enc -i input-volume-file [*.bin,*.pgx,*.img] -o jp3d-file [*.jp3d,*.j2k] (+ options)\n");
		return 1;
	}
	if((parameters->cod_format != J3D_CFMT) && (parameters->cod_format != J2K_CFMT)) {
		fprintf(stdout, "usage: jp3d_vm_enc -i input-volume-file [*.bin,*.pgx,*.img] -o jp3d-file [*.jp3d,*.j2k] (+ options)\n");
		return 1;
	}

	if((parameters->encoding_format == ENCOD_2GR || parameters->encoding_format == ENCOD_3GR) && parameters->transform_format != TRF_3D_LSE && parameters->transform_format != TRF_3D_RLS) {
		fprintf(stdout, "[ERROR] Entropy coding options -C [2GR,3GR] are only compatible with predictive-based transform algorithms: -T [3RLS,3LSE].\n");
		return 1;
	}
	if (parameters->encoding_format == ENCOD_3EB)
		parameters->mode |= (1 << 6);

	if ((parameters->mode >> 6) & 1) {
		parameters->encoding_format = ENCOD_3EB;
	}

	if((parameters->numresolution[2] == 0 || (parameters->numresolution[1] == 0) || (parameters->numresolution[0] == 0))) {
		fprintf(stdout, "[ERROR] -n 'resolution levels' argument error ! Resolutions must be greater than 1 in order to perform DWT.\n");
		return 1;
	}
	if (parameters->numresolution[1] != parameters->numresolution[0]) {
		fprintf(stdout, "[ERROR] -n 'resolution levels' argument error ! Resolutions in X and Y axis must be the same in this implementation.\n");
		return 1;
	}
	
	if (parameters->numresolution[2] > parameters->numresolution[0]) {
		fprintf(stdout, "[ERROR] -n 'resolution levels' argument error ! Resolutions in Z axis must be lower than in X-Y axis.\n");
		return 1;
	}
	
	if (parameters->dcoffset >= 128 && parameters->dcoffset <= -128) {
		fprintf(stdout, "[ERROR] -D 'DC offset' argument error ! Value must be -128<=DCO<=128.\n");
		return 1;
	}

	if(parameters->numresolution[2] != 1) {
		parameters->transform_format = TRF_3D_DWT;
		//fprintf(stdout, "[Warning] Resolution level in axial dim > 1 : 3D-DWT will be performed... \n");
	} else if (parameters->numresolution[2] == 1) {
		parameters->transform_format = TRF_2D_DWT;
		//fprintf(stdout, "[Warning] Resolution level in axial dim == 1 : 2D-DWT will be performed... \n");
	}
	
	if ((parameters->cod_format == J2K_CFMT) && (parameters->transform_format != TRF_2D_DWT || parameters->encoding_format != ENCOD_2EB)) {
		fprintf(stdout, "[WARNING] Incompatible options -o *.j2k and defined transform or encoding algorithm. Latter will be ignored\n");
		parameters->transform_format = TRF_2D_DWT;
		parameters->encoding_format = ENCOD_2EB;
	}

	if ((parameters->cp_disto_alloc || parameters->cp_fixed_alloc || parameters->cp_fixed_quality) && (!(parameters->cp_disto_alloc ^ parameters->cp_fixed_quality))) {
		fprintf(stdout, "[ERROR] Options -r and -q cannot be used together !!\n");
		return 1;
	}				/* mod fixed_quality */

	/* if no rate entered, lossless by default */
	if (parameters->tcp_numlayers == 0) {
		parameters->tcp_rates[0] = 0.0;	/* MOD antonin : losslessbug */
		parameters->tcp_numlayers++;
		parameters->cp_disto_alloc = 1;
	}

	if((parameters->cp_tx0 > parameters->volume_offset_x0) || (parameters->cp_ty0 > parameters->volume_offset_y0) || (parameters->cp_tz0 > parameters->volume_offset_z0)) {
		fprintf(stdout,	"[ERROR] Tile offset dimension is unnappropriate --> TX0(%d)<=IMG_X0(%d) TYO(%d)<=IMG_Y0(%d) TZO(%d)<=IMG_Z0(%d)\n",
			parameters->cp_tx0, parameters->volume_offset_x0, parameters->cp_ty0, parameters->volume_offset_y0, 
			parameters->cp_tz0, parameters->volume_offset_z0);
		return 1;
	}

	for (i = 0; i < parameters->numpocs; i++) {
		if (parameters->POC[i].prg == -1) {
			fprintf(stdout,"[ERROR] Unrecognized progression order in option -P (POC n %d) [LRCP, RLCP, RPCL, PCRL, CPRL] !!\n",i + 1);
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
	bool delete_comment = true;
	opj_cparameters_t parameters;	/* compression parameters */
	opj_event_mgr_t event_mgr;		/* event manager */
	opj_volume_t *volume = NULL;

	/* 
	configure the event callbacks (not required)
	setting of each callback is optionnal 
	*/
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = error_callback;
	event_mgr.warning_handler = warning_callback;
	event_mgr.info_handler = info_callback;

	/* set encoding parameters to default values */
	opj_set_default_encoder_parameters(&parameters);

	/* parse input and get user encoding parameters */
	if(parse_cmdline_encoder(argc, argv, &parameters) == 1) {
		return 0;
	}

	if(parameters.cp_comment == NULL) {
		parameters.cp_comment = "Created by OpenJPEG version JP3D";
		/* no need to delete parameters.cp_comment on exit */
		delete_comment = false;
	}
	
	/* encode the destination volume */
	/* ---------------------------- */
	if (parameters.cod_format == J3D_CFMT || parameters.cod_format == J2K_CFMT)	{
		int codestream_length, pixels, bitsin;
		opj_cio_t *cio = NULL;
		FILE *f = NULL;
		opj_cinfo_t* cinfo = NULL;
		
		/* decode the source volume */
		/* ----------------------- */
		switch (parameters.decod_format) {
			case PGX_DFMT: 
				fprintf(stdout, "[INFO] Loading pgx file(s)\n");
				volume = pgxtovolume(parameters.infile, &parameters);
				if (!volume) {
					fprintf(stdout, "[ERROR] Unable to load pgx files\n");
					return 1;
				}
				break;
			
			case BIN_DFMT:
				fprintf(stdout, "[INFO] Loading bin file\n");
				volume = bintovolume(parameters.infile, parameters.imgfile, &parameters);
				if (!volume) {
					fprintf(stdout, "[ERROR] Unable to load bin file\n");
					return 1;
				}
				break;

			case IMG_DFMT:
				fprintf(stdout, "[INFO] Loading img file\n");
				volume = imgtovolume(parameters.infile, &parameters);
				if (!volume) {
					fprintf(stderr, "[ERROR] Unable to load img file\n");
					return 1;
				}
				break;
		}
		
		/* get a JP3D or J2K compressor handle */
		if (parameters.cod_format == J3D_CFMT) 
            cinfo = opj_create_compress(CODEC_J3D);
		else if (parameters.cod_format == J2K_CFMT) 
            cinfo = opj_create_compress(CODEC_J2K);

		/* catch events using our callbacks and give a local context */
		opj_set_event_mgr((opj_common_ptr)cinfo, &event_mgr, stdout);			

		/* setup the encoder parameters using the current volume and using user parameters */
		opj_setup_encoder(cinfo, &parameters, volume);
		
		/* open a byte stream for writing */
		/* allocate memory for all tiles */
		cio = opj_cio_open((opj_common_ptr)cinfo, NULL, 0);

		/* encode the volume */
		//fprintf(stdout, "[INFO] Encode the volume\n");
		bSuccess = opj_encode(cinfo, cio, volume, parameters.index);
		if (!bSuccess) {
			opj_cio_close(cio);
			fprintf(stdout, "[ERROR] Failed to encode volume\n");
			return 1;
		}
		codestream_length = cio_tell(cio);
		pixels =(volume->x1 - volume->x0) * (volume->y1 - volume->y0) * (volume->z1 - volume->z0);
		bitsin = pixels * volume->comps[0].prec;
		fprintf(stdout, "[RESULT] Volume: %d x %d x %d (x %d bpv)\n Codestream: %d B,  Ratio: %5.3f bpv,  (%5.3f : 1) \n", 
			(volume->x1 - volume->x0),(volume->y1 - volume->y0),(volume->z1 - volume->z0),volume->comps[0].prec,
			codestream_length, ((double)codestream_length * 8.0/(double)pixels), ((double)bitsin/(8.0*(double)codestream_length)));

		/* write the buffer to disk */
		f = fopen(parameters.outfile, "wb");
		if (!f) {
			fprintf(stdout, "[ERROR] Failed to open %s for writing\n", parameters.outfile);
			return 1;
		}
		fwrite(cio->buffer, 1, codestream_length, f);
		fclose(f);

		/* close and free the byte stream */
		opj_cio_close(cio);

		/* free remaining compression structures */
		opj_destroy_compress(cinfo);
	} else {
		fprintf(stdout, "[ERROR] Cod_format != JP3d !!! \n");
		return 1;
	}

	/* free user parameters structure */
	if(delete_comment) {
		if(parameters.cp_comment) free(parameters.cp_comment);
	}
	if(parameters.cp_matrice) free(parameters.cp_matrice);

	/* free volume data */
	opj_volume_destroy(volume);
	
	return 0;
}
