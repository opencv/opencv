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
#include <stdlib.h>
#include <string.h>
#include "../libjp3dvm/openjpeg3d.h"
#ifdef _WIN32
#include "windirent.h"
#else
#include <dirent.h>
#endif /* _WIN32 */



void dump_volume(FILE *fd, opj_volume_t * vol) {
	int compno;
	fprintf(fd, "volume {\n");
	fprintf(fd, "  x0=%d, y0=%d, z0=%d, x1=%d, y1=%d, z1=%d\n", vol->x0, vol->y0, vol->z0,vol->x1, vol->y1,  vol->z1);
	fprintf(fd, "  numcomps=%d\n", vol->numcomps);
	for (compno = 0; compno < vol->numcomps; compno++) {
		opj_volume_comp_t *comp = &vol->comps[compno];
		fprintf(fd, "  comp %d {\n", compno);
		fprintf(fd, "    dx=%d, dy=%d, dz=%d\n", comp->dx, comp->dy, comp->dz);
		fprintf(fd, "    prec=%d\n", comp->prec);
		fprintf(fd, "    sgnd=%d\n", comp->sgnd);
		fprintf(fd, "  }\n");
	}
	fprintf(fd, "}\n");
}

/*
 * Get logarithm of an integer and round downwards.
 *
 * log2(a)
 */
static int int_floorlog2(int a) {
	int l;
	for (l = 0; a > 1; l++) {
		a >>= 1;
	}
	return l;
}

/*
 * Divide an integer by a power of 2 and round upwards.
 *
 * a divided by 2^b
 */
static int int_ceildivpow2(int a, int b) {
	return (a + (1 << b) - 1) >> b;
}

/*
 * Divide an integer and round upwards.
 *
 * a divided by b
 */
static int int_ceildiv(int a, int b) {
	return (a + b - 1) / b;
}


/* -->> -->> -->> -->>

PGX IMAGE FORMAT

<<-- <<-- <<-- <<-- */


unsigned char readuchar(FILE * f)
{
  unsigned char c1;
  fread(&c1, 1, 1, f);
  return c1;
}

unsigned short readushort(FILE * f, int bigendian)
{
  unsigned char c1, c2;
  fread(&c1, 1, 1, f);
  fread(&c2, 1, 1, f);
  if (bigendian)
    return (c1 << 8) + c2;
  else
    return (c2 << 8) + c1;
}

unsigned int readuint(FILE * f, int bigendian)
{
  unsigned char c1, c2, c3, c4;
  fread(&c1, 1, 1, f);
  fread(&c2, 1, 1, f);
  fread(&c3, 1, 1, f);
  fread(&c4, 1, 1, f);
  if (bigendian)
    return (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
  else
    return (c4 << 24) + (c3 << 16) + (c2 << 8) + c1;
}
/*****************************************/
static unsigned short ShortSwap(unsigned short v)
{
  unsigned char c1, c2;
  c1 = v & 0xff;
  c2 = (v >> 8) & 0xff;
  return (c1 << 8) + c2;
}

static unsigned int LongSwap (unsigned int i)
{
  unsigned char b1, b2, b3, b4;
  b1 = i & 255;
  b2 = ( i >> 8 ) & 255;
  b3 = ( i>>16 ) & 255;
  b4 = ( i>>24 ) & 255;
  return ((int)b1 << 24) + ((int)b2 << 16) + ((int)b3 << 8) + b4;
}
/*****************************************/

opj_volume_t* pgxtovolume(char *relpath, opj_cparameters_t *parameters) {
	
	FILE *f = NULL;
	int w, h, prec;
	unsigned long offset;
	int i, s, numcomps, maxvalue, sliceno, slicepos, maxslice = 0;
	
	OPJ_COLOR_SPACE color_space;
	opj_volume_cmptparm_t cmptparm;	// maximum of 1 component 
	opj_volume_t * volume = NULL;

	char endian1,endian2,sign;
	char signtmp[32];
	char temp[32];
	opj_volume_comp_t *comp = NULL;

		DIR *dirp;
    struct dirent *direntp;
	
	char *tmp = NULL, *tmp2 = NULL,
		*point = NULL, *pgx = NULL;
	char tmpdirpath[MAX_PATH];
	char dirpath[MAX_PATH];
	char pattern[MAX_PATH];
	char pgxfiles[MAX_SLICES][MAX_PATH];
	int pgxslicepos[MAX_SLICES];
	char tmpno[3];
	
	numcomps = 1;
	color_space = CLRSPC_GRAY;
	sliceno = 0;
	maxvalue = 0;
	memset(pgxfiles, 0, MAX_SLICES * MAX_PATH * sizeof(char));
	memset(&cmptparm, 0, sizeof(opj_volume_cmptparm_t));
	
	/* Separación del caso de un único slice frente al de muchos */
	if ((tmp = strrchr(relpath,'-')) == NULL){ 
		//fprintf(stdout,"[INFO] A volume of only one slice....\n");
		sliceno = 1;
		maxslice = 1;
		strcpy(pgxfiles[0],relpath);
	
	} else {
		//Fetch only the path 
		strcpy(tmpdirpath,relpath);
		if ((tmp = strrchr(tmpdirpath,'/')) != NULL){
			tmp++; *tmp='\0';
			strcpy(dirpath,tmpdirpath);
		} else {
			strcpy(dirpath,"./");
		}

		//Fetch the pattern of the volume slices
		if ((tmp = strrchr (relpath,'/')) != NULL) 
			tmp++;	
		else 
			tmp = relpath;
        if ((tmp2 = strrchr(tmp,'-')) != NULL)
            *tmp2='\0';
		else{ 
			fprintf(stdout, "[ERROR] tmp2 ha dado null. no ha encontrado el * %s %s",tmp,relpath);
			return NULL;
		}
        strcpy(pattern,tmp);

		dirp = opendir( dirpath );
		if (dirp == NULL){
			fprintf(stdout, "[ERROR] Infile must be a .pgx file or a directory that contain pgx files");
			return NULL;
		}

		/*Read all .pgx files of directory */
		while ( (direntp = readdir( dirp )) != NULL )
		{
			/* Found a directory, but ignore . and .. */
			if(strcmp(".",direntp->d_name) == 0 || strcmp("..",direntp->d_name) == 0)
					continue;
			
			if( ((pgx = strstr(direntp->d_name,pattern)) != NULL) && ((tmp2 = strstr(direntp->d_name,".pgx")) != NULL) ){
			
				strcpy(tmp,dirpath);
				tmp = strcat(tmp,direntp->d_name);
						
				//Obtenemos el index de la secuencia de slices
				if ((tmp2 = strpbrk (direntp->d_name, "0123456789")) == NULL) 
					continue;
				i = 0;
				while (tmp2 != NULL) {					
					tmpno[i++] = *tmp2;
					point = tmp2;
					tmp2 = strpbrk (tmp2+1,"0123456789");
				}tmpno[i]='\0';

				//Comprobamos que no estamos leyendo algo raro como pattern.jp3d
				if ((point = strpbrk (point,".")) == NULL){
					break;
				}
				//Slicepos --> index de slice; Sliceno --> no de slices hasta el momento
				slicepos = atoi(tmpno);
				pgxslicepos[sliceno] = slicepos - 1;
				sliceno++;
				if (slicepos>maxslice)
					maxslice = slicepos;
				
				//Colocamos el slices en su posicion correspondiente
				strcpy(pgxfiles[slicepos-1],tmp);
			}
		}
	
	}/* else if pattern*.pgx */

	if (!sliceno) {
		fprintf(stdout,"[ERROR] No slices with this pattern founded !! Please check input volume name\n");
		return NULL;
	}
	/*if ( maxslice != sliceno) {
		fprintf(stdout,"[ERROR] Slices are not sequentially numbered !! Please rename them accordingly\n");
		return NULL;
	}*/
	
	for (s=0;s<sliceno;s++)
	{
			int pos = maxslice == sliceno ? s: pgxslicepos[s];
			f = fopen(pgxfiles[pos], "rb");
			if (!f) {
				fprintf(stdout, "[ERROR] Failed to open %s for reading !\n", pgxfiles[s]);
				return NULL;
			}
			fprintf(stdout, "[INFO] Loading %s \n",pgxfiles[pos]);

			fseek(f, 0, SEEK_SET);
			fscanf(f, "PG%[ \t]%c%c%[ \t+-]%d%[ \t]%d%[ \t]%d",temp,&endian1,&endian2,signtmp,&prec,temp,&w,temp,&h);

			i=0;
			sign='+';		
			while (signtmp[i]!='\0') {
				if (signtmp[i]=='-') sign='-';
				i++;
			}
		
			fgetc(f);
			if (endian1=='M' && endian2=='L') {
				cmptparm.bigendian = 1;
			} else if (endian2=='M' && endian1=='L') {
				cmptparm.bigendian = 0;
			} else {
				fprintf(stdout, "[ERROR] Bad pgx header, please check input file\n");
				return NULL;
			}

			if (s==0){
				/* initialize volume component */

				cmptparm.x0 = parameters->volume_offset_x0;
				cmptparm.y0 = parameters->volume_offset_y0;
				cmptparm.z0 = parameters->volume_offset_z0;
				cmptparm.w = !cmptparm.x0 ? (w - 1) * parameters->subsampling_dx + 1 : cmptparm.x0 + (w - 1) * parameters->subsampling_dx + 1;
				cmptparm.h = !cmptparm.y0 ? (h - 1) * parameters->subsampling_dy + 1 : cmptparm.y0 + (h - 1) * parameters->subsampling_dy + 1;
				cmptparm.l = !cmptparm.z0 ? (sliceno - 1) * parameters->subsampling_dz + 1 : cmptparm.z0 + (sliceno - 1) * parameters->subsampling_dz + 1;
				
				if (sign == '-') {
					cmptparm.sgnd = 1;
				} else {
					cmptparm.sgnd = 0;
				}
				cmptparm.prec = prec;
				cmptparm.bpp = prec;
				cmptparm.dcoffset = parameters->dcoffset;
				cmptparm.dx = parameters->subsampling_dx;
				cmptparm.dy = parameters->subsampling_dy;
				cmptparm.dz = parameters->subsampling_dz;
				
				/* create the volume */
				volume = opj_volume_create(numcomps, &cmptparm, color_space);
				if(!volume) {
					fclose(f);
					return NULL;
				}
				/* set volume offset and reference grid */
				volume->x0 = cmptparm.x0;
				volume->y0 = cmptparm.y0;
				volume->z0 = cmptparm.z0;
				volume->x1 = cmptparm.w;
				volume->y1 = cmptparm.h;
				volume->z1 = cmptparm.l;
				
				/* set volume data :only one component, that is a volume*/
				comp = &volume->comps[0];
			
			}//if sliceno==1
			
			offset = w * h * s;
			
			for (i = 0; i < w * h; i++) {
				int v;
				if (comp->prec <= 8) {
					if (!comp->sgnd) {
						v = readuchar(f);
					} else {
						v = (char) readuchar(f);
					}
				} else if (comp->prec <= 16) {
					if (!comp->sgnd) {
						v = readushort(f, cmptparm.bigendian);
					} else {
						v = (short) readushort(f, cmptparm.bigendian);
					}
				} else {
					if (!comp->sgnd) {
						v = readuint(f, cmptparm.bigendian);
					} else {
						v = (int) readuint(f, cmptparm.bigendian);
					}
				}				
				if (v > maxvalue)
					maxvalue = v;
				comp->data[i + offset] = v;
				
			}
			fclose(f);
	} // for s --> sliceno
	comp->bpp = int_floorlog2(maxvalue) + 1;
	if (sliceno != 1)
		closedir( dirp );
	//dump_volume(stdout, volume);
	return volume;
}


int volumetopgx(opj_volume_t * volume, char *outfile) {
	int w, wr, wrr, h, hr, hrr, l, lr, lrr;
	int i, j, compno, offset, sliceno;
	FILE *fdest = NULL;

	for (compno = 0; compno < volume->numcomps; compno++) {
		opj_volume_comp_t *comp = &volume->comps[compno];
		char name[256];
		int nbytes = 0;
		char *tmp = outfile;
		while (*tmp) {
			tmp++;
		}
		while (*tmp!='.') {
			tmp--;
		}
		*tmp='\0';
		for(sliceno = 0; sliceno < volume->z1 - volume->z0; sliceno++) {

			if (volume->numcomps > 1) {
				sprintf(name, "%s%d-%d.pgx", outfile, sliceno+1, compno);
			} else if ((volume->z1 - volume->z0) > 1) {
				sprintf(name, "%s%d.pgx", outfile, sliceno+1);
			} else {
				sprintf(name, "%s.pgx", outfile);
			}

			fdest = fopen(name, "wb");
			if (!fdest) {
				fprintf(stdout, "[ERROR] Failed to open %s for writing \n", name);
				return 1;
			}

			fprintf(stdout,"[INFO] Writing in %s (%s)\n",name,volume->comps[0].bigendian ? "Bigendian" : "Little-endian");

			w = int_ceildiv(volume->x1 - volume->x0, volume->comps[compno].dx);
			wr = volume->comps[compno].w;
			wrr = int_ceildivpow2(volume->comps[compno].w, volume->comps[compno].factor[0]);
			
			h = int_ceildiv(volume->y1 - volume->y0, volume->comps[compno].dy);
			hr = volume->comps[compno].h;
			hrr = int_ceildivpow2(volume->comps[compno].h, volume->comps[compno].factor[1]);
			
			l = int_ceildiv(volume->z1 - volume->z0, volume->comps[compno].dz);
			lr = volume->comps[compno].l;
			lrr = int_ceildivpow2(volume->comps[compno].l, volume->comps[compno].factor[2]);

			fprintf(fdest, "PG %c%c %c%d %d %d\n", comp->bigendian ? 'M':'L', comp->bigendian ? 'L':'M',comp->sgnd ? '-' : '+', comp->prec, wr, hr);
			if (comp->prec <= 8) {
				nbytes = 1;
			} else if (comp->prec <= 16) {
				nbytes = 2;
			} else {
				nbytes = 4;
			}

			offset = (sliceno / lrr * l) + (sliceno % lrr);
			offset = wrr * hrr * offset;
			//fprintf(stdout,"%d %d %d %d\n",offset,wrr*hrr,wrr,w);
			for (i = 0; i < wrr * hrr; i++) {
				int v = volume->comps[0].data[(i / wrr * w) + (i % wrr) + offset];
				if (volume->comps[0].bigendian) {
					for (j = nbytes - 1; j >= 0; j--) {
                        char byte = (char) ((v >> (j * 8)) & 0xff);
                        fwrite(&byte, 1, 1, fdest);
					}
				} else {
					for (j = 0; j <= nbytes - 1; j++) {
                        char byte = (char) ((v >> (j * 8)) & 0xff);
						fwrite(&byte, 1, 1, fdest);
					}
				}
			}

			fclose(fdest);
		}//for sliceno
	}//for compno

	return 0;
}

/* -->> -->> -->> -->>

BIN IMAGE FORMAT

<<-- <<-- <<-- <<-- */

opj_volume_t* bintovolume(char *filename, char *fileimg, opj_cparameters_t *parameters) {
	int subsampling_dx =  parameters->subsampling_dx;
	int subsampling_dy =  parameters->subsampling_dy;
	int subsampling_dz =  parameters->subsampling_dz;
	
	int i, compno, w, h, l, numcomps = 1;
	int prec, max = 0;

//	char temp[32];
	char line[100];
	int bigendian;
	
	FILE *f = NULL;
	FILE *fimg = NULL;
	OPJ_COLOR_SPACE color_space;
	opj_volume_cmptparm_t cmptparm;	/* maximum of 1 component */
	opj_volume_t * volume = NULL;
	opj_volume_comp_t *comp = NULL;

	bigendian = 0;
	color_space = CLRSPC_GRAY;

	fimg = fopen(fileimg,"r");
	if (!fimg) { 
		fprintf(stdout, "[ERROR] Failed to open %s for reading !!\n", fileimg);
		return 0;
	}

	fseek(fimg, 0, SEEK_SET);
	while (!feof(fimg)) {
        fgets(line,100,fimg);
		//fprintf(stdout,"%s %d \n",line,feof(fimg));
		if (strncmp(line,"Bpp",3) == 0){
			sscanf(line,"%*s%*[ \t]%d",&prec);
		} else if (strncmp(line,"Color",5) == 0){
			sscanf(line, "%*s%*[ \t]%d",&color_space);
		} else if (strncmp(line,"Dim",3) == 0){
			sscanf(line, "%*s%*[ \t]%d%*[ \t]%d%*[ \t]%d",&w,&h,&l);
		}
	}
	//fscanf(fimg, "Bpp%[ \t]%d%[ \t\n]",temp,&prec,temp);
	//fscanf(fimg, "Color Map%[ \t]%d%[ \n\t]Dimensions%[ \t]%d%[ \t]%d%[ \t]%d%[ \n\t]",temp,&color_space,temp,temp,&w,temp,&h,temp,&l,temp);
	//fscanf(fimg, "Resolution(mm)%[ \t]%d%[ \t]%d%[ \t]%d%[ \n\t]",temp,&subsampling_dx,temp,&subsampling_dy,temp,&subsampling_dz,temp);

	#ifdef VERBOSE
		fprintf(stdout, "[INFO] %d \t %d %d %d \t %3.2f %2.2f %2.2f \t %d \n",color_space,w,h,l,subsampling_dx,subsampling_dy,subsampling_dz,prec);
	#endif
	fclose(fimg);
	
	/* initialize volume components */
	memset(&cmptparm, 0, sizeof(opj_volume_cmptparm_t));
	
	cmptparm.prec = prec;
	cmptparm.bpp = prec;
	cmptparm.sgnd = 0;
	cmptparm.bigendian = bigendian;
	cmptparm.dcoffset = parameters->dcoffset;
	cmptparm.dx = subsampling_dx;
	cmptparm.dy = subsampling_dy;
	cmptparm.dz = subsampling_dz;
	cmptparm.w = w;
	cmptparm.h = h;
	cmptparm.l = l;
	
	/* create the volume */
	volume = opj_volume_create(numcomps, &cmptparm, color_space);
	if(!volume) {
		fprintf(stdout,"[ERROR] Unable to create volume");	
		fclose(f);
		return NULL;
	}
	
	/* set volume offset and reference grid */
	volume->x0 = parameters->volume_offset_x0;
	volume->y0 = parameters->volume_offset_y0;
	volume->z0 = parameters->volume_offset_z0;
	volume->x1 = parameters->volume_offset_x0 + (w - 1) *	subsampling_dx + 1;
	volume->y1 = parameters->volume_offset_y0 + (h - 1) *	subsampling_dy + 1;
	volume->z1 = parameters->volume_offset_z0 + (l - 1) *	subsampling_dz + 1;
	
	/* set volume data */
	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stdout, "[ERROR] Failed to open %s for reading !!\n", filename);
		return 0;
	}
	
	/* BINARY */
	for (compno = 0; compno < volume->numcomps; compno++) {
		int whl = w * h * l;
		/* set volume data */
		comp = &volume->comps[compno];
		
		/*if (comp->prec <= 8) {
			if (!comp->sgnd) {
                unsigned char *data = (unsigned char *) malloc(whl * sizeof(unsigned char));
				fread(data, 1, whl, f);
				for (i = 0; i < whl; i++) {
					comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				char *data = (char *) malloc(whl);
				fread(data, 1, whl, f);
				for (i = 0; i < whl; i++) {
					comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			}
		} else if (comp->prec <= 16) {
			if (!comp->sgnd) {
                unsigned short *data = (unsigned short *) malloc(whl * sizeof(unsigned short));
				int leido = fread(data, 2, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}
				
				for (i = 0; i < whl; i++) {
					if (bigendian)	//(c1 << 8) + c2;
						comp->data[i] = data[i];
					else{			//(c2 << 8) + c1;
						comp->data[i] = ShortSwap(data[i]);
					}
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				short *data = (short *) malloc(whl);
				int leido = fread(data, 2, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}
				for (i = 0; i < whl; i++) {
					if (bigendian){	//(c1 << 8) + c2;
						comp->data[i] = data[i];
					}else{			//(c2 << 8) + c1;
						comp->data[i] = (short) ShortSwap((unsigned short) data[i]);
					}
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			}
		} else {
			if (!comp->sgnd) {
                unsigned int *data = (unsigned int *) malloc(whl * sizeof(unsigned int));
				int leido = fread(data, 4, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}				for (i = 0; i < whl; i++) {
					if (!bigendian)
						comp->data[i] = LongSwap(data[i]);
					else
						comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				int leido = fread(comp->data, 4, whl, f);
				if (!leido)	{
					fclose(f);
					return NULL;
				}				
				for (i = 0; i < whl; i++) {
					if (!bigendian) 
						comp->data[i] = (int) LongSwap((unsigned int) comp->data[i]);
					if (comp->data[i] > max)
						max = comp->data[i];
				}
			}
		}*/
		
		for (i = 0; i < whl; i++) {
			int v;
			if (comp->prec <= 8) {
				if (!comp->sgnd) {
					v = readuchar(f);
				} else {
					v = (char) readuchar(f);
				}
			} else if (comp->prec <= 16) {
				if (!comp->sgnd) {
					v = readushort(f, bigendian);
				} else {
					v = (short) readushort(f, bigendian);
				}
			} else {
				if (!comp->sgnd) {
					v = readuint(f, bigendian);
				} else {
					v = (int) readuint(f, bigendian);
				}
			}
			if (v > max)
				max = v;
			comp->data[i] = v;
		}
		comp->bpp = int_floorlog2(max) + 1;
	}
	fclose(f);
	return volume;
}

int volumetobin(opj_volume_t * volume, char *outfile) {
	int w, wr, wrr, h, hr, hrr, l, lr, lrr, max;
	int i,j, compno, nbytes;
	int offset, sliceno;
	FILE *fdest = NULL;
	FILE *fimgdest = NULL;
//	char *imgtemp;
	char name[256];

	for (compno = 0; compno < 1; compno++) { //Only one component
		
		fdest = fopen(outfile, "wb");
		if (!fdest) {
			fprintf(stdout, "[ERROR] Failed to open %s for writing\n", outfile);
			return 1;
		}
        fprintf(stdout,"[INFO] Writing outfile %s (%s) \n",outfile, volume->comps[0].bigendian ? "Bigendian" : "Little-endian");

		w = int_ceildiv(volume->x1 - volume->x0, volume->comps[compno].dx);
		wr = volume->comps[compno].w;
		wrr = int_ceildivpow2(volume->comps[compno].w, volume->comps[compno].factor[0]);
		
		h = int_ceildiv(volume->y1 - volume->y0, volume->comps[compno].dy);
		hr = volume->comps[compno].h;
		hrr = int_ceildivpow2(volume->comps[compno].h, volume->comps[compno].factor[1]);
		
		l = int_ceildiv(volume->z1 - volume->z0, volume->comps[compno].dz);
		lr = volume->comps[compno].l;
		lrr = int_ceildivpow2(volume->comps[compno].l, volume->comps[compno].factor[2]);

		max = (volume->comps[compno].prec <= 8) ? 255 : (1 << volume->comps[compno].prec) - 1;
		
		volume->comps[compno].x0 = int_ceildivpow2(volume->comps[compno].x0 - int_ceildiv(volume->x0, volume->comps[compno].dx), volume->comps[compno].factor[0]);
		volume->comps[compno].y0 = int_ceildivpow2(volume->comps[compno].y0 - int_ceildiv(volume->y0, volume->comps[compno].dy), volume->comps[compno].factor[1]);
		volume->comps[compno].z0 = int_ceildivpow2(volume->comps[compno].z0 - int_ceildiv(volume->z0, volume->comps[compno].dz), volume->comps[compno].factor[2]);
		
		if (volume->comps[0].prec <= 8) {
			nbytes = 1;
		} else if (volume->comps[0].prec <= 16) {
			nbytes = 2;
		} else {
			nbytes = 4;
		}

		//fprintf(stdout,"w %d wr %d wrr %d h %d hr %d hrr %d l %d lr %d lrr %d max %d nbytes %d\n Factor %d %d %d",w,wr,wrr,h,hr,hrr,l,lr,lrr,max,nbytes,volume->comps[compno].factor[0],volume->comps[compno].factor[1],volume->comps[compno].factor[2]);

		for(sliceno = 0; sliceno < lrr; sliceno++) {
			offset = (sliceno / lrr * l) + (sliceno % lrr);
            offset = wrr * hrr * offset;
			for (i = 0; i < wrr * hrr; i++) {
				int v = volume->comps[0].data[(i / wrr * w) + (i % wrr) + offset];
				if (volume->comps[0].bigendian) {
					for (j = nbytes - 1; j >= 0; j--) {
                        char byte = (char) ((v >> (j * 8)) & 0xff);
                        fwrite(&byte, 1, 1, fdest);
					}
				} else {
					for (j = 0; j <= nbytes - 1; j++) {
                        char byte = (char) ((v >> (j * 8)) & 0xff);
						fwrite(&byte, 1, 1, fdest);
					}
				}
			}
		}
	
	}
	
	fclose(fdest);

	sprintf(name,"%s.img",outfile);
	fimgdest = fopen(name, "w");
		if (!fimgdest) {
			fprintf(stdout, "[ERROR] Failed to open %s for writing\n", name);
			return 1;
		}
	fprintf(fimgdest, "Bpp\t%d\nColor Map\t2\nDimensions\t%d\t%d\t%d\nResolution(mm)\t%d\t%d\t%d\t\n",
		volume->comps[0].prec,wrr,hrr,lrr,volume->comps[0].dx,volume->comps[0].dy,volume->comps[0].dz);

	fclose(fimgdest);
	return 0;
}
/* -->> -->> -->> -->>

IMG IMAGE FORMAT

<<-- <<-- <<-- <<-- */
opj_volume_t* imgtovolume(char *fileimg, opj_cparameters_t *parameters) {
	int subsampling_dx =  parameters->subsampling_dx;
	int subsampling_dy =  parameters->subsampling_dy;
	int subsampling_dz =  parameters->subsampling_dz;
	
	int i, compno, w, h, l, numcomps = 1;
	int prec, max = 0, min = 0;
	float dx, dy, dz;
	char filename[100], tmpdirpath[100], dirpath[100], *tmp;
	char line[100], datatype[100];
	int bigendian;
	
	FILE *f = NULL;
	FILE *fimg = NULL;
	OPJ_COLOR_SPACE color_space;
	opj_volume_cmptparm_t cmptparm;	/* maximum of 1 component */
	opj_volume_t * volume = NULL;
	opj_volume_comp_t *comp = NULL;

	bigendian = 0;
	color_space = CLRSPC_GRAY;

	fimg = fopen(fileimg,"r");
	if (!fimg) { 
		fprintf(stderr, "[ERROR] Failed to open %s for reading !!\n", fileimg);
		return 0;
	}

	//Fetch only the path 
	strcpy(tmpdirpath,fileimg);
	if ((tmp = strrchr(tmpdirpath,'/')) != NULL){
		tmp++; *tmp='\0';
		strcpy(dirpath,tmpdirpath);
	} else {
		strcpy(dirpath,"./");
	}

	fseek(fimg, 0, SEEK_SET);
	while (!feof(fimg)) {
        fgets(line,100,fimg);
		//fprintf(stdout,"%s %d \n",line,feof(fimg));
		if (strncmp(line,"Image",5) == 0){
			sscanf(line,"%*s%*[ \t]%s",datatype);
		} else if (strncmp(line,"File",4) == 0){
			sscanf(line,"%*s %*s%*[ \t]%s",filename);
			strcat(dirpath, filename);
			strcpy(filename,dirpath);
		} else if (strncmp(line,"Min",3) == 0){
			sscanf(line,"%*s %*s%*[ \t]%d%*[ \t]%d",&min,&max);
			prec = int_floorlog2(max - min + 1);
		} else if (strncmp(line,"Bpp",3) == 0){
			sscanf(line,"%*s%*[ \t]%d",&prec);
		} else if (strncmp(line,"Color",5) == 0){
			sscanf(line, "%*s %*s%*[ \t]%d",&color_space);
		} else if (strncmp(line,"Dim",3) == 0){
			sscanf(line, "%*s%*[ \t]%d%*[ \t]%d%*[ \t]%d",&w,&h,&l);
		} else if (strncmp(line,"Res",3) == 0){
			sscanf(line,"%*s%*[ \t]%f%*[ \t]%f%*[ \t]%f",&dx,&dy,&dz);
		}

	}
	#ifdef VERBOSE
		fprintf(stdout, "[INFO] %s %d \t %d %d %d \t %f %f %f \t %d %d %d \n",filename,color_space,w,h,l,dx,dy,dz,max,min,prec);
	#endif
	fclose(fimg);

	/* error control */
	if ( !prec || !w || !h || !l ){
		fprintf(stderr,"[ERROR] Unable to read IMG file correctly. Found some null values.");	
		return NULL;
	}

	/* initialize volume components */
	memset(&cmptparm, 0, sizeof(opj_volume_cmptparm_t));
	
	cmptparm.prec = prec;
	cmptparm.bpp = prec;
	cmptparm.sgnd = 0;
	cmptparm.bigendian = bigendian;
	cmptparm.dcoffset = parameters->dcoffset;
	cmptparm.dx = subsampling_dx;
	cmptparm.dy = subsampling_dy;
	cmptparm.dz = subsampling_dz;
	cmptparm.w = w;
	cmptparm.h = h;
	cmptparm.l = l;
	
	/* create the volume */
	volume = opj_volume_create(numcomps, &cmptparm, color_space);
	if(!volume) {
		fprintf(stdout,"[ERROR] Unable to create volume");	
		return NULL;
	}
	
	/* set volume offset and reference grid */
	volume->x0 = parameters->volume_offset_x0;
	volume->y0 = parameters->volume_offset_y0;
	volume->z0 = parameters->volume_offset_z0;
	volume->x1 = parameters->volume_offset_x0 + (w - 1) *	subsampling_dx + 1;
	volume->y1 = parameters->volume_offset_y0 + (h - 1) *	subsampling_dy + 1;
	volume->z1 = parameters->volume_offset_z0 + (l - 1) *	subsampling_dz + 1;
	
	max = 0;
	/* set volume data */
	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "[ERROR] Failed to open %s for reading !!\n", filename);
		fclose(f);
		return 0;
	}
	
	/* BINARY */
	for (compno = 0; compno < volume->numcomps; compno++) {
		int whl = w * h * l;
		/* set volume data */
		comp = &volume->comps[compno];
		
		/*if (comp->prec <= 8) {
			if (!comp->sgnd) {
                unsigned char *data = (unsigned char *) malloc(whl * sizeof(unsigned char));
				fread(data, 1, whl, f);
				for (i = 0; i < whl; i++) {
					comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				char *data = (char *) malloc(whl);
				fread(data, 1, whl, f);
				for (i = 0; i < whl; i++) {
					comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			}
		} else if (comp->prec <= 16) {
			if (!comp->sgnd) {
                unsigned short *data = (unsigned short *) malloc(whl * sizeof(unsigned short));
				int leido = fread(data, 2, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}
				
				for (i = 0; i < whl; i++) {
					if (bigendian)	//(c1 << 8) + c2;
						comp->data[i] = data[i];
					else{			//(c2 << 8) + c1;
						comp->data[i] = ShortSwap(data[i]);
					}
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				short *data = (short *) malloc(whl);
				int leido = fread(data, 2, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}
				for (i = 0; i < whl; i++) {
					if (bigendian){	//(c1 << 8) + c2;
						comp->data[i] = data[i];
					}else{			//(c2 << 8) + c1;
						comp->data[i] = (short) ShortSwap((unsigned short) data[i]);
					}
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			}
		} else {
			if (!comp->sgnd) {
                unsigned int *data = (unsigned int *) malloc(whl * sizeof(unsigned int));
				int leido = fread(data, 4, whl, f);
				if (!leido)	{
					free(data);	fclose(f);
					return NULL;
				}				for (i = 0; i < whl; i++) {
					if (!bigendian)
						comp->data[i] = LongSwap(data[i]);
					else
						comp->data[i] = data[i];
					if (comp->data[i] > max)
						max = comp->data[i];
				}
				free(data);
			} else {
				int leido = fread(comp->data, 4, whl, f);
				if (!leido)	{
					fclose(f);
					return NULL;
				}				
				for (i = 0; i < whl; i++) {
					if (!bigendian) 
						comp->data[i] = (int) LongSwap((unsigned int) comp->data[i]);
					if (comp->data[i] > max)
						max = comp->data[i];
				}
			}
		}*/
		
		for (i = 0; i < whl; i++) {
			int v;
			if (comp->prec <= 8) {
				if (!comp->sgnd) {
					v = readuchar(f);
				} else {
					v = (char) readuchar(f);
				}
			} else if (comp->prec <= 16) {
				if (!comp->sgnd) {
					v = readushort(f, bigendian);
				} else {
					v = (short) readushort(f, bigendian);
				}
			} else {
				if (!comp->sgnd) {
					v = readuint(f, bigendian);
				} else {
					v = (int) readuint(f, bigendian);
				}
			}
			if (v > max)
				max = v;
			comp->data[i] = v;
		}
		comp->bpp = int_floorlog2(max) + 1;
	}
	fclose(f);
	return volume;
}

