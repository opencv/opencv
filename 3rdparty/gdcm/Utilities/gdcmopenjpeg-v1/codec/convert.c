/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux and Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2006-2007, Parvatha Elangovan
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
#include "opj_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_LIBTIFF
#ifdef _WIN32
#include "../libs/libtiff/tiffio.h"
#else
#include <tiffio.h>
#endif /* _WIN32 */
#endif /* HAVE_LIBTIFF */

#ifdef HAVE_LIBPNG
#ifdef _WIN32
#include "../libs/png/png.h"
#else
#include <png.h>
#endif /* _WIN32 */
#endif /* HAVE_LIBPNG */

#include "../libopenjpeg/openjpeg.h"
#include "convert.h"

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

  TGA IMAGE FORMAT

 <<-- <<-- <<-- <<-- */

// TGA header definition.
#pragma pack(push,1) // Pack structure byte aligned
typedef struct tga_header
{                           
    unsigned char   id_length;              /* Image id field length    */
    unsigned char   colour_map_type;        /* Colour map type          */
    unsigned char   image_type;             /* Image type               */
    /*
    ** Colour map specification
    */
    unsigned short  colour_map_index;       /* First entry index        */
    unsigned short  colour_map_length;      /* Colour map length        */
    unsigned char   colour_map_entry_size;  /* Colour map entry size    */
    /*
    ** Image specification
    */
    unsigned short  x_origin;               /* x origin of image        */
    unsigned short  y_origin;               /* u origin of image        */
    unsigned short  image_width;            /* Image width              */
    unsigned short  image_height;           /* Image height             */
    unsigned char   pixel_depth;            /* Pixel depth              */
    unsigned char   image_desc;             /* Image descriptor         */
} tga_header;
#pragma pack(pop) // Return to normal structure packing alignment.

int tga_readheader(FILE *fp, unsigned int *bits_per_pixel, 
	unsigned int *width, unsigned int *height, int *flip_image)
{
	int palette_size;
	tga_header tga ;

	if (!bits_per_pixel || !width || !height || !flip_image)
		return 0;
	
	// Read TGA header
	fread((unsigned char*)&tga, sizeof(tga_header), 1, fp);

	*bits_per_pixel = tga.pixel_depth;
	
	*width  = tga.image_width;
	*height = tga.image_height ;

	// Ignore tga identifier, if present ...
	if (tga.id_length)
	{
		unsigned char *id = (unsigned char *) malloc(tga.id_length);
		fread(id, tga.id_length, 1, fp);
		free(id);  
	}

	// Test for compressed formats ... not yet supported ...
	// Note :-  9 - RLE encoded palettized.
	//	  	   10 - RLE encoded RGB.
	if (tga.image_type > 8)
	{
		fprintf(stderr, "Sorry, compressed tga files are not currently supported.\n");
		return 0 ;
	}

	*flip_image = !(tga.image_desc & 32);

	// Palettized formats are not yet supported, skip over the palette, if present ... 
	palette_size = tga.colour_map_length * (tga.colour_map_entry_size/8);
	
	if (palette_size>0)
	{
		fprintf(stderr, "File contains a palette - not yet supported.");
		fseek(fp, palette_size, SEEK_CUR);
	}
	return 1;
}

int tga_writeheader(FILE *fp, int bits_per_pixel, int width, int height, 
	bool flip_image)
{
	tga_header tga;

	if (!bits_per_pixel || !width || !height)
		return 0;

	memset(&tga, 0, sizeof(tga_header));

	tga.pixel_depth = bits_per_pixel;
	tga.image_width  = width;
	tga.image_height = height;
	tga.image_type = 2; // Uncompressed.
	tga.image_desc = 8; // 8 bits per component.

	if (flip_image)
		tga.image_desc |= 32;

	// Write TGA header
	fwrite((unsigned char*)&tga, sizeof(tga_header), 1, fp);

	return 1;
}

opj_image_t* tgatoimage(const char *filename, opj_cparameters_t *parameters) {
	FILE *f;
	opj_image_t *image;
	unsigned int image_width, image_height, pixel_bit_depth;
	unsigned int x, y;
	int flip_image=0;
	opj_image_cmptparm_t cmptparm[4];	/* maximum 4 components */
	int numcomps;
	OPJ_COLOR_SPACE color_space;
	bool mono ;
	bool save_alpha;
	int subsampling_dx, subsampling_dy;
	int i;	

	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		return 0;
	}

	if (!tga_readheader(f, &pixel_bit_depth, &image_width, &image_height, &flip_image))
		return NULL;

	// We currently only support 24 & 32 bit tga's ...
	if (!((pixel_bit_depth == 24) || (pixel_bit_depth == 32)))
		return NULL;

	/* initialize image components */   
	memset(&cmptparm[0], 0, 4 * sizeof(opj_image_cmptparm_t));

	mono = (pixel_bit_depth == 8) || (pixel_bit_depth == 16);  // Mono with & without alpha.
	save_alpha = (pixel_bit_depth == 16) || (pixel_bit_depth == 32); // Mono with alpha, or RGB with alpha

	if (mono) {
		color_space = CLRSPC_GRAY;
		numcomps = save_alpha ? 2 : 1;
	}	
	else {
		numcomps = save_alpha ? 4 : 3;
		color_space = CLRSPC_SRGB;
	}

	subsampling_dx = parameters->subsampling_dx;
	subsampling_dy = parameters->subsampling_dy;

	for (i = 0; i < numcomps; i++) {
		cmptparm[i].prec = 8;
		cmptparm[i].bpp = 8;
		cmptparm[i].sgnd = 0;
		cmptparm[i].dx = subsampling_dx;
		cmptparm[i].dy = subsampling_dy;
		cmptparm[i].w = image_width;
		cmptparm[i].h = image_height;
	}

	/* create the image */
	image = opj_image_create(numcomps, &cmptparm[0], color_space);

	if (!image)
		return NULL;

	/* set image offset and reference grid */
	image->x0 = parameters->image_offset_x0;
	image->y0 = parameters->image_offset_y0;
	image->x1 =	!image->x0 ? (image_width - 1) * subsampling_dx + 1 : image->x0 + (image_width - 1) * subsampling_dx + 1;
	image->y1 =	!image->y0 ? (image_height - 1) * subsampling_dy + 1 : image->y0 + (image_height - 1) * subsampling_dy + 1;

	/* set image data */
	for (y=0; y < image_height; y++) 
	{
		int index;

		if (flip_image)
			index = (image_height-y-1)*image_width;
		else
			index = y*image_width;

		if (numcomps==3)
		{
			for (x=0;x<image_width;x++) 
			{
				unsigned char r,g,b;
				fread(&b, 1, 1, f);
				fread(&g, 1, 1, f);
				fread(&r, 1, 1, f);

				image->comps[0].data[index]=r;
				image->comps[1].data[index]=g;
				image->comps[2].data[index]=b;
				index++;
			}
		}
		else if (numcomps==4)
		{
			for (x=0;x<image_width;x++) 
			{
				unsigned char r,g,b,a;
				fread(&b, 1, 1, f);
				fread(&g, 1, 1, f);
				fread(&r, 1, 1, f);
				fread(&a, 1, 1, f);

				image->comps[0].data[index]=r;
				image->comps[1].data[index]=g;
				image->comps[2].data[index]=b;
				image->comps[3].data[index]=a;
				index++;
			}
		}
		else {
			fprintf(stderr, "Currently unsupported bit depth : %s\n", filename);
		}
	}	
	return image;
}

int imagetotga(opj_image_t * image, const char *outfile) {
	int width, height, bpp, x, y;
	bool write_alpha;
	int i;
	unsigned int alpha_channel;
	float r,g,b,a;
	unsigned char value;
	float scale;
	FILE *fdest;

	fdest = fopen(outfile, "wb");
	if (!fdest) {
		fprintf(stderr, "ERROR -> failed to open %s for writing\n", outfile);
		return 1;
	}

	for (i = 0; i < image->numcomps-1; i++)	{
		if ((image->comps[0].dx != image->comps[i+1].dx) 
			||(image->comps[0].dy != image->comps[i+1].dy) 
			||(image->comps[0].prec != image->comps[i+1].prec))	{
      fprintf(stderr, "Unable to create a tga file with such J2K image charateristics.");
      return 1;
   }
	}

	width = image->comps[0].w;
	height = image->comps[0].h; 

	// Mono with alpha, or RGB with alpha.
	write_alpha = (image->numcomps==2) || (image->numcomps==4);   

	// Write TGA header 
	bpp = write_alpha ? 32 : 24;
	if (!tga_writeheader(fdest, bpp, width , height, true))
		return 1;

	alpha_channel = image->numcomps-1; 

	scale = 255.0f / (float)((1<<image->comps[0].prec)-1);

	for (y=0; y < height; y++) {
		unsigned int index=y*width;

		for (x=0; x < width; x++, index++)	{
			r = (float)(image->comps[0].data[index]);

			if (image->numcomps>2) {
				g = (float)(image->comps[1].data[index]);
				b = (float)(image->comps[2].data[index]);
			}
			else  {// Greyscale ...
				g = r;
				b = r;
			}

			// TGA format writes BGR ...
			value = (unsigned char)(b*scale);
			fwrite(&value,1,1,fdest);

			value = (unsigned char)(g*scale);
			fwrite(&value,1,1,fdest);

			value = (unsigned char)(r*scale);
			fwrite(&value,1,1,fdest);

			if (write_alpha) {
				a = (float)(image->comps[alpha_channel].data[index]);
				value = (unsigned char)(a*scale);
				fwrite(&value,1,1,fdest);
			}
		}
	}

	return 0;
}

/* -->> -->> -->> -->>

  BMP IMAGE FORMAT

 <<-- <<-- <<-- <<-- */

/* WORD defines a two byte word */
typedef unsigned short int WORD;

/* DWORD defines a four byte word */
typedef unsigned long int DWORD;

typedef struct {
  WORD bfType;			/* 'BM' for Bitmap (19776) */
  DWORD bfSize;			/* Size of the file        */
  WORD bfReserved1;		/* Reserved : 0            */
  WORD bfReserved2;		/* Reserved : 0            */
  DWORD bfOffBits;		/* Offset                  */
} BITMAPFILEHEADER_t;

typedef struct {
  DWORD biSize;			/* Size of the structure in bytes */
  DWORD biWidth;		/* Width of the image in pixels */
  DWORD biHeight;		/* Heigth of the image in pixels */
  WORD biPlanes;		/* 1 */
  WORD biBitCount;		/* Number of color bits by pixels */
  DWORD biCompression;		/* Type of encoding 0: none 1: RLE8 2: RLE4 */
  DWORD biSizeImage;		/* Size of the image in bytes */
  DWORD biXpelsPerMeter;	/* Horizontal (X) resolution in pixels/meter */
  DWORD biYpelsPerMeter;	/* Vertical (Y) resolution in pixels/meter */
  DWORD biClrUsed;		/* Number of color used in the image (0: ALL) */
  DWORD biClrImportant;		/* Number of important color (0: ALL) */
} BITMAPINFOHEADER_t;

opj_image_t* bmptoimage(const char *filename, opj_cparameters_t *parameters) {
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	int i, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm[3];	/* maximum of 3 components */
	opj_image_t * image = NULL;

	FILE *IN;
	BITMAPFILEHEADER_t File_h;
	BITMAPINFOHEADER_t Info_h;
	unsigned char *RGB;
	unsigned char *table_R, *table_G, *table_B;
	unsigned int j, PAD = 0;

	int x, y, index;
	int gray_scale = 1, not_end_file = 1; 

	unsigned int line = 0, col = 0;
	unsigned char v, v2;
	DWORD W, H;
  
	IN = fopen(filename, "rb");
	if (!IN) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		return 0;
	}
	
	File_h.bfType = getc(IN);
	File_h.bfType = (getc(IN) << 8) + File_h.bfType;
	
	if (File_h.bfType != 19778) {
		fprintf(stderr,"Error, not a BMP file!\n");
		return 0;
	} else {
		/* FILE HEADER */
		/* ------------- */
		File_h.bfSize = getc(IN);
		File_h.bfSize = (getc(IN) << 8) + File_h.bfSize;
		File_h.bfSize = (getc(IN) << 16) + File_h.bfSize;
		File_h.bfSize = (getc(IN) << 24) + File_h.bfSize;

		File_h.bfReserved1 = getc(IN);
		File_h.bfReserved1 = (getc(IN) << 8) + File_h.bfReserved1;

		File_h.bfReserved2 = getc(IN);
		File_h.bfReserved2 = (getc(IN) << 8) + File_h.bfReserved2;

		File_h.bfOffBits = getc(IN);
		File_h.bfOffBits = (getc(IN) << 8) + File_h.bfOffBits;
		File_h.bfOffBits = (getc(IN) << 16) + File_h.bfOffBits;
		File_h.bfOffBits = (getc(IN) << 24) + File_h.bfOffBits;

		/* INFO HEADER */
		/* ------------- */

		Info_h.biSize = getc(IN);
		Info_h.biSize = (getc(IN) << 8) + Info_h.biSize;
		Info_h.biSize = (getc(IN) << 16) + Info_h.biSize;
		Info_h.biSize = (getc(IN) << 24) + Info_h.biSize;

		Info_h.biWidth = getc(IN);
		Info_h.biWidth = (getc(IN) << 8) + Info_h.biWidth;
		Info_h.biWidth = (getc(IN) << 16) + Info_h.biWidth;
		Info_h.biWidth = (getc(IN) << 24) + Info_h.biWidth;
		w = Info_h.biWidth;

		Info_h.biHeight = getc(IN);
		Info_h.biHeight = (getc(IN) << 8) + Info_h.biHeight;
		Info_h.biHeight = (getc(IN) << 16) + Info_h.biHeight;
		Info_h.biHeight = (getc(IN) << 24) + Info_h.biHeight;
		h = Info_h.biHeight;

		Info_h.biPlanes = getc(IN);
		Info_h.biPlanes = (getc(IN) << 8) + Info_h.biPlanes;

		Info_h.biBitCount = getc(IN);
		Info_h.biBitCount = (getc(IN) << 8) + Info_h.biBitCount;

		Info_h.biCompression = getc(IN);
		Info_h.biCompression = (getc(IN) << 8) + Info_h.biCompression;
		Info_h.biCompression = (getc(IN) << 16) + Info_h.biCompression;
		Info_h.biCompression = (getc(IN) << 24) + Info_h.biCompression;

		Info_h.biSizeImage = getc(IN);
		Info_h.biSizeImage = (getc(IN) << 8) + Info_h.biSizeImage;
		Info_h.biSizeImage = (getc(IN) << 16) + Info_h.biSizeImage;
		Info_h.biSizeImage = (getc(IN) << 24) + Info_h.biSizeImage;

		Info_h.biXpelsPerMeter = getc(IN);
		Info_h.biXpelsPerMeter = (getc(IN) << 8) + Info_h.biXpelsPerMeter;
		Info_h.biXpelsPerMeter = (getc(IN) << 16) + Info_h.biXpelsPerMeter;
		Info_h.biXpelsPerMeter = (getc(IN) << 24) + Info_h.biXpelsPerMeter;

		Info_h.biYpelsPerMeter = getc(IN);
		Info_h.biYpelsPerMeter = (getc(IN) << 8) + Info_h.biYpelsPerMeter;
		Info_h.biYpelsPerMeter = (getc(IN) << 16) + Info_h.biYpelsPerMeter;
		Info_h.biYpelsPerMeter = (getc(IN) << 24) + Info_h.biYpelsPerMeter;

		Info_h.biClrUsed = getc(IN);
		Info_h.biClrUsed = (getc(IN) << 8) + Info_h.biClrUsed;
		Info_h.biClrUsed = (getc(IN) << 16) + Info_h.biClrUsed;
		Info_h.biClrUsed = (getc(IN) << 24) + Info_h.biClrUsed;

		Info_h.biClrImportant = getc(IN);
		Info_h.biClrImportant = (getc(IN) << 8) + Info_h.biClrImportant;
		Info_h.biClrImportant = (getc(IN) << 16) + Info_h.biClrImportant;
		Info_h.biClrImportant = (getc(IN) << 24) + Info_h.biClrImportant;

		/* Read the data and store them in the OUT file */
    
		if (Info_h.biBitCount == 24) {
			numcomps = 3;
			color_space = CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */

			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			W = Info_h.biWidth;
			H = Info_h.biHeight;

			/* PAD = 4 - (3 * W) % 4; */
			/* PAD = (PAD == 4) ? 0 : PAD; */
			PAD = (3 * W) % 4 ? 4 - (3 * W) % 4 : 0;
			
			RGB = (unsigned char *) malloc((3 * W + PAD) * H * sizeof(unsigned char));
			
			fread(RGB, sizeof(unsigned char), (3 * W + PAD) * H, IN);
			
			index = 0;

			for(y = 0; y < (int)H; y++) {
				unsigned char *scanline = RGB + (3 * W + PAD) * (H - 1 - y);
				for(x = 0; x < (int)W; x++) {
					unsigned char *pixel = &scanline[3 * x];
					image->comps[0].data[index] = pixel[2];	/* R */
					image->comps[1].data[index] = pixel[1];	/* G */
					image->comps[2].data[index] = pixel[0];	/* B */
					index++;
				}
			}

			free(RGB);

		} else if (Info_h.biBitCount == 8 && Info_h.biCompression == 0) {
			table_R = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_G = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_B = (unsigned char *) malloc(256 * sizeof(unsigned char));
			
			for (j = 0; j < Info_h.biClrUsed; j++) {
				table_B[j] = getc(IN);
				table_G[j] = getc(IN);
				table_R[j] = getc(IN);
				getc(IN);
				if (table_R[j] != table_G[j] && table_R[j] != table_B[j] && table_G[j] != table_B[j])
					gray_scale = 0;
			}
			
			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			W = Info_h.biWidth;
			H = Info_h.biHeight;
			if (Info_h.biWidth % 2)
				W++;
			
			numcomps = gray_scale ? 1 : 3;
			color_space = gray_scale ? CLRSPC_GRAY : CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */

			RGB = (unsigned char *) malloc(W * H * sizeof(unsigned char));
			
			fread(RGB, sizeof(unsigned char), W * H, IN);
			if (gray_scale) {
				index = 0;
				for (j = 0; j < W * H; j++) {
					if ((j % W < W - 1 && Info_h.biWidth % 2) || !(Info_h.biWidth % 2)) {
						image->comps[0].data[index] = table_R[RGB[W * H - ((j) / (W) + 1) * W + (j) % (W)]];
						index++;
					}
				}

			} else {		
				index = 0;
				for (j = 0; j < W * H; j++) {
					if ((j % W < W - 1 && Info_h.biWidth % 2) || !(Info_h.biWidth % 2)) {
						unsigned char pixel_index = RGB[W * H - ((j) / (W) + 1) * W + (j) % (W)];
						image->comps[0].data[index] = table_R[pixel_index];
						image->comps[1].data[index] = table_G[pixel_index];
						image->comps[2].data[index] = table_B[pixel_index];
						index++;
					}
				}
			}
			free(RGB);
      free(table_R);
      free(table_G);
      free(table_B);
		} else if (Info_h.biBitCount == 8 && Info_h.biCompression == 1) {				
			table_R = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_G = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_B = (unsigned char *) malloc(256 * sizeof(unsigned char));
			
			for (j = 0; j < Info_h.biClrUsed; j++) {
				table_B[j] = getc(IN);
				table_G[j] = getc(IN);
				table_R[j] = getc(IN);
				getc(IN);
				if (table_R[j] != table_G[j] && table_R[j] != table_B[j] && table_G[j] != table_B[j])
					gray_scale = 0;
			}

			numcomps = gray_scale ? 1 : 3;
			color_space = gray_scale ? CLRSPC_GRAY : CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */
			
			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			RGB = (unsigned char *) malloc(Info_h.biWidth * Info_h.biHeight * sizeof(unsigned char));
            
			while (not_end_file) {
				v = getc(IN);
				if (v) {
					v2 = getc(IN);
					for (i = 0; i < (int) v; i++) {
						RGB[line * Info_h.biWidth + col] = v2;
						col++;
					}
				} else {
					v = getc(IN);
					switch (v) {
						case 0:
							col = 0;
							line++;
							break;
						case 1:
							line++;
							not_end_file = 0;
							break;
						case 2:
							fprintf(stderr,"No Delta supported\n");
							opj_image_destroy(image);
							fclose(IN);
							return NULL;
						default:
							for (i = 0; i < v; i++) {
								v2 = getc(IN);
								RGB[line * Info_h.biWidth + col] = v2;
								col++;
							}
							if (v % 2)
								v2 = getc(IN);
							break;
					}
				}
			}
			if (gray_scale) {
				index = 0;
				for (line = 0; line < Info_h.biHeight; line++) {
					for (col = 0; col < Info_h.biWidth; col++) {
						image->comps[0].data[index] = table_R[(int)RGB[(Info_h.biHeight - line - 1) * Info_h.biWidth + col]];
						index++;
					}
				}
			} else {
				index = 0;
				for (line = 0; line < Info_h.biHeight; line++) {
					for (col = 0; col < Info_h.biWidth; col++) {
						unsigned char pixel_index = (int)RGB[(Info_h.biHeight - line - 1) * Info_h.biWidth + col];
						image->comps[0].data[index] = table_R[pixel_index];
						image->comps[1].data[index] = table_G[pixel_index];
						image->comps[2].data[index] = table_B[pixel_index];
						index++;
					}
				}
			}
			free(RGB);
      free(table_R);
      free(table_G);
      free(table_B);
	} else {
		fprintf(stderr, 
			"Other system than 24 bits/pixels or 8 bits (no RLE coding) is not yet implemented [%d]\n", Info_h.biBitCount);
	}
	fclose(IN);
 }
 
 return image;
}

int imagetobmp(opj_image_t * image, const char *outfile) {
	int w, h;
	int i, pad;
	FILE *fdest = NULL;
	int adjustR, adjustG, adjustB;

	if (image->numcomps == 3 && image->comps[0].dx == image->comps[1].dx
		&& image->comps[1].dx == image->comps[2].dx
		&& image->comps[0].dy == image->comps[1].dy
		&& image->comps[1].dy == image->comps[2].dy
		&& image->comps[0].prec == image->comps[1].prec
		&& image->comps[1].prec == image->comps[2].prec) {
		
		/* -->> -->> -->> -->>    
		24 bits color	    
		<<-- <<-- <<-- <<-- */
	    
		fdest = fopen(outfile, "wb");
		if (!fdest) {
			fprintf(stderr, "ERROR -> failed to open %s for writing\n", outfile);
			return 1;
		}
	    
		w = image->comps[0].w;	    
		h = image->comps[0].h;
	    
		fprintf(fdest, "BM");
	    
		/* FILE HEADER */
		/* ------------- */
		fprintf(fdest, "%c%c%c%c",
			(unsigned char) (h * w * 3 + 3 * h * (w % 2) + 54) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2) + 54)	>> 8) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2) + 54)	>> 16) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2) + 54)	>> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (54) & 0xff, ((54) >> 8) & 0xff,((54) >> 16) & 0xff, ((54) >> 24) & 0xff);
	    
		/* INFO HEADER   */
		/* ------------- */
		fprintf(fdest, "%c%c%c%c", (40) & 0xff, ((40) >> 8) & 0xff,	((40) >> 16) & 0xff, ((40) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) ((w) & 0xff),
			(unsigned char) ((w) >> 8) & 0xff,
			(unsigned char) ((w) >> 16) & 0xff,
			(unsigned char) ((w) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) ((h) & 0xff),
			(unsigned char) ((h) >> 8) & 0xff,
			(unsigned char) ((h) >> 16) & 0xff,
			(unsigned char) ((h) >> 24) & 0xff);
		fprintf(fdest, "%c%c", (1) & 0xff, ((1) >> 8) & 0xff);
		fprintf(fdest, "%c%c", (24) & 0xff, ((24) >> 8) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) (3 * h * w + 3 * h * (w % 2)) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2)) >> 8) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2)) >> 16) & 0xff,
			(unsigned char) ((h * w * 3 + 3 * h * (w % 2)) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff, ((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff,	((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
	    
		if (image->comps[0].prec > 8) {
			adjustR = image->comps[0].prec - 8;
			printf("BMP CONVERSION: Truncating component 0 from %d bits to 8 bits\n", image->comps[0].prec);
		}
		else 
			adjustR = 0;
		if (image->comps[1].prec > 8) {
			adjustG = image->comps[1].prec - 8;
			printf("BMP CONVERSION: Truncating component 1 from %d bits to 8 bits\n", image->comps[1].prec);
		}
		else 
			adjustG = 0;
		if (image->comps[2].prec > 8) {
			adjustB = image->comps[2].prec - 8;
			printf("BMP CONVERSION: Truncating component 2 from %d bits to 8 bits\n", image->comps[2].prec);
		}
		else 
			adjustB = 0;

		for (i = 0; i < w * h; i++) {
			unsigned char rc, gc, bc;
			int r, g, b;
							
			r = image->comps[0].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
			r += (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);
			rc = (unsigned char) ((r >> adjustR)+((r >> (adjustR-1))%2));
			g = image->comps[1].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
			g += (image->comps[1].sgnd ? 1 << (image->comps[1].prec - 1) : 0);
			gc = (unsigned char) ((g >> adjustG)+((g >> (adjustG-1))%2));
			b = image->comps[2].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
			b += (image->comps[2].sgnd ? 1 << (image->comps[2].prec - 1) : 0);
			bc = (unsigned char) ((b >> adjustB)+((b >> (adjustB-1))%2));

			fprintf(fdest, "%c%c%c", bc, gc, rc);
			
			if ((i + 1) % w == 0) {
				for (pad = (3 * w) % 4 ? 4 - (3 * w) % 4 : 0; pad > 0; pad--)	/* ADD */
					fprintf(fdest, "%c", 0);
			}
		}
		fclose(fdest);
	} else {			/* Gray-scale */

		/* -->> -->> -->> -->>
		8 bits non code (Gray scale)
		<<-- <<-- <<-- <<-- */

		fdest = fopen(outfile, "wb");
		w = image->comps[0].w;	    
		h = image->comps[0].h;
	    
		fprintf(fdest, "BM");
	    
		/* FILE HEADER */
		/* ------------- */
		fprintf(fdest, "%c%c%c%c", (unsigned char) (h * w + 54 + 1024 + h * (w % 2)) & 0xff,
			(unsigned char) ((h * w + 54 + 1024 + h * (w % 2)) >> 8) & 0xff,
			(unsigned char) ((h * w + 54 + 1024 + h * (w % 2)) >> 16) & 0xff,
			(unsigned char) ((h * w + 54 + 1024 + w * (w % 2)) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (54 + 1024) & 0xff, ((54 + 1024) >> 8) & 0xff, 
			((54 + 1024) >> 16) & 0xff,
			((54 + 1024) >> 24) & 0xff);
	    
		/* INFO HEADER */
		/* ------------- */
		fprintf(fdest, "%c%c%c%c", (40) & 0xff, ((40) >> 8) & 0xff,	((40) >> 16) & 0xff, ((40) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) ((w) & 0xff),
			(unsigned char) ((w) >> 8) & 0xff,
			(unsigned char) ((w) >> 16) & 0xff,
			(unsigned char) ((w) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) ((h) & 0xff),
			(unsigned char) ((h) >> 8) & 0xff,
			(unsigned char) ((h) >> 16) & 0xff,
			(unsigned char) ((h) >> 24) & 0xff);
		fprintf(fdest, "%c%c", (1) & 0xff, ((1) >> 8) & 0xff);
		fprintf(fdest, "%c%c", (8) & 0xff, ((8) >> 8) & 0xff);
		fprintf(fdest, "%c%c%c%c", (0) & 0xff, ((0) >> 8) & 0xff, ((0) >> 16) & 0xff, ((0) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (unsigned char) (h * w + h * (w % 2)) & 0xff,
			(unsigned char) ((h * w + h * (w % 2)) >> 8) &	0xff,
			(unsigned char) ((h * w + h * (w % 2)) >> 16) &	0xff,
			(unsigned char) ((h * w + h * (w % 2)) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff,	((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (7834) & 0xff, ((7834) >> 8) & 0xff,	((7834) >> 16) & 0xff, ((7834) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (256) & 0xff, ((256) >> 8) & 0xff, ((256) >> 16) & 0xff, ((256) >> 24) & 0xff);
		fprintf(fdest, "%c%c%c%c", (256) & 0xff, ((256) >> 8) & 0xff, ((256) >> 16) & 0xff, ((256) >> 24) & 0xff);

		if (image->comps[0].prec > 8) {
			adjustR = image->comps[0].prec - 8;
			printf("BMP CONVERSION: Truncating component 0 from %d bits to 8 bits\n", image->comps[0].prec);
		}else 
			adjustR = 0;

		for (i = 0; i < 256; i++) {
			fprintf(fdest, "%c%c%c%c", i, i, i, 0);
		}

		for (i = 0; i < w * h; i++) {
			unsigned char rc;
			int r;
			
			r = image->comps[0].data[w * h - ((i) / (w) + 1) * w + (i) % (w)];
			r += (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);
			rc = (unsigned char) ((r >> adjustR)+((r >> (adjustR-1))%2));
			
			fprintf(fdest, "%c", rc);

			if ((i + 1) % w == 0) {
				for (pad = w % 4 ? 4 - w % 4 : 0; pad > 0; pad--)	/* ADD */
					fprintf(fdest, "%c", 0);
			}
		}
		fclose(fdest);
	}

	return 0;
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

opj_image_t* pgxtoimage(const char *filename, opj_cparameters_t *parameters) {
	FILE *f = NULL;
	int w, h, prec;
	int i, numcomps, max;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm;	/* maximum of 1 component  */
	opj_image_t * image = NULL;

	char endian1,endian2,sign;
	char signtmp[32];

	char temp[32];
	int bigendian;
	opj_image_comp_t *comp = NULL;

	numcomps = 1;
	color_space = CLRSPC_GRAY;

	memset(&cmptparm, 0, sizeof(opj_image_cmptparm_t));

	max = 0;

	f = fopen(filename, "rb");
	if (!f) {
	  fprintf(stderr, "Failed to open %s for reading !\n", filename);
	  return NULL;
	}

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
		bigendian = 1;
	} else if (endian2=='M' && endian1=='L') {
		bigendian = 0;
	} else {
		fprintf(stderr, "Bad pgx header, please check input file\n");
		return NULL;
	}

	/* initialize image component */

	cmptparm.x0 = parameters->image_offset_x0;
	cmptparm.y0 = parameters->image_offset_y0;
	cmptparm.w = !cmptparm.x0 ? (w - 1) * parameters->subsampling_dx + 1 : cmptparm.x0 + (w - 1) * parameters->subsampling_dx + 1;
	cmptparm.h = !cmptparm.y0 ? (h - 1) * parameters->subsampling_dy + 1 : cmptparm.y0 + (h - 1) * parameters->subsampling_dy + 1;
	
	if (sign == '-') {
		cmptparm.sgnd = 1;
	} else {
		cmptparm.sgnd = 0;
	}
	cmptparm.prec = prec;
	cmptparm.bpp = prec;
	cmptparm.dx = parameters->subsampling_dx;
	cmptparm.dy = parameters->subsampling_dy;
	
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm, color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}
	/* set image offset and reference grid */
	image->x0 = cmptparm.x0;
	image->y0 = cmptparm.x0;
	image->x1 = cmptparm.w;
	image->y1 = cmptparm.h;

	/* set image data */

	comp = &image->comps[0];

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
	fclose(f);
	comp->bpp = int_floorlog2(max) + 1;

	return image;
}

int imagetopgx(opj_image_t * image, const char *outfile) {
	int w, h;
	int i, j, compno;
	FILE *fdest = NULL;

	for (compno = 0; compno < image->numcomps; compno++) {
		opj_image_comp_t *comp = &image->comps[compno];
		char bname[256]; /* buffer for name */
    char *name = bname; /* pointer */
    int nbytes = 0;
    const size_t olen = strlen(outfile);
    const size_t dotpos = olen - 4;
    const size_t total = dotpos + 1 + 1 + 4; /* '-' + '[1-3]' + '.pgx' */
    if( outfile[dotpos] != '.' ) {
      /* `pgx` was recognized but there is no dot at expected position */
      fprintf(stderr, "ERROR -> Impossible happen." );
      return 1;
      }
    if( total > 256 ) {
      name = (char*)malloc(total+1);
      }
    strncpy(name, outfile, dotpos);
		if (image->numcomps > 1) {
			sprintf(name+dotpos, "-%d.pgx", compno);
		} else {
			strcpy(name+dotpos, ".pgx");
		}
		fdest = fopen(name, "wb");
		if (!fdest) {
			fprintf(stderr, "ERROR -> failed to open %s for writing\n", name);
			return 1;
		}
    /* dont need name anymore */
    if( total > 256 ) {
      free(name);
      }

		w = image->comps[compno].w;
		h = image->comps[compno].h;
	    
		fprintf(fdest, "PG ML %c %d %d %d\n", comp->sgnd ? '-' : '+', comp->prec, w, h);
		if (comp->prec <= 8) {
			nbytes = 1;
		} else if (comp->prec <= 16) {
			nbytes = 2;
		} else {
			nbytes = 4;
		}
		for (i = 0; i < w * h; i++) {
			int v = image->comps[compno].data[i];
			for (j = nbytes - 1; j >= 0; j--) {
				char byte = (char) (v >> (j * 8));
				fwrite(&byte, 1, 1, fdest);
			}
		}
		fclose(fdest);
	}

	return 0;
}

/* -->> -->> -->> -->>

PNM IMAGE FORMAT

<<-- <<-- <<-- <<-- */

opj_image_t* pnmtoimage(const char *filename, opj_cparameters_t *parameters) {
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	FILE *f = NULL;
	int i, compno, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm[3];	/* maximum of 3 components */
	opj_image_t * image = NULL;
	char value;
	
	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		return 0;
	}

	if (fgetc(f) != 'P')
		return 0;
	value = fgetc(f);

		switch(value) {
			case '2':	/* greyscale image type */
			case '5':
				numcomps = 1;
				color_space = CLRSPC_GRAY;
				break;
				
			case '3':	/* RGB image type */
			case '6':
				numcomps = 3;
				color_space = CLRSPC_SRGB;
				break;
				
			default:
				fclose(f);
				return NULL;
		}
		
		fgetc(f);
		
		/* skip comments */
		while(fgetc(f) == '#') while(fgetc(f) != '\n');
		
		fseek(f, -1, SEEK_CUR);
		fscanf(f, "%d %d\n255", &w, &h);			
		fgetc(f);	/* <cr><lf> */
		
	/* initialize image components */
	memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
	for(i = 0; i < numcomps; i++) {
		cmptparm[i].prec = 8;
		cmptparm[i].bpp = 8;
		cmptparm[i].sgnd = 0;
		cmptparm[i].dx = subsampling_dx;
		cmptparm[i].dy = subsampling_dy;
		cmptparm[i].w = w;
		cmptparm[i].h = h;
	}
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm[0], color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}

	/* set image offset and reference grid */
	image->x0 = parameters->image_offset_x0;
	image->y0 = parameters->image_offset_y0;
	image->x1 = parameters->image_offset_x0 + (w - 1) *	subsampling_dx + 1;
	image->y1 = parameters->image_offset_y0 + (h - 1) *	subsampling_dy + 1;

	/* set image data */

	if ((value == '2') || (value == '3')) {	/* ASCII */
		for (i = 0; i < w * h; i++) {
			for(compno = 0; compno < numcomps; compno++) {
				unsigned int index = 0;
				fscanf(f, "%u", &index);
				/* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
				image->comps[compno].data[i] = index;
			}
		}
	} else if ((value == '5') || (value == '6')) {	/* BINARY */
		for (i = 0; i < w * h; i++) {
			for(compno = 0; compno < numcomps; compno++) {
				unsigned char index = 0;
				fread(&index, 1, 1, f);
				/* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
				image->comps[compno].data[i] = index;
			}
		}
	}

	fclose(f);

	return image;
}

int imagetopnm(opj_image_t * image, const char *outfile) {
	int w, wr, h, hr, max;
	int i, compno;
	int adjustR, adjustG, adjustB, adjustX;
	FILE *fdest = NULL;
	char S2;
	const char *tmp = outfile;

	while (*tmp) {
		tmp++;
	}
	tmp--;
	tmp--;
	S2 = *tmp;

	if (image->numcomps == 3 && image->comps[0].dx == image->comps[1].dx
		&& image->comps[1].dx == image->comps[2].dx
		&& image->comps[0].dy == image->comps[1].dy
		&& image->comps[1].dy == image->comps[2].dy
		&& image->comps[0].prec == image->comps[1].prec
		&& image->comps[1].prec == image->comps[2].prec
		&& S2 !='g' && S2 !='G') {

		fdest = fopen(outfile, "wb");
		if (!fdest) {
			fprintf(stderr, "ERROR -> failed to open %s for writing\n", outfile);
			return 1;
		}

		w = int_ceildiv(image->x1 - image->x0, image->comps[0].dx);
		wr = image->comps[0].w;
        
		h = int_ceildiv(image->y1 - image->y0, image->comps[0].dy);
		hr = image->comps[0].h;
	    
		max = image->comps[0].prec > 8 ? 255 : (1 << image->comps[0].prec) - 1;
	    
		image->comps[0].x0 = int_ceildivpow2(image->comps[0].x0 - int_ceildiv(image->x0, image->comps[0].dx), image->comps[0].factor);
		image->comps[0].y0 = int_ceildivpow2(image->comps[0].y0 -	int_ceildiv(image->y0, image->comps[0].dy), image->comps[0].factor);

		fprintf(fdest, "P6\n%d %d\n%d\n", wr, hr, max);

		if (image->comps[0].prec > 8) {
			adjustR = image->comps[0].prec - 8;
			printf("PNM CONVERSION: Truncating component 0 from %d bits to 8 bits\n", image->comps[0].prec);
		}
		else 
			adjustR = 0;
		if (image->comps[1].prec > 8) {
			adjustG = image->comps[1].prec - 8;
			printf("PNM CONVERSION: Truncating component 1 from %d bits to 8 bits\n", image->comps[1].prec);
		}
		else 
			adjustG = 0;
		if (image->comps[2].prec > 8) {
			adjustB = image->comps[2].prec - 8;
			printf("PNM CONVERSION: Truncating component 2 from %d bits to 8 bits\n", image->comps[2].prec);
		}
		else 
			adjustB = 0;


		for (i = 0; i < wr * hr; i++) {
			int r, g, b;
			unsigned char rc,gc,bc;
			r = image->comps[0].data[i];
			r += (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);
			rc = (unsigned char) ((r >> adjustR)+((r >> (adjustR-1))%2));

			g = image->comps[1].data[i];
			g += (image->comps[1].sgnd ? 1 << (image->comps[1].prec - 1) : 0);
			gc = (unsigned char) ((g >> adjustG)+((g >> (adjustG-1))%2));
			
			b = image->comps[2].data[i];
			b += (image->comps[2].sgnd ? 1 << (image->comps[2].prec - 1) : 0);
			bc = (unsigned char) ((b >> adjustB)+((b >> (adjustB-1))%2));
			
			fprintf(fdest, "%c%c%c", rc, gc, bc);
		}
		fclose(fdest);

	} else {
		int ncomp=(S2=='g' || S2=='G')?1:image->numcomps;
		if (image->numcomps > ncomp) {
			fprintf(stderr,"WARNING -> [PGM files] Only the first component\n");
			fprintf(stderr,"           is written to the file\n");
		}
		for (compno = 0; compno < ncomp; compno++) {
			char name[256];
			if (ncomp > 1) {
				sprintf(name, "%d.%s", compno, outfile);
			} else {
				sprintf(name, "%s", outfile);
			}
			
			fdest = fopen(name, "wb");
			if (!fdest) {
				fprintf(stderr, "ERROR -> failed to open %s for writing\n", name);
				return 1;
			}
            
			w = int_ceildiv(image->x1 - image->x0, image->comps[compno].dx);
			wr = image->comps[compno].w;
			
			h = int_ceildiv(image->y1 - image->y0, image->comps[compno].dy);
			hr = image->comps[compno].h;
			
			max = image->comps[compno].prec > 8 ? 255 : (1 << image->comps[compno].prec) - 1;
			
			image->comps[compno].x0 = int_ceildivpow2(image->comps[compno].x0 - int_ceildiv(image->x0, image->comps[compno].dx), image->comps[compno].factor);
			image->comps[compno].y0 = int_ceildivpow2(image->comps[compno].y0 - int_ceildiv(image->y0, image->comps[compno].dy), image->comps[compno].factor);
			
			fprintf(fdest, "P5\n%d %d\n%d\n", wr, hr, max);
			
			if (image->comps[compno].prec > 8) {
				adjustX = image->comps[0].prec - 8;
				printf("PNM CONVERSION: Truncating component %d from %d bits to 8 bits\n",compno, image->comps[compno].prec);
			}
			else 
				adjustX = 0;
			
			for (i = 0; i < wr * hr; i++) {
				int l;
				unsigned char lc;
				l = image->comps[compno].data[i];
				l += (image->comps[compno].sgnd ? 1 << (image->comps[compno].prec - 1) : 0);
				lc = (unsigned char) ((l >> adjustX)+((l >> (adjustX-1))%2));
				fprintf(fdest, "%c", lc);
			}
			fclose(fdest);
		}
	}

	return 0;
}

#ifdef HAVE_LIBTIFF
/* -->> -->> -->> -->>

	TIFF IMAGE FORMAT

 <<-- <<-- <<-- <<-- */

typedef struct tiff_infoheader{
	DWORD tiWidth;  // Width of Image in pixel
	DWORD tiHeight; // Height of Image in pixel
	DWORD tiPhoto;	// Photometric
	WORD  tiBps;	// Bits per sample
	WORD  tiSf;		// Sample Format
	WORD  tiSpp;	// Sample per pixel 1-bilevel,gray scale , 2- RGB
	WORD  tiPC;	// Planar config (1-Interleaved, 2-Planarcomp)
}tiff_infoheader_t;

int imagetotif(opj_image_t * image, const char *outfile) {
	int width, height, imgsize;
	int bps,index,adjust = 0;
	int last_i=0;
	TIFF *tif;
	tdata_t buf;
	tstrip_t strip;
	tsize_t strip_size;

	if (image->numcomps == 3 && image->comps[0].dx == image->comps[1].dx
		&& image->comps[1].dx == image->comps[2].dx
		&& image->comps[0].dy == image->comps[1].dy
		&& image->comps[1].dy == image->comps[2].dy
		&& image->comps[0].prec == image->comps[1].prec
		&& image->comps[1].prec == image->comps[2].prec) {

			/* -->> -->> -->>    
			RGB color	    
			<<-- <<-- <<-- */

			tif = TIFFOpen(outfile, "wb"); 
			if (!tif) {
				fprintf(stderr, "ERROR -> failed to open %s for writing\n", outfile);
				return 1;
			}

			width	= image->comps[0].w;
			height	= image->comps[0].h;
			imgsize = width * height ;
			bps		= image->comps[0].prec;
			/* Set tags */
			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bps);
			TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);

			/* Get a buffer for the data */
			strip_size=TIFFStripSize(tif);
			buf = _TIFFmalloc(strip_size);
			index=0;		
			adjust = image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0;
			for (strip = 0; strip < TIFFNumberOfStrips(tif); strip++) {
				unsigned char *dat8;
				tsize_t i, ssize;
				ssize = TIFFStripSize(tif);
				dat8 = (unsigned char*)buf;
				if (image->comps[0].prec == 8){
					for (i=0; i<ssize-2; i+=3) {	// 8 bits per pixel 
						int r = 0,g = 0,b = 0;
						if(index < imgsize){
							r = image->comps[0].data[index];
							g = image->comps[1].data[index];
							b = image->comps[2].data[index];
							if (image->comps[0].sgnd){			
								r += adjust;
								g += adjust;
								b += adjust;
							}
							dat8[i+0] = r ;	// R 
							dat8[i+1] = g ;	// G 
							dat8[i+2] = b ;	// B 
							index++;
							last_i = i+3;
						}else
							break;
					}
					if(last_i < ssize){
						for (i=last_i; i<ssize; i+=3) {	// 8 bits per pixel 
							int r = 0,g = 0,b = 0;
							if(index < imgsize){
								r = image->comps[0].data[index];
								g = image->comps[1].data[index];
								b = image->comps[2].data[index];
								if (image->comps[0].sgnd){			
									r += adjust;
									g += adjust;
									b += adjust;
								}
								dat8[i+0] = r ;	// R 
								if(i+1 <ssize) dat8[i+1] = g ;	else break;// G 
								if(i+2 <ssize) dat8[i+2] = b ;	else break;// B 
								index++;
							}else
								break;
						}
					}
				}else if (image->comps[0].prec == 12){
					for (i=0; i<ssize-8; i+=9) {	// 12 bits per pixel 
						int r = 0,g = 0,b = 0;
						int r1 = 0,g1 = 0,b1 = 0;
						if((index < imgsize)&(index+1 < imgsize)){
							r  = image->comps[0].data[index];
							g  = image->comps[1].data[index];
							b  = image->comps[2].data[index];
							r1 = image->comps[0].data[index+1];
							g1 = image->comps[1].data[index+1];
							b1 = image->comps[2].data[index+1];
							if (image->comps[0].sgnd){														
								r  += adjust;
								g  += adjust;
								b  += adjust;
								r1 += adjust;
								g1 += adjust;
								b1 += adjust;
							}
							dat8[i+0] = (r >> 4);
							dat8[i+1] = ((r & 0x0f) << 4 )|((g >> 8)& 0x0f);
							dat8[i+2] = g ;		
							dat8[i+3] = (b >> 4);
							dat8[i+4] = ((b & 0x0f) << 4 )|((r1 >> 8)& 0x0f);
							dat8[i+5] = r1;		
							dat8[i+6] = (g1 >> 4);
							dat8[i+7] = ((g1 & 0x0f)<< 4 )|((b1 >> 8)& 0x0f);
							dat8[i+8] = b1;
							index+=2;
							last_i = i+9;
						}else
							break;
					}
					if(last_i < ssize){
						for (i= last_i; i<ssize; i+=9) {	// 12 bits per pixel 
							int r = 0,g = 0,b = 0;
							int r1 = 0,g1 = 0,b1 = 0;
							if((index < imgsize)&(index+1 < imgsize)){
								r  = image->comps[0].data[index];
								g  = image->comps[1].data[index];
								b  = image->comps[2].data[index];
								r1 = image->comps[0].data[index+1];
								g1 = image->comps[1].data[index+1];
								b1 = image->comps[2].data[index+1];
								if (image->comps[0].sgnd){														
									r  += adjust;
									g  += adjust;
									b  += adjust;
									r1 += adjust;
									g1 += adjust;
									b1 += adjust;
								}
								dat8[i+0] = (r >> 4);
								if(i+1 <ssize) dat8[i+1] = ((r & 0x0f) << 4 )|((g >> 8)& 0x0f); else break;
								if(i+2 <ssize) dat8[i+2] = g ;			else break;
								if(i+3 <ssize) dat8[i+3] = (b >> 4);	else break;
								if(i+4 <ssize) dat8[i+4] = ((b & 0x0f) << 4 )|((r1 >> 8)& 0x0f);else break;
								if(i+5 <ssize) dat8[i+5] = r1;			else break;
								if(i+6 <ssize) dat8[i+6] = (g1 >> 4);	else break;
								if(i+7 <ssize) dat8[i+7] = ((g1 & 0x0f)<< 4 )|((b1 >> 8)& 0x0f);else break;
								if(i+8 <ssize) dat8[i+8] = b1;			else break;
								index+=2;
							}else
								break;
						}
					}
				}else if (image->comps[0].prec == 16){
					for (i=0 ; i<ssize-5 ; i+=6) {	// 16 bits per pixel 
						int r = 0,g = 0,b = 0;
						if(index < imgsize){
							r = image->comps[0].data[index];
							g = image->comps[1].data[index];
							b = image->comps[2].data[index];
							if (image->comps[0].sgnd){
							r += adjust;
							g += adjust;
							b += adjust;
							}
							dat8[i+0] =  r;//LSB
							dat8[i+1] = (r >> 8);//MSB	 
							dat8[i+2] =  g;		
							dat8[i+3] = (g >> 8);
							dat8[i+4] =  b;	
							dat8[i+5] = (b >> 8);
							index++;
							last_i = i+6;
						}else
							break; 
					}
					if(last_i < ssize){
						for (i=0 ; i<ssize ; i+=6) {	// 16 bits per pixel 
							int r = 0,g = 0,b = 0;
							if(index < imgsize){
								r = image->comps[0].data[index];
								g = image->comps[1].data[index];
								b = image->comps[2].data[index];
								if (image->comps[0].sgnd){
									r += adjust;
									g += adjust;
									b += adjust;
								}
								dat8[i+0] =  r;//LSB
								if(i+1 <ssize) dat8[i+1] = (r >> 8);else break;//MSB	 
								if(i+2 <ssize) dat8[i+2] =  g;		else break;
								if(i+3 <ssize) dat8[i+3] = (g >> 8);else break;
								if(i+4 <ssize) dat8[i+4] =  b;		else break;
								if(i+5 <ssize) dat8[i+5] = (b >> 8);else break;
								index++;
							}else
								break; 
						}						
					}
				}else{
					fprintf(stderr,"Bits=%d, Only 8,12,16 bits implemented\n",image->comps[0].prec);
					fprintf(stderr,"Aborting\n");
					return 1;
				}
				(void)TIFFWriteEncodedStrip(tif, strip, (void*)buf, strip_size);
			}
			_TIFFfree((void*)buf);
			TIFFClose(tif);
		}else if (image->numcomps == 1){
			/* -->> -->> -->>    
			Black and White	    
			<<-- <<-- <<-- */

			tif = TIFFOpen(outfile, "wb"); 
			if (!tif) {
				fprintf(stderr, "ERROR -> failed to open %s for writing\n", outfile);
				return 1;
			}

			width	= image->comps[0].w;
			height	= image->comps[0].h;
			imgsize = width * height;
			bps		= image->comps[0].prec;

			/* Set tags */
			TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
			TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
			TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bps);
			TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
			TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);

			/* Get a buffer for the data */
			strip_size = TIFFStripSize(tif);
			buf = _TIFFmalloc(strip_size);
			index = 0;			
			for (strip = 0; strip < TIFFNumberOfStrips(tif); strip++) {
				unsigned char *dat8;
				tsize_t i;
				dat8 = (unsigned char*)buf;
				if (image->comps[0].prec == 8){
					for (i=0; i<TIFFStripSize(tif); i+=1) {	// 8 bits per pixel 
						if(index < imgsize){
							int r = 0;
							r = image->comps[0].data[index];
							if (image->comps[0].sgnd){
								r  += adjust;
							}
							dat8[i+0] = r;
							index++;
						}else
							break; 
					}
				}else if (image->comps[0].prec == 12){
					for (i = 0; i<TIFFStripSize(tif); i+=3) {	// 12 bits per pixel 
						if(index < imgsize){
							int r = 0, r1 = 0;
							r  = image->comps[0].data[index];
							r1 = image->comps[0].data[index+1];
							if (image->comps[0].sgnd){
								r  += adjust;
								r1 += adjust;
							}
							dat8[i+0] = (r >> 4);
							dat8[i+1] = ((r & 0x0f) << 4 )|((r1 >> 8)& 0x0f);
							dat8[i+2] = r1 ;
							index+=2;
						}else
							break; 
					}
				}else if (image->comps[0].prec == 16){
					for (i=0; i<TIFFStripSize(tif); i+=2) {	// 16 bits per pixel 
						if(index < imgsize){
							int r = 0;
							r = image->comps[0].data[index];
							if (image->comps[0].sgnd){
								r  += adjust;
							}
							dat8[i+0] = r;
							dat8[i+1] = r >> 8;
							index++;
						}else
							break; 
					}
				}else{
					fprintf(stderr,"TIFF file creation. Bits=%d, Only 8,12,16 bits implemented\n",image->comps[0].prec);
					fprintf(stderr,"Aborting\n");
					return 1;
				}
				(void)TIFFWriteEncodedStrip(tif, strip, (void*)buf, strip_size);
			}
			_TIFFfree(buf);
			TIFFClose(tif);
		}else{
			fprintf(stderr,"TIFF file creation. Bad color format. Only RGB & Grayscale has been implemented\n");
			fprintf(stderr,"Aborting\n");
			return 1;
		}
		return 0;
}

opj_image_t* tiftoimage(const char *filename, opj_cparameters_t *parameters)
{
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;
	TIFF *tif;
	tiff_infoheader_t Info;
	tdata_t buf;
	tstrip_t strip;
	tsize_t strip_size;
	int j, numcomps, w, h,index;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm[3];
	opj_image_t * image = NULL;
	int imgsize = 0;

	tif = TIFFOpen(filename, "r");

	if (!tif) {
		fprintf(stderr, "Failed to open %s for reading\n", filename);
		return 0;
	}

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &Info.tiWidth);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &Info.tiHeight);
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &Info.tiBps);
	TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &Info.tiSf);
	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &Info.tiSpp);
	Info.tiPhoto = 0;
	TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &Info.tiPhoto);
	TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &Info.tiPC);
	w= Info.tiWidth;
	h= Info.tiHeight;
	
	if (Info.tiPhoto == 2) { 
		/* -->> -->> -->>    
		RGB color	    
		<<-- <<-- <<-- */

		numcomps = 3;
		color_space = CLRSPC_SRGB;
		/* initialize image components*/ 
		memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
		for(j = 0; j < numcomps; j++) {
			if (parameters->cp_cinema) {
				cmptparm[j].prec = 12;
				cmptparm[j].bpp = 12;
			}else{
				cmptparm[j].prec = Info.tiBps;
				cmptparm[j].bpp = Info.tiBps;
			}
			cmptparm[j].sgnd = 0;
			cmptparm[j].dx = subsampling_dx;
			cmptparm[j].dy = subsampling_dy;
			cmptparm[j].w = w;
			cmptparm[j].h = h;
		}
		/* create the image*/ 
		image = opj_image_create(numcomps, &cmptparm[0], color_space);
		if(!image) {
			TIFFClose(tif);
			return NULL;
		}

		/* set image offset and reference grid */
		image->x0 = parameters->image_offset_x0;
		image->y0 = parameters->image_offset_y0;
		image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
		image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

		buf = _TIFFmalloc(TIFFStripSize(tif));
		strip_size=0;
		strip_size=TIFFStripSize(tif);
		index = 0;
		imgsize = image->comps[0].w * image->comps[0].h ;
		/* Read the Image components*/
		for (strip = 0; strip < TIFFNumberOfStrips(tif); strip++) {
			unsigned char *dat8;
			int i, ssize;
			ssize = TIFFReadEncodedStrip(tif, strip, buf, strip_size);
			dat8 = (unsigned char*)buf;

			if (Info.tiBps==12){
				for (i=0; i<ssize; i+=9) {	/*12 bits per pixel*/
					if((index < imgsize)&(index+1 < imgsize)){
						image->comps[0].data[index]   = ( dat8[i+0]<<4 )		|(dat8[i+1]>>4);
						image->comps[1].data[index]   = ((dat8[i+1]& 0x0f)<< 8)	| dat8[i+2];
						image->comps[2].data[index]   = ( dat8[i+3]<<4)			|(dat8[i+4]>>4);
						image->comps[0].data[index+1] = ((dat8[i+4]& 0x0f)<< 8)	| dat8[i+5];
						image->comps[1].data[index+1] = ( dat8[i+6] <<4)		|(dat8[i+7]>>4);
						image->comps[2].data[index+1] = ((dat8[i+7]& 0x0f)<< 8)	| dat8[i+8];
						index+=2;
					}else
						break;
				}
			}
			else if( Info.tiBps==16){
				for (i=0; i<ssize; i+=6) {	/* 16 bits per pixel */
					if(index < imgsize){
						image->comps[0].data[index] = ( dat8[i+1] << 8 ) | dat8[i+0]; // R 
						image->comps[1].data[index] = ( dat8[i+3] << 8 ) | dat8[i+2]; // G 
						image->comps[2].data[index] = ( dat8[i+5] << 8 ) | dat8[i+4]; // B 
						if(parameters->cp_cinema){/* Rounding to 12 bits*/
							image->comps[0].data[index] = (image->comps[0].data[index] + 0x08) >> 4 ;
							image->comps[1].data[index] = (image->comps[1].data[index] + 0x08) >> 4 ;
							image->comps[2].data[index] = (image->comps[2].data[index] + 0x08) >> 4 ;
						}
						index++;
					}else
						break;
				}
			}
			else if ( Info.tiBps==8){
				for (i=0; i<ssize; i+=3) {	/* 8 bits per pixel */
					if(index < imgsize){
						image->comps[0].data[index] = dat8[i+0];// R 
						image->comps[1].data[index] = dat8[i+1];// G 
						image->comps[2].data[index] = dat8[i+2];// B 
						if(parameters->cp_cinema){/* Rounding to 12 bits*/
							image->comps[0].data[index] = image->comps[0].data[index] << 4 ;
							image->comps[1].data[index] = image->comps[1].data[index] << 4 ;
							image->comps[2].data[index] = image->comps[2].data[index] << 4 ;
						}
						index++;
					}else
						break;
				}
			}
			else{
				fprintf(stderr,"TIFF file creation. Bits=%d, Only 8,12,16 bits implemented\n",Info.tiBps);
				fprintf(stderr,"Aborting\n");
				return NULL;
			}
		}

		_TIFFfree(buf);
		TIFFClose(tif);
	}else if(Info.tiPhoto == 1) { 
		/* -->> -->> -->>    
		Black and White
		<<-- <<-- <<-- */

		numcomps = 1;
		color_space = CLRSPC_GRAY;
		/* initialize image components*/ 
		memset(&cmptparm[0], 0, sizeof(opj_image_cmptparm_t));
		cmptparm[0].prec = Info.tiBps;
		cmptparm[0].bpp = Info.tiBps;
		cmptparm[0].sgnd = 0;
		cmptparm[0].dx = subsampling_dx;
		cmptparm[0].dy = subsampling_dy;
		cmptparm[0].w = w;
		cmptparm[0].h = h;

		/* create the image*/ 
		image = opj_image_create(numcomps, &cmptparm[0], color_space);
		if(!image) {
			TIFFClose(tif);
			return NULL;
		}
		/* set image offset and reference grid */
		image->x0 = parameters->image_offset_x0;
		image->y0 = parameters->image_offset_y0;
		image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
		image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

		buf = _TIFFmalloc(TIFFStripSize(tif));
		strip_size = 0;
		strip_size = TIFFStripSize(tif);
		index = 0;
		imgsize = image->comps[0].w * image->comps[0].h ;
		/* Read the Image components*/
		for (strip = 0; strip < TIFFNumberOfStrips(tif); strip++) {
			unsigned char *dat8;
			int i, ssize;
			ssize = TIFFReadEncodedStrip(tif, strip, buf, strip_size);
			dat8 = (unsigned char*)buf;

			if (Info.tiBps==12){
				for (i=0; i<ssize; i+=3) {	/* 12 bits per pixel*/
					if(index < imgsize){
						image->comps[0].data[index]   = ( dat8[i+0]<<4 )		|(dat8[i+1]>>4) ;
						image->comps[0].data[index+1] = ((dat8[i+1]& 0x0f)<< 8)	| dat8[i+2];
						index+=2;
					}else
						break;
				}
			}
			else if( Info.tiBps==16){
				for (i=0; i<ssize; i+=2) {	/* 16 bits per pixel */
					if(index < imgsize){
						image->comps[0].data[index] = ( dat8[i+1] << 8 ) | dat8[i+0];
						index++;
					}else
						break;
				}
			}
			else if ( Info.tiBps==8){
				for (i=0; i<ssize; i+=1) {	/* 8 bits per pixel */
					if(index < imgsize){
						image->comps[0].data[index] = dat8[i+0];
						index++;
					}else
						break;
				}
			}
			else{
				fprintf(stderr,"TIFF file creation. Bits=%d, Only 8,12,16 bits implemented\n",Info.tiBps);
				fprintf(stderr,"Aborting\n");
				return NULL;
			}
		}

		_TIFFfree(buf);
		TIFFClose(tif);
	}else{
		fprintf(stderr,"TIFF file creation. Bad color format. Only RGB & Grayscale has been implemented\n");
		fprintf(stderr,"Aborting\n");
		return NULL;
	}
	return image;
}

#endif /* HAVE_LIBTIFF */

/* -->> -->> -->> -->>

	RAW IMAGE FORMAT

 <<-- <<-- <<-- <<-- */

opj_image_t* rawtoimage(const char *filename, opj_cparameters_t *parameters, raw_cparameters_t *raw_cp) {
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	FILE *f = NULL;
	int i, compno, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t *cmptparm;	
	opj_image_t * image = NULL;
	unsigned short ch;
	
	if((! (raw_cp->rawWidth & raw_cp->rawHeight & raw_cp->rawComp & raw_cp->rawBitDepth)) == 0)
	{
		fprintf(stderr,"\nError: invalid raw image parameters\n");
		fprintf(stderr,"Please use the Format option -F:\n");
		fprintf(stderr,"-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
		fprintf(stderr,"Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
		fprintf(stderr,"Aborting\n");
		return NULL;
	}

	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		fprintf(stderr,"Aborting\n");
		return NULL;
	}
	numcomps = raw_cp->rawComp;
	color_space = CLRSPC_SRGB;
	w = raw_cp->rawWidth;
	h = raw_cp->rawHeight;
	cmptparm = (opj_image_cmptparm_t*) malloc(numcomps * sizeof(opj_image_cmptparm_t));
	
	/* initialize image components */	
	memset(&cmptparm[0], 0, numcomps * sizeof(opj_image_cmptparm_t));
	for(i = 0; i < numcomps; i++) {		
		cmptparm[i].prec = raw_cp->rawBitDepth;
		cmptparm[i].bpp = raw_cp->rawBitDepth;
		cmptparm[i].sgnd = raw_cp->rawSigned;
		cmptparm[i].dx = subsampling_dx;
		cmptparm[i].dy = subsampling_dy;
		cmptparm[i].w = w;
		cmptparm[i].h = h;
	}
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm[0], color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}
	/* set image offset and reference grid */
	image->x0 = parameters->image_offset_x0;
	image->y0 = parameters->image_offset_y0;
	image->x1 = parameters->image_offset_x0 + (w - 1) *	subsampling_dx + 1;
	image->y1 = parameters->image_offset_y0 + (h - 1) *	subsampling_dy + 1;

	if(raw_cp->rawBitDepth <= 8)
	{
		unsigned char value = 0;
		for(compno = 0; compno < numcomps; compno++) {
			for (i = 0; i < w * h; i++) {
				if (!fread(&value, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				image->comps[compno].data[i] = raw_cp->rawSigned?(char)value:value;
			}
		}
	}
	else if(raw_cp->rawBitDepth <= 16)
	{
		unsigned short value;
		for(compno = 0; compno < numcomps; compno++) {
			for (i = 0; i < w * h; i++) {
				unsigned char temp;
				if (!fread(&temp, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				value = temp << 8;
				if (!fread(&temp, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				value += temp;
				image->comps[compno].data[i] = raw_cp->rawSigned?(short)value:value;
			}
		}
	}
	else {
		fprintf(stderr,"OpenJPEG cannot encode raw components with bit depth higher than 16 bits.\n");
		return NULL;
	}

	if (fread(&ch, 1, 1, f)) {
		fprintf(stderr,"Warning. End of raw file not reached... processing anyway\n");
	}
	fclose(f);

	return image;
}

int imagetoraw(opj_image_t * image, const char *outfile)
{
	FILE *rawFile = NULL;
	int compno;
	int w, h;
	int line, row;
	int *ptr;

	if((image->numcomps * image->x1 * image->y1) == 0)
	{
		fprintf(stderr,"\nError: invalid raw image parameters\n");
		return 1;
	}

	rawFile = fopen(outfile, "wb");
	if (!rawFile) {
		fprintf(stderr, "Failed to open %s for writing !!\n", outfile);
		return 1;
	}

	fprintf(stdout,"Raw image characteristics: %d components\n", image->numcomps);

	for(compno = 0; compno < image->numcomps; compno++)
	{
		fprintf(stdout,"Component %d characteristics: %dx%dx%d %s\n", compno, image->comps[compno].w,
			image->comps[compno].h, image->comps[compno].prec, image->comps[compno].sgnd==1 ? "signed": "unsigned");

		w = image->comps[compno].w;
		h = image->comps[compno].h;

		if(image->comps[compno].prec <= 8)
		{
			if(image->comps[compno].sgnd == 1)
			{
				signed char curr;
				int mask = (1 << image->comps[compno].prec) - 1;
				ptr = image->comps[compno].data;
				for (line = 0; line < h; line++) {
					for(row = 0; row < w; row++)	{				
						curr = (signed char) (*ptr & mask);
						fwrite(&curr, sizeof(signed char), 1, rawFile);
						ptr++;
					}
				}
			}
			else if(image->comps[compno].sgnd == 0)
			{
				unsigned char curr;
				int mask = (1 << image->comps[compno].prec) - 1;
				ptr = image->comps[compno].data;
				for (line = 0; line < h; line++) {
					for(row = 0; row < w; row++)	{	
						curr = (unsigned char) (*ptr & mask);
						fwrite(&curr, sizeof(unsigned char), 1, rawFile);
						ptr++;
					}
				}
			}
		}
		else if(image->comps[compno].prec <= 16)
		{
			if(image->comps[compno].sgnd == 1)
			{
				signed short int curr;
				int mask = (1 << image->comps[compno].prec) - 1;
				ptr = image->comps[compno].data;
				for (line = 0; line < h; line++) {
					for(row = 0; row < w; row++)	{					
						unsigned char temp;
						curr = (signed short int) (*ptr & mask);
						temp = (unsigned char) (curr >> 8);
						fwrite(&temp, 1, 1, rawFile);
						temp = (unsigned char) curr;
						fwrite(&temp, 1, 1, rawFile);
						ptr++;
					}
				}
			}
			else if(image->comps[compno].sgnd == 0)
			{
				unsigned short int curr;
				int mask = (1 << image->comps[compno].prec) - 1;
				ptr = image->comps[compno].data;
				for (line = 0; line < h; line++) {
					for(row = 0; row < w; row++)	{				
						unsigned char temp;
						curr = (unsigned short int) (*ptr & mask);
						temp = (unsigned char) (curr >> 8);
						fwrite(&temp, 1, 1, rawFile);
						temp = (unsigned char) curr;
						fwrite(&temp, 1, 1, rawFile);
						ptr++;
					}
				}
			}
		}
		else if (image->comps[compno].prec <= 32)
		{
			fprintf(stderr,"More than 16 bits per component no handled yet\n");
			return 1;
		}
		else
		{
			fprintf(stderr,"Error: invalid precision: %d\n", image->comps[compno].prec);
			return 1;
		}
	}
	fclose(rawFile);
	return 0;
}

#ifdef HAVE_LIBPNG

#define PNG_MAGIC "\x89PNG\x0d\x0a\x1a\x0a"
#define MAGIC_SIZE 8
/* PNG allows bits per sample: 1, 2, 4, 8, 16 */

opj_image_t *pngtoimage(const char *read_idf, opj_cparameters_t * params)
{
	png_structp  png;
	png_infop    info;
	double gamma, display_exponent;
	int bit_depth, interlace_type,compression_type, filter_type;
	int unit;
	png_uint_32 resx, resy;
	unsigned int i, j;
	png_uint_32  width, height;
	int color_type, has_alpha, is16;
	unsigned char *s;
	FILE *reader;
	unsigned char **rows;
/* j2k: */
	opj_image_t *image;
	opj_image_cmptparm_t cmptparm[4];
	int sub_dx, sub_dy;
	unsigned int nr_comp;
	int *r, *g, *b, *a;
	unsigned char sigbuf[8];

	if((reader = fopen(read_idf, "rb")) == NULL)
   {
	fprintf(stderr,"pngtoimage: can not open %s\n",read_idf);
	return NULL;
   }
	image = NULL; png = NULL; rows = NULL;

	if(fread(sigbuf, 1, MAGIC_SIZE, reader) != MAGIC_SIZE
	|| memcmp(sigbuf, PNG_MAGIC, MAGIC_SIZE) != 0)
   {
	fprintf(stderr,"pngtoimage: %s is no valid PNG file\n",read_idf);
	goto fin;
   }
/* libpng-VERSION/example.c: 
 * PC : screen_gamma = 2.2;
 * Mac: screen_gamma = 1.7 or 1.0;
*/
	display_exponent = 2.2;

	if((png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
				    NULL, NULL, NULL)) == NULL)
	  goto fin;
	if((info = png_create_info_struct(png)) == NULL)
	  goto fin;

	if(setjmp(png_jmpbuf(png)))
	  goto fin;

	png_init_io(png, reader);
	png_set_sig_bytes(png, MAGIC_SIZE);

	png_read_info(png, info);

	if(png_get_IHDR(png, info, &width, &height,
		&bit_depth, &color_type, &interlace_type, 
		&compression_type, &filter_type) == 0)
	 goto fin;

/* png_set_expand():
 * expand paletted images to RGB, expand grayscale images of
 * less than 8-bit depth to 8-bit depth, and expand tRNS chunks
 * to alpha channels.
*/
	if(color_type == PNG_COLOR_TYPE_PALETTE)
	  png_set_expand(png);
	else
	if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
	  png_set_expand(png);

	if(png_get_valid(png, info, PNG_INFO_tRNS))
	  png_set_expand(png);

	is16 = (bit_depth == 16);

/* GRAY => RGB; GRAY_ALPHA => RGBA
*/
	if(color_type == PNG_COLOR_TYPE_GRAY
	|| color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
   {
	png_set_gray_to_rgb(png);
	color_type = 
	 (color_type == PNG_COLOR_TYPE_GRAY? PNG_COLOR_TYPE_RGB:
		PNG_COLOR_TYPE_RGB_ALPHA);
   }
	if( !png_get_gAMA(png, info, &gamma))
	  gamma = 0.45455;

	png_set_gamma(png, display_exponent, gamma);

	png_read_update_info(png, info);

	png_get_pHYs(png, info, &resx, &resy, &unit);

	color_type = png_get_color_type(png, info);

	has_alpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

	nr_comp = 3 + has_alpha;

	bit_depth = png_get_bit_depth(png, info);

	rows = (unsigned char**)calloc(height+1, sizeof(unsigned char*));
	for(i = 0; i < height; ++i)
	 rows[i] = (unsigned char*)malloc(png_get_rowbytes(png,info));

	png_read_image(png, rows);

	memset(&cmptparm, 0, 4 * sizeof(opj_image_cmptparm_t));

	sub_dx = params->subsampling_dx; sub_dy = params->subsampling_dy;

	for(i = 0; i < nr_comp; ++i)
   {
	cmptparm[i].prec = bit_depth;
/* bits_per_pixel: 8 or 16 */
	cmptparm[i].bpp = bit_depth;
	cmptparm[i].sgnd = 0;
	cmptparm[i].dx = sub_dx;
	cmptparm[i].dy = sub_dy;
	cmptparm[i].w = width;
	cmptparm[i].h = height;
   }

	image = opj_image_create(nr_comp, &cmptparm[0], CLRSPC_SRGB);

	if(image == NULL) goto fin;

    image->x0 = params->image_offset_x0;
    image->y0 = params->image_offset_y0;
    image->x1 = image->x0 + (width  - 1) * sub_dx + 1 + image->x0;
    image->y1 = image->y0 + (height - 1) * sub_dy + 1 + image->y0;

	r = image->comps[0].data;
	g = image->comps[1].data;
	b = image->comps[2].data;
	a = image->comps[3].data;

	for(i = 0; i < height; ++i)
   {
	s = rows[i];

	for(j = 0; j < width; ++j)
  {
	if(is16)
 {
	*r++ = s[0]<<8|s[1]; s += 2;

	*g++ = s[0]<<8|s[1]; s += 2;
	
	*b++ = s[0]<<8|s[1]; s += 2;
	
	if(has_alpha) { *a++ = s[0]<<8|s[1]; s += 2; }

	continue;
 }
	*r++ = *s++; *g++ = *s++; *b++ = *s++;

	if(has_alpha) *a++ = *s++;
  }
   }
fin:
	if(rows)
   {
	for(i = 0; i < height; ++i)
	 free(rows[i]);
	free(rows);
   }
	if(png)
	  png_destroy_read_struct(&png, &info, NULL);

	fclose(reader);

	return image;

}/* pngtoimage() */

int imagetopng(opj_image_t * image, const char *write_idf)
{
	FILE *writer;
	png_structp png;
	png_infop info;
	int *red, *green, *blue, *alpha;
	unsigned char *row_buf, *d;
	int has_alpha, width, height, nr_comp, color_type;
	int adjustR, adjustG, adjustB, x, y, fails, is16, force16;
  int opj_prec, prec, ushift, dshift;
	unsigned short mask = 0xffff;
	png_color_8 sig_bit;

	is16 = force16 = ushift = dshift = 0; fails = 1;
	prec = opj_prec = image->comps[0].prec;

	if(prec > 8 && prec < 16)
   {
	 prec = 16; force16 = 1;
   }
	if(prec != 1 && prec != 2 && prec != 4 && prec != 8 && prec != 16)
   {
	fprintf(stderr,"imagetopng: can not create %s"
	 "\n\twrong bit_depth %d\n", write_idf, prec);
	return fails;
   }
	writer = fopen(write_idf, "wb");

	if(writer == NULL) return fails;

	info = NULL; has_alpha = 0;

/* Create and initialize the png_struct with the desired error handler
 * functions.  If you want to use the default stderr and longjump method,
 * you can supply NULL for the last three parameters.  We also check that
 * the library version is compatible with the one used at compile time,
 * in case we are using dynamically linked libraries.  REQUIRED.
*/
	png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
		NULL, NULL, NULL);
/*png_voidp user_error_ptr, user_error_fn, user_warning_fn); */

	if(png == NULL) goto fin;

/* Allocate/initialize the image information data.  REQUIRED 
*/
	info = png_create_info_struct(png);

	if(info == NULL) goto fin;

/* Set error handling.  REQUIRED if you are not supplying your own
 * error handling functions in the png_create_write_struct() call.
*/
	if(setjmp(png_jmpbuf(png))) goto fin;

/* I/O initialization functions is REQUIRED 
*/
	png_init_io(png, writer);

/* Set the image information here.  Width and height are up to 2^31,
 * bit_depth is one of 1, 2, 4, 8, or 16, but valid values also depend on
 * the color_type selected. color_type is one of PNG_COLOR_TYPE_GRAY,
 * PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_PALETTE, PNG_COLOR_TYPE_RGB,
 * or PNG_COLOR_TYPE_RGB_ALPHA.  interlace is either PNG_INTERLACE_NONE or
 * PNG_INTERLACE_ADAM7, and the compression_type and filter_type MUST
 * currently be PNG_COMPRESSION_TYPE_BASE and PNG_FILTER_TYPE_BASE. 
 * REQUIRED
*/
	png_set_compression_level(png, Z_BEST_COMPRESSION);

	if(prec == 16) mask = 0xffff;
	else
	if(prec == 8) mask = 0x00ff;
	else
	if(prec == 4) mask = 0x000f;
	else
	if(prec == 2) mask = 0x0003;
	else
	if(prec == 1) mask = 0x0001;

	nr_comp = image->numcomps;

	if(nr_comp >= 3
    && image->comps[0].dx == image->comps[1].dx
    && image->comps[1].dx == image->comps[2].dx
    && image->comps[0].dy == image->comps[1].dy
    && image->comps[1].dy == image->comps[2].dy
    && image->comps[0].prec == image->comps[1].prec
    && image->comps[1].prec == image->comps[2].prec)
   {
	int v;

    has_alpha = (nr_comp > 3); 

	is16 = (prec == 16);
	
    width = image->comps[0].w;
    height = image->comps[0].h;

	red = image->comps[0].data;
	green = image->comps[1].data;
	blue = image->comps[2].data;

    sig_bit.red = sig_bit.green = sig_bit.blue = prec;

	if(has_alpha) 
  {
	sig_bit.alpha = prec;
	alpha = image->comps[3].data; 
	color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  }
	else 
  {
	sig_bit.alpha = 0; alpha = NULL;
	color_type = PNG_COLOR_TYPE_RGB;
  }
	png_set_sBIT(png, info, &sig_bit);

	png_set_IHDR(png, info, width, height, prec, 
	 color_type,
	 PNG_INTERLACE_NONE,
	 PNG_COMPRESSION_TYPE_BASE,  PNG_FILTER_TYPE_BASE);

/*=============================*/
	png_write_info(png, info);
/*=============================*/
	if(opj_prec < 8)
  {
	png_set_packing(png);
  }
	if(force16)
  {
	ushift = 16 - opj_prec; dshift = opj_prec - ushift;	
  }
    adjustR = (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);
    adjustG = (image->comps[1].sgnd ? 1 << (image->comps[1].prec - 1) : 0);
    adjustB = (image->comps[2].sgnd ? 1 << (image->comps[2].prec - 1) : 0);

	row_buf = (unsigned char*)malloc(width * nr_comp * 2);

	for(y = 0; y < height; ++y)
  {
	d = row_buf;

	for(x = 0; x < width; ++x)
 {
		if(is16)
	   {
/* Network byte order */
		v = *red + adjustR; ++red;
		
		if(force16) { v = (v<<ushift) + (v>>dshift); }

	    *d++ = (unsigned char)(v>>8); *d++ = (unsigned char)v;

		v = *green + adjustG; ++green;
		
		if(force16) { v = (v<<ushift) + (v>>dshift); }

	    *d++ = (unsigned char)(v>>8); *d++ = (unsigned char)v;

		v =  *blue + adjustB; ++blue;
		
		if(force16) { v = (v<<ushift) + (v>>dshift); }

	    *d++ = (unsigned char)(v>>8); *d++ = (unsigned char)v;

		if(has_alpha)
	  {
		v = *alpha++;
		
		if(force16) { v = (v<<ushift) + (v>>dshift); }

		*d++ = (unsigned char)(v>>8); *d++ = (unsigned char)v;
	  }
		continue;
	   }
		*d++ = (unsigned char)((*red + adjustR) & mask); ++red;
		*d++ = (unsigned char)((*green + adjustG) & mask); ++green;
		*d++ = (unsigned char)((*blue + adjustB) & mask); ++blue;

		if(has_alpha)
	   {
		*d++ = (unsigned char)(*alpha & mask); ++alpha;
	   }
 }	/* for(x) */

	png_write_row(png, row_buf);

  }	/* for(y) */
	free(row_buf);

   }/* nr_comp >= 3 */
	else
	if(nr_comp == 1 /* GRAY */
	|| (   nr_comp == 2 /* GRAY_ALPHA */
		&& image->comps[0].dx == image->comps[1].dx
		&& image->comps[0].dy == image->comps[1].dy
		&& image->comps[0].prec == image->comps[1].prec))
   {
	int v;

	red = image->comps[0].data;

    if(force16)
  {
    ushift = 16 - opj_prec; dshift = opj_prec - ushift;
  }

    sig_bit.gray = prec;
    sig_bit.red = sig_bit.green = sig_bit.blue = sig_bit.alpha = 0;
	alpha = NULL;
	color_type = PNG_COLOR_TYPE_GRAY;

    if(nr_comp == 2) 
  { 
	has_alpha = 1; sig_bit.alpha = prec;
	alpha = image->comps[1].data;
	color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
  }
    width = image->comps[0].w;
    height = image->comps[0].h;

	png_set_IHDR(png, info, width, height, sig_bit.gray,
     color_type,
     PNG_INTERLACE_NONE,
     PNG_COMPRESSION_TYPE_BASE,  PNG_FILTER_TYPE_BASE);

	png_set_sBIT(png, info, &sig_bit);
/*=============================*/
	png_write_info(png, info);
/*=============================*/
	adjustR = (image->comps[0].sgnd ? 1 << (image->comps[0].prec - 1) : 0);

	if(opj_prec < 8)
  {
	png_set_packing(png);
  }

	if(prec > 8)
  {
/* Network byte order */


	row_buf = (unsigned char*)
	 malloc(width * nr_comp * sizeof(unsigned short));

	for(y = 0; y < height; ++y)
 {
	d = row_buf;

		for(x = 0; x < width; ++x)
	   {
		v = *red + adjustR; ++red;

		if(force16) { v = (v<<ushift) + (v>>dshift); }

		*d++ = (unsigned char)(v>>8); *d++ = (unsigned char)(v & 0xff);

		if(has_alpha)
	  {
		v = *alpha++;

		if(force16) { v = (v<<ushift) + (v>>dshift); }

		*d++ = (unsigned char)(v>>8); *d++ = (unsigned char)(v & 0xff);
	  }
	   }/* for(x) */
	png_write_row(png, row_buf);

 }	/* for(y) */
	free(row_buf);
  }
	else /* prec <= 8 */
  {
	row_buf = (unsigned char*)calloc(width, nr_comp * 2);

	for(y = 0; y < height; ++y)
 {
	d = row_buf;

		for(x = 0; x < width; ++x)
	   {
		*d++ = (unsigned char)((*red + adjustR) & mask); ++red;

		if(has_alpha)
	  {
		*d++ = (unsigned char)(*alpha & mask); ++alpha;
	  }
	   }/* for(x) */

	png_write_row(png, row_buf);

 }	/* for(y) */
	free(row_buf);
  }
   }
	else
   {
	fprintf(stderr,"imagetopng: can not create %s\n",write_idf);
	goto fin;
   }
	png_write_end(png, info);

	fails = 0;

fin:

	if(png)
   {
    png_destroy_write_struct(&png, &info);
   }
	fclose(writer);

	if(fails) remove(write_idf);

	return fails;
}/* imagetopng() */
#endif /* HAVE_LIBPNG */

