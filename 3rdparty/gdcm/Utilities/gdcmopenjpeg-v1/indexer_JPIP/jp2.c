/*
 * Copyright (c) 2003-2004, Yannick Verschueren
 * Copyright (c) 2003-2004,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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

#include "j2k.h"
#include "cio.h"
#include "tcd.h"
#include "int.h"

#define JPIP_JPIP 0x6a706970

#define JP2_JP   0x6a502020
#define JP2_FTYP 0x66747970
#define JP2_JP2H 0x6a703268
#define JP2_IHDR 0x69686472
#define JP2_COLR 0x636f6c72
#define JP2_JP2C 0x6a703263
#define JP2_URL  0x75726c20
#define JP2_DBTL 0x6474626c
#define JP2_BPCC 0x62706363
#define JP2      0x6a703220


void jp2_write_url(char *Idx_file)
{
  int len, lenp; 
  unsigned int i;
  char str[256];

  sprintf(str, "%s", Idx_file);
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_URL, 4);  // DBTL
  cio_write(0,1);          // VERS
  cio_write(0,3);          // FLAG

  for (i=0; i<strlen(str); i++) {
        cio_write(str[i], 1);
    }

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
}

void jp2_write_dbtl(char *Idx_file)
{
  int len, lenp;

  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_DBTL, 4);  // DBTL
  cio_write(1,2);           // NDR : Only 1
  
  jp2_write_url(Idx_file); // URL Box

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
}

int jp2_write_ihdr(j2k_image_t *j2k_img)
{
  int len, lenp,i; 
  int depth_0,depth, sign, BPC_ok=1;

  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_IHDR, 4);  // IHDR

  cio_write(j2k_img->y1-j2k_img->x0,4);   // HEIGHT
  cio_write(j2k_img->x1-j2k_img->x0,4);   // WIDTH
  cio_write(j2k_img->numcomps,2);   // NC

  depth_0=j2k_img->comps[0].prec-1;
  sign=j2k_img->comps[0].sgnd;

  for(i=1;i<j2k_img->numcomps;i++)
    {
      depth=j2k_img->comps[i].prec-1;
      sign=j2k_img->comps[i].sgnd;
      if(depth_0!=depth) BPC_ok=0;
    }
  
  if (BPC_ok)
    cio_write(depth_0+(sign<<7),1);
  else
    cio_write(255,1);

  cio_write(7,1);          // C : Always 7
  cio_write(1,1);          // UnkC, colorspace unknow
  cio_write(0,1);          // IPR, no intellectual property

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);

  return BPC_ok;
}

void jp2_write_bpcc(j2k_image_t *j2k_img)
{
  int len, lenp, i;
  
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_BPCC, 4);  // BPCC
  
  for(i=0;i<j2k_img->numcomps;i++)
    cio_write(j2k_img->comps[i].prec-1+(j2k_img->comps[i].sgnd<<7),1);

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
}

void jp2_write_colr(int BPC_ok, j2k_image_t *j2k_img)
{
  int len, lenp, meth;
  
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_COLR, 4);  // COLR

  if ((j2k_img->numcomps==1 || j2k_img->numcomps==3) && (BPC_ok && j2k_img->comps[0].prec==8))
    meth=1;
  else
    meth=2;

  cio_write(meth,1);       // METH
  cio_write(0,1);          // PREC
  cio_write(0,1);          // APPROX
  
  if (meth==1)
    cio_write(j2k_img->numcomps>1?16:17,4);          // EnumCS

  if (meth==2)
    cio_write(0,1);        // PROFILE (??) 

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
}

/*
 * Write the JP2H box
 *
 * JP2 Header box
 *
 */
void jp2_write_jp2h(j2k_image_t *j2k_img)
{
  int len, lenp, BPC_ok;
  
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_JP2H, 4);           /* JP2H */

  BPC_ok=jp2_write_ihdr(j2k_img);

  if (!BPC_ok)
    jp2_write_bpcc(j2k_img);
  jp2_write_colr(BPC_ok, j2k_img);

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         /* L */
  cio_seek(lenp+len);
}

/*
 * Write the FTYP box
 *
 * File type box
 *
 */
void jp2_write_ftyp()
{
  int len, lenp;
  
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_FTYP, 4);   /* FTYP       */

  cio_write(JP2,4);         /* BR         */
  cio_write(0,4);           /* MinV       */
  cio_write(JP2,4);         /* CL0 : JP2  */
  cio_write(JPIP_JPIP,4);   /* CL1 : JPIP */

  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         /* L          */
  cio_seek(lenp+len);
}

/*
 * Read the FTYP box
 *
 * File type box
 *
 */
void jp2_read_ftyp(int length)
{
  int BR, MinV, type, i;

  BR = cio_read(4);         /* BR              */
  MinV = cio_read(4);       /* MinV            */
  length-=8;
  
  for (i=length/4;i>0;i--)
    type = cio_read(4);     /* CLi : JP2, JPIP */
}

int jp2_write_jp2c(char *J2K_file)
{
  int len, lenp, totlen, i;
  FILE *src;
  char *j2kfile;

  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_JP2C, 4);  // JP2C

  src=fopen(J2K_file, "rb");
  fseek(src, 0, SEEK_END);
  totlen=ftell(src);
  fseek(src, 0, SEEK_SET);
  
  j2kfile=(char*)malloc(totlen);
  fread(j2kfile, 1, totlen, src);
  fclose(src);

  for (i=0;i<totlen;i++)
    cio_write(j2kfile[i],1);
  
  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
  return lenp;
}

void jp2_write_jp()
{
  int len, lenp;
  
  lenp=cio_tell();
  cio_skip(4);
  cio_write(JP2_JP, 4);  // JP
  cio_write(0x0d0a870a,4);
  len=cio_tell()-lenp;
  cio_seek(lenp);
  cio_write(len,4);         // L
  cio_seek(lenp+len);
}

/*
 * Read the JP box
 *
 * JPEG 2000 signature
 *
 * return 1 if error else 0
 */
int jp2_read_jp()
{
  if (0x0d0a870a!=cio_read(4))
    return 1;
  else
    return 0;
}
