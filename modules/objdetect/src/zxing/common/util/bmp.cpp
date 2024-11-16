//
//  BMP.cpp
//  QQView
//
//  Created by Tencent Research on 9/30/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "bmp.hpp"

#include<string.h>

#include <stdio.h>
#include <memory.h>

typedef unsigned long       DWORD;
typedef unsigned char       BYTE;
typedef unsigned short      WORD;
typedef long 		    LONG;
typedef void		*LPVOID;

#define NULL 0

typedef struct tagBITMAPFILEHEADER {
    WORD    bfType;
    DWORD   bfSize;
    WORD    bfReserved1;
    WORD    bfReserved2;
    DWORD   bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER{
    DWORD      biSize;
    LONG       biWidth;
    LONG       biHeight;
    WORD       biPlanes;
    WORD       biBitCount;
    DWORD      biCompression;
    DWORD      biSizeImage;
    LONG       biXPelsPerMeter;
    LONG       biYPelsPerMeter;
    DWORD      biClrUsed;
    DWORD      biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBQUAD {
    BYTE    rgbBlue;
    BYTE    rgbGreen;
    BYTE    rgbRed;
    BYTE    rgbReserved;
} RGBQUAD;



bool SaveBMP(const char* BMPfname, int nWidth, int nHeight, unsigned char* buffer)
{
    BITMAPFILEHEADER	BMFH;
    BITMAPINFOHEADER	BMIH;
    RGBQUAD             *aColors=NULL;
    BYTE                *ptrbmp=NULL;
    BYTE				*pbyPads=NULL;
    int					i = 0 /* , j = 0 */;
    int					nResidue=0;
    int					nPad=0;
    
    FILE* fpOut;
    if ((fpOut = fopen(BMPfname, "wb"))==NULL)
        return false;
    
    // fill in the fields of info header 
    BMIH.biSize    = DWORD(sizeof(BITMAPINFOHEADER));
    BMIH.biWidth     = nWidth;
    BMIH.biHeight    = nHeight;
    BMIH.biPlanes    = 1;
    BMIH.biBitCount  = 8;
    BMIH.biCompression = 0;
    BMIH.biSizeImage = 0;  // nWidth*nHeight;
    BMIH.biXPelsPerMeter = 0;
    BMIH.biYPelsPerMeter = 0;
    BMIH.biClrUsed     = 256;
    BMIH.biClrImportant = 0;
    
    // Fill in the fields of file header 
    BMFH.bfType		= ((WORD) ('M' << 8) | 'B');	// is always "BM"
    int bfSize = 14;
    // bMFH.bfSize		=  DWORD(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)+ BMIH.biClrUsed*sizeof(RGBQUAD) + nWidth*nHeight);
    BMFH.bfSize		=  DWORD(bfSize + sizeof(BITMAPINFOHEADER) + BMIH.biClrUsed*sizeof(RGBQUAD) + nWidth*nHeight);
    BMFH.bfReserved1 = 0;
    BMFH.bfReserved2 = 0;
    // bMFH.bfOffBits	= (DWORD) (sizeof(BMFH) + BMIH.biSize + BMIH.biClrUsed * sizeof(RGBQUAD));
    BMFH.bfOffBits	= (DWORD) (bfSize + BMIH.biSize +
                               BMIH.biClrUsed * sizeof(RGBQUAD));
    
    aColors = new RGBQUAD[BMIH.biClrUsed];
    
    
    // fill in the field of palette
    for (i = 0; i < static_cast<int>(BMIH.biClrUsed); i++)
    {
        aColors[i].rgbBlue  = BYTE(i);
        aColors[i].rgbGreen = BYTE(i);
        aColors[i].rgbRed   = BYTE(i);
        aColors[i].rgbReserved = 0;
    }
    
    
    // bitmap files from bottom to up
    // linear constrast stretch, 'Data' range from -1 to 1, 
    // while 'aBitmapBits' range from 0 to 255
    // fwrite((LPVOID)&BMFH, sizeof(BITMAPFILEHEADER), 1, fpOut);
    fwrite((LPVOID)&(BMFH.bfType), sizeof(WORD), 1, fpOut);
    fwrite((LPVOID)&(BMFH.bfSize), sizeof(DWORD), 1, fpOut);
    fwrite((LPVOID)&(BMFH.bfReserved1), sizeof(WORD), 1, fpOut);
    fwrite((LPVOID)&(BMFH.bfReserved2), sizeof(WORD), 1, fpOut);
    fwrite((LPVOID)&(BMFH.bfOffBits), sizeof(DWORD), 1, fpOut);
    
    fwrite((LPVOID)&BMIH,  sizeof(BITMAPINFOHEADER), 1, fpOut);
    fwrite((LPVOID)aColors, sizeof(RGBQUAD), BMIH.biClrUsed, fpOut);
    
    
    //Note:A scan line must be zero-padded to end on a 32-bit boundary.
    //(4-byte boundary)!!!!!
    
    nResidue = nWidth%4;
    if (nResidue != 0)
    {
        nPad = 4 - nResidue;
        pbyPads = new BYTE[nPad];
        memset(pbyPads, 0, sizeof(BYTE)*nPad);
        
        ptrbmp = buffer + (nHeight-1)*nWidth;
        for (i=nHeight-1; i>=0; i--, ptrbmp -= nWidth)  // write row by row
        {
            fwrite((LPVOID) ptrbmp, sizeof(BYTE), nWidth, fpOut);
            fwrite((LPVOID) pbyPads, sizeof(BYTE), nPad, fpOut);
        }
        delete[] pbyPads;
    }
    else
    {
        ptrbmp = buffer + (nHeight-1)*nWidth;
        for (i=nHeight-1; i>=0; i--, ptrbmp -= nWidth)  // write row by row
        {
            fwrite((LPVOID) ptrbmp, sizeof(BYTE), nWidth, fpOut);
        }
    }
    
    
    delete[] aColors;
    fclose(fpOut);
    
    return true;
}

bool LoadBMP(const char* BMPfname, int &nWidth, int &nHeight, unsigned char* buffer)
{		
    BITMAPINFOHEADER	BMIH;
    BYTE                *ptrbmp=NULL;
    // BYTE				*pbyPads=NULL;
    int					i = 0;  // , j = 0;
    int					nResidue=0;
    // int					nPad=0;
    
    FILE* fpIn;
    if ((fpIn = fopen(BMPfname, "rb"))==NULL)
        return false;
    
    fseek(fpIn, 14, SEEK_CUR);
    fread((LPVOID)&BMIH,  sizeof(BITMAPINFOHEADER), 1, fpIn);
    fseek(fpIn, sizeof(RGBQUAD)*256, SEEK_CUR);
    
    nWidth = BMIH.biWidth;
    nHeight= BMIH.biHeight;
    
    //Note:A scan line must be zero-padded to end on a 32-bit boundary.
    //(4-byte boundary)!!!!!
    
    nResidue = nWidth%4;
    if (nResidue != 0)
    {
        ptrbmp = buffer + (nHeight-1)*nWidth;
        for (i=nHeight-1; i>=0; i--, ptrbmp -= nWidth)  // write row by row
        {
            fread((LPVOID) ptrbmp, sizeof(BYTE), nWidth, fpIn);
            fseek(fpIn, nResidue, SEEK_CUR);
        }
    }
    else
    {
        ptrbmp = buffer + (nHeight-1)*nWidth;
        for (i=nHeight-1; i>=0; i--, ptrbmp -= nWidth)  // write row by row
        {
            fread((LPVOID) ptrbmp, sizeof(BYTE), nWidth, fpIn);
        }
    }
    
    fclose(fpIn);
    return true;
}
