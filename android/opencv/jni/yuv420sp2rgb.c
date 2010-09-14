#include <string.h>
#include <jni.h>
#include <yuv420sp2rgb.h>


#ifndef max
#define max(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({typeof(a) _a = (a); typeof(b) _b = (b); _a < _b ? _a : _b; })
#endif



/*
   YUV 4:2:0 image with a plane of 8 bit Y samples followed by an interleaved
   U/V plane containing 8 bit 2x2 subsampled chroma samples.
   except the interleave order of U and V is reversed.

                        H V
   Y Sample Period      1 1
   U (Cb) Sample Period 2 2
   V (Cr) Sample Period 2 2
 */


/*
 size of a char:
 find . -name limits.h -exec grep CHAR_BIT {} \;
 */

const int bytes_per_pixel = 2;
void color_convert_common(
    unsigned char *pY, unsigned char *pUV,
    int width, int height, unsigned char *buffer,
    int grey)
{
	int i, j;
	int nR, nG, nB;
	int nY, nU, nV;
	unsigned char *out = buffer;
	int offset = 0;

	if(grey){
		  for (i = 0; i < height; i++) {
		            for (j = 0; j < width; j++) {
		                unsigned char nB = *(pY + i * width + j);

		                out[offset++] = (unsigned char)nB;
					//	out[offset++] = (unsigned char)nB;
					//	out[offset++] = (unsigned char)nB;
		            }
		        }
	}else
	// YUV 4:2:0
	for (i = 0; i < height; i++) {
	    for (j = 0; j < width; j++) {
		nY = *(pY + i * width + j);
		nV = *(pUV + (i/2) * width + bytes_per_pixel * (j/2));
		nU = *(pUV + (i/2) * width + bytes_per_pixel * (j/2) + 1);
	    
		// Yuv Convert
		nY -= 16;
		nU -= 128;
		nV -= 128;
	    
		if (nY < 0)
		    nY = 0;
	    
		// nR = (int)(1.164 * nY + 2.018 * nU);
		// nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
		// nB = (int)(1.164 * nY + 1.596 * nV);
	    
		nB = (int)(1192 * nY + 2066 * nU);
		nG = (int)(1192 * nY - 833 * nV - 400 * nU);
		nR = (int)(1192 * nY + 1634 * nV);
	    
		nR = min(262143, max(0, nR));
		nG = min(262143, max(0, nG));
		nB = min(262143, max(0, nB));
	    
       
		nR >>= 10; nR &= 0xff;
		nG >>= 10; nG &= 0xff;
		nB >>= 10; nB &= 0xff;

		out[offset++] = (unsigned char)nR;
		out[offset++] = (unsigned char)nG;
		out[offset++] = (unsigned char)nB;

		//out[offset++] = 0xff; //set alpha for ARGB 8888 format


	    }
	    //offset = i * width * 3;        //non power of two
	    //offset = i * texture_size + j;//power of two
	    //offset *= 3; //3 byte per pixel
	    //out = buffer + offset;
	}
}   
