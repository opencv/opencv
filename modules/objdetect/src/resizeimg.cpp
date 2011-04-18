#include "precomp.hpp"
#include "_lsvm_resizeimg.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

IplImage* resize_opencv(IplImage* img, float scale)
{
    IplImage* imgTmp;

    int W, H, tW, tH;

    W = img->width;
    H = img->height;

    tW = (int)(((float)W) * scale + 0.5);
    tH = (int)(((float)H) * scale + 0.5);

    imgTmp = cvCreateImage(cvSize(tW , tH), img->depth, img->nChannels);
    cvResize(img, imgTmp, CV_INTER_AREA);

    return imgTmp;
}

//
///*
// * Fast image subsampling.
// * This is used to construct the feature pyramid.
// */
//
//// struct used for caching interpolation values
//typedef struct  {
//  int si, di;
//  float alpha;
//}alphainfo;
//
//// copy src into dst using pre-computed interpolation values
//void alphacopy(float *src, float *dst, alphainfo *ofs, int n) {
//    int i;
//    for(i = 0; i < n; i++){
//        dst[ofs[i].di] += ofs[i].alpha * src[ofs[i].si];
//    }
//}
//
//int round(float val){
//    return (int)(val + 0.5);
//}
//void bzero(float * arr, int cnt){
//    int i;
//    for(i = 0; i < cnt; i++){
//        arr[i] = 0.0f;
//    }
//}
//// resize along each column
//// result is transposed, so we can apply it twice for a complete resize
//void resize1dtran(float *src, int sheight, float *dst, int dheight, 
//		  int width, int chan) {
//  alphainfo *ofs;
//  float scale = (float)dheight/(float)sheight;
//  float invscale = (float)sheight/(float)dheight;
//  
//  // we cache the interpolation values since they can be 
//  // shared among different columns
//  int len = (int)ceilf(dheight*invscale) + 2*dheight;
//  int k = 0;
//  int dy;
//  float fsy1;
//  float fsy2;
//  int sy1;
//  int sy2;
//  int sy;
//  int c, x;
//  float *s, *d;
//
//  ofs = (alphainfo *) malloc (sizeof(alphainfo) * len);
//  for (dy = 0; dy < dheight; dy++) {
//    fsy1 = dy * invscale;
//    fsy2 = fsy1 + invscale;
//    sy1 = (int)ceilf(fsy1);
//    sy2 = (int)floorf(fsy2);
//
//    if (sy1 - fsy1 > 1e-3) {
//      assert(k < len);
//      assert(sy1 - 1 >= 0);
//      ofs[k].di = dy*width;
//      ofs[k].si = sy1-1;
//      ofs[k++].alpha = (sy1 - fsy1) * scale;
//    }
//
//    for (sy = sy1; sy < sy2; sy++) {
//      assert(k < len);
//      assert(sy < sheight);
//      ofs[k].di = dy*width;
//      ofs[k].si = sy;
//      ofs[k++].alpha = scale;
//    }
//
//    if (fsy2 - sy2 > 1e-3) {
//      assert(k < len);
//      assert(sy2 < sheight);
//      ofs[k].di = dy*width;
//      ofs[k].si = sy2;
//      ofs[k++].alpha = (fsy2 - sy2) * scale;
//    }
//  }
//
//  // resize each column of each color channel
//  bzero(dst, chan*width*dheight);
//  for (c = 0; c < chan; c++) {
//    for (x = 0; x < width; x++) {
//      s = src + c*width*sheight + x*sheight;
//      d = dst + c*width*dheight + x;
//      alphacopy(s, d, ofs, k);
//    }
//  }
//  free(ofs);
//}
//
//IplImage * resize_article_dp(IplImage * img, float scale, const int k){
//    IplImage * imgTmp;
//    float W, H;
//    unsigned  char   *dataSrc;
//    float * dataf;
//    float *src, *dst, *tmp;
//    int i, j, kk, channels;
//	int index;
//    int widthStep;
//    int tW, tH;
//    
//    W = (float)img->width;
//    H = (float)img->height;
//    channels  = img->nChannels;
//	widthStep = img->widthStep;
//
//    tW = (int)(((float)W) * scale + 0.5f);
//    tH = (int)(((float)H) * scale + 0.5f);
//
//    src = (float *)malloc(sizeof(float) * (int)(W * H * 3));
//
//    dataSrc = (unsigned char*)(img->imageData);
//	index = 0;
//	for (kk = 0; kk < channels; kk++)
//	{
//		for (i = 0; i < W; i++)
//		{
//			for (j = 0; j < H; j++)
//			{
//				src[index++] = (float)dataSrc[j * widthStep + i * channels + kk];
//			}
//		}
//	}
//	
//    imgTmp = cvCreateImage(cvSize(tW , tH), IPL_DEPTH_32F, channels);
//
//    dst = (float *)malloc(sizeof(float) * (int)(tH * tW) * channels);
//    tmp = (float *)malloc(sizeof(float) * (int)(tH *  W) * channels);
//
//    resize1dtran(src, (int)H, tmp, (int)tH, (int)W , 3);
//	
//    resize1dtran(tmp, (int)W, dst, (int)tW, (int)tH, 3);
//    
//	index = 0;
//	//dataf = (float*)imgTmp->imageData;
//	for (kk = 0; kk < channels; kk++)
//	{
//		for (i = 0; i < tW; i++)
//		{
//			for (j = 0; j < tH; j++)
//			{
//                dataf = (float*)(imgTmp->imageData + j * imgTmp->widthStep);
//				dataf[ i * channels + kk] = dst[index++];
//			}
//		}
//	}
//
//    free(src);
//    free(dst);
//    free(tmp);
//    return imgTmp;
//}
//
//IplImage * resize_article_dp1(IplImage * img, float scale, const int k){
//    IplImage * imgTmp;
//    float W, H;
//    float * dataf;
//    float *src, *dst, *tmp;
//    int i, j, kk, channels;
//	int index;
//	int widthStep;
//    int tW, tH;
//    
//    W = (float)img->width;
//    H = (float)img->height;
//    channels  = img->nChannels;
//	widthStep = img->widthStep;
//
//    tW = (int)(((float)W) * scale + 0.5f);
//    tH = (int)(((float)H) * scale + 0.5f);
//
//    src = (float *)malloc(sizeof(float) * (int)(W * H) * 3);
//
//	index = 0;
//	for (kk = 0; kk < channels; kk++)
//	{
//		for (i = 0; i < W; i++)
//		{
//			for (j = 0; j < H; j++)
//			{
//				src[index++] = (float)(*( (float *)(img->imageData + j * widthStep) + i * channels + kk));
//			}
//		}
//	}
//	
//    imgTmp = cvCreateImage(cvSize(tW , tH), IPL_DEPTH_32F, channels);
//
//    dst = (float *)malloc(sizeof(float) * (int)(tH * tW) * channels);
//    tmp = (float *)malloc(sizeof(float) * (int)(tH *  W) * channels);
//
//    resize1dtran(src, (int)H, tmp, (int)tH, (int)W , 3);
//	
//    resize1dtran(tmp, (int)W, dst, (int)tW, (int)tH, 3);
//	
//	index = 0;
//	for (kk = 0; kk < channels; kk++)
//	{
//		for (i = 0; i < tW; i++)
//		{
//			for (j = 0; j < tH; j++)
//			{
//                dataf = (float *)(imgTmp->imageData + j * imgTmp->widthStep);
//				dataf[ i * channels + kk] = dst[index++];
//			}
//		}
//	}
// 
//    free(src);
//    free(dst);
//    free(tmp);
//    return imgTmp;
//}
//
