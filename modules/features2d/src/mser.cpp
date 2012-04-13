/* Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * 	Redistributions of source code must retain the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer.
 * 	Redistributions in binary form must reproduce the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer in the documentation and/or other materials
 * 	provided with the distribution.
 * 	The name of Contributor may not be used to endorse or
 * 	promote products derived from this software without
 * 	specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 * Copyright© 2009, Liu Liu All rights reserved.
 * 
 * OpenCV functions for MSER extraction
 * 
 * 1. there are two different implementation of MSER, one for grey image, one for color image
 * 2. the grey image algorithm is taken from: Linear Time Maximally Stable Extremal Regions;
 *    the paper claims to be faster than union-find method;
 *    it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.
 * 3. the color image algorithm is taken from: Maximally Stable Colour Regions for Recognition and Match;
 *    it should be much slower than grey image method ( 3~4 times );
 *    the chi_table.h file is taken directly from paper's source code which is distributed under GPL.
 * 4. though the name is *contours*, the result actually is a list of point set.
 */

#include "precomp.hpp"

namespace cv
{

const int TABLE_SIZE = 400;

static double chitab3[]={0,  0.0150057,  0.0239478,  0.0315227,
                  0.0383427,  0.0446605,  0.0506115,  0.0562786,
                  0.0617174,  0.0669672,  0.0720573,  0.0770099,
                  0.081843,  0.0865705,  0.0912043,  0.0957541,
                  0.100228,  0.104633,  0.108976,  0.113261,
                  0.117493,  0.121676,  0.125814,  0.12991,
                  0.133967,  0.137987,  0.141974,  0.145929,
                  0.149853,  0.15375,  0.15762,  0.161466,
                  0.165287,  0.169087,  0.172866,  0.176625,
                  0.180365,  0.184088,  0.187794,  0.191483,
                  0.195158,  0.198819,  0.202466,  0.2061,
                  0.209722,  0.213332,  0.216932,  0.220521,
                  0.2241,  0.22767,  0.231231,  0.234783,
                  0.238328,  0.241865,  0.245395,  0.248918,
                  0.252435,  0.255947,  0.259452,  0.262952,
                  0.266448,  0.269939,  0.273425,  0.276908,
                  0.280386,  0.283862,  0.287334,  0.290803,
                  0.29427,  0.297734,  0.301197,  0.304657,
                  0.308115,  0.311573,  0.315028,  0.318483,
                  0.321937,  0.32539,  0.328843,  0.332296,
                  0.335749,  0.339201,  0.342654,  0.346108,
                  0.349562,  0.353017,  0.356473,  0.35993,
                  0.363389,  0.366849,  0.37031,  0.373774,
                  0.377239,  0.380706,  0.384176,  0.387648,
                  0.391123,  0.3946,  0.39808,  0.401563,
                  0.405049,  0.408539,  0.412032,  0.415528,
                  0.419028,  0.422531,  0.426039,  0.429551,
                  0.433066,  0.436586,  0.440111,  0.44364,
                  0.447173,  0.450712,  0.454255,  0.457803,
                  0.461356,  0.464915,  0.468479,  0.472049,
                  0.475624,  0.479205,  0.482792,  0.486384,
                  0.489983,  0.493588,  0.4972,  0.500818,
                  0.504442,  0.508073,  0.511711,  0.515356,
                  0.519008,  0.522667,  0.526334,  0.530008,
                  0.533689,  0.537378,  0.541075,  0.54478,
                  0.548492,  0.552213,  0.555942,  0.55968,
                  0.563425,  0.56718,  0.570943,  0.574715,
                  0.578497,  0.582287,  0.586086,  0.589895,
                  0.593713,  0.597541,  0.601379,  0.605227,
                  0.609084,  0.612952,  0.61683,  0.620718,
                  0.624617,  0.628526,  0.632447,  0.636378,
                  0.64032,  0.644274,  0.648239,  0.652215,
                  0.656203,  0.660203,  0.664215,  0.668238,
                  0.672274,  0.676323,  0.680384,  0.684457,
                  0.688543,  0.692643,  0.696755,  0.700881,
                  0.70502,  0.709172,  0.713339,  0.717519,
                  0.721714,  0.725922,  0.730145,  0.734383,
                  0.738636,  0.742903,  0.747185,  0.751483,
                  0.755796,  0.760125,  0.76447,  0.768831,
                  0.773208,  0.777601,  0.782011,  0.786438,
                  0.790882,  0.795343,  0.799821,  0.804318,
                  0.808831,  0.813363,  0.817913,  0.822482,
                  0.827069,  0.831676,  0.836301,  0.840946,
                  0.84561,  0.850295,  0.854999,  0.859724,
                  0.864469,  0.869235,  0.874022,  0.878831,
                  0.883661,  0.888513,  0.893387,  0.898284,
                  0.903204,  0.908146,  0.913112,  0.918101,
                  0.923114,  0.928152,  0.933214,  0.938301,
                  0.943413,  0.94855,  0.953713,  0.958903,
                  0.964119,  0.969361,  0.974631,  0.979929,
                  0.985254,  0.990608,  0.99599,  1.0014,
                  1.00684,  1.01231,  1.01781,  1.02335,
                  1.02891,  1.0345,  1.04013,  1.04579,
                  1.05148,  1.05721,  1.06296,  1.06876,
                  1.07459,  1.08045,  1.08635,  1.09228,
                  1.09826,  1.10427,  1.11032,  1.1164,
                  1.12253,  1.1287,  1.1349,  1.14115,
                  1.14744,  1.15377,  1.16015,  1.16656,
                  1.17303,  1.17954,  1.18609,  1.19269,
                  1.19934,  1.20603,  1.21278,  1.21958,
                  1.22642,  1.23332,  1.24027,  1.24727,
                  1.25433,  1.26144,  1.26861,  1.27584,
                  1.28312,  1.29047,  1.29787,  1.30534,
                  1.31287,  1.32046,  1.32812,  1.33585,
                  1.34364,  1.3515,  1.35943,  1.36744,
                  1.37551,  1.38367,  1.39189,  1.4002,
                  1.40859,  1.41705,  1.42561,  1.43424,
                  1.44296,  1.45177,  1.46068,  1.46967,
                  1.47876,  1.48795,  1.49723,  1.50662,
                  1.51611,  1.52571,  1.53541,  1.54523,
                  1.55517,  1.56522,  1.57539,  1.58568,
                  1.59611,  1.60666,  1.61735,  1.62817,
                  1.63914,  1.65025,  1.66152,  1.67293,
                  1.68451,  1.69625,  1.70815,  1.72023,
                  1.73249,  1.74494,  1.75757,  1.77041,
                  1.78344,  1.79669,  1.81016,  1.82385,
                  1.83777,  1.85194,  1.86635,  1.88103,
                  1.89598,  1.91121,  1.92674,  1.94257,
                  1.95871,  1.97519,  1.99201,  2.0092,
                  2.02676,  2.04471,  2.06309,  2.08189,
                  2.10115,  2.12089,  2.14114,  2.16192,
                  2.18326,  2.2052,  2.22777,  2.25101,
                  2.27496,  2.29966,  2.32518,  2.35156,
                  2.37886,  2.40717,  2.43655,  2.46709,
                  2.49889,  2.53206,  2.56673,  2.60305,
                  2.64117,  2.6813,  2.72367,  2.76854,
                  2.81623,  2.86714,  2.92173,  2.98059,
                  3.04446,  3.1143,  3.19135,  3.27731,
                  3.37455,  3.48653,  3.61862,  3.77982,
                  3.98692,  4.2776,  4.77167,  133.333 };

typedef struct LinkedPoint
{
	struct LinkedPoint* prev;
	struct LinkedPoint* next;
	Point pt;
}
LinkedPoint;

// the history of region grown
typedef struct MSERGrowHistory
{
	struct MSERGrowHistory* shortcut;
	struct MSERGrowHistory* child;
	int stable; // when it ever stabled before, record the size
	int val;
	int size;
}
MSERGrowHistory;

typedef struct MSERConnectedComp
{
	LinkedPoint* head;
	LinkedPoint* tail;
	MSERGrowHistory* history;
	unsigned long grey_level;
	int size;
	int dvar; // the derivative of last var
	float var; // the current variation (most time is the variation of one-step back)
}
MSERConnectedComp;

// Linear Time MSER claims by using bsf can get performance gain, here is the implementation
// however it seems that will not do any good in real world test
inline void _bitset(unsigned long * a, unsigned long b)
{
	*a |= 1<<b;
}
inline void _bitreset(unsigned long * a, unsigned long b)
{
	*a &= ~(1<<b);
}

struct MSERParams
{
    MSERParams( int _delta, int _minArea, int _maxArea, double _maxVariation,
                double _minDiversity, int _maxEvolution, double _areaThreshold,
                double _minMargin, int _edgeBlurSize )
        : delta(_delta), minArea(_minArea), maxArea(_maxArea), maxVariation(_maxVariation),
        minDiversity(_minDiversity), maxEvolution(_maxEvolution), areaThreshold(_areaThreshold),
        minMargin(_minMargin), edgeBlurSize(_edgeBlurSize)
    {}
    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};

// clear the connected component in stack
static void
initMSERComp( MSERConnectedComp* comp )
{
	comp->size = 0;
	comp->var = 0;
	comp->dvar = 1;
	comp->history = NULL;
}

// add history of size to a connected component
static void
MSERNewHistory( MSERConnectedComp* comp, MSERGrowHistory* history )
{
	history->child = history;
	if ( NULL == comp->history )
	{
		history->shortcut = history;
		history->stable = 0;
	} else {
		comp->history->child = history;
		history->shortcut = comp->history->shortcut;
		history->stable = comp->history->stable;
	}
	history->val = comp->grey_level;
	history->size = comp->size;
	comp->history = history;
}

// merging two connected component
static void
MSERMergeComp( MSERConnectedComp* comp1,
		  MSERConnectedComp* comp2,
		  MSERConnectedComp* comp,
		  MSERGrowHistory* history )
{
	LinkedPoint* head;
	LinkedPoint* tail;
	comp->grey_level = comp2->grey_level;
	history->child = history;
	// select the winner by size
	if ( comp1->size >= comp2->size )
	{
		if ( NULL == comp1->history )
		{
			history->shortcut = history;
			history->stable = 0;
		} else {
			comp1->history->child = history;
			history->shortcut = comp1->history->shortcut;
			history->stable = comp1->history->stable;
		}
		if ( NULL != comp2->history && comp2->history->stable > history->stable )
			history->stable = comp2->history->stable;
		history->val = comp1->grey_level;
		history->size = comp1->size;
		// put comp1 to history
		comp->var = comp1->var;
		comp->dvar = comp1->dvar;
		if ( comp1->size > 0 && comp2->size > 0 )
		{
			comp1->tail->next = comp2->head;
			comp2->head->prev = comp1->tail;
		}
		head = ( comp1->size > 0 ) ? comp1->head : comp2->head;
		tail = ( comp2->size > 0 ) ? comp2->tail : comp1->tail;
		// always made the newly added in the last of the pixel list (comp1 ... comp2)
	} else {
		if ( NULL == comp2->history )
		{
			history->shortcut = history;
			history->stable = 0;
		} else {
			comp2->history->child = history;
			history->shortcut = comp2->history->shortcut;
			history->stable = comp2->history->stable;
		}
		if ( NULL != comp1->history && comp1->history->stable > history->stable )
			history->stable = comp1->history->stable;
		history->val = comp2->grey_level;
		history->size = comp2->size;
		// put comp2 to history
		comp->var = comp2->var;
		comp->dvar = comp2->dvar;
		if ( comp1->size > 0 && comp2->size > 0 )
		{
			comp2->tail->next = comp1->head;
			comp1->head->prev = comp2->tail;
		}
		head = ( comp2->size > 0 ) ? comp2->head : comp1->head;
		tail = ( comp1->size > 0 ) ? comp1->tail : comp2->tail;
		// always made the newly added in the last of the pixel list (comp2 ... comp1)
	}
	comp->head = head;
	comp->tail = tail;
	comp->history = history;
	comp->size = comp1->size + comp2->size;
}

static float
MSERVariationCalc( MSERConnectedComp* comp, int delta )
{
	MSERGrowHistory* history = comp->history;
	int val = comp->grey_level;
	if ( NULL != history )
	{
		MSERGrowHistory* shortcut = history->shortcut;
		while ( shortcut != shortcut->shortcut && shortcut->val + delta > val )
			shortcut = shortcut->shortcut;
		MSERGrowHistory* child = shortcut->child;
		while ( child != child->child && child->val + delta <= val )
		{
			shortcut = child;
			child = child->child;
		}
		// get the position of history where the shortcut->val <= delta+val and shortcut->child->val >= delta+val
		history->shortcut = shortcut;
		return (float)(comp->size-shortcut->size)/(float)shortcut->size;
		// here is a small modification of MSER where cal ||R_{i}-R_{i-delta}||/||R_{i-delta}||
		// in standard MSER, cal ||R_{i+delta}-R_{i-delta}||/||R_{i}||
		// my calculation is simpler and much easier to implement
	}
	return 1.;
}

static bool MSERStableCheck( MSERConnectedComp* comp, MSERParams params )
{
	// tricky part: it actually check the stablity of one-step back
	if ( comp->history == NULL || comp->history->size <= params.minArea || comp->history->size >= params.maxArea )
		return 0;
	float div = (float)(comp->history->size-comp->history->stable)/(float)comp->history->size;
	float var = MSERVariationCalc( comp, params.delta );
	int dvar = ( comp->var < var || (unsigned long)(comp->history->val + 1) < comp->grey_level );
	int stable = ( dvar && !comp->dvar && comp->var < params.maxVariation && div > params.minDiversity );
	comp->var = var;
	comp->dvar = dvar;
	if ( stable )
		comp->history->stable = comp->history->size;
	return stable != 0;
}

// add a pixel to the pixel list
static void accumulateMSERComp( MSERConnectedComp* comp, LinkedPoint* point )
{
	if ( comp->size > 0 )
	{
		point->prev = comp->tail;
		comp->tail->next = point;
		point->next = NULL;
	} else {
		point->prev = NULL;
		point->next = NULL;
		comp->head = point;
	}
	comp->tail = point;
	comp->size++;
}

// convert the point set to CvSeq
static CvContour* MSERToContour( MSERConnectedComp* comp, CvMemStorage* storage )
{
	CvSeq* _contour = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage );
	CvContour* contour = (CvContour*)_contour;
	cvSeqPushMulti( _contour, 0, comp->history->size );
	LinkedPoint* lpt = comp->head;
	for ( int i = 0; i < comp->history->size; i++ )
	{
		CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, _contour, i );
		pt->x = lpt->pt.x;
		pt->y = lpt->pt.y;
		lpt = lpt->next;
	}
	cvBoundingRect( contour );
	return contour;
}

// to preprocess src image to following format
// 32-bit image
// > 0 is available, < 0 is visited
// 17~19 bits is the direction
// 8~11 bits is the bucket it falls to (for BitScanForward)
// 0~8 bits is the color
static int* preprocessMSER_8UC1( CvMat* img,
			int*** heap_cur,
			CvMat* src,
			CvMat* mask )
{
	int srccpt = src->step-src->cols;
	int cpt_1 = img->cols-src->cols-1;
	int* imgptr = img->data.i;
	int* startptr;

	int level_size[256];
	for ( int i = 0; i < 256; i++ )
		level_size[i] = 0;

	for ( int i = 0; i < src->cols+2; i++ )
	{
		*imgptr = -1;
		imgptr++;
	}
	imgptr += cpt_1-1;
	uchar* srcptr = src->data.ptr;
	if ( mask )
	{
		startptr = 0;
		uchar* maskptr = mask->data.ptr;
		for ( int i = 0; i < src->rows; i++ )
		{
			*imgptr = -1;
			imgptr++;
			for ( int j = 0; j < src->cols; j++ )
			{
				if ( *maskptr )
				{
					if ( !startptr )
						startptr = imgptr;
					*srcptr = 0xff-*srcptr;
					level_size[*srcptr]++;
					*imgptr = ((*srcptr>>5)<<8)|(*srcptr);
				} else {
					*imgptr = -1;
				}
				imgptr++;
				srcptr++;
				maskptr++;
			}
			*imgptr = -1;
			imgptr += cpt_1;
			srcptr += srccpt;
			maskptr += srccpt;
		}
	} else {
		startptr = imgptr+img->cols+1;
		for ( int i = 0; i < src->rows; i++ )
		{
			*imgptr = -1;
			imgptr++;
			for ( int j = 0; j < src->cols; j++ )
			{
				*srcptr = 0xff-*srcptr;
				level_size[*srcptr]++;
				*imgptr = ((*srcptr>>5)<<8)|(*srcptr);
				imgptr++;
				srcptr++;
			}
			*imgptr = -1;
			imgptr += cpt_1;
			srcptr += srccpt;
		}
	}
	for ( int i = 0; i < src->cols+2; i++ )
	{
		*imgptr = -1;
		imgptr++;
	}

	heap_cur[0][0] = 0;
	for ( int i = 1; i < 256; i++ )
	{
		heap_cur[i] = heap_cur[i-1]+level_size[i-1]+1;
		heap_cur[i][0] = 0;
	}
	return startptr;
}

static void extractMSER_8UC1_Pass( int* ioptr,
			  int* imgptr,
			  int*** heap_cur,
			  LinkedPoint* ptsptr,
			  MSERGrowHistory* histptr,
			  MSERConnectedComp* comptr,
			  int step,
			  int stepmask,
			  int stepgap,
			  MSERParams params,
			  int color,
			  CvSeq* contours,
			  CvMemStorage* storage )
{
	comptr->grey_level = 256;
	comptr++;
	comptr->grey_level = (*imgptr)&0xff;
	initMSERComp( comptr );
	*imgptr |= 0x80000000;
	heap_cur += (*imgptr)&0xff;
	int dir[] = { 1, step, -1, -step };
#ifdef __INTRIN_ENABLED__
	unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned long* bit_cur = heapbit+(((*imgptr)&0x700)>>8);
#endif
	for ( ; ; )
	{
		// take tour of all the 4 directions
		while ( ((*imgptr)&0x70000) < 0x40000 )
		{
			// get the neighbor
			int* imgptr_nbr = imgptr+dir[((*imgptr)&0x70000)>>16];
			if ( *imgptr_nbr >= 0 ) // if the neighbor is not visited yet
			{
				*imgptr_nbr |= 0x80000000; // mark it as visited
				if ( ((*imgptr_nbr)&0xff) < ((*imgptr)&0xff) )
				{
					// when the value of neighbor smaller than current
					// push current to boundary heap and make the neighbor to be the current one
					// create an empty comp
					(*heap_cur)++;
					**heap_cur = imgptr;
					*imgptr += 0x10000;
					heap_cur += ((*imgptr_nbr)&0xff)-((*imgptr)&0xff);
#ifdef __INTRIN_ENABLED__
					_bitset( bit_cur, (*imgptr)&0x1f );
					bit_cur += (((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8;
#endif
					imgptr = imgptr_nbr;
					comptr++;
					initMSERComp( comptr );
					comptr->grey_level = (*imgptr)&0xff;
					continue;
				} else {
					// otherwise, push the neighbor to boundary heap
					heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)]++;
					*heap_cur[((*imgptr_nbr)&0xff)-((*imgptr)&0xff)] = imgptr_nbr;
#ifdef __INTRIN_ENABLED__
					_bitset( bit_cur+((((*imgptr_nbr)&0x700)-((*imgptr)&0x700))>>8), (*imgptr_nbr)&0x1f );
#endif
				}
			}
			*imgptr += 0x10000;
		}
		int i = (int)(imgptr-ioptr);
		ptsptr->pt = cvPoint( i&stepmask, i>>stepgap );
		// get the current location
		accumulateMSERComp( comptr, ptsptr );
		ptsptr++;
		// get the next pixel from boundary heap
		if ( **heap_cur )
		{
			imgptr = **heap_cur;
			(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
			if ( !**heap_cur )
				_bitreset( bit_cur, (*imgptr)&0x1f );
#endif
		} else {
#ifdef __INTRIN_ENABLED__
			bool found_pixel = 0;
			unsigned long pixel_val;
			for ( int i = ((*imgptr)&0x700)>>8; i < 8; i++ )
			{
				if ( _BitScanForward( &pixel_val, *bit_cur ) )
				{
					found_pixel = 1;
					pixel_val += i<<5;
					heap_cur += pixel_val-((*imgptr)&0xff);
					break;
				}
				bit_cur++;
			}
			if ( found_pixel )
#else
			heap_cur++;
			unsigned long pixel_val = 0;
			for ( unsigned long i = ((*imgptr)&0xff)+1; i < 256; i++ )
			{
				if ( **heap_cur )
				{
					pixel_val = i;
					break;
				}
				heap_cur++;
			}
			if ( pixel_val )
#endif
			{
				imgptr = **heap_cur;
				(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
				if ( !**heap_cur )
					_bitreset( bit_cur, pixel_val&0x1f );
#endif
				if ( pixel_val < comptr[-1].grey_level )
				{
					// check the stablity and push a new history, increase the grey level
					if ( MSERStableCheck( comptr, params ) )
					{
						CvContour* contour = MSERToContour( comptr, storage );
						contour->color = color;
						cvSeqPush( contours, &contour );
					}
					MSERNewHistory( comptr, histptr );
					comptr[0].grey_level = pixel_val;
					histptr++;
				} else {
					// keep merging top two comp in stack until the grey level >= pixel_val
					for ( ; ; )
					{
						comptr--;
						MSERMergeComp( comptr+1, comptr, comptr, histptr );
						histptr++;
						if ( pixel_val <= comptr[0].grey_level )
							break;
						if ( pixel_val < comptr[-1].grey_level )
						{
							// check the stablity here otherwise it wouldn't be an ER
							if ( MSERStableCheck( comptr, params ) )
							{
								CvContour* contour = MSERToContour( comptr, storage );
								contour->color = color;
								cvSeqPush( contours, &contour );
							}
							MSERNewHistory( comptr, histptr );
							comptr[0].grey_level = pixel_val;
							histptr++;
							break;
						}
					}
				}
			} else
				break;
		}
	}
}

static void extractMSER_8UC1( CvMat* src,
		     CvMat* mask,
		     CvSeq* contours,
		     CvMemStorage* storage,
		     MSERParams params )
{
	int step = 8;
	int stepgap = 3;
	while ( step < src->step+2 )
	{
		step <<= 1;
		stepgap++;
	}
	int stepmask = step-1;

	// to speedup the process, make the width to be 2^N
	CvMat* img = cvCreateMat( src->rows+2, step, CV_32SC1 );
	int* ioptr = img->data.i+step+1;
	int* imgptr;

	// pre-allocate boundary heap
	int** heap = (int**)cvAlloc( (src->rows*src->cols+256)*sizeof(heap[0]) );
	int** heap_start[256];
	heap_start[0] = heap;

	// pre-allocate linked point and grow history
	LinkedPoint* pts = (LinkedPoint*)cvAlloc( src->rows*src->cols*sizeof(pts[0]) );
	MSERGrowHistory* history = (MSERGrowHistory*)cvAlloc( src->rows*src->cols*sizeof(history[0]) );
	MSERConnectedComp comp[257];

	// darker to brighter (MSER-)
	imgptr = preprocessMSER_8UC1( img, heap_start, src, mask );
	extractMSER_8UC1_Pass( ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, -1, contours, storage );
	// brighter to darker (MSER+)
	imgptr = preprocessMSER_8UC1( img, heap_start, src, mask );
	extractMSER_8UC1_Pass( ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap, params, 1, contours, storage );

	// clean up
	cvFree( &history );
	cvFree( &heap );
	cvFree( &pts );
	cvReleaseMat( &img );
}

struct MSCRNode;

struct TempMSCR
{
	MSCRNode* head;
	MSCRNode* tail;
	double m; // the margin used to prune area later
	int size;
};

struct MSCRNode
{
	MSCRNode* shortcut;
	// to make the finding of root less painful
	MSCRNode* prev;
	MSCRNode* next;
	// a point double-linked list
	TempMSCR* tmsr;
	// the temporary msr (set to NULL at every re-initialise)
	TempMSCR* gmsr;
	// the global msr (once set, never to NULL)
	int index;
	// the index of the node, at this point, it should be x at the first 16-bits, and y at the last 16-bits.
	int rank;
	int reinit;
	int size, sizei;
	double dt, di;
	double s;
};

struct MSCREdge
{
	double chi;
	MSCRNode* left;
	MSCRNode* right;
};

static double ChiSquaredDistance( uchar* x, uchar* y )
{
	return (double)((x[0]-y[0])*(x[0]-y[0]))/(double)(x[0]+y[0]+1e-10)+
	       (double)((x[1]-y[1])*(x[1]-y[1]))/(double)(x[1]+y[1]+1e-10)+
	       (double)((x[2]-y[2])*(x[2]-y[2]))/(double)(x[2]+y[2]+1e-10);
}

static void initMSCRNode( MSCRNode* node )
{
	node->gmsr = node->tmsr = NULL;
	node->reinit = 0xffff;
	node->rank = 0;
	node->sizei = node->size = 1;
	node->prev = node->next = node->shortcut = node;
}

// the preprocess to get the edge list with proper gaussian blur
static int preprocessMSER_8UC3( MSCRNode* node,
			MSCREdge* edge,
			double* total,
			CvMat* src,
			CvMat* mask,
			CvMat* dx,
			CvMat* dy,
			int Ne,
			int edgeBlurSize )
{
	int srccpt = src->step-src->cols*3;
	uchar* srcptr = src->data.ptr;
	uchar* lastptr = src->data.ptr+3;
	double* dxptr = dx->data.db;
	for ( int i = 0; i < src->rows; i++ )
	{
		for ( int j = 0; j < src->cols-1; j++ )
		{
			*dxptr = ChiSquaredDistance( srcptr, lastptr );
			dxptr++;
			srcptr += 3;
			lastptr += 3;
		}
		srcptr += srccpt+3;
		lastptr += srccpt+3;
	}
	srcptr = src->data.ptr;
	lastptr = src->data.ptr+src->step;
	double* dyptr = dy->data.db;
	for ( int i = 0; i < src->rows-1; i++ )
	{
		for ( int j = 0; j < src->cols; j++ )
		{
			*dyptr = ChiSquaredDistance( srcptr, lastptr );
			dyptr++;
			srcptr += 3;
			lastptr += 3;
		}
		srcptr += srccpt;
		lastptr += srccpt;
	}
	// get dx and dy and blur it
	if ( edgeBlurSize >= 1 )
	{
		cvSmooth( dx, dx, CV_GAUSSIAN, edgeBlurSize, edgeBlurSize );
		cvSmooth( dy, dy, CV_GAUSSIAN, edgeBlurSize, edgeBlurSize );
	}
	dxptr = dx->data.db;
	dyptr = dy->data.db;
	// assian dx, dy to proper edge list and initialize mscr node
	// the nasty code here intended to avoid extra loops
	if ( mask )
	{
		Ne = 0;
		int maskcpt = mask->step-mask->cols+1;
		uchar* maskptr = mask->data.ptr;
		MSCRNode* nodeptr = node;
		initMSCRNode( nodeptr );
		nodeptr->index = 0;
		*total += edge->chi = *dxptr;
		if ( maskptr[0] && maskptr[1] )
		{
			edge->left = nodeptr;
			edge->right = nodeptr+1;
			edge++;
			Ne++;
		}
		dxptr++;
		nodeptr++;
		maskptr++;
		for ( int i = 1; i < src->cols-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = i;
			if ( maskptr[0] && maskptr[1] )
			{
				*total += edge->chi = *dxptr;
				edge->left = nodeptr;
				edge->right = nodeptr+1;
				edge++;
				Ne++;
			}
			dxptr++;
			nodeptr++;
			maskptr++;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = src->cols-1;
		nodeptr++;
		maskptr += maskcpt;
		for ( int i = 1; i < src->rows-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = i<<16;
			if ( maskptr[0] )
			{
				if ( maskptr[-mask->step] )
				{
					*total += edge->chi = *dyptr;
					edge->left = nodeptr-src->cols;
					edge->right = nodeptr;
					edge++;
					Ne++;
				}
				if ( maskptr[1] )
				{
					*total += edge->chi = *dxptr;
					edge->left = nodeptr;
					edge->right = nodeptr+1;
					edge++;
					Ne++;
				}
			}
			dyptr++;
			dxptr++;
			nodeptr++;
			maskptr++;
			for ( int j = 1; j < src->cols-1; j++ )
			{
				initMSCRNode( nodeptr );
				nodeptr->index = (i<<16)|j;
				if ( maskptr[0] )
				{
					if ( maskptr[-mask->step] )
					{
						*total += edge->chi = *dyptr;
						edge->left = nodeptr-src->cols;
						edge->right = nodeptr;
						edge++;
						Ne++;
					}
					if ( maskptr[1] )
					{
						*total += edge->chi = *dxptr;
						edge->left = nodeptr;
						edge->right = nodeptr+1;
						edge++;
						Ne++;
					}
				}
				dyptr++;
				dxptr++;
				nodeptr++;
				maskptr++;
			}
			initMSCRNode( nodeptr );
			nodeptr->index = (i<<16)|(src->cols-1);
			if ( maskptr[0] && maskptr[-mask->step] )
			{
				*total += edge->chi = *dyptr;
				edge->left = nodeptr-src->cols;
				edge->right = nodeptr;
				edge++;
				Ne++;
			}
			dyptr++;
			nodeptr++;
			maskptr += maskcpt;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = (src->rows-1)<<16;
		if ( maskptr[0] )
		{
			if ( maskptr[1] )
			{
				*total += edge->chi = *dxptr;
				edge->left = nodeptr;
				edge->right = nodeptr+1;
				edge++;
				Ne++;
			}
			if ( maskptr[-mask->step] )
			{
				*total += edge->chi = *dyptr;
				edge->left = nodeptr-src->cols;
				edge->right = nodeptr;
				edge++;
				Ne++;
			}
		}
		dxptr++;
		dyptr++;
		nodeptr++;
		maskptr++;
		for ( int i = 1; i < src->cols-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = ((src->rows-1)<<16)|i;
			if ( maskptr[0] )
			{
				if ( maskptr[1] )
				{
					*total += edge->chi = *dxptr;
					edge->left = nodeptr;
					edge->right = nodeptr+1;
					edge++;
					Ne++;
				}
				if ( maskptr[-mask->step] )
				{
					*total += edge->chi = *dyptr;
					edge->left = nodeptr-src->cols;
					edge->right = nodeptr;
					edge++;
					Ne++;
				}
			}
			dxptr++;
			dyptr++;
			nodeptr++;
			maskptr++;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = ((src->rows-1)<<16)|(src->cols-1);
		if ( maskptr[0] && maskptr[-mask->step] )
		{
			*total += edge->chi = *dyptr;
			edge->left = nodeptr-src->cols;
			edge->right = nodeptr;
			Ne++;
		}
	} else {
		MSCRNode* nodeptr = node;
		initMSCRNode( nodeptr );
		nodeptr->index = 0;
		*total += edge->chi = *dxptr;
		dxptr++;
		edge->left = nodeptr;
		edge->right = nodeptr+1;
		edge++;
		nodeptr++;
		for ( int i = 1; i < src->cols-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = i;
			*total += edge->chi = *dxptr;
			dxptr++;
			edge->left = nodeptr;
			edge->right = nodeptr+1;
			edge++;
			nodeptr++;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = src->cols-1;
		nodeptr++;
		for ( int i = 1; i < src->rows-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = i<<16;
			*total += edge->chi = *dyptr;
			dyptr++;
			edge->left = nodeptr-src->cols;
			edge->right = nodeptr;
			edge++;
			*total += edge->chi = *dxptr;
			dxptr++;
			edge->left = nodeptr;
			edge->right = nodeptr+1;
			edge++;
			nodeptr++;
			for ( int j = 1; j < src->cols-1; j++ )
			{
				initMSCRNode( nodeptr );
				nodeptr->index = (i<<16)|j;
				*total += edge->chi = *dyptr;
				dyptr++;
				edge->left = nodeptr-src->cols;
				edge->right = nodeptr;
				edge++;
				*total += edge->chi = *dxptr;
				dxptr++;
				edge->left = nodeptr;
				edge->right = nodeptr+1;
				edge++;
				nodeptr++;
			}
			initMSCRNode( nodeptr );
			nodeptr->index = (i<<16)|(src->cols-1);
			*total += edge->chi = *dyptr;
			dyptr++;
			edge->left = nodeptr-src->cols;
			edge->right = nodeptr;
			edge++;
			nodeptr++;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = (src->rows-1)<<16;
		*total += edge->chi = *dxptr;
		dxptr++;
		edge->left = nodeptr;
		edge->right = nodeptr+1;
		edge++;
		*total += edge->chi = *dyptr;
		dyptr++;
		edge->left = nodeptr-src->cols;
		edge->right = nodeptr;
		edge++;
		nodeptr++;
		for ( int i = 1; i < src->cols-1; i++ )
		{
			initMSCRNode( nodeptr );
			nodeptr->index = ((src->rows-1)<<16)|i;
			*total += edge->chi = *dxptr;
			dxptr++;
			edge->left = nodeptr;
			edge->right = nodeptr+1;
			edge++;
			*total += edge->chi = *dyptr;
			dyptr++;
			edge->left = nodeptr-src->cols;
			edge->right = nodeptr;
			edge++;
			nodeptr++;
		}
		initMSCRNode( nodeptr );
		nodeptr->index = ((src->rows-1)<<16)|(src->cols-1);
		*total += edge->chi = *dyptr;
		edge->left = nodeptr-src->cols;
		edge->right = nodeptr;
	}
	return Ne;
}

#define cmp_mscr_edge(edge1, edge2) \
	((edge1).chi < (edge2).chi)

static CV_IMPLEMENT_QSORT( QuickSortMSCREdge, MSCREdge, cmp_mscr_edge )

// to find the root of one region
static MSCRNode* findMSCR( MSCRNode* x )
{
	MSCRNode* prev = x;
	MSCRNode* next;
	for ( ; ; )
	{
		next = x->shortcut;
		x->shortcut = prev;
		if ( next == x ) break;
		prev= x;
		x = next;
	}
	MSCRNode* root = x;
	for ( ; ; )
	{
		prev = x->shortcut;
		x->shortcut = root;
		if ( prev == x ) break;
		x = prev;
	}
	return root;
}

// the stable mscr should be:
// bigger than minArea and smaller than maxArea
// differ from its ancestor more than minDiversity
static bool MSCRStableCheck( MSCRNode* x, MSERParams params )
{
	if ( x->size <= params.minArea || x->size >= params.maxArea )
		return 0;
	if ( x->gmsr == NULL )
		return 1;
	double div = (double)(x->size-x->gmsr->size)/(double)x->size;
	return div > params.minDiversity;
}

static void
extractMSER_8UC3( CvMat* src,
		     CvMat* mask,
		     CvSeq* contours,
		     CvMemStorage* storage,
		     MSERParams params )
{
	MSCRNode* map = (MSCRNode*)cvAlloc( src->cols*src->rows*sizeof(map[0]) );
	int Ne = src->cols*src->rows*2-src->cols-src->rows;
	MSCREdge* edge = (MSCREdge*)cvAlloc( Ne*sizeof(edge[0]) );
	TempMSCR* mscr = (TempMSCR*)cvAlloc( src->cols*src->rows*sizeof(mscr[0]) );
	double emean = 0;
	CvMat* dx = cvCreateMat( src->rows, src->cols-1, CV_64FC1 );
	CvMat* dy = cvCreateMat( src->rows-1, src->cols, CV_64FC1 );
	Ne = preprocessMSER_8UC3( map, edge, &emean, src, mask, dx, dy, Ne, params.edgeBlurSize );
	emean = emean / (double)Ne;
	QuickSortMSCREdge( edge, Ne, 0 );
	MSCREdge* edge_ub = edge+Ne;
	MSCREdge* edgeptr = edge;
	TempMSCR* mscrptr = mscr;
	// the evolution process
	for ( int i = 0; i < params.maxEvolution; i++ )
	{
		double k = (double)i/(double)params.maxEvolution*(TABLE_SIZE-1);
		int ti = cvFloor(k);
		double reminder = k-ti;
		double thres = emean*(chitab3[ti]*(1-reminder)+chitab3[ti+1]*reminder);
		// to process all the edges in the list that chi < thres
		while ( edgeptr < edge_ub && edgeptr->chi < thres )
		{
			MSCRNode* lr = findMSCR( edgeptr->left );
			MSCRNode* rr = findMSCR( edgeptr->right );
			// get the region root (who is responsible)
			if ( lr != rr )
			{
				// rank idea take from: N-tree Disjoint-Set Forests for Maximally Stable Extremal Regions
				if ( rr->rank > lr->rank )
				{
					MSCRNode* tmp;
					CV_SWAP( lr, rr, tmp );
				} else if ( lr->rank == rr->rank ) {
					// at the same rank, we will compare the size
					if ( lr->size > rr->size )
					{
						MSCRNode* tmp;
						CV_SWAP( lr, rr, tmp );
					}
					lr->rank++;
				}
				rr->shortcut = lr;
				lr->size += rr->size;
				// join rr to the end of list lr (lr is a endless double-linked list)
				lr->prev->next = rr;
				lr->prev = rr->prev;
				rr->prev->next = lr;
				rr->prev = lr;
				// area threshold force to reinitialize
				if ( lr->size > (lr->size-rr->size)*params.areaThreshold )
				{
					lr->sizei = lr->size;
					lr->reinit = i;
					if ( lr->tmsr != NULL )
					{
						lr->tmsr->m = lr->dt-lr->di;
						lr->tmsr = NULL;
					}
					lr->di = edgeptr->chi;
					lr->s = 1e10;
				}
				lr->dt = edgeptr->chi;
				if ( i > lr->reinit )
				{
					double s = (double)(lr->size-lr->sizei)/(lr->dt-lr->di);
					if ( s < lr->s )
					{
						// skip the first one and check stablity
						if ( i > lr->reinit+1 && MSCRStableCheck( lr, params ) )
						{
							if ( lr->tmsr == NULL )
							{
								lr->gmsr = lr->tmsr = mscrptr;
								mscrptr++;
							}
							lr->tmsr->size = lr->size;
							lr->tmsr->head = lr;
							lr->tmsr->tail = lr->prev;
							lr->tmsr->m = 0;
						}
						lr->s = s;
					}
				}
			}
			edgeptr++;
		}
		if ( edgeptr >= edge_ub )
			break;
	}
	for ( TempMSCR* ptr = mscr; ptr < mscrptr; ptr++ )
		// to prune area with margin less than minMargin
		if ( ptr->m > params.minMargin )
		{
			CvSeq* _contour = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage );
			cvSeqPushMulti( _contour, 0, ptr->size );
			MSCRNode* lpt = ptr->head;
			for ( int i = 0; i < ptr->size; i++ )
			{
				CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, _contour, i );
				pt->x = (lpt->index)&0xffff;
				pt->y = (lpt->index)>>16;
				lpt = lpt->next;
			}
			CvContour* contour = (CvContour*)_contour;
			cvBoundingRect( contour );
			contour->color = 0;
			cvSeqPush( contours, &contour );
		}
	cvReleaseMat( &dx );
	cvReleaseMat( &dy );
	cvFree( &mscr );
	cvFree( &edge );
	cvFree( &map );
}

static void
extractMSER( CvArr* _img,
	       CvArr* _mask,
	       CvSeq** _contours,
	       CvMemStorage* storage,
	       MSERParams params )
{
	CvMat srchdr, *src = cvGetMat( _img, &srchdr );
	CvMat maskhdr, *mask = _mask ? cvGetMat( _mask, &maskhdr ) : 0;
	CvSeq* contours = 0;

	CV_Assert(src != 0);
	CV_Assert(CV_MAT_TYPE(src->type) == CV_8UC1 || CV_MAT_TYPE(src->type) == CV_8UC3);
	CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
	CV_Assert(storage != 0);

	contours = *_contours = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), storage );

	// choose different method for different image type
	// for grey image, it is: Linear Time Maximally Stable Extremal Regions
	// for color image, it is: Maximally Stable Colour Regions for Recognition and Matching
	switch ( CV_MAT_TYPE(src->type) )
	{
		case CV_8UC1:
			extractMSER_8UC1( src, mask, contours, storage, params );
			break;
		case CV_8UC3:
			extractMSER_8UC3( src, mask, contours, storage, params );
			break;
	}
}


MSER::MSER( int _delta, int _min_area, int _max_area,
      double _max_variation, double _min_diversity,
      int _max_evolution, double _area_threshold,
      double _min_margin, int _edge_blur_size )
    : delta(_delta), minArea(_min_area), maxArea(_max_area),
    maxVariation(_max_variation), minDiversity(_min_diversity),
    maxEvolution(_max_evolution), areaThreshold(_area_threshold),
    minMargin(_min_margin), edgeBlurSize(_edge_blur_size)
{
}

void MSER::operator()( const Mat& image, vector<vector<Point> >& dstcontours, const Mat& mask ) const
{
    CvMat _image = image, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSeq*> contours;
    extractMSER( &_image, pmask, &contours.seq, storage,
                 MSERParams(delta, minArea, maxArea, maxVariation, minDiversity,
                            maxEvolution, areaThreshold, minMargin, edgeBlurSize));
    SeqIterator<CvSeq*> it = contours.begin();
    size_t i, ncontours = contours.size();
    dstcontours.resize(ncontours);
    for( i = 0; i < ncontours; i++, ++it )
        Seq<Point>(*it).copyTo(dstcontours[i]);
}
    

void MserFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    vector<vector<Point> > msers;
    
    (*this)(image, msers, mask);
    
    vector<vector<Point> >::const_iterator contour_it = msers.begin();
    for( ; contour_it != msers.end(); ++contour_it )
    {
        // TODO check transformation from MSER region to KeyPoint
        RotatedRect rect = fitEllipse(Mat(*contour_it));
        float diam = sqrt(rect.size.height*rect.size.width);
        
        if( diam > std::numeric_limits<float>::epsilon() )
            keypoints.push_back( KeyPoint( rect.center, diam, rect.angle) );
    }
}
    
}
