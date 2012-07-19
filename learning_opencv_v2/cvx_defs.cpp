#include "cvx_defs.h"

CvScalar cvx_hsv2rgb( CvScalar hsv ) {

	// H is given on [0, 180]. S and V are given on [0, 255].  
	// RGB are each returned on [0, 255].
	//
	float h = hsv.val[0]/30.0f;
	float s = hsv.val[1]/255.0f;
	float v = hsv.val[2]/255.0f;
	while( h>6.0f )	h-=6.0f;
	while( h<0.0f ) h+=6.0f;
	float m, n, f;  
	int i;  

	CvScalar rgb;

	i = floor(h);  
	f = h - i;  
	if ( !(i&1) ) f = 1 - f; // if i is even  
	m = v * (1 - s);  
	n = v * (1 - s * f);  
	switch (i) {  
		case 6:  
		case 0: rgb = CV_RGB(v, n, m); break;
		case 1: rgb = CV_RGB(n, v, m); break;
		case 2: rgb = CV_RGB(m, v, n); break;
		case 3: rgb = CV_RGB(m, n, v); break;
		case 4: rgb = CV_RGB(n, m, v); break;
		case 5: rgb = CV_RGB(v, m, n); break;
	}  

	rgb.val[0] *= 255.0f;
	rgb.val[1] *= 255.0f;
	rgb.val[2] *= 255.0f;

	return rgb;
} 
