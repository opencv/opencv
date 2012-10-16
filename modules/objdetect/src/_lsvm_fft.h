#ifndef _LSVM_FFT_H_
#define _LSVM_FFT_H_

#include "_lsvm_types.h"
#include "_lsvm_error.h"
#include <math.h>

/*
// 1-dimensional FFT
//
// API
// int fft(float *x_in, float *x_out, int n, int shift);
// INPUT
// x_in              - input signal
// n                 - number of elements for searching Fourier image
// shift             - shift between input elements
// OUTPUT
// x_out             - output signal (contains 2n elements in order
                       Re(x_in[0]), Im(x_in[0]), Re(x_in[1]), Im(x_in[1]) and etc.)
// RESULT
// Error status
*/
int fft(float *x_in, float *x_out, int n, int shift);

/*
// Inverse 1-dimensional FFT
//
// API
// int fftInverse(float *x_in, float *x_out, int n, int shift);
// INPUT
// x_in              - Fourier image of 1d input signal(contains 2n elements
                       in order Re(x_in[0]), Im(x_in[0]),
                       Re(x_in[1]), Im(x_in[1]) and etc.)
// n                 - number of elements for searching counter FFT image
// shift             - shift between input elements
// OUTPUT
// x_in              - input signal (contains n elements)
// RESULT
// Error status
*/
int fftInverse(float *x_in, float *x_out, int n, int shift);

/*
// 2-dimensional FFT
//
// API
// int fft2d(float *x_in, float *x_out, int numRows, int numColls);
// INPUT
// x_in              - input signal (matrix, launched by rows)
// numRows           - number of rows
// numColls          - number of collumns
// OUTPUT
// x_out             - output signal (contains (2 * numRows * numColls) elements
                       in order Re(x_in[0][0]), Im(x_in[0][0]),
                       Re(x_in[0][1]), Im(x_in[0][1]) and etc.)
// RESULT
// Error status
*/
int fft2d(float *x_in, float *x_out, int numRows, int numColls);

/*
// Inverse 2-dimensional FFT
//
// API
// int fftInverse2d(float *x_in, float *x_out, int numRows, int numColls);
// INPUT
// x_in              - Fourier image of matrix (contains (2 * numRows * numColls)
                       elements in order Re(x_in[0][0]), Im(x_in[0][0]),
                       Re(x_in[0][1]), Im(x_in[0][1]) and etc.)
// numRows           - number of rows
// numColls          - number of collumns
// OUTPUT
// x_out             - initial signal (matrix, launched by rows)
// RESULT
// Error status
*/
int fftInverse2d(float *x_in, float *x_out, int numRows, int numColls);

#endif
