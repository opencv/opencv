#include "precomp.hpp"
#include "_lsvm_fft.h"

// static int getEntireRes(int number, int divisor, int *entire, int *res)
// {
//     *entire = number / divisor;
//     *res = number % divisor;
//     return FFT_OK;
// }

static int getMultipliers(int n, int *n1, int *n2)
{
    int multiplier, i;
    if (n == 1)
    {
        *n1 = 1;
        *n2 = 1;
        return FFT_ERROR; // n = 1
    }
    multiplier = n / 2;
    for (i = multiplier; i >= 2; i--)
    {
        if (n % i == 0)
        {
            *n1 = i;
            *n2 = n / i;
            return FFT_OK; // n = n1 * n2
        }
    }
    *n1 = 1;
    *n2 = n;
    return FFT_ERROR; // n - prime number
}

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
int fft(float *x_in, float *x_out, int n, int shift)
{
    int n1, n2, res, k1, k2, m1, m2, index, idx;
    float alpha, beta, gamma, angle, cosAngle, sinAngle;
    float tmpGamma, tmpAlpha, tmpBeta;
    float tmpRe, tmpIm, phaseRe, phaseIm;
    res = getMultipliers(n, &n1, &n2);
    if (res == FFT_OK)
    {
        fft(x_in, x_out, n1, shift);
        fft(x_in, x_out, n2, shift);
    }
    alpha = (float)(2.0 * PI / ((float)n));
    beta = (float)(2.0 * PI / ((float)n1));
    gamma = (float)(2.0 * PI / ((float)n2));
    for (k1 = 0; k1 < n1; k1++)
    {
        tmpBeta = beta * k1;
        for (k2 = 0; k2 < n2; k2++)
        {
            idx = shift * (n2 * k1 + k2);
            x_out[idx] = 0.0;
            x_out[idx + 1] = 0.0;
            tmpGamma = gamma * k2;
            tmpAlpha = alpha * k2;
            for (m1 = 0; m1 < n1; m1++)
            {
                tmpRe = 0.0;
                tmpIm = 0.0;
                for (m2 = 0; m2 < n2; m2++)
                {
                    angle = tmpGamma * m2;
                    index = shift * (n1 * m2 + m1);
                    cosAngle = cosf(angle);
                    sinAngle = sinf(angle);
                    tmpRe += x_in[index] * cosAngle + x_in[index + 1] * sinAngle;
                    tmpIm += x_in[index + 1] * cosAngle - x_in[index] * sinAngle;
                }
                angle = tmpAlpha * m1;
                cosAngle = cosf(angle);
                sinAngle = sinf(angle);
                phaseRe = cosAngle * tmpRe + sinAngle * tmpIm;
                phaseIm = cosAngle * tmpIm - sinAngle * tmpRe;
                angle = tmpBeta * m1;
                cosAngle = cosf(angle);
                sinAngle = sinf(angle);
                x_out[idx] += (cosAngle * phaseRe + sinAngle * phaseIm);
                x_out[idx + 1] += (cosAngle * phaseIm - sinAngle * phaseRe);
            }
        }
    }
    return FFT_OK;
}

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
int fftInverse(float *x_in, float *x_out, int n, int shift)
{
    int n1, n2, res, k1, k2, m1, m2, index, idx;
    float alpha, beta, gamma, angle, cosAngle, sinAngle;
    float tmpRe, tmpIm, phaseRe, phaseIm;
    res = getMultipliers(n, &n1, &n2);
    if (res == FFT_OK)
    {
        fftInverse(x_in, x_out, n1, shift);
        fftInverse(x_in, x_out, n2, shift);
    }
    alpha = (float)(2.0f * PI / ((float)n));
    beta = (float)(2.0f * PI / ((float)n1));
    gamma = (float)(2.0f * PI / ((float)n2));
    for (m1 = 0; m1 < n1; m1++)
    {
        for (m2 = 0; m2 < n2; m2++)
        {
            idx = (n1 * m2 + m1) * shift;
            x_out[idx] = 0.0;
            x_out[idx + 1] = 0.0;
            for (k2 = 0; k2 < n2; k2++)
            {
                tmpRe = 0.0;
                tmpIm = 0.0;
                for (k1 = 0; k1 < n1; k1++)
                {
                    angle = beta * k1 * m1;
                    index = shift *(n2 * k1 + k2);
                    sinAngle = sinf(angle);
                    cosAngle = cosf(angle);
                    tmpRe += x_in[index] * cosAngle - x_in[index + 1] * sinAngle;
                    tmpIm += x_in[index] * sinAngle + x_in[index + 1] * cosAngle;
                }
                angle = alpha * m1 * k2;
                sinAngle = sinf(angle);
                cosAngle = cosf(angle);
                phaseRe = cosAngle * tmpRe - sinAngle * tmpIm;
                phaseIm = cosAngle * tmpIm + sinAngle * tmpRe;
                angle = gamma * k2 * m2;
                sinAngle = sinf(angle);
                cosAngle = cosf(angle);
                x_out[idx] += cosAngle * phaseRe - sinAngle * phaseIm;
                x_out[idx + 1] += cosAngle * phaseIm + sinAngle * phaseRe;
            }
            x_out[idx] /= n;
            x_out[idx + 1] /= n;
        }
    }
    return FFT_OK;
}

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
int fft2d(float *x_in, float *x_out, int numRows, int numColls)
{
    int i, size;
    float *x_outTmp;
    size = numRows * numColls;
    x_outTmp = (float *)malloc(sizeof(float) * (2 * size));
    for (i = 0; i < numRows; i++)
    {
        fft(x_in + i * 2 * numColls,
            x_outTmp + i * 2 * numColls,
            numColls, 2);
    }
    for (i = 0; i < numColls; i++)
    {
        fft(x_outTmp + 2 * i,
            x_out + 2 * i,
            numRows, 2 * numColls);
    }
    free(x_outTmp);
    return FFT_OK;
}

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
int fftInverse2d(float *x_in, float *x_out, int numRows, int numColls)
{
    int i, size;
    float *x_outTmp;
    size = numRows * numColls;
    x_outTmp = (float *)malloc(sizeof(float) * (2 * size));
    for (i = 0; i < numRows; i++)
    {
        fftInverse(x_in + i * 2 * numColls,
            x_outTmp + i * 2 * numColls,
            numColls, 2);
    }
    for (i = 0; i < numColls; i++)
    {
        fftInverse(x_outTmp + 2 * i,
            x_out + 2 * i,
            numRows, 2 * numColls);
    }
    free(x_outTmp);
    return FFT_OK;
}

