/*#******************************************************************************
** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
** 
** By downloading, copying, installing or using the software you agree to this license.
** If you do not agree to this license, do not download, install,
** copy or use the software.
** 
** 
** HVStools : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
** Use: extract still images & image sequences features, from contours details to motion spatio-temporal features, etc. for high level visual scene analysis. Also contribute to image enhancement/compression such as tone mapping.
** 
** Maintainers : Listic lab (code author current affiliation & applications) and Gipsa Lab (original research origins & applications)
** 
**  Creation - enhancement process 2007-2011
**      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
** 
** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
** Refer to the following research paper for more information:
** Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
** This work have been carried out thanks to Jeanny Herault who's research and great discussions are the basis of all this work, please take a look at his book:
** Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
** 
** The retina filter includes the research contributions of phd/research collegues from which code has been redrawn by the author :
** _take a look at the retinacolor.hpp module to discover Brice Chaix de Lavarene color mosaicing/demosaicing and the reference paper:
** ====> B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007
** _take a look at imagelogpolprojection.hpp to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions.
** ====> more informations in the above cited Jeanny Heraults's book.
** 
**                          License Agreement
**               For Open Source Computer Vision Library
** 
** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
** 
**               For Human Visual System tools (hvstools)
** Copyright (C) 2007-2011, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
** 
** Third party copyrights are property of their respective owners.
** 
** Redistribution and use in source and binary forms, with or without modification,
** are permitted provided that the following conditions are met:
** 
** * Redistributions of source code must retain the above copyright notice,
**    this list of conditions and the following disclaimer.
** 
** * Redistributions in binary form must reproduce the above copyright notice,
**    this list of conditions and the following disclaimer in the documentation
**    and/or other materials provided with the distribution.
** 
** * The name of the copyright holders may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
** 
** This software is provided by the copyright holders and contributors "as is" and
** any express or implied warranties, including, but not limited to, the implied
** warranties of merchantability and fitness for a particular purpose are disclaimed.
** In no event shall the Intel Corporation or contributors be liable for any direct,
** indirect, incidental, special, exemplary, or consequential damages
** (including, but not limited to, procurement of substitute goods or services;
** loss of use, data, or profits; or business interruption) however caused
** and on any theory of liability, whether in contract, strict liability,
** or tort (including negligence or otherwise) arising in any way out of
** the use of this software, even if advised of the possibility of such damage.
*******************************************************************************/

#ifndef __TEMPLATEBUFFER_HPP__
#define __TEMPLATEBUFFER_HPP__

#include <valarray>
#include <cstdlib>
#include <iostream>
#include <cmath>

//#define __TEMPLATEBUFFERDEBUG //define TEMPLATEBUFFERDEBUG in order to display debug information

namespace cv
{
/**
* @class TemplateBuffer
* @brief this class is a simple template memory buffer which contains basic functions to get information on or normalize the buffer content
* note that thanks to the parent STL template class "valarray", it is possible to perform easily operations on the full array such as addition, product etc.
* @author Alexandre BENOIT (benoit.alexandre.vision@gmail.com), helped by Gelu IONESCU (gelu.ionescu@lis.inpg.fr)
* creation date: september 2007
*/
template <class type> class TemplateBuffer : public std::valarray<type>
{
public:

	/**
	* constructor for monodimensional array
	* @param dim: the size of the vector
	*/
	TemplateBuffer(const size_t dim=0)
	: std::valarray<type>((type)0, dim)
	  {
		_NBrows=1;
		_NBcolumns=dim;
		_NBdepths=1;
		_NBpixels=dim;
		_doubleNBpixels=2*dim;
	  }

	/**
	* constructor by copy for monodimensional array
	* @param pVal: the pointer to a buffer to copy
	* @param dim: the size of the vector
	*/
	TemplateBuffer(const type* pVal, const size_t dim)
	: std::valarray<type>(pVal, dim)
	  {
		_NBrows=1;
		_NBcolumns=dim;
		_NBdepths=1;
		_NBpixels=dim;
		_doubleNBpixels=2*dim;
	  }

	/**
	* constructor for bidimensional array
	* @param dimRows: the size of the vector
	* @param dimColumns: the size of the vector
	* @param depth: the number of layers of the buffer in its third dimension (3 of color images, 1 for gray images.
	*/
	TemplateBuffer(const size_t dimRows, const size_t dimColumns, const size_t depth=1)
	: std::valarray<type>((type)0, dimRows*dimColumns*depth)
	  {
#ifdef TEMPLATEBUFFERDEBUG
		std::cout<<"TemplateBuffer::TemplateBuffer: new buffer, size="<<dimRows<<", "<<dimColumns<<", "<<depth<<"valarraySize="<<this->size()<<std::endl;
#endif
		_NBrows=dimRows;
		_NBcolumns=dimColumns;
		_NBdepths=depth;
		_NBpixels=dimRows*dimColumns;
		_doubleNBpixels=2*dimRows*dimColumns;
		//_createTableIndex();
#ifdef TEMPLATEBUFFERDEBUG
		std::cout<<"TemplateBuffer::TemplateBuffer: construction successful"<<std::endl;
#endif

	  }

	/**
	* copy constructor
	* @param toCopy
	* @return thenconstructed instance
	*emplateBuffer(const TemplateBuffer &toCopy)
	:_NBrows(toCopy.getNBrows()),_NBcolumns(toCopy.getNBcolumns()),_NBdepths(toCopy.getNBdephs()), _NBpixels(toCopy.getNBpixels()), _doubleNBpixels(toCopy.getNBpixels()*2)
	//std::valarray<type>(toCopy)
	{
	memcpy(Buffer(), toCopy.Buffer(), this->size());
	}*/
	/**
	* destructor
	*/
	virtual ~TemplateBuffer()
	{
#ifdef TEMPLATEBUFFERDEBUG
		std::cout<<"~TemplateBuffer"<<std::endl;
#endif
	}

	/**
	* delete the buffer content (set zeros)
	*/
	inline void setZero(){std::valarray<type>::operator=(0);};//memset(Buffer(), 0, sizeof(type)*_NBpixels);};

	/**
	* @return the numbers of rows (height) of the images used by the object
	*/
	inline unsigned int getNBrows(){return _NBrows;};

	/**
	* @return the numbers of columns (width) of the images used by the object
	*/
	inline unsigned int getNBcolumns(){return _NBcolumns;};

	/**
	* @return the numbers of pixels (width*height) of the images used by the object
	*/
	inline unsigned int getNBpixels(){return _NBpixels;};

	/**
	* @return the numbers of pixels (width*height) of the images used by the object
	*/
	inline unsigned int getDoubleNBpixels(){return _doubleNBpixels;};

	/**
	* @return the numbers of depths (3rd dimension: 1 for gray images, 3 for rgb images) of the images used by the object
	*/
	inline unsigned int getDepthSize(){return _NBdepths;};

	/**
	* resize the buffer and recompute table index etc.
	*/
	void resizeBuffer(const size_t dimRows, const size_t dimColumns, const size_t depth=1)
	{
		this->resize(dimRows*dimColumns*depth);
		_NBrows=dimRows;
		_NBcolumns=dimColumns;
		_NBdepths=depth;
		_NBpixels=dimRows*dimColumns;
		_doubleNBpixels=2*dimRows*dimColumns;
	}

	inline TemplateBuffer<type> & operator=(const std::valarray<type> &b)
	{
		//std::cout<<"TemplateBuffer<type> & operator= affect vector: "<<std::endl;
		std::valarray<type>::operator=(b);
		return *this;
	}

	inline TemplateBuffer<type> & operator=(const type &b)
	{
		//std::cout<<"TemplateBuffer<type> & operator= affect value: "<<b<<std::endl;
		std::valarray<type>::operator=(b);
		return *this;
	}

	/*  inline const type  &operator[](const unsigned int &b)
  {
	  return (*this)[b];
  }
	 */
	/**
	* @return the buffer adress in non const mode
	*/
	inline type*    Buffer()            {    return &(*this)[0];    }

	///////////////////////////////////////////////////////
	// Standard Image manipulation functions

	/**
	* standard 0 to 255 image normalization function
	* @param inputOutputBuffer: the image to be normalized (rewrites the input), if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param nbPixels: specifies the number of pixel on which the normalization should be performed, if 0, then all pixels specified in the constructor are processed
	* @param maxOutputValue: the maximum output value
	*/
	static void normalizeGrayOutput_0_maxOutputValue(type *inputOutputBuffer, const size_t nbPixels, const type maxOutputValue=(type)255.0);

	/**
	* standard 0 to 255 image normalization function
	* @param inputOutputBuffer: the image to be normalized (rewrites the input), if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param nbPixels: specifies the number of pixel on which the normalization should be performed, if 0, then all pixels specified in the constructor are processed
	* @param maxOutputValue: the maximum output value
	*/
	void normalizeGrayOutput_0_maxOutputValue(const type maxOutputValue=(type)255.0){normalizeGrayOutput_0_maxOutputValue(this->Buffer(), this->size(), maxOutputValue);};

	/**
	* sigmoide image normalization function (saturates min and max values)
	* @param meanValue: specifies the mean value of th pixels to be processed
	* @param sensitivity: strenght of the sigmoide
	* @param inputPicture: the image to be normalized if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param outputBuffer: the ouput buffer on which the result is writed, if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param maxOutputValue: the maximum output value
	*/
	static void normalizeGrayOutputCentredSigmoide(const type meanValue, const type sensitivity, const type maxOutputValue, type *inputPicture, type *outputBuffer, const unsigned int nbPixels);

	/**
	* sigmoide image normalization function on the current buffer (saturates min and max values)
	* @param meanValue: specifies the mean value of th pixels to be processed
	* @param sensitivity: strenght of the sigmoide
	* @param maxOutputValue: the maximum output value
	*/
	inline void normalizeGrayOutputCentredSigmoide(const type meanValue=(type)0.0, const type sensitivity=(type)2.0, const type maxOutputValue=(type)255.0){normalizeGrayOutputCentredSigmoide(meanValue, sensitivity, 255.0, this->Buffer(), this->Buffer(), this->getNBpixels());};

	/**
	* sigmoide image normalization function (saturates min and max values), in this function, the sigmoide is centered on low values (high saturation of the medium and high values
	* @param inputPicture: the image to be normalized if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param outputBuffer: the ouput buffer on which the result is writed, if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param sensitivity: strenght of the sigmoide
	* @param maxOutputValue: the maximum output value
	*/
	void normalizeGrayOutputNearZeroCentreredSigmoide(type *inputPicture=(type*)NULL, type *outputBuffer=(type*)NULL, const type sensitivity=(type)40, const type maxOutputValue=(type)255.0);

	/**
	* center and reduct the image (image-mean)/std
	* @param inputOutputBuffer: the image to be normalized if no parameter, the result is rewrited on it
	*/
	void centerReductImageLuminance(type *inputOutputBuffer=(type*)NULL);

	/**
	* @return standard deviation of the buffer
	*/
	const double getStandardDeviation()
	{
		double standardDeviation=0;
		double meanValue=getMean();

		type *bufferPTR=Buffer();
		for (unsigned int i=0;i<this->size();++i)
		{
			double diff=(*(bufferPTR++)-meanValue);
			standardDeviation+=diff*diff;
		}
		return sqrt(standardDeviation/this->size());
	};

	/**
	* Clip buffer histogram
	* @param minRatio: the minimum ratio of the lower pixel values, range=[0,1] and lower than maxRatio
	* @param maxRatio: the aximum ratio of the higher pixel values, range=[0,1] and higher than minRatio
	*/
	void clipHistogram(double minRatio, double maxRatio, double maxOutputValue)
	{

		if (minRatio>=maxRatio)
		{
			std::cerr<<"TemplateBuffer::clipHistogram: minRatio must be inferior to maxRatio, buffer unchanged"<<std::endl;
			return;
		}

		/*    minRatio=min(max(minRatio, 1.0),0.0);
    maxRatio=max(max(maxRatio, 0.0),1.0);
		 */

		// find the pixel value just above the threshold
		const double maxThreshold=this->max()*maxRatio;
		const double minThreshold=(this->max()-this->min())*minRatio+this->min();

		type *bufferPTR=this->Buffer();

		double deltaH=maxThreshold;
		double deltaL=maxThreshold;

		double updatedHighValue=maxThreshold;
		double updatedLowValue=maxThreshold;

		for (unsigned int i=0;i<this->size();++i)
		{
			double curentValue=(double)*(bufferPTR++);

			// updating "closest to the high threshold" pixel value
			double highValueTest=maxThreshold-curentValue;
			if (highValueTest>0)
			{
				if (deltaH>highValueTest)
				{
					deltaH=highValueTest;
					updatedHighValue=curentValue;
				}
			}

			// updating "closest to the low threshold" pixel value
			double lowValueTest=curentValue-minThreshold;
			if (lowValueTest>0)
			{
				if (deltaL>lowValueTest)
				{
					deltaL=lowValueTest;
					updatedLowValue=curentValue;
				}
			}
		}

		std::cout<<"Tdebug"<<std::endl;
		std::cout<<"deltaL="<<deltaL<<", deltaH="<<deltaH<<std::endl;
		std::cout<<"this->max()"<<this->max()<<"maxThreshold="<<maxThreshold<<"updatedHighValue="<<updatedHighValue<<std::endl;
		std::cout<<"this->min()"<<this->min()<<"minThreshold="<<minThreshold<<"updatedLowValue="<<updatedLowValue<<std::endl;
		// clipping values outside than the updated thresholds
		bufferPTR=this->Buffer();
		for (unsigned int i=0;i<this->size();++i, ++bufferPTR)
		{
			if (*bufferPTR<updatedLowValue)
				*bufferPTR=updatedLowValue;
			else if (*bufferPTR>updatedHighValue)
				*bufferPTR=updatedHighValue;
		}

		normalizeGrayOutput_0_maxOutputValue(this->Buffer(), this->size(), maxOutputValue);

	}

	/**
	* @return the mean value of the vector
	*/
	inline const double getMean(){return this->sum()/this->size();};

protected:
	size_t _NBrows;
	size_t _NBcolumns;
	size_t _NBdepths;
	size_t _NBpixels;
	size_t _doubleNBpixels;
	// utilities
	static type _abs(const type x);

};

///////////////////////////////////////////////////////////////////////
/// normalize output between 0 and 255, can be applied on images of different size that the declared size if nbPixels parameters is setted up;
template <class type>
void TemplateBuffer<type>::normalizeGrayOutput_0_maxOutputValue(type *inputOutputBuffer, const size_t processedPixels, const type maxOutputValue)
{
	type maxValue=inputOutputBuffer[0], minValue=inputOutputBuffer[0];

	// get the min and max value
	register type *inputOutputBufferPTR=inputOutputBuffer;
	for (register size_t j = 0; j<processedPixels; ++j)
	{
		type pixValue = *(inputOutputBufferPTR++);
		if (maxValue < pixValue)
			maxValue = pixValue;
		else if (minValue > pixValue)
			minValue = pixValue;
	}
	// change the range of the data to 0->255

	type factor = maxOutputValue/(maxValue-minValue);
	type offset = -1.0*minValue*factor;

	inputOutputBufferPTR=inputOutputBuffer;
	for (register size_t j = 0; j < processedPixels; ++j, ++inputOutputBufferPTR)
		*inputOutputBufferPTR=*(inputOutputBufferPTR)*factor+offset;

}
// normalize data with a sigmoide close to 0 (saturates values for those superior to 0)
template <class type>
void TemplateBuffer<type>::normalizeGrayOutputNearZeroCentreredSigmoide(type *inputBuffer, type *outputBuffer, const type sensitivity, const type maxOutputValue)
{
	if (inputBuffer==NULL)
		inputBuffer=Buffer();
	if (outputBuffer==NULL)
		outputBuffer=Buffer();

	type X0cube=sensitivity*sensitivity*sensitivity;

	register type *inputBufferPTR=inputBuffer;
	register type *outputBufferPTR=outputBuffer;

	for (register size_t j = 0; j < _NBpixels; ++j, ++inputBufferPTR)
	{

		type currentCubeLuminance=*inputBufferPTR**inputBufferPTR**inputBufferPTR;
		*(outputBufferPTR++)=maxOutputValue*currentCubeLuminance/(currentCubeLuminance+X0cube);
	}
}

// normalize and adjust luminance with a centered to 128 sigmode
template <class type>
void TemplateBuffer<type>::normalizeGrayOutputCentredSigmoide(const type meanValue, const type sensitivity, const type maxOutputValue, type *inputBuffer, type *outputBuffer, const unsigned int nbPixels)
{

	if (sensitivity==1.0)
	{
		std::cerr<<"TemplateBuffer::TemplateBuffer<type>::normalizeGrayOutputCentredSigmoide error: 2nd parameter (sensitivity) must not equal 0, copying original data..."<<std::endl;
		memcpy(outputBuffer, inputBuffer, sizeof(type)*nbPixels);
		return;
	}

	type X0=maxOutputValue/(sensitivity-(type)1.0);

	register type *inputBufferPTR=inputBuffer;
	register type *outputBufferPTR=outputBuffer;

	for (register size_t j = 0; j < nbPixels; ++j, ++inputBufferPTR)
		*(outputBufferPTR++)=(meanValue+(meanValue+X0)*(*(inputBufferPTR)-meanValue)/(_abs(*(inputBufferPTR)-meanValue)+X0));

}

// center and reduct the image (image-mean)/std
template <class type>
void TemplateBuffer<type>::centerReductImageLuminance(type *inputOutputBuffer)
{
	// if outputBuffer unsassigned, the rewrite the buffer
	if (inputOutputBuffer==NULL)
		inputOutputBuffer=Buffer();
	type meanValue=0, stdValue=0;

	// compute mean value
	for (register size_t j = 0; j < _NBpixels; ++j)
		meanValue+=inputOutputBuffer[j];
	meanValue/=((type)_NBpixels);

	// compute std value
	register type *inputOutputBufferPTR=inputOutputBuffer;
	for (size_t index=0;index<_NBpixels;++index)
	{
		type inputMinusMean=*(inputOutputBufferPTR++)-meanValue;
		stdValue+=inputMinusMean*inputMinusMean;
	}

	stdValue=sqrt(stdValue/((type)_NBpixels));
	// adjust luminance in regard of mean and std value;
	inputOutputBufferPTR=inputOutputBuffer;
	for (size_t index=0;index<_NBpixels;++index, ++inputOutputBufferPTR)
		*inputOutputBufferPTR=(*(inputOutputBufferPTR)-meanValue)/stdValue;
}


template <class type>
type TemplateBuffer<type>::_abs(const type x)
{

	if (x>0)
		return x;
	else
		return -x;
}

template < >
inline int TemplateBuffer<int>::_abs(const int x)
{
	return std::abs(x);
}
template < >
inline double TemplateBuffer<double>::_abs(const double x)
{
	return std::fabs(x);
}

template < >
inline float TemplateBuffer<float>::_abs(const float x)
{
	return std::fabs(x);
}

}
#endif



