// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 
#ifndef CHARLS_PROCESSLINE
#define CHARLS_PROCESSLINE

#include "colortransform.h"

//
// This file defines the ProcessLine base class, its derivitives and helper functions.
// During coding/decoding, CharLS process one line at a time. The different Processline implementations
// convert the uncompressed format to and from the internal format for encoding.
// Conversions include color transforms, line interleaved vs sample interleaved, masking out unused bits,
// accounting for line padding etc.
// This mechanism could be used to encode/decode images as they are received.
//

class ProcessLine
{
public:
	virtual ~ProcessLine() {}
	virtual void NewLineDecoded(const void* pSrc, int pixelCount, int bytesperline) = 0;
	virtual void NewLineRequested(void* pSrc, int pixelCount, int bytesperline) = 0;
};


class PostProcesSingleComponent : public ProcessLine
{
public:
	PostProcesSingleComponent(void* pbyteOutput, const JlsParameters& info, int bytesPerPixel) :
		_pbyteOutput((BYTE*)pbyteOutput), 
		_bytesPerPixel(bytesPerPixel),
		_bytesPerLine(info.bytesperline)
	{
	}

	void NewLineRequested(void* pDst, int pixelCount, int /*byteStride*/)
	{
		::memcpy(pDst, _pbyteOutput, pixelCount * _bytesPerPixel);
		_pbyteOutput += _bytesPerLine;
	}

	void NewLineDecoded(const void* pSrc, int pixelCount, int /*byteStride*/)
	{
		::memcpy(_pbyteOutput, pSrc, pixelCount * _bytesPerPixel);
		_pbyteOutput += _bytesPerLine;		
	}

private:
	BYTE* _pbyteOutput;
	int _bytesPerPixel;
	int _bytesPerLine;
};


template<class TRANSFORM, class SAMPLE> 
void TransformLineToQuad(const SAMPLE* ptypeInput, LONG pixelStrideIn, Quad<SAMPLE>* pbyteBuffer, LONG pixelStride, TRANSFORM& transform)
{
	int cpixel = MIN(pixelStride, pixelStrideIn);
	Quad<SAMPLE>* ptypeBuffer = (Quad<SAMPLE>*)pbyteBuffer;

	for (int x = 0; x < cpixel; ++x)
	{
		Quad<SAMPLE> pixel(transform(ptypeInput[x], ptypeInput[x + pixelStrideIn], ptypeInput[x + 2*pixelStrideIn]),ptypeInput[x + 3*pixelStrideIn]) ;
		
		ptypeBuffer[x] = pixel;
	}
}


template<class TRANSFORM, class SAMPLE> 
void TransformQuadToLine(const Quad<SAMPLE>* pbyteInput, LONG pixelStrideIn, SAMPLE* ptypeBuffer, LONG pixelStride, TRANSFORM& transform)
{
	int cpixel = MIN(pixelStride, pixelStrideIn);
	const Quad<SAMPLE>* ptypeBufferIn = (Quad<SAMPLE>*)pbyteInput;

	for (int x = 0; x < cpixel; ++x)
	{
		Quad<SAMPLE> color = ptypeBufferIn[x];
		Quad<SAMPLE> colorTranformed(transform(color.v1, color.v2, color.v3), color.v4);

		ptypeBuffer[x] = colorTranformed.v1;
		ptypeBuffer[x + pixelStride] = colorTranformed.v2;
		ptypeBuffer[x + 2 *pixelStride] = colorTranformed.v3;
		ptypeBuffer[x + 3 *pixelStride] = colorTranformed.v4;
	}
}


template<class SAMPLE> 
void TransformRgbToBgr(SAMPLE* pDest, int samplesPerPixel, int pixelCount)
{
	for (int i = 0; i < pixelCount; ++i)
	{
		std::swap(pDest[0], pDest[2]);		
		pDest += samplesPerPixel;
	}
}


template<class TRANSFORM, class SAMPLE> 
void TransformLine(Triplet<SAMPLE>* pDest, const Triplet<SAMPLE>* pSrc, int pixelCount, TRANSFORM& transform) 
{	
	for (int i = 0; i < pixelCount; ++i)
	{
		pDest[i] = transform(pSrc[i].v1, pSrc[i].v2, pSrc[i].v3);
	}
}


template<class TRANSFORM, class SAMPLE> 
void TransformLineToTriplet(const SAMPLE* ptypeInput, LONG pixelStrideIn, Triplet<SAMPLE>* pbyteBuffer, LONG pixelStride, TRANSFORM& transform)
{
	int cpixel = MIN(pixelStride, pixelStrideIn);
	Triplet<SAMPLE>* ptypeBuffer = (Triplet<SAMPLE>*)pbyteBuffer;

	for (int x = 0; x < cpixel; ++x)
	{
		ptypeBuffer[x] = transform(ptypeInput[x], ptypeInput[x + pixelStrideIn], ptypeInput[x + 2*pixelStrideIn]);
	}
}


template<class TRANSFORM, class SAMPLE> 
void TransformTripletToLine(const Triplet<SAMPLE>* pbyteInput, LONG pixelStrideIn, SAMPLE* ptypeBuffer, LONG pixelStride, TRANSFORM& transform)
{
	int cpixel = MIN(pixelStride, pixelStrideIn);
	const Triplet<SAMPLE>* ptypeBufferIn = (Triplet<SAMPLE>*)pbyteInput;

	for (int x = 0; x < cpixel; ++x)
	{
		Triplet<SAMPLE> color = ptypeBufferIn[x];
		Triplet<SAMPLE> colorTranformed = transform(color.v1, color.v2, color.v3);

		ptypeBuffer[x] = colorTranformed.v1;
		ptypeBuffer[x + pixelStride] = colorTranformed.v2;
		ptypeBuffer[x + 2 *pixelStride] = colorTranformed.v3;
	}
}


template<class TRANSFORM> 
class ProcessTransformed : public ProcessLine
{
	typedef typename TRANSFORM::SAMPLE SAMPLE;

	ProcessTransformed(const ProcessTransformed&);
public:
	ProcessTransformed(void* pbyteOutput, const JlsParameters& info, TRANSFORM transform) :
		_pbyteOutput((BYTE*)pbyteOutput),
		_info(info),
		_templine(info.width *  info.components),
		_transform(transform),
		_inverseTransform(transform)
	{
//		ASSERT(_info.components == sizeof(TRIPLET)/sizeof(TRIPLET::SAMPLE));
	}


	void NewLineRequested(void* pDst, int pixelCount, int stride)
	{
		SAMPLE* pLine = (SAMPLE*)_pbyteOutput;
		if (_info.outputBgr)
		{
			pLine = &_templine[0]; 
			memcpy(pLine, _pbyteOutput, sizeof(Triplet<SAMPLE>)*pixelCount);
			TransformRgbToBgr(pLine, _info.components, pixelCount);
		}

		if (_info.components == 3)
		{
			if (_info.ilv == ILV_SAMPLE)
			{
				TransformLine((Triplet<SAMPLE>*)pDst, (const Triplet<SAMPLE>*)pLine, pixelCount, _transform);
			}
			else
			{
				TransformTripletToLine((const Triplet<SAMPLE>*)pLine, pixelCount, (SAMPLE*)pDst, stride, _transform);
			}
		}
		else if (_info.components == 4 && _info.ilv == ILV_LINE)
		{
			TransformQuadToLine((const Quad<SAMPLE>*)pLine, pixelCount, (SAMPLE*)pDst, stride, _transform);
		}
		_pbyteOutput += _info.bytesperline;
	}


	void NewLineDecoded(const void* pSrc, int pixelCount, int byteStride)
	{
		if (_info.components == 3)
		{	
			if (_info.ilv == ILV_SAMPLE)
			{
				TransformLine((Triplet<SAMPLE>*)_pbyteOutput, (const Triplet<SAMPLE>*)pSrc, pixelCount, _inverseTransform);
			}
			else
			{
				TransformLineToTriplet((const SAMPLE*)pSrc, byteStride, (Triplet<SAMPLE>*)_pbyteOutput, pixelCount, _inverseTransform);
			}
		}
		else if (_info.components == 4 && _info.ilv == ILV_LINE)
		{
			TransformLineToQuad((const SAMPLE*)pSrc, byteStride, (Quad<SAMPLE>*)_pbyteOutput, pixelCount, _inverseTransform);
		}

		if (_info.outputBgr)
		{
			TransformRgbToBgr(_pbyteOutput, _info.components, pixelCount);
		}
		_pbyteOutput += _info.bytesperline;		
	}


private:
	BYTE* _pbyteOutput;
	const JlsParameters& _info;	
	std::vector<SAMPLE> _templine;
	TRANSFORM _transform;	
	typename TRANSFORM::INVERSE _inverseTransform;
};



#endif
