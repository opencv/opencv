// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 

#include "config.h"
#include "util.h"
#include "header.h"
#include "streams.h"
#include "decoderstrategy.h"
#include "encoderstrategy.h"
#include <memory>


// JFIF\0
BYTE jfifID[] = {'J','F','I','F','\0'};


bool IsDefault(const JlsCustomParameters* pcustom)
{
	if (pcustom->MAXVAL != 0)
		return false;

	if (pcustom->T1 != 0)
		return false;

	if (pcustom->T2 != 0)
		return false;

	if (pcustom->T3 != 0)
		return false;

	if (pcustom->RESET != 0)
		return false;

	return true;
}


LONG CLAMP(LONG i, LONG j, LONG MAXVAL)
{
	if (i > MAXVAL || i < j)
		return j;

	return i;
}


JlsCustomParameters ComputeDefault(LONG MAXVAL, LONG NEAR)
{
	JlsCustomParameters preset = JlsCustomParameters();

	LONG FACTOR = (MIN(MAXVAL, 4095) + 128)/256;

	preset.T1 = CLAMP(FACTOR * (BASIC_T1 - 2) + 2 + 3*NEAR, NEAR + 1, MAXVAL);
	preset.T2 = CLAMP(FACTOR * (BASIC_T2 - 3) + 3 + 5*NEAR, preset.T1, MAXVAL);
	preset.T3 = CLAMP(FACTOR * (BASIC_T3 - 4) + 4 + 7*NEAR, preset.T2, MAXVAL);
	preset.MAXVAL = MAXVAL;
	preset.RESET = BASIC_RESET;
	return preset;
}


JLS_ERROR CheckParameterCoherent(const JlsParameters* pparams)
{
	if (pparams->bitspersample < 6 || pparams->bitspersample > 16)
		return ParameterValueNotSupported;

	if (pparams->ilv < 0 || pparams->ilv > 2)
		throw JlsException(InvalidCompressedData);

	if (pparams->bitspersample < 6 || pparams->bitspersample > 16)
		return ParameterValueNotSupported;

	switch (pparams->components)
	{
		case 4: return pparams->ilv == ILV_SAMPLE ? ParameterValueNotSupported : OK; 
		case 3: return OK;
		case 1: return pparams->ilv != ILV_NONE ? ParameterValueNotSupported : OK;
		case 0: return InvalidJlsParameters;

		default: return pparams->ilv != ILV_NONE ? ParameterValueNotSupported : OK; 
	}
}

//
// JpegMarkerSegment
//
class JpegMarkerSegment : public JpegSegment
{
public:
	JpegMarkerSegment(BYTE marker, std::vector<BYTE> vecbyte)
	{
		_marker = marker;
		std::swap(_vecbyte, vecbyte);
	}

	virtual void Write(JLSOutputStream* pstream)
	{
		pstream->WriteByte(0xFF);
		pstream->WriteByte(_marker);
		pstream->WriteWord(USHORT(_vecbyte.size() + 2));
		pstream->WriteBytes(_vecbyte);		
	}

	BYTE _marker;
	std::vector<BYTE> _vecbyte;
};


//
// push_back()
//
void push_back(std::vector<BYTE>& vec, USHORT value)
{
	vec.push_back(BYTE(value / 0x100));
	vec.push_back(BYTE(value % 0x100));
}				   


//
// CreateMarkerStartOfFrame()
//
JpegSegment* CreateMarkerStartOfFrame(Size size, LONG bitsPerSample, LONG ccomp)
{

	std::vector<BYTE> vec;
	vec.push_back(static_cast<BYTE>(bitsPerSample));
	push_back(vec, static_cast<USHORT>(size.cy));
	push_back(vec, static_cast<USHORT>(size.cx));
	
	// components
	vec.push_back(static_cast<BYTE>(ccomp));
	for (BYTE component = 0; component < ccomp; component++)
	{
		// rescaling
		vec.push_back(component + 1);
		vec.push_back(0x11); 
		//"Tq1" reserved, 0
		vec.push_back(0);		
	}

	return new JpegMarkerSegment(JPEG_SOF, vec);
}




//
// ctor()
//
JLSOutputStream::JLSOutputStream() :
	_bCompare(false),
	_pdata(NULL),
	_cbyteOffset(0),
	_cbyteLength(0),
	_icompLast(0)
{
}



//
// dtor()
//
JLSOutputStream::~JLSOutputStream()
{
	for (size_t i = 0; i < _segments.size(); ++i)
	{
		delete _segments[i];
	}
	_segments.empty();
}




//
// Init()
//
void JLSOutputStream::Init(Size size, LONG bitsPerSample, LONG ccomp)
{
		_segments.push_back(CreateMarkerStartOfFrame(size, bitsPerSample, ccomp));
}


void JLSOutputStream::AddColorTransform(int i)
{
	std::vector<BYTE> rgbyteXform;
	rgbyteXform.push_back('m');
	rgbyteXform.push_back('r');
	rgbyteXform.push_back('f');
	rgbyteXform.push_back('x');
	rgbyteXform.push_back((BYTE)i);
			
	_segments.push_back(new JpegMarkerSegment(JPEG_APP8, rgbyteXform));	
}


//
// Write()
//
size_t JLSOutputStream::Write(BYTE* pdata, size_t cbyteLength)
{
	_pdata = pdata;
	_cbyteLength = cbyteLength;

	WriteByte(0xFF);
	WriteByte(JPEG_SOI);
	
	for (size_t i = 0; i < _segments.size(); ++i)
	{
		_segments[i]->Write(this);
	}

	//_bCompare = false;

	WriteByte(0xFF);
	WriteByte(JPEG_EOI);

	return _cbyteOffset;
}



JLSInputStream::JLSInputStream(const BYTE* pdata, size_t cbyteLength) :
		_pdata(pdata),
		_cbyteOffset(0),
		_cbyteLength(cbyteLength),
		_bCompare(false),
		_info(),
		_rect()
{
}

//
// Read()
//
void JLSInputStream::Read(void* pvoid, size_t cbyteAvailable)
{
	ReadHeader();

	JLS_ERROR error = CheckParameterCoherent(&_info);
	if (error != OK)
		throw JlsException(error);

	ReadPixels(pvoid, cbyteAvailable);
}





//
// ReadPixels()
//
void JLSInputStream::ReadPixels(void* pvoid, size_t cbyteAvailable)
{

	if (_rect.Width <= 0)
	{
		_rect.Width = _info.width;
		_rect.Height = _info.height;
	}

	int64_t cbytePlane = (int64_t)(_rect.Width) * _rect.Height * ((_info.bitspersample + 7)/8);

	if (int64_t(cbyteAvailable) < cbytePlane * _info.components)
		throw JlsException(UncompressedBufferTooSmall);
 	
	int scancount = _info.ilv == ILV_NONE ? _info.components : 1;

	BYTE* pbyte = (BYTE*)pvoid;
	for (LONG scan = 0; scan < scancount; ++scan)
	{
		ReadScan(pbyte);
		pbyte += cbytePlane;
	}	
}

// ReadNBytes()
//
void JLSInputStream::ReadNBytes(std::vector<char>& dst, int byteCount)
{
	for (int i = 0; i < byteCount; ++i)
	{
		dst.push_back((char)ReadByte());
	}
}


//
// ReadHeader()
//
void JLSInputStream::ReadHeader()
{
	if (ReadByte() != 0xFF)
		throw JlsException(InvalidCompressedData);

	if (ReadByte() != JPEG_SOI)
		throw JlsException(InvalidCompressedData);
	
	for (;;)
	{
		if (ReadByte() != 0xFF)
			throw JlsException(InvalidCompressedData);

		BYTE marker = (BYTE)ReadByte();

		size_t cbyteStart = _cbyteOffset;
		LONG cbyteMarker = ReadWord();

		switch (marker)
		{
			case JPEG_SOS: ReadStartOfScan();  break;
			case JPEG_SOF: ReadStartOfFrame(); break;
			case JPEG_COM: ReadComment();	   break;
			case JPEG_LSE: ReadPresetParameters();	break;
			case JPEG_APP0: ReadJfif(); break;
			case JPEG_APP7: ReadColorSpace(); break;
			case JPEG_APP8: ReadColorXForm(); break;			
			// Other tags not supported (among which DNL DRI)
			default: 		throw JlsException(ImageTypeNotSupported);
		}

		if (marker == JPEG_SOS)
		{				
			_cbyteOffset = cbyteStart - 2;
			return;
		}
		_cbyteOffset = cbyteStart + cbyteMarker;
	}
}


JpegMarkerSegment* EncodeStartOfScan(const JlsParameters* pparams, LONG icomponent)
{
	BYTE itable		= 0;
	
	std::vector<BYTE> rgbyte;

	if (icomponent < 0)
	{
		rgbyte.push_back((BYTE)pparams->components);
		for (LONG icomponent_ = 0; icomponent_ < pparams->components; ++icomponent_ )
		{
			rgbyte.push_back(BYTE(icomponent_ + 1));
			rgbyte.push_back(itable);
		}
	}
	else
	{
		rgbyte.push_back(1);
		rgbyte.push_back((BYTE)icomponent);
		rgbyte.push_back(itable);	
	}

	rgbyte.push_back(BYTE(pparams->allowedlossyerror));
	rgbyte.push_back(BYTE(pparams->ilv));
	rgbyte.push_back(0); // transform

	return new JpegMarkerSegment(JPEG_SOS, rgbyte);
}



JpegMarkerSegment* CreateLSE(const JlsCustomParameters* pcustom)
{
	std::vector<BYTE> rgbyte;

	rgbyte.push_back(1);
	push_back(rgbyte, (USHORT)pcustom->MAXVAL);
	push_back(rgbyte, (USHORT)pcustom->T1);
	push_back(rgbyte, (USHORT)pcustom->T2);
	push_back(rgbyte, (USHORT)pcustom->T3);
	push_back(rgbyte, (USHORT)pcustom->RESET);
	
	return new JpegMarkerSegment(JPEG_LSE, rgbyte);
}

//
// ReadPresetParameters()
//
void JLSInputStream::ReadPresetParameters()
{
	LONG type = ReadByte();


	switch (type)
	{
	case 1:
		{
			_info.custom.MAXVAL = ReadWord();
			_info.custom.T1 = ReadWord();
			_info.custom.T2 = ReadWord();
			_info.custom.T3 = ReadWord();
			_info.custom.RESET = ReadWord();
			return;
		}
	}

	
}


//
// ReadStartOfScan()
//
void JLSInputStream::ReadStartOfScan()
{
	LONG ccomp = ReadByte();
	for (LONG i = 0; i < ccomp; ++i)
	{
		ReadByte();
		ReadByte();
	}
	_info.allowedlossyerror = ReadByte();
	_info.ilv = interleavemode(ReadByte());
	
	if(_info.bytesperline == 0)
	{
		int width = _rect.Width != 0 ? _rect.Width : _info.width;
		int components = _info.ilv == ILV_NONE ? 1 : _info.components;
		_info.bytesperline = components * width * ((_info.bitspersample + 7)/8);
	}
}


//
// ReadComment()
//
void JLSInputStream::ReadComment()
{}


//
// ReadJfif()
//
void JLSInputStream::ReadJfif()
{
	for(int i = 0; i < (int)sizeof(jfifID); i++)
	{
		if(jfifID[i] != ReadByte())
			return;
	}
	_info.jfif.Ver   = ReadWord();

	// DPI or DPcm
	_info.jfif.units = ReadByte();
	_info.jfif.XDensity = ReadWord();
	_info.jfif.YDensity = ReadWord();
	
	// thumbnail
	_info.jfif.Xthumb = ReadByte();
	_info.jfif.Ythumb = ReadByte();
	if(_info.jfif.Xthumb > 0 && _info.jfif.pdataThumbnail) 
	{
		std::vector<char> tempbuff((char*)_info.jfif.pdataThumbnail, (char*)_info.jfif.pdataThumbnail+3*_info.jfif.Xthumb*_info.jfif.Ythumb);
		ReadNBytes(tempbuff, 3*_info.jfif.Xthumb*_info.jfif.Ythumb);
	}
}

//
// CreateJFIF()
//
JpegMarkerSegment* CreateJFIF(const JfifParameters* jfif)
{
	std::vector<BYTE> rgbyte;
	for(int i = 0; i < (int)sizeof(jfifID); i++)
	{
		rgbyte.push_back(jfifID[i]);
	}

	push_back(rgbyte, (USHORT)jfif->Ver);

	rgbyte.push_back(jfif->units);	
	push_back(rgbyte, (USHORT)jfif->XDensity);
	push_back(rgbyte, (USHORT)jfif->YDensity);

	// thumbnail
	rgbyte.push_back((BYTE)jfif->Xthumb);
	rgbyte.push_back((BYTE)jfif->Ythumb);
	if(jfif->Xthumb > 0) 
	{
		if(jfif->pdataThumbnail)
			throw JlsException(InvalidJlsParameters);

		rgbyte.insert(rgbyte.end(), (BYTE*)jfif->pdataThumbnail, (BYTE*)jfif->pdataThumbnail+3*jfif->Xthumb*jfif->Ythumb
		);
	}
	
	return new JpegMarkerSegment(JPEG_APP0, rgbyte);
}


//
// ReadStartOfFrame()
//
void JLSInputStream::ReadStartOfFrame()
{
	_info.bitspersample = ReadByte();
	int cline = ReadWord();
	int ccol = ReadWord();
	_info.width = ccol;
	_info.height = cline;
	_info.components= ReadByte();
}


//
// ReadByte()
//
BYTE JLSInputStream::ReadByte()
{  
    if (_cbyteOffset >= _cbyteLength)
	throw JlsException(InvalidCompressedData);

    return _pdata[_cbyteOffset++]; 
}


//
// ReadWord()
//
int JLSInputStream::ReadWord()
{
	int i = ReadByte() * 256;
	return i + ReadByte();
}


void JLSInputStream::ReadScan(void* pvout) 
{
	std::auto_ptr<DecoderStrategy> qcodec = JlsCodecFactory<DecoderStrategy>().GetCodec(_info, _info.custom);
	
	_cbyteOffset += qcodec->DecodeScan(pvout, _rect, _pdata + _cbyteOffset, _cbyteLength - _cbyteOffset, _bCompare); 
}


class JpegImageDataSegment: public JpegSegment
{
public:
	JpegImageDataSegment(const void* pvoidRaw, const JlsParameters& info, LONG icompStart, int ccompScan)  :
		_ccompScan(ccompScan),
		_icompStart(icompStart),
		_pvoidRaw(pvoidRaw),
		_info(info)
	{
	}

	void Write(JLSOutputStream* pstream)
	{		
		JlsParameters info = _info;
		info.components = _ccompScan;	
		std::auto_ptr<EncoderStrategy> qcodec =JlsCodecFactory<EncoderStrategy>().GetCodec(info, _info.custom);
		size_t cbyteWritten = qcodec->EncodeScan((BYTE*)_pvoidRaw, pstream->GetPos(), pstream->GetLength(), pstream->_bCompare ? pstream->GetPos() : NULL); 
		pstream->Seek(cbyteWritten);
	}


	int _ccompScan;
	LONG _icompStart;
	const void* _pvoidRaw;
	JlsParameters _info;
};


void JLSOutputStream::AddScan(const void* compareData, const JlsParameters* pparams)
{
	if (pparams->jfif.Ver)
	{
		_segments.push_back(CreateJFIF(&pparams->jfif));
	}
	if (!IsDefault(&pparams->custom))
	{
		_segments.push_back(CreateLSE(&pparams->custom));		
	}
	else if (pparams->bitspersample > 12)
	{
		JlsCustomParameters preset = ComputeDefault((1 << pparams->bitspersample) - 1, pparams->allowedlossyerror);
        _segments.push_back(CreateLSE(&preset));
	}

	_icompLast += 1;
	_segments.push_back(EncodeStartOfScan(pparams,pparams->ilv == ILV_NONE ? _icompLast : -1));

	//Size size = Size(pparams->width, pparams->height);
	int ccomp = pparams->ilv == ILV_NONE ? 1 : pparams->components;
		_segments.push_back(new JpegImageDataSegment(compareData, *pparams, _icompLast, ccomp));
}


//
// ReadColorSpace()
//
void JLSInputStream::ReadColorSpace()
{}



//
// ReadColorXForm()
//
void JLSInputStream::ReadColorXForm()
{
	std::vector<char> sourceTag;
	ReadNBytes(sourceTag, 4);

	if(strncmp(&sourceTag[0],"mrfx", 4) != 0)
		return;
	
	int xform = ReadByte();
	switch(xform) 
	{
		case COLORXFORM_NONE:
		case COLORXFORM_HP1:
		case COLORXFORM_HP2:
		case COLORXFORM_HP3:
			_info.colorTransform = xform;
			return;
		case COLORXFORM_RGB_AS_YUV_LOSSY:
		case COLORXFORM_MATRIX:
			throw JlsException(ImageTypeNotSupported);
		default:
			throw JlsException(InvalidCompressedData);
	}
}

