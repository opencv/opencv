// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 
#ifndef CHARLS_STREAMS
#define CHARLS_STREAMS

#include <memory>
#include <vector>
#include "util.h"



// This file defines JPEG-LS streams: The header and the actual pixel data. Header markers have fixed length, the pixeldata not. 



class JpegSegment;

enum JPEGLS_ColorXForm
{
	// default (RGB)
	COLORXFORM_NONE = 0,	

	// Color transforms as defined by HP
	COLORXFORM_HP1,
	COLORXFORM_HP2,
	COLORXFORM_HP3,

	// Defined by HP but not supported by CharLS
	COLORXFORM_RGB_AS_YUV_LOSSY,
	COLORXFORM_MATRIX
};
	
//
// JLSOutputStream: minimal implementation to write JPEG header streams
//
class JLSOutputStream
{
	friend class JpegMarkerSegment;
	friend class JpegImageDataSegment;

public:
	JLSOutputStream();
	virtual ~JLSOutputStream();

	void Init(Size size, LONG bitsPerSample, LONG ccomp);
	void AddScan(const void* compareData, const JlsParameters* pparams);
	void AddLSE(const JlsCustomParameters* pcustom);
	void AddColorTransform(int i);
	size_t GetBytesWritten()
		{ return _cbyteOffset; }

	size_t GetLength()
		{ return _cbyteLength - _cbyteOffset; }

	size_t Write(BYTE* pdata, size_t cbyteLength);
	
	void EnableCompare(bool bCompare) 
	{ _bCompare = bCompare; }
private:
	BYTE* GetPos() const
		{ return _pdata + _cbyteOffset; }

	void WriteByte(BYTE val)
	{ 
		ASSERT(!_bCompare || _pdata[_cbyteOffset] == val);
		
		_pdata[_cbyteOffset++] = val; 
	}

	void WriteBytes(const std::vector<BYTE>& rgbyte)
	{
		for (size_t i = 0; i < rgbyte.size(); ++i)
		{
			WriteByte(rgbyte[i]);
		}		
	}

	void WriteWord(USHORT val)
	{
		WriteByte(BYTE(val / 0x100));
		WriteByte(BYTE(val % 0x100));
	}


    void Seek(size_t byteCount)
		{ _cbyteOffset += byteCount; }

	bool _bCompare;

private:
	BYTE* _pdata;
	size_t _cbyteOffset;
	size_t _cbyteLength;
	LONG _icompLast;
	std::vector<JpegSegment*> _segments;
};


//
// JLSInputStream: minimal implementation to read JPEG header streams
//
class JLSInputStream
{
public:
	JLSInputStream(const BYTE* pdata, size_t cbyteLength);

	size_t GetBytesRead()
		{ return _cbyteOffset; }

	const JlsParameters& GetMetadata() const
		{ return _info; } 

	const JlsCustomParameters& GetCustomPreset() const
	{ return _info.custom; } 

	void Read(void* pvoid, size_t cbyteAvailable);
	void ReadHeader();
	
	void EnableCompare(bool bCompare)
		{ _bCompare = bCompare;	}

	void SetInfo(JlsParameters* info) { _info = *info; }

	void SetRect(JlsRect rect) { _rect = rect; }

private:
	void ReadPixels(void* pvoid, size_t cbyteAvailable);
	void ReadScan(void*);	
	void ReadStartOfScan();
	void ReadPresetParameters();
	void ReadComment();
	void ReadStartOfFrame();
	BYTE ReadByte();
	int ReadWord();
	void ReadNBytes(std::vector<char>& dst, int byteCount);

	// JFIF
	void ReadJfif();
	// Color Transform Application Markers & Code Stream (HP extension)
	void ReadColorSpace();
	void ReadColorXForm();
	
private:
	const BYTE* _pdata;
	size_t _cbyteOffset;
	size_t _cbyteLength;
	bool _bCompare;
	JlsParameters _info;
	JlsRect _rect;
};




#endif
