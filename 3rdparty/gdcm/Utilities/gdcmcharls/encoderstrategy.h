// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 

#ifndef CHARLS_ENCODERSTRATEGY
#define CHARLS_ENCODERSTRATEGY

#include "processline.h"
#include "decoderstrategy.h"

// Implements encoding to stream of bits. In encoding mode JpegLsCodec inherits from EncoderStrategy

class EncoderStrategy
{

public:
	explicit EncoderStrategy(const JlsParameters& info) :
		 _qdecoder(0),
		 _info(info),
		 _processLine(0),
 		 valcurrent(0),
		 bitpos(0),
		 _isFFWritten(false),
		 _bytesWritten(0)
		
	{
	}

	virtual ~EncoderStrategy() 
	{
	}

	LONG PeekByte();
	
	void OnLineBegin(LONG cpixel, void* ptypeBuffer, LONG pixelStride)
	{
		_processLine->NewLineRequested(ptypeBuffer, cpixel, pixelStride);
	}

	void OnLineEnd(LONG /*cpixel*/, void* /*ptypeBuffer*/, LONG /*pixelStride*/) { }

    virtual void SetPresets(const JlsCustomParameters& presets) = 0;
		
	virtual size_t EncodeScan(const void* pvoid, void* pvoidOut, size_t byteCount, void* pvoidCompare) = 0;

protected:

	void Init(BYTE* compressedBytes, size_t byteCount)
	{
		bitpos = 32;
		valcurrent = 0;
		_position = compressedBytes;
   		_compressedLength = byteCount;
	}


	void AppendToBitStream(LONG value, LONG length)
	{	
		ASSERT(length < 32 && length >= 0);

		ASSERT((_qdecoder.get() == NULL) || (length == 0 && value == 0) ||( _qdecoder->ReadLongValue(length) == value));

#ifndef NDEBUG
		if (length < 32)
		{
			int mask = (1 << (length)) - 1;
			ASSERT((value | mask) == mask);
		}
#endif

		bitpos -= length;
		if (bitpos >= 0)
		{
			valcurrent = valcurrent | (value << bitpos);
			return;
		}
		valcurrent |= value >> -bitpos;

		Flush();
	        
		ASSERT(bitpos >=0);
		valcurrent |= value << bitpos;	

	}

	void EndScan()
	{
		Flush();

		// if a 0xff was written, Flush() will force one unset bit anyway
		if (_isFFWritten)
			AppendToBitStream(0, (bitpos - 1) % 8);
		else
			AppendToBitStream(0, bitpos % 8);
		
		Flush();
		ASSERT(bitpos == 0x20);
	}

	void Flush()
	{
		for (LONG i = 0; i < 4; ++i)
		{
			if (bitpos >= 32)
				break;

			if (_isFFWritten)
			{
				// insert highmost bit
				*_position = BYTE(valcurrent >> 25);
				valcurrent = valcurrent << 7;			
				bitpos += 7;	
				_isFFWritten = false;
			}
			else
			{
				*_position = BYTE(valcurrent >> 24);
				valcurrent = valcurrent << 8;			
				bitpos += 8;			
				_isFFWritten = *_position == 0xFF;			
			}
			
			_position++;
			_compressedLength--;
			_bytesWritten++;

		}
		
	}

	size_t GetLength() 
	{ 
		return _bytesWritten - (bitpos -32)/8; 
	}


	inlinehint void AppendOnesToBitStream(LONG length)
	{
		AppendToBitStream((1 << length) - 1, length);	
	}


	std::auto_ptr<DecoderStrategy> _qdecoder; 

protected:
	JlsParameters _info;
	std::auto_ptr<ProcessLine> _processLine;
private:

	unsigned int valcurrent;
	LONG bitpos;
	size_t _compressedLength;
	
	// encoding
	BYTE* _position;
	bool _isFFWritten;
	size_t _bytesWritten;

};

#endif
