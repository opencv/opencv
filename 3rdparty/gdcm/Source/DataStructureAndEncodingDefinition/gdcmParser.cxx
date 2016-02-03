/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmParser.h"

namespace gdcm
{

static const char *ErrorStrings[] =
{
  "FATAL",
  NULL
};

bool Parser::Parse(const char *buffer, int len, bool isFinal)
{
  if (len == 0)
    {
    if (!isFinal)
      {
      return true;
      }
    //PositionPtr = BufferPtr;
    //ErrorCode = processor(BufferPtr, ParseEndPtr = BufferEnd, 0);
    //if (ErrorCode == NoError)
    //  return true;
    //EventEndPtr = EventPtr;
    //ProcessorPtr = ErrorProcessor;
    return false;
    }
  else
    {
    memcpy(GetBuffer(len), buffer, len);
    return ParseBuffer(len, isFinal);
    }
}


char *Parser::GetBuffer(int len)
{
  return Buffer.Get(len);
}

Parser::ErrorType Parser::Process()
{
  return Parser::NoError;
}

bool Parser::ParseBuffer(int len, bool isFinal)
{
  const char *start = Buffer.GetStart();
  const char *positionPtr = start;
  (void)positionPtr;
  Buffer.ShiftEnd(len); //bufferEnd += len;
  //parseEndByteIndex += len;
  ErrorCode = Process();
  if (ErrorCode == Parser::NoError)
    {
    if (!isFinal)
      Buffer.UpdatePosition();
    return 1;
    }
  //else
    {
    //eventEndPtr = eventPtr;
    //processor = errorProcessor;
    }
  return 0;
}


void Parser::SetUserData(void *userData)
{
  (void)userData;
}

void * Parser::GetUserData() const
{
  return UserData;
}

void Parser::SetElementHandler(StartElementHandler start, EndElementHandler end)
{
  StartElement = start;
  EndElement = end;
}

unsigned long Parser::GetCurrentByteIndex() const
{
  return 0;
}

Parser::ErrorType Parser::GetErrorCode() const
{
  return ErrorCode;
}

const char *Parser::GetErrorString(ErrorType const &err)
{
  return ErrorStrings[(int)err];
}

} // end namespace gdcm
