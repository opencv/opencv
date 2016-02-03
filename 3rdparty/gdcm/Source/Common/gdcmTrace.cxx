/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTrace.h"

#include <iostream>
#include <fstream>

namespace gdcm
{
//-----------------------------------------------------------------------------
// Warning message level to be displayed
static bool DebugFlag   = false;
static bool WarningFlag = true;
static bool ErrorFlag   = true;
// Stream based API:
static std::ostream * DebugStream   = &std::cerr;
static std::ostream * WarningStream = &std::cerr;
static std::ostream * ErrorStream   = &std::cerr;
// File based API:
static bool UseStreamToFile       = false;
static std::ofstream * FileStream = NULL;

void Trace::SetStreamToFile( const char *filename )
{
  if( !filename ) return;
  if( UseStreamToFile )
    {
    assert( FileStream );
    FileStream->close();
    FileStream = NULL;
    UseStreamToFile = false;
    }
  std::ofstream * out = new std::ofstream;
  if( !out ) return;
  out->open( filename );
  if( !out->good() ) return;
  assert( !FileStream && !UseStreamToFile );
  FileStream = out;
  UseStreamToFile = true;
  DebugStream   = FileStream;
  WarningStream = FileStream;
  ErrorStream   = FileStream;
}

void Trace::SetStream(std::ostream &os)
{
  if( !os.good() ) return;
  if( UseStreamToFile )
    {
    assert( FileStream );
    FileStream->close();
    FileStream = NULL;
    UseStreamToFile = false;
    }
  DebugStream   = &os;
  WarningStream = &os;
  ErrorStream   = &os;
}

std::ostream &Trace::GetStream()
{
  return *DebugStream;
}

void Trace::SetDebugStream(std::ostream &os)
{
  DebugStream = &os;
}

std::ostream &Trace::GetDebugStream()
{
  return *DebugStream;
}

void Trace::SetWarningStream(std::ostream &os)
{
  WarningStream = &os;
}

std::ostream &Trace::GetWarningStream()
{
  return *WarningStream;
}

void Trace::SetErrorStream(std::ostream &os)
{
  ErrorStream = &os;
}

std::ostream &Trace::GetErrorStream()
{
  return *ErrorStream;
}

//-----------------------------------------------------------------------------
// Constructor / Destructor
Trace::Trace()
{
  DebugFlag = WarningFlag = ErrorFlag = false;
}

Trace::~Trace()
{
  if( UseStreamToFile )
    {
    assert( FileStream );
    FileStream->close();
    FileStream = NULL;
    }
}

void Trace::SetDebug(bool debug)  { DebugFlag = debug; }
void Trace::DebugOn()  { DebugFlag = true; }
void Trace::DebugOff() { DebugFlag = false; }
bool Trace::GetDebugFlag()
{
  return DebugFlag;
}

void Trace::SetWarning(bool warning)  { WarningFlag = warning; }
void Trace::WarningOn()  { WarningFlag = true; }
void Trace::WarningOff() { WarningFlag = false; }
bool Trace::GetWarningFlag()
{
  return WarningFlag;
}

void Trace::SetError(bool error)  { ErrorFlag = error; }
void Trace::ErrorOn()  { ErrorFlag = true; }
void Trace::ErrorOff() { ErrorFlag = false; }
bool Trace::GetErrorFlag()
{
  return ErrorFlag;
}

} // end namespace gdcm
