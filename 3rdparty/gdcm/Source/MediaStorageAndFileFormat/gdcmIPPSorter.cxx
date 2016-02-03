/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmIPPSorter.h"
#include "gdcmScanner.h"
#include "gdcmElement.h"
#include "gdcmDirectionCosines.h"

#include <map>
#include <math.h>

namespace gdcm
{

IPPSorter::IPPSorter()
{
  ComputeZSpacing = true;
  DropDuplicatePositions = false;
  ZSpacing = 0;
  ZTolerance = 1e-6;
  DirCosTolerance = 0.;
}


inline double spacing_round(double n, int d) /* pow is defined as pow( double, double) or pow(double int) on M$ comp */
{
  return floor(n * pow(10., d) + .5) / pow(10., d);
}

bool IPPSorter::Sort(std::vector<std::string> const & filenames)
{
  // BUG: I cannot clear Filenames since input filenames could also be the output of ourself...
  // Filenames.clear();
  ZSpacing = 0;
  if( filenames.empty() )
    {
    Filenames.clear();
    return true;
    }

  Scanner scanner;
  const Tag tipp(0x0020,0x0032); // Image Position (Patient)
  const Tag tiop(0x0020,0x0037); // Image Orientation (Patient)
  const Tag tframe(0x0020,0x0052); // Frame of Reference UID
  // Temporal Position Identifier (0020,0100) 3 Temporal order of a dynamic or functional set of Images.
  //const Tag tpi(0x0020,0x0100);
  scanner.AddTag( tipp );
  scanner.AddTag( tiop );
  scanner.AddTag( tframe );
  bool b = scanner.Scan( filenames );
  if( !b )
    {
    gdcmDebugMacro( "Scanner failed" );
    return false;
    }
  Scanner::ValuesType iops = scanner.GetValues(tiop);
  Scanner::ValuesType frames = scanner.GetValues(tframe);
  if( DirCosTolerance == 0. )
    {
    if( iops.size() != 1 )
      {
      gdcmDebugMacro( "More than one IOP (or no IOP): " << iops.size() );
      //std::copy(iops.begin(), iops.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
      return false;
      }
    }
  if( frames.size() > 1 ) // Should I really tolerate no Frame of Reference UID ?
    {
    gdcmDebugMacro( "More than one Frame Of Reference UID" );
    return false;
    }

  const char *reference = filenames[0].c_str();
  // we cannot simply consider the first file, what if this is not DICOM ?
  for(std::vector<std::string>::const_iterator it1 = filenames.begin();
    it1 != filenames.end(); ++it1)
    {
    const char *filename = it1->c_str();
    bool iskey = scanner.IsKey(filename);
    if( iskey )
      {
      reference = filename;
      }
    }
  Scanner::TagToValue const &t2v = scanner.GetMapping(reference);
  Scanner::TagToValue::const_iterator it = t2v.find( tiop );
  // Take the first file in the list of filenames, if not IOP is found, simply gives up:
  if( it == t2v.end() )
    {
    // DEAD CODE
    gdcmDebugMacro( "No iop in: " << reference );
    return false;
    }
  if( it->first != tiop )
    {
    // first file does not contains Image Orientation (Patient), let's give up
    gdcmDebugMacro( "No iop in first file ");
    return false;
    }
  const char *dircos = it->second;
  if( !dircos )
    {
    // first file does contains Image Orientation (Patient), but it is empty
    gdcmDebugMacro( "Empty iop in first file ");
    return false;
    }

  // http://www.itk.org/pipermail/insight-users/2003-September/004762.html
  // Compute normal:
  // The steps I take when reconstructing a volume are these: First,
  // calculate the slice normal from IOP:
  double normal[3];

  DirectionCosines dc;
  dc.SetFromString( dircos );
  if( !dc.IsValid() ) return false;
  dc.Cross( normal );
  // You only have to do this once for all slices in the volume. Next, for
  // each slice, calculate the distance along the slice normal using the IPP
  // tag ("dist" is initialized to zero before reading the first slice) :
  //typedef std::multimap<double, const char*> SortedFilenames;
  typedef std::map<double, const char*> SortedFilenames;
  SortedFilenames sorted;
{
  std::vector<std::string>::const_iterator it1 = filenames.begin();
  DirectionCosines dc2;
  for(; it1 != filenames.end(); ++it1)
    {
    const char *filename = it1->c_str();
    bool iskey = scanner.IsKey(filename);
    if( iskey )
      {
      const char *value =  scanner.GetValue(filename, tipp);
      if( value )
        {
        if( DirCosTolerance != 0. )
          {
          const char *value2 =  scanner.GetValue(filename, tiop);
          if( !dc2.SetFromString( value2 ) )
            {
            if( value2 )
              gdcmWarningMacro( filename << " cant read IOP: " << value2 );
            return false;
            }
          double cd = dc2.CrossDot( dc );
          // result should be as close to 1 as possible:
          if( fabs(1 - cd) > DirCosTolerance )
            {
            gdcmWarningMacro( filename << " Problem with DirCosTolerance: " );
            // Cant print cd since 0.9999 is printed as 1... may confuse user
            return false;
            }
          //dc2.Normalize();
          //dc2.Print( std::cout << std::endl );
          }
        //gdcmDebugMacro( filename << " has " << ipp << " = " << value );
        Element<VR::DS,VM::VM3> ipp;
        std::stringstream ss;
        ss.str( value );
        ipp.Read( ss );
        double dist = 0;
        for (int i = 0; i < 3; ++i) dist += normal[i]*ipp[i];
        // FIXME: This test is weak, since implicitely we are doing a != on floating point value
        if( sorted.find(dist) != sorted.end() )
          {
            if( this->DropDuplicatePositions )
            {
              gdcmWarningMacro( "dropping file " << filename << " since Z position: " << dist << " already found" );
              continue;
            }
            gdcmWarningMacro( "dist: " << dist << " for " << filename <<
              " already found in " << sorted.find(dist)->second );
            return false;
          }
        sorted.insert(
          SortedFilenames::value_type(dist,filename) );
        }
      else
        {
        gdcmDebugMacro( "File: " << filename << " has no Tag" << tipp << ". Skipping." );
        }
      }
    else
      {
      gdcmDebugMacro( "File: " << filename << " could not be read. Skipping." );
      }
    }
}
  assert( !sorted.empty() );
{
  SortedFilenames::const_iterator it2 = sorted.begin();
  double prev = it2->first;
  Filenames.push_back( it2->second );
  if( sorted.size() > 1 )
    {
    bool spacingisgood = true;
    ++it2;
    double current = it2->first;
    double zspacing = current - prev;
    for( ; it2 != sorted.end(); ++it2)
      {
      //std::cout << it2->first << " " << it2->second << std::endl;
      current = it2->first;
      Filenames.push_back( it2->second );
      if( fabs((current - prev) - zspacing) > ZTolerance )
        {
        gdcmDebugMacro( "ZTolerance test failed. You need to decrease ZTolerance." );
        spacingisgood = false;
        }
      // update prev for the next for-loop
      prev = current;
      }
    // is spacing good ?
    if( spacingisgood && ComputeZSpacing )
      {
      // If user ask for a ZTolerance of 1e-4, there is no need for us to
      // store the extra digits... this will make sure to return 2.2 from a 2.1999938551239993 value
      const int l = (int)( -log10(ZTolerance) );
      ZSpacing = spacing_round(zspacing, l);
      }
    if( !spacingisgood )
      {
      std::ostringstream os;
      os << "Filenames and 'z' positions" << std::endl;
      double prev1 = 0.;
      for(SortedFilenames::const_iterator it1 = sorted.begin(); it1 != sorted.end(); ++it1)
        {
        std::string f = it1->second;
        if( f.length() > 32 )
          {
          f = f.substr(0,10) + " ... " + f.substr(f.length()-17);
          }
        double d = it1->first - prev1;
        if( it1 != sorted.begin() && fabs(d - zspacing) > ZTolerance) os << "* ";
        else os << "  ";
        os << it1->first << "\t" << f << std::endl;
        prev1 = it1->first;
        }
      gdcmDebugMacro( os.str() );
      }
    assert( spacingisgood == false ||  (ComputeZSpacing ? (ZSpacing > ZTolerance && ZTolerance > 0) : ZTolerance > 0) );
    }
}

  // return true: means sorting succeed, it does not mean spacing computation succeded !
  return true;
}

bool IPPSorter::ComputeSpacing(std::vector<std::string> const & filenames)
{
  (void)filenames;
  return false;
}

} // end namespace gdcm
