/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSERIEHELPER_H
#define GDCMSERIEHELPER_H

#include "gdcmTag.h"
#include "gdcmSmartPointer.h"
#include "gdcmFile.h"
#include <vector>
#include <string>
#include <map>

namespace gdcm
{

enum CompOperators {
   GDCM_EQUAL = 0,
   GDCM_DIFFERENT,
   GDCM_GREATER,
   GDCM_GREATEROREQUAL,
   GDCM_LESS,
   GDCM_LESSOREQUAL
};
enum LodModeType
{
   LD_ALL         = 0x00000000,
   LD_NOSEQ       = 0x00000001,
   LD_NOSHADOW    = 0x00000002,
   LD_NOSHADOWSEQ = 0x00000004
};


/**
 * \brief FileWithName
 *
 * \details
 * Backward only class do not use in newer code
 */
class GDCM_EXPORT FileWithName : public File
{
public:
  FileWithName(File &f):File(f),filename(){}
  std::string filename;
};

typedef std::vector< SmartPointer<FileWithName> > FileList;
typedef bool (*BOOL_FUNCTION_PFILE_PFILE_POINTER)(File *, File *);
class Scanner;

/**
 * \brief SerieHelper
 * DO NOT USE this class, it is only a temporary solution for ITK migration from GDCM 1.x to GDCM 2.x
 * It will disapear soon, you've been warned.
 *
 * Instead see ImageHelper or IPPSorter
 */
class GDCM_EXPORT SerieHelper
{
public:
  SerieHelper();
  ~SerieHelper();

  void Clear();
  void SetLoadMode (int ) {}
  void SetDirectory(std::string const &dir, bool recursive=false);

  void AddRestriction(const std::string & tag);
  void SetUseSeriesDetails( bool useSeriesDetails );
  void CreateDefaultUniqueSeriesIdentifier();
  FileList *GetFirstSingleSerieUIDFileSet();
  FileList *GetNextSingleSerieUIDFileSet();
  std::string CreateUniqueSeriesIdentifier( File * inFile );
  void OrderFileList(FileList *fileSet);
  void AddRestriction(uint16_t group, uint16_t elem, std::string const &value, int op);

protected:
  bool UserOrdering(FileList *fileSet);
  void AddFileName(std::string const &filename);
  bool AddFile(FileWithName &header);
  void AddRestriction(const Tag& tag);
  bool ImagePositionPatientOrdering(FileList *fileSet);
  bool FileNameOrdering( FileList *fileList );

  typedef struct {
    uint16_t group;
    uint16_t elem;
    std::string value;
    int op;
  } Rule;
  typedef std::vector<Rule> SerieRestrictions;

  typedef std::map<std::string, FileList *> SingleSerieUIDFileSetmap;
  SingleSerieUIDFileSetmap SingleSerieUIDFileSetHT;
  SingleSerieUIDFileSetmap::iterator ItFileSetHt;

private:
  SerieRestrictions Restrictions;
  SerieRestrictions Refine;

  bool UseSeriesDetails;
  bool DirectOrder;

  BOOL_FUNCTION_PFILE_PFILE_POINTER UserLessThanFunction;
};

// backward compat
} // end namespace gdcm


#endif //GDCMSERIEHELPER_H
