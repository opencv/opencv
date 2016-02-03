/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * HISTORY:
 * In GDCM 1.X the prefered terms was 'ReWrite', however one author of GDCM dislike
 * the term ReWrite since it is associated with the highly associated with the Rewrite
 * notion in software programming where using reinvent the wheel and rewrite from scratch code
 * the term convert was prefered
 *
 * Tools to conv. Goals being to 'purify' a DICOM file.
 * For now it will do the minimum:
 * - If Group Length is present, the length is guarantee to be correct
 * - If Element with Group Tag 0x1, 0x3, 0x5 or 0x7 are present they are
 *   simply discarded (not written).
 * - Elements are written in alphabetical order
 * - 32bits VR have the residue bytes sets to 0x0,0x0
 * - Same goes from Item Length end delimitor, sets to 0x0,0x0
 * - All buggy files (wrong length: GE, 13 and Siemens Leonardo) are fixed
 * - All size are even (no odd length from gdcm 1.x)
 *
 * // \todo:
 * // --preamble: clean preamble
 * // --meta: clean meta (meta info version...)
 * // --dicomV3 (use TS unless not supported)
 * // --recompute group-length
 * // --undefined sq
 * // --explicit sq *
 * \todo in a close future:
 * - Set appropriate VR from DICOM dict
 * - Rewrite PMS SQ into DICOM SQ
 * - Rewrite Implicit SQ with defined length as undefined length
 * - PixelData with `overlay` in unused bits should be cleanup
 * - Any broken JPEG file (wrong bits) should be fixed
 * - DicomObject bug should be fixed
 * - Meta and Dataset should have a matching UID (more generally File Meta
 *   should be correct (Explicit!) and consistant with DataSet)
 * - User should be able to specify he wants Group Length (or remove them)
 * - Media SOP should be correct (deduct from something else or set to
 *   SOP Secondary if all else fail).
 * - Padding character should be correct
 *
 * \todo distant future:
 * - Later on, it should run through a Validator
 *   which will make sure all field 1, 1C are present and those only
 * - In a perfect world I should remove private tags and transform them into
 *   public fields.
 * - DA should be correct, PN should be correct (no space!)
 * - Enumerated Value should be correct
 */
/*
 check-meta is ideal for image like:

  gdcmconv -C gdcmData/PICKER-16-MONO2-No_DicomV3_Preamble.dcm bla.dcm
*/
#include "gdcmReader.h"
#include "gdcmFileDerivation.h"
#include "gdcmAnonymizer.h"
#include "gdcmVersion.h"
#include "gdcmPixmapReader.h"
#include "gdcmPixmapWriter.h"
#include "gdcmWriter.h"
#include "gdcmSystem.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmIconImageGenerator.h"
#include "gdcmAttribute.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmUIDGenerator.h"
#include "gdcmImage.h"
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmImageApplyLookupTable.h"
#include "gdcmImageFragmentSplitter.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmImageChangePhotometricInterpretation.h"
#include "gdcmFileExplicitFilter.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmSequenceOfFragments.h"

#include <string>
#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

struct SetSQToUndefined
{
  void operator() (gdcm::DataElement &de) {
    de.SetVLToUndefined();
  }
};

static void PrintVersion()
{
  std::cout << "gdcmconv: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintLossyWarning()
{
  std::cout << "You have selected a lossy compression transfer syntax." << std::endl;
  std::cout << "This will degrade the quality of your input image, and can." << std::endl;
  std::cout << "impact professional interpretation of the image." << std::endl;
  std::cout << "Do not use if you do not understand the risk." << std::endl;
  std::cout << "WARNING: this mode is very experimental." << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmconv [OPTION] input.dcm output.dcm" << std::endl;
  std::cout << "Convert a DICOM file into another DICOM file.\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input      DICOM filename" << std::endl;
  std::cout << "  -o --output     DICOM filename" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -X --explicit            Change Transfer Syntax to explicit." << std::endl;
  std::cout << "  -M --implicit            Change Transfer Syntax to implicit." << std::endl;
  std::cout << "  -U --use-dict            Use dict for VR (only public by default)." << std::endl;
  std::cout << "     --with-private-dict   Use private dict for VR (advanced user only)." << std::endl;
  std::cout << "  -C --check-meta          Check File Meta Information (advanced user only)." << std::endl;
  std::cout << "     --root-uid            Root UID." << std::endl;
  std::cout << "     --remove-gl           Remove group length (deprecated in DICOM 2008)." << std::endl;
  std::cout << "     --remove-private-tags Remove private tags." << std::endl;
  std::cout << "     --remove-retired      Remove retired tags." << std::endl;
  std::cout << "Image only Options:" << std::endl;
  std::cout << "  -l --apply-lut                      Apply LUT (non-standard, advanced user only)." << std::endl;
  std::cout << "  -P --photometric-interpretation %s  Change Photometric Interpretation (when possible)." << std::endl;
  std::cout << "  -w --raw                            Decompress image." << std::endl;
  std::cout << "  -d --deflated                       Compress using deflated (gzip)." << std::endl;
  std::cout << "  -J --jpeg                           Compress image in jpeg." << std::endl;
  std::cout << "  -K --j2k                            Compress image in j2k." << std::endl;
  std::cout << "  -L --jpegls                         Compress image in jpeg-ls." << std::endl;
  std::cout << "  -R --rle                            Compress image in rle (lossless only)." << std::endl;
  std::cout << "  -F --force                          Force decompression/merging before recompression/splitting." << std::endl;
  std::cout << "     --generate-icon                  Generate icon." << std::endl;
  std::cout << "     --icon-minmax %d,%d              Min/Max value for icon." << std::endl;
  std::cout << "     --icon-auto-minmax               Automatically commpute best Min/Max values for icon." << std::endl;
  std::cout << "     --compress-icon                  Decide whether icon follows main TransferSyntax or remains uncompressed." << std::endl;
  std::cout << "     --planar-configuration [01]      Change planar configuration." << std::endl;
  std::cout << "  -Y --lossy                          Use the lossy (if possible) compressor." << std::endl;
  std::cout << "  -S --split %d                       Write 2D image with multiple fragments (using max size)" << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose    more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning    print warning info." << std::endl;
  std::cout << "  -D --debug      print debug info." << std::endl;
  std::cout << "  -E --error      print error info." << std::endl;
  std::cout << "  -h --help       print help." << std::endl;
  std::cout << "  -v --version    print version." << std::endl;
  std::cout << "     --quiet      do not print to stdout." << std::endl;
  std::cout << "JPEG Options:" << std::endl;
  std::cout << "  -q --quality %*f           set quality." << std::endl;
  std::cout << "JPEG-LS Options:" << std::endl;
  std::cout << "  -e --lossy-error %*i             set error." << std::endl;
  std::cout << "J2K Options:" << std::endl;
  std::cout << "  -r --rate    %*f           set rate." << std::endl;
  std::cout << "  -q --quality %*f           set quality." << std::endl;
  std::cout << "  -t --tile %d,%d            set tile size." << std::endl;
  std::cout << "  -n --number-resolution %d  set number of resolution." << std::endl;
  std::cout << "     --irreversible          set irreversible." << std::endl;
  std::cout << "Special Options:" << std::endl;
  std::cout << "  -I --ignore-errors   convert even if file is corrupted (advanced users only, see disclaimers)." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_ROOT_UID Root UID" << std::endl;
/*
 * Default behavior for root UID is:
 * By default the GDCM one is used
 * If GDCM_ROOT_UID is set, then use this one instead
 * If --root-uid is explicitly set on the command line, it will override any other defined behavior
 */
}

template <typename T>
static size_t readvector(std::vector<T> &v, const char *str)
{
  if( !str ) return 0;
  std::istringstream os( str );
  T f;
  while( os >> f )
    {
    v.push_back( f );
    os.get(); //  == ","
    }
  return v.size();
}

namespace gdcm
{
static bool derives( File & file, const Pixmap& compressed_image )
{
#if 1
  DataSet &ds = file.GetDataSet();

  if( !ds.FindDataElement( Tag(0x0008,0x0016) )
    || ds.GetDataElement( Tag(0x0008,0x0016) ).IsEmpty() )
    {
    return false;
    }
  if( !ds.FindDataElement( Tag(0x0008,0x0018) )
    || ds.GetDataElement( Tag(0x0008,0x0018) ).IsEmpty() )
    {
    return false;
    }
  const DataElement &sopclassuid = ds.GetDataElement( Tag(0x0008,0x0016) );
  const DataElement &sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0018) );
  // Make sure that const char* pointer will be properly padded with \0 char:
  std::string sopclassuid_str( sopclassuid.GetByteValue()->GetPointer(), sopclassuid.GetByteValue()->GetLength() );
  std::string sopinstanceuid_str( sopinstanceuid.GetByteValue()->GetPointer(), sopinstanceuid.GetByteValue()->GetLength() );
  ds.Remove( Tag(0x8,0x18) );

  FileDerivation fd;
  fd.SetFile( file );
  fd.AddReference( sopclassuid_str.c_str(), sopinstanceuid_str.c_str() );

  // CID 7202 Source Image Purposes of Reference
  // {"DCM",121320,"Uncompressed predecessor"},
  fd.SetPurposeOfReferenceCodeSequenceCodeValue( 121320 );

  // CID 7203 Image Derivation
  // { "DCM",113040,"Lossy Compression" },
  fd.SetDerivationCodeSequenceCodeValue( 113040 );
  fd.SetDerivationDescription( "lossy conversion" );
  if( !fd.Derive() )
    {
    std::cerr << "Sorry could not derive using input info" << std::endl;
    return false;
    }


#else
/*
(0008,2111) ST [Lossy compression with JPEG extended sequential 8 bit, IJG quality... # 102, 1 DerivationDescription
(0008,2112) SQ (Sequence with explicit length #=1)      # 188, 1 SourceImageSequence
  (fffe,e000) na (Item with explicit length #=3)          # 180, 1 Item
    (0008,1150) UI =UltrasoundImageStorage                  #  28, 1 ReferencedSOPClassUID
    (0008,1155) UI [1.2.840.1136190195280574824680000700.3.0.1.19970424140438] #  58, 1 ReferencedSOPInstanceUID
    (0040,a170) SQ (Sequence with explicit length #=1)      #  66, 1 PurposeOfReferenceCodeSequence
      (fffe,e000) na (Item with explicit length #=3)          #  58, 1 Item
        (0008,0100) SH [121320]                                 #   6, 1 CodeValue
        (0008,0102) SH [DCM]                                    #   4, 1 CodingSchemeDesignator
        (0008,0104) LO [Uncompressed predecessor]               #  24, 1 CodeMeaning
      (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
*/
    const Tag sisq(0x8,0x2112);
    SequenceOfItems * sqi;
      sqi = new SequenceOfItems;
      DataElement de( sisq);
      de.SetVR( VR::SQ );
      de.SetValue( *sqi );
      de.SetVLToUndefined();

  DataSet &ds = file.GetDataSet();
      ds.Insert( de );
{
    // (0008,0008) CS [ORIGINAL\SECONDARY]                     #  18, 2 ImageType
    gdcm::Attribute<0x0008,0x0008> at3;
    static const gdcm::CSComp values[] = {"DERIVED","SECONDARY"};
    at3.SetValues( values, 2, true ); // true => copy data !
    if( ds.FindDataElement( at3.GetTag() ) )
      {
      const gdcm::DataElement &de = ds.GetDataElement( at3.GetTag() );
      at3.SetFromDataElement( de );
      // Make sure that value #1 is at least 'DERIVED', so override in all cases:
      at3.SetValue( 0, values[0] );
      }
    ds.Replace( at3.GetAsDataElement() );

}
{
    Attribute<0x0008,0x2111> at1;
    at1.SetValue( "lossy conversion" );
    ds.Replace( at1.GetAsDataElement() );
}

    sqi = (SequenceOfItems*)ds.GetDataElement( sisq ).GetSequenceOfItems();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }

    Item &item1 = sqi->GetItem(1);
    DataSet &subds = item1.GetNestedDataSet();
/*
    (0008,1150) UI =UltrasoundImageStorage                  #  28, 1 ReferencedSOPClassUID
    (0008,1155) UI [1.2.840.1136190195280574824680000700.3.0.1.19970424140438] #  58, 1 ReferencedSOPInstanceUID
*/
{
    DataElement sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0016) );
    sopinstanceuid.SetTag( Tag(0x8,0x1150 ) );
    subds.Replace( sopinstanceuid );
    DataElement sopclassuid = ds.GetDataElement( Tag(0x0008,0x0018) );
    sopclassuid.SetTag( Tag(0x8,0x1155 ) );
    subds.Replace( sopclassuid );
    ds.Remove( Tag(0x8,0x18) );
}

    const Tag prcs(0x0040,0xa170);
    if( !subds.FindDataElement( prcs) )
      {
      SequenceOfItems *sqi2 = new SequenceOfItems;
      DataElement de( prcs );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi2 );
      de.SetVLToUndefined();
      subds.Insert( de );
      }

    sqi = (SequenceOfItems*)subds.GetDataElement( prcs ).GetSequenceOfItems();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }
    Item &item2 = sqi->GetItem(1);
    DataSet &subds2 = item2.GetNestedDataSet();

/*
        (0008,0100) SH [121320]                                 #   6, 1 CodeValue
        (0008,0102) SH [DCM]                                    #   4, 1 CodingSchemeDesignator
        (0008,0104) LO [Uncompressed predecessor]               #  24, 1 CodeMeaning
*/

    Attribute<0x0008,0x0100> at1;
    at1.SetValue( "121320" );
    subds2.Replace( at1.GetAsDataElement() );
    Attribute<0x0008,0x0102> at2;
    at2.SetValue( "DCM" );
    subds2.Replace( at2.GetAsDataElement() );
    Attribute<0x0008,0x0104> at3;
    at3.SetValue( "Uncompressed predecessor" );
    subds2.Replace( at3.GetAsDataElement() );

/*
(0008,9215) SQ (Sequence with explicit length #=1)      #  98, 1 DerivationCodeSequence
  (fffe,e000) na (Item with explicit length #=3)          #  90, 1 Item
    (0008,0100) SH [121327]                                 #   6, 1 CodeValue
    (0008,0102) SH [DCM]                                    #   4, 1 CodingSchemeDesignator
    (0008,0104) LO [Full fidelity image, uncompressed or lossless compressed] #  56, 1 CodeMeaning
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
*/
{
    const Tag sisq(0x8,0x9215);
    SequenceOfItems * sqi;
      sqi = new SequenceOfItems;
      DataElement de( sisq );
      de.SetVR( VR::SQ );
      de.SetValue( *sqi );
      de.SetVLToUndefined();
      ds.Insert( de );
    sqi = (SequenceOfItems*)ds.GetDataElement( sisq ).GetSequenceOfItems();
    sqi->SetLengthToUndefined();

    if( !sqi->GetNumberOfItems() )
      {
      Item item; //( Tag(0xfffe,0xe000) );
      item.SetVLToUndefined();
      sqi->AddItem( item );
      }

    Item &item1 = sqi->GetItem(1);
    DataSet &subds3 = item1.GetNestedDataSet();

    Attribute<0x0008,0x0100> at1;
    at1.SetValue( "121327" );
    subds3.Replace( at1.GetAsDataElement() );
    Attribute<0x0008,0x0102> at2;
    at2.SetValue( "DCM" );
    subds3.Replace( at2.GetAsDataElement() );
    Attribute<0x0008,0x0104> at3;
    at3.SetValue( "Full fidelity image, uncompressed or lossless compressed" );
    subds3.Replace( at3.GetAsDataElement() );
}
#endif

{
  /*
  (0028,2110) CS [01]                                     #   2, 1 LossyImageCompression
  (0028,2112) DS [15.95]                                  #   6, 1 LossyImageCompressionRatio
  (0028,2114) CS [ISO_10918_1]                            #  12, 1 LossyImageCompressionMethod
   */
  const DataElement & pixeldata = compressed_image.GetDataElement();
  size_t len = pixeldata.GetSequenceOfFragments()->ComputeByteLength();
  size_t reflen = compressed_image.GetBufferLength();
  double ratio = (double)reflen / (double)len;
  Attribute<0x0028,0x2110> at1;
  at1.SetValue( "01" );
  ds.Replace( at1.GetAsDataElement() );
  Attribute<0x0028,0x2112> at2;
  at2.SetValues( &ratio, 1);
  ds.Replace( at2.GetAsDataElement() );
  Attribute<0x0028,0x2114> at3;

  // ImageWriter will properly set attribute 0028,2114 (Lossy Image Compression Method)
}

return true;

}
} // end namespace gdcm

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  std::string filename;
  std::string outfilename;
  std::string root;
  int explicitts = 0; // explicit is a reserved keyword
  int implicit = 0;
  int quiet = 0;
  int lut = 0;
  int raw = 0;
  int deflated = 0;
  int rootuid = 0;
  int checkmeta = 0;
  int jpeg = 0;
  int jpegls = 0;
  int j2k = 0;
  int lossy = 0;
  int split = 0;
  int fragmentsize = 0;
  int rle = 0;
  int force = 0;
  int planarconf = 0;
  int planarconfval = 0;
  double iconmin = 0;
  double iconmax = 0;
  int usedict = 0;
  int compressicon = 0;
  int generateicon = 0;
  int iconminmax = 0;
  int iconautominmax = 0;
  int removegrouplength = 0;
  int removeprivate = 0;
  int removeretired = 0;
  int photometricinterpretation = 0;
  std::string photometricinterpretation_str;
  int quality = 0;
  int rate = 0;
  int tile = 0;
  int nres = 0;
  int nresvalue = 6; // ??
  std::vector<float> qualities;
  std::vector<float> rates;
  std::vector<unsigned int> tilesize;
  int irreversible = 0;
  int changeprivatetags = 0;

  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;
  int ignoreerrors = 0;
  int jpeglserror = 0;
  int jpeglserror_value = 0;

  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},
        {"group-length", 1, 0, 0}, // valid / create / remove
        {"preamble", 1, 0, 0}, // valid / create / remove
        {"padding", 1, 0, 0}, // valid (\0 -> space) / optimize (at most 1 byte of padding)
        {"vr", 1, 0, 0}, // valid
        {"sop", 1, 0, 0}, // default to SC...
        {"iod", 1, 0, 0}, // valid
        {"meta", 1, 0, 0}, // valid / create / remove
        {"dataset", 1, 0, 0}, // valid / create / remove?
        {"sequence", 1, 0, 0}, // defined / undefined
        {"deflate", 1, 0, 0}, // 1 - 9 / best = 9 / fast = 1
        {"tag", 1, 0, 0}, // need to specify a tag xxxx,yyyy = value to override default
        {"name", 1, 0, 0}, // same as tag but explicit use of name
        {"root-uid", 1, &rootuid, 1}, // specific Root (not GDCM)
        {"check-meta", 0, &checkmeta, 1}, // specific Root (not GDCM)
// Image specific options:
        {"pixeldata", 1, 0, 0}, // valid
        {"apply-lut", 0, &lut, 1}, // default (implicit VR, LE) / Explicit LE / Explicit BE
        {"raw", 0, &raw, 1}, // default (implicit VR, LE) / Explicit LE / Explicit BE
        {"deflated", 0, &deflated, 1}, // DeflatedExplicitVRLittleEndian
        {"lossy", 0, &lossy, 1}, // Specify lossy comp
        {"force", 0, &force, 1}, // force decompression even if target compression is identical
        {"jpeg", 0, &jpeg, 1}, // JPEG lossy / lossless
        {"jpegls", 0, &jpegls, 1}, // JPEG-LS: lossy / lossless
        {"j2k", 0, &j2k, 1}, // J2K: lossy / lossless
        {"rle", 0, &rle, 1}, // lossless !
        {"mpeg2", 0, 0, 0}, // lossy !
        {"jpip", 0, 0, 0}, // ??
        {"split", 1, &split, 1}, // split fragments
        {"planar-configuration", 1, &planarconf, 1}, // Planar Configuration
        {"explicit", 0, &explicitts, 1}, //
        {"implicit", 0, &implicit, 1}, //
        {"use-dict", 0, &usedict, 1}, //
        {"generate-icon", 0, &generateicon, 1}, //
        {"icon-minmax", 1, &iconminmax, 1}, //
        {"icon-auto-minmax", 0, &iconautominmax, 1}, //
        {"compress-icon", 0, &compressicon, 1}, //
        {"remove-gl", 0, &removegrouplength, 1}, //
        {"remove-private-tags", 0, &removeprivate, 1}, //
        {"remove-retired", 0, &removeretired, 1}, //
        {"photometric-interpretation", 1, &photometricinterpretation, 1}, //
        {"with-private-dict", 0, &changeprivatetags, 1}, //
// j2k :
        {"rate", 1, &rate, 1}, //
        {"quality", 1, &quality, 1}, // will also work for regular jpeg compressor
        {"tile", 1, &tile, 1}, //
        {"number-resolution", 1, &nres, 1}, //
        {"irreversible", 0, &irreversible, 1}, //
        {"allowed-error", 1, &jpeglserror, 1}, //

// General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {"ignore-errors", 0, &ignoreerrors, 1},
        {"quiet", 0, &quiet, 1},

        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "i:o:XMUClwdJKLRFYS:P:VWDEhvIr:q:t:n:e:",
      long_options, &option_index);
    if (c == -1)
      {
      break;
      }

    switch (c)
      {
    case 0:
        {
        const char *s = long_options[option_index].name; (void)s;
        //printf ("option %s", s);
        if (optarg)
          {
          if( option_index == 0 ) /* input */
            {
            assert( strcmp(s, "input") == 0 );
            assert( filename.empty() );
            filename = optarg;
            }
          else if( option_index == 14 ) /* root-uid */
            {
            assert( strcmp(s, "root-uid") == 0 );
            assert( root.empty() );
            root = optarg;
            }
          else if( option_index == 28 ) /* split */
            {
            assert( strcmp(s, "split") == 0 );
            fragmentsize = atoi(optarg);
            }
          else if( option_index == 29 ) /* planar conf*/
            {
            assert( strcmp(s, "planar-configuration") == 0 );
            planarconfval = atoi(optarg);
            }
          else if( option_index == 34 ) /* icon minmax*/
            {
            assert( strcmp(s, "icon-minmax") == 0 );
            std::stringstream ss;
            ss.str( optarg );
            ss >> iconmin;
            char comma;
            ss >> comma;
            ss >> iconmax;
            }
          else if( option_index == 40 ) /* photometricinterpretation */
            {
            assert( strcmp(s, "photometric-interpretation") == 0 );
            photometricinterpretation_str = optarg;
            }
          else if( option_index == 42 ) /* rate */
            {
            assert( strcmp(s, "rate") == 0 );
            readvector(rates, optarg);
            }
          else if( option_index == 43 ) /* quality */
            {
            assert( strcmp(s, "quality") == 0 );
            readvector(qualities, optarg);
            }
          else if( option_index == 44 ) /* tile */
            {
            assert( strcmp(s, "tile") == 0 );
            size_t n = readvector(tilesize, optarg);
            assert( n == 2 ); (void)n;
            }
          else if( option_index == 45 ) /* number of resolution */
            {
            assert( strcmp(s, "number-resolution") == 0 );
            nresvalue = atoi(optarg);
            }
          else if( option_index == 47 ) /* JPEG-LS error */
            {
            assert( strcmp(s, "allowed-error") == 0 );
            jpeglserror_value = atoi(optarg);
            }
          //printf (" with arg %s, index = %d", optarg, option_index);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( filename.empty() );
      filename = optarg;
      break;

    case 'o':
      //printf ("option o with value '%s'\n", optarg);
      assert( outfilename.empty() );
      outfilename = optarg;
      break;

    case 'X':
      explicitts = 1;
      break;

    case 'M':
      implicit = 1;
      break;

    case 'U':
      usedict = 1;
      break;

    case 'C':
      checkmeta = 1;
      break;

    // root-uid

    case 'l':
      lut = 1;
      break;

    case 'w':
      raw = 1;
      break;

    case 'e':
      jpeglserror = 1;
      jpeglserror_value = atoi(optarg);
      break;

    case 'd':
      deflated = 1;
      break;

    case 'J':
      jpeg = 1;
      break;

    case 'K':
      j2k = 1;
      break;

    case 'L':
      jpegls = 1;
      break;

    case 'R':
      rle = 1;
      break;

    case 'F':
      force = 1;
      break;

    case 'Y':
      lossy = 1;
      break;

    case 'S':
      split = 1;
      fragmentsize = atoi(optarg);
      break;

    case 'P':
      photometricinterpretation = 1;
      photometricinterpretation_str = optarg;
      break;

    case 'r':
      rate = 1;
      readvector(rates, optarg);
      break;

    case 'q':
      quality = 1;
      readvector(qualities, optarg);
      break;

    case 't':
      tile = 1;
      readvector(tilesize, optarg);
      break;

    case 'n':
      nres = 1;
      nresvalue = atoi(optarg);
      break;

    // General option
    case 'V':
      verbose = 1;
      break;

    case 'W':
      warning = 1;
      break;

    case 'D':
      debug = 1;
      break;

    case 'E':
      error = 1;
      break;

    case 'h':
      help = 1;
      break;

    case 'v':
      version = 1;
      break;

    case 'I':
      ignoreerrors = 1;
      break;

    case '?':
      break;

    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
      }
  }

  // For now only support one input / one output
  if (optind < argc)
    {
    //printf ("non-option ARGV-elements: ");
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind++]);
      files.push_back( argv[optind++] );
      }
    //printf ("\n");
    if( files.size() == 2
      && filename.empty()
      && outfilename.empty()
    )
      {
      filename = files[0];
      outfilename = files[1];
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }

  if( version )
    {
    //std::cout << "version" << std::endl;
    PrintVersion();
    return 0;
    }

  if( help )
    {
    //std::cout << "help" << std::endl;
    PrintHelp();
    return 0;
    }

  if( filename.empty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }
  if( outfilename.empty() )
    {
    //std::cerr << "Need output file (-o)\n";
    PrintHelp();
    return 1;
    }

  // Debug is a little too verbose
  gdcm::Trace::SetDebug( (debug  > 0 ? true : false));
  gdcm::Trace::SetWarning(  (warning  > 0 ? true : false));
  gdcm::Trace::SetError(  (error  > 0 ? true : false));
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( (verbose  > 0 ? true : false) );
    gdcm::Trace::SetError( (verbose  > 0 ? true : false) );
    }

  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmconv" );
  if( !rootuid )
    {
    // only read the env var is no explicit cmd line option
    // maybe there is an env var defined... let's check
    const char *rootuid_env = getenv("GDCM_ROOT_UID");
    if( rootuid_env )
      {
      rootuid = 1;
      root = rootuid_env;
      }
    }
  if( rootuid )
    {
    // root is set either by the cmd line option or the env var
    if( !gdcm::UIDGenerator::IsValid( root.c_str() ) )
      {
      std::cerr << "specified Root UID is not valid: " << root << std::endl;
      return 1;
      }
    gdcm::UIDGenerator::SetRoot( root.c_str() );
    }

  if( removegrouplength || removeprivate || removeretired )
    {
    gdcm::Reader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read: " << filename << std::endl;
      return 1;
      }
    gdcm::MediaStorage ms;
    ms.SetFromFile( reader.GetFile() );
    if( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage )
      {
      std::cerr << "Sorry DICOMDIR is not supported" << std::endl;
      return 1;
      }

    gdcm::Anonymizer ano;
    ano.SetFile( reader.GetFile() );
    if( removegrouplength )
      {
      if( !ano.RemoveGroupLength() )
        {
        std::cerr << "Could not remove group length" << std::endl;
        }
      }
    if( removeretired )
      {
      if( !ano.RemoveRetired() )
        {
        std::cerr << "Could not remove retired tags" << std::endl;
        }
      }
    if( removeprivate )
      {
      if( !ano.RemovePrivateTags() )
        {
        std::cerr << "Could not remove private tags" << std::endl;
        }
      }

    gdcm::Writer writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( ano.GetFile() );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }

    return 0;
    }

  // Handle here the general file (not required to be image)
  if ( explicitts || implicit || deflated )
    {
    if( explicitts && implicit ) return 1; // guard
    if( explicitts && deflated ) return 1; // guard
    if( implicit && deflated ) return 1; // guard
    gdcm::Reader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read: " << filename << std::endl;
      return 1;
      }
    gdcm::MediaStorage ms;
    ms.SetFromFile( reader.GetFile() );
    if( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage )
      {
      std::cerr << "Sorry DICOMDIR is not supported" << std::endl;
      return 1;
      }

    gdcm::Writer writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( reader.GetFile() );
    gdcm::File & file = writer.GetFile();
    gdcm::FileMetaInformation &fmi = file.GetHeader();

    const gdcm::TransferSyntax &orits = fmi.GetDataSetTransferSyntax();
    if( orits != gdcm::TransferSyntax::ExplicitVRLittleEndian
      && orits != gdcm::TransferSyntax::ImplicitVRLittleEndian
      && orits != gdcm::TransferSyntax::DeflatedExplicitVRLittleEndian )
      {
      std::cerr << "Sorry input Transfer Syntax not supported for this conversion: " << orits << std::endl;
      return 1;
      }

    gdcm::TransferSyntax ts = gdcm::TransferSyntax::ImplicitVRLittleEndian;
    if( explicitts )
      {
      ts = gdcm::TransferSyntax::ExplicitVRLittleEndian;
      }
    else if( deflated )
      {
      ts = gdcm::TransferSyntax::DeflatedExplicitVRLittleEndian;
      }
    std::string tsuid = gdcm::TransferSyntax::GetTSString( ts );
    if( tsuid.size() % 2 == 1 )
      {
      tsuid.push_back( 0 ); // 0 padding
      }
    gdcm::DataElement de( gdcm::Tag(0x0002,0x0010) );
    de.SetByteValue( &tsuid[0], (uint32_t)tsuid.size() );
    de.SetVR( gdcm::Attribute<0x0002, 0x0010>::GetVR() );
    fmi.Clear();
    fmi.Replace( de );

    fmi.SetDataSetTransferSyntax(ts);

    if( explicitts || deflated )
      {
      gdcm::FileExplicitFilter fef;
      fef.SetChangePrivateTags( (changeprivatetags > 0 ? true: false));
      fef.SetFile( reader.GetFile() );
      if( !fef.Change() )
        {
        std::cerr << "Failed to change: " << filename << std::endl;
        return 1;
        }
      }

    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }

    return 0;
    }

  // split fragments
  if( split )
    {
    gdcm::PixmapReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read (pixmap): " << filename << std::endl;
      return 1;
      }
    const gdcm::Pixmap &image = reader.GetPixmap();

    gdcm::ImageFragmentSplitter splitter;
    splitter.SetInput( image );
    splitter.SetFragmentSizeMax( fragmentsize );
    splitter.SetForce( (force > 0 ? true: false));
    bool b = splitter.Split();
    if( !b )
      {
      std::cerr << "Could not split: " << filename << std::endl;
      return 1;
      }
    gdcm::PixmapWriter writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( reader.GetFile() );
    writer.SetPixmap( splitter.PixmapToPixmapFilter::GetOutput() );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }
    }
  else if( photometricinterpretation )
    {
    gdcm::PixmapReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read (pixmap): " << filename << std::endl;
      return 1;
      }
    const gdcm::Pixmap &image = reader.GetPixmap();

    // Just in case:
    if( gdcm::PhotometricInterpretation::GetPIType(photometricinterpretation_str.c_str())
      == gdcm::PhotometricInterpretation::PI_END )
      {
      std::cerr << "Do not handle PhotometricInterpretation: " << photometricinterpretation_str << std::endl;
      return 1;
      }
    gdcm::PhotometricInterpretation pi (
      gdcm::PhotometricInterpretation::GetPIType(photometricinterpretation_str.c_str()) );
    gdcm::ImageChangePhotometricInterpretation pifilt;
    pifilt.SetInput( image );
    pifilt.SetPhotometricInterpretation( pi );
    bool b = pifilt.Change();
    if( !b )
      {
      std::cerr << "Could not apply PhotometricInterpretation: " << filename << std::endl;
      return 1;
      }
    gdcm::PixmapWriter writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( reader.GetFile() );
    writer.SetPixmap( pifilt.PixmapToPixmapFilter::GetOutput() );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }
    }
  else if( lut )
    {
    gdcm::PixmapReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read (pixmap): " << filename << std::endl;
      return 1;
      }
    const gdcm::Pixmap &image = reader.GetPixmap();

    gdcm::ImageApplyLookupTable lutfilt;
    lutfilt.SetInput( image );
    bool b = lutfilt.Apply();
    if( !b )
      {
      std::cerr << "Could not apply LUT: " << filename << std::endl;
      return 1;
      }
    gdcm::PixmapWriter writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( reader.GetFile() );
    writer.SetPixmap( lutfilt.PixmapToPixmapFilter::GetOutput() );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }
    }
  else if( jpeg || j2k || jpegls || rle || raw || force /*|| deflated*/ /*|| planarconf*/ )
    {
    gdcm::PixmapReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read (pixmap): " << filename << std::endl;
      return 1;
      }
    gdcm::Pixmap &image = reader.GetPixmap();
    //const gdcm::IconImage &icon = image.GetIconImage();
    //if( !icon.IsEmpty() )
    //  {
    //  std::cerr << "Icons are not supported" << std::endl;
    //  return 1;
    //  }
    if( generateicon )
      {
      gdcm::IconImageGenerator iig;
      iig.SetPixmap( image );
      const unsigned int idims[2] = { 64, 64 };
      iig.SetOutputDimensions( idims );
      if( iconminmax )
        {
        iig.SetPixelMinMax( iconmin, iconmax );
        }
      iig.AutoPixelMinMax( iconautominmax ? true : false );
      bool b = iig.Generate();
      if( !b ) return 1;
      const gdcm::IconImage &icon = iig.GetIconImage();
      image.SetIconImage( icon );
      }

    gdcm::JPEG2000Codec j2kcodec;
    gdcm::JPEGCodec jpegcodec;
    gdcm::JPEGLSCodec jpeglscodec;
    gdcm::ImageChangeTransferSyntax change;
    change.SetForce( (force > 0 ? true: false));
    change.SetCompressIconImage( (compressicon > 0 ? true: false));
    if( jpeg )
      {
      if( lossy )
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEGBaselineProcess1 );
        jpegcodec.SetLossless( false );
        if( quality )
          {
          assert( qualities.size() == 1 );
          jpegcodec.SetQuality( qualities[0] );
          }
        change.SetUserCodec( &jpegcodec );
        }
      else
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEGLosslessProcess14_1 );
        }
      }
    else if( jpegls )
      {
      if( lossy )
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEGLSNearLossless );
        jpeglscodec.SetLossless( false );
        if( jpeglserror )
          {
          jpeglscodec.SetLossyError( jpeglserror_value );
          }
        change.SetUserCodec( &jpeglscodec );
        }
      else
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEGLSLossless );
        }
      }
    else if( j2k )
      {
      if( lossy )
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEG2000 );
        if( rate )
          {
          int i = 0;
          for(std::vector<float>::const_iterator it = rates.begin(); it != rates.end(); ++it )
            {
            j2kcodec.SetRate(i++, *it );
            }
          }
        if( quality )
          {
          int i = 0;
          for(std::vector<float>::const_iterator it = qualities.begin(); it != qualities.end(); ++it )
            {
            j2kcodec.SetQuality( i++, *it );
            }
          }
        if( tile )
          {
          j2kcodec.SetTileSize( tilesize[0], tilesize[1] );
          }
        if( nres )
          {
          j2kcodec.SetNumberOfResolutions( nresvalue );
          }
        j2kcodec.SetReversible( !irreversible );
        change.SetUserCodec( &j2kcodec );
        }
      else
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::JPEG2000Lossless );
        }
      }
    else if( raw )
      {
      if( lossy )
        {
        std::cerr << "no such thing as raw & lossy" << std::endl;
        return 1;
        }
      const gdcm::TransferSyntax &ts = image.GetTransferSyntax();
#ifdef GDCM_WORDS_BIGENDIAN
	(void)ts;
      change.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRBigEndian );
#else
      if( ts.IsExplicit() )
        {
        change.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
        }
      else
        {
        assert( ts.IsImplicit() );
        change.SetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );
        }
#endif
      }
    else if( rle )
      {
      if( lossy )
        {
        std::cerr << "no such thing as rle & lossy" << std::endl;
        return 1;
        }
      change.SetTransferSyntax( gdcm::TransferSyntax::RLELossless );
      }
    else if( deflated )
      {
      if( lossy )
        {
        std::cerr << "no such thing as deflated & lossy" << std::endl;
        return 1;
        }
      change.SetTransferSyntax( gdcm::TransferSyntax::DeflatedExplicitVRLittleEndian );
      }
    else if( force )
      {
      // If image is encapsulated it will check some attribute (col/row/pi/pf) and
      // some attributes...
      }
    else
      {
      std::cerr << "unhandled action" << std::endl;
      return 1;
      }
    if( raw && planarconf )
      {
      gdcm::ImageChangePlanarConfiguration icpc;
      icpc.SetPlanarConfiguration( planarconfval );
      icpc.SetInput( image );
      bool b = icpc.Change();
      if( !b )
        {
        std::cerr << "Could not change the Planar Configuration: " << filename << std::endl;
        return 1;
        }
      change.SetInput( icpc.PixmapToPixmapFilter::GetOutput() );
      }
    else
      {
      change.SetInput( image );
      }
    bool b = change.Change();
    if( !b )
      {
      std::cerr << "Could not change the Transfer Syntax: " << filename << std::endl;
      return 1;
      }
    if( lossy )
      {
      if(!quiet)
        PrintLossyWarning();
      if( !gdcm::derives( reader.GetFile(), change.PixmapToPixmapFilter::GetOutput() ) )
        {
        std::cerr << "Failed to derives: " << filename << std::endl;
        return 1;
        }
      }
    if( usedict /*ts.IsImplicit()*/ )
      {
      gdcm::FileExplicitFilter fef;
      fef.SetChangePrivateTags( (changeprivatetags > 0 ? true : false));
      fef.SetFile( reader.GetFile() );
      if(!fef.Change())
        {
        std::cerr << "Failed to change: " << filename << std::endl;
        return 1;
        }
      }

    gdcm::PixmapWriter writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetFile( reader.GetFile() );
    //writer.SetFile( fef.GetFile() );

    gdcm::File & file = writer.GetFile();
    gdcm::FileMetaInformation &fmi = file.GetHeader();
    fmi.Remove( gdcm::Tag(0x0002,0x0100) ); //  '   '    ' // PrivateInformationCreatorUID
    fmi.Remove( gdcm::Tag(0x0002,0x0102) ); //  '   '    ' // PrivateInformation

    const gdcm::Pixmap &pixout = change.PixmapToPixmapFilter::GetOutput();
    writer.SetPixmap( pixout );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      return 1;
      }

    }
  else if( raw && false )
    {
    gdcm::PixmapReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read (pixmap): " << filename << std::endl;
      return 1;
      }

    const gdcm::Pixmap &ir = reader.GetPixmap();

    gdcm::Pixmap image( ir );
    const gdcm::TransferSyntax &ts = ir.GetTransferSyntax();
    if( ts.IsExplicit() )
      {
      image.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
      }
    else
      {
      assert( ts.IsImplicit() );
      image.SetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );
      }

/*
    image.SetNumberOfDimensions( ir.GetNumberOfDimensions() );

    const unsigned int *dims = ir.GetDimensions();
    image.SetDimension(0, dims[0] );
    image.SetDimension(1, dims[1] );

    const gdcm::PixelFormat &pixeltype = ir.GetPixelFormat();
    image.SetPixelFormat( pixeltype );

    const gdcm::PhotometricInterpretation &pi = ir.GetPhotometricInterpretation();
    image.SetPhotometricInterpretation( pi );
*/

    unsigned long len = ir.GetBufferLength();
    //assert( len = ir.GetBufferLength() );
    std::vector<char> buffer;
    buffer.resize(len); // black image

    ir.GetBuffer( &buffer[0] );
    gdcm::ByteValue *bv = new gdcm::ByteValue(buffer);
    gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );
    pixeldata.SetValue( *bv );
    image.SetDataElement( pixeldata );

    gdcm::PixmapWriter writer;
    writer.SetFile( reader.GetFile() );
    writer.SetPixmap( image );
    writer.SetFileName( outfilename.c_str() );

    if( !writer.Write() )
      {
      std::cerr << "could not write: " << outfilename << std::endl;
      return 1;
      }
    }
  else
    {
    gdcm::Reader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      if( ignoreerrors )
        {
        std::cerr << "WARNING: an error was found during the reading of your DICOM file." << std::endl;
        std::cerr << "gdcmconv will still try to continue and rewrite your DICOM file." << std::endl;
        std::cerr << "There is absolutely no guarantee that your output file will be valid." << std::endl;
        }
      else
        {
        std::cerr << "Failed to read: " << filename << std::endl;
        return 1;
        }
      }
    gdcm::MediaStorage ms;
    ms.SetFromFile( reader.GetFile() );
    if( ms == gdcm::MediaStorage::MediaStorageDirectoryStorage )
      {
      std::cerr << "Sorry DICOMDIR is not supported" << std::endl;
      return 1;
      }

#if 0
    // if preamble create:
    gdcm::File f(reader.GetFile());
    gdcm::Preamble p;
    p.Create();
    f.SetPreamble(p);
    gdcm::DataSet ds = reader.GetFile().GetDataSet();
    SetSQToUndefined undef;
    ds.ExecuteOperation(undef);

    gdcm::File f(reader.GetFile());
    f.SetDataSet(ds);
#endif

#if 0
    gdcm::DataSet& ds = reader.GetFile().GetDataSet();
    gdcm::DataElement de = ds.GetDataElement( gdcm::Tag(0x0010,0x0010) );
    const char patname[] = "John^Doe";
    de.SetByteValue(patname, strlen(patname));
    std::cout << de << std::endl;

    ds.Replace( de );
    std::cout << ds.GetDataElement( gdcm::Tag(0x0010,0x0010) ) << std::endl;
#endif

    /*
    //(0020,0032) DS [-158.135803\-179.035797\-75.699997]     #  34, 3 ImagePositionPatient
    //(0020,0037) DS [1.000000\0.000000\0.000000\0.000000\1.000000\0.000000] #  54, 6 ImageOrientationPatient
    gdcm::Attribute<0x0020,0x0032> at = { -158.135803, -179.035797, -75.699997 };
    gdcm::DataElement ipp = at.GetAsDataElement();
    ds.Remove( at.GetTag() );
    ds.Remove( ipp.GetTag() );
    ds.Replace( ipp );
     */

    gdcm::Writer writer;
    writer.SetFileName( outfilename.c_str() );
    writer.SetCheckFileMetaInformation( (checkmeta > 0 ? true : false));
    //writer.SetFile( f );
    writer.SetFile( reader.GetFile() );
    if( !writer.Write() )
      {
      std::cerr << "Failed to write: " << outfilename << std::endl;
      // remove file to avoid any temptation
      if( filename != outfilename )
        {
        gdcm::System::RemoveFile( outfilename.c_str() );
        }
      else
        {
        std::cerr << "gdcmconv just corrupted: " << filename << " for you (data lost)." << std::endl;
        }
      return 1;
      }
    }

  return 0;
}
