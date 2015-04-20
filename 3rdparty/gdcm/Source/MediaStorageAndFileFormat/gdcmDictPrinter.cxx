/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDictPrinter.h"
#include "gdcmDict.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmSequenceOfItems.h"

namespace gdcm
{

//-----------------------------------------------------------------------------
DictPrinter::DictPrinter()
{
}

//-----------------------------------------------------------------------------
DictPrinter::~DictPrinter()
{
}

VM GuessVMType(DataElement const &de)
{
  if( de.IsEmpty() ) return VM::VM1;
  const VR &vr = de.GetVR();
  const VL &vl = de.GetVL();
  const Value& value = de.GetValue();
  VM vm;
  if( VR::IsBinary( vr ) )
    {
    if( vr == VR::SQ  )
      {
      vm = VM::VM1; // ??
      }
    else if( vr & VR::OB_OW )
      {
      vm = VM::VM1;
      }
    else
      {
      vm = VM::GetVMTypeFromLength( value.GetLength(), vr.GetSize() );
      }
    }
  else
    {
    assert( VR::IsASCII( vr ) || vr == VR::INVALID );
    switch(vr)
      {
    case VR::INVALID:
      vm = VM::VM1;
      break;
    case VR::UN: // TODO need to do some magic guessing
      vm = VM::VM1;
      break;
    case VR::DA: case VR::TM: case VR::LT:
    case VR::SH: case VR::UI: case VR::LO: case VR::ST:
    case VR::UT: case VR::AE: case VR::AS:
      vm = VM::VM1;
      break;
    case VR::PN:
      vm = VM::VM1; // I think it is ok
      break;
    case VR::IS: case VR::DS: case VR::DT: case VR::CS:
        {
        // Need to count \\ character
        const ByteValue *bv = dynamic_cast<const ByteValue*>(&value);
        vm = VM::VM1; // why not ?
        if(!de.IsEmpty())
          {
          assert( bv && "not bv" );
          const char *array = bv->GetPointer();
          unsigned int c = VM::GetNumberOfElementsFromArray(array, vl);
          vm = VM::GetVMTypeFromLength( c, 1 );
          }
        }
      break;
    default:
      vm = VM::VM0;
      assert( 0 ); // Impossible happen ! (someone added new VR and forgot this switch)
      }
    }

  return vm;
}

struct OWNER_VERSION
{
  const char* owner;
  const char* version;
};

// See getowner.xsl
static const OWNER_VERSION OwnerVersionTable[] ={
{"","EMPTY!"},
{"","2"},
{"","3"},
{"1.2.840.113663.1","ATL"},
{"1.2.840.113681","DUP"},
{"1.2.840.113708.794.1.1.2.0","ARM"},
{"2.16.840.1.114059.1.1.6.1.50.1","DEX"},
{"ACUSON","ACU"},
{"ADAC_IMG","ADAC"},
{"AEGIS_DICOM_2.00","AEG"},
{"AGFA PACS Archive Mirroring 1.0 ","MIT"},
{"AGFA","AGFA"},
{"AGFA_ADC_Compact","AGFA"},
{"Applicare/RadWorks/Version 5.0","APL"},
{"Applicare/RadWorks/Version 6.0","APL"},
{"Applicare/RadWorks/Version 6.0/Summary","APL"},
{"BioPri","BIO"},
{"CAMTRONICS IP ","CMT"},
{"CAMTRONICS","CMT"},
{"CARDIO-D.R. 1.0 ","PDIC"},
{"Canon Inc.","CAN"},
{"DIDI TO PCR 1.1 ","PSPI"},
{"DIGISCAN IMAGE","SSPI"},
{"DLX_ANNOT_01","DLX"},
{"DLX_EXAMS_01","DLX"},
{"DLX_LKUP_01 ","GEM"},
{"DLX_PATNT_01","DLX"},
{"DLX_SERIE_01","DLX"},
{"DLX_SERIE_01","GEM"},
{"ELSCINT1","EL1"},
{"FDMS 1.0","FUJ"},
{"FFP DATA","SSPI"},
{"GE ??? From Adantage Review CS","GEM"},
{"GEIIS PACS","GEM"},
{"GEIIS ","GEM"},
{"GEMS_3DSTATE_001","GEM"},
{"GEMS_ACQU_01","GEM"},
{"GEMS_ACRQA_1.0 BLOCK1 ","GEM"},
{"GEMS_ACRQA_1.0 BLOCK2 ","GEM"},
{"GEMS_ACRQA_1.0 BLOCK3 ","GEM"},
{"GEMS_ACRQA_2.0 BLOCK1 ","GEM"},
{"GEMS_ACRQA_2.0 BLOCK2 ","GEM"},
{"GEMS_ACRQA_2.0 BLOCK3 ","GEM"},
{"GEMS_ADWSoft_3D1","GEM"},
{"GEMS_ADWSoft_DPO","GEM"},
{"GEMS_AWSOFT_CD1 ","GEM"},
{"GEMS_AWSoft_SB1 ","GEM"},
{"GEMS_CTHD_01","GEM"},
{"GEMS_CT_CARDIAC_001 ","GEM"},
{"GEMS_CT_HINO_01 ","GEM"},
{"GEMS_CT_VES_01","GEM"},
{"GEMS_DL_FRAME_01","GEM"},
{"GEMS_DL_IMG_01","GEM"},
{"GEMS_DL_PATNT_01","GEM"},
{"GEMS_DL_SERIES_01 ","GEM"},
{"GEMS_DL_STUDY_01","GEM"},
{"GEMS_DRS_1","GEM"},
{"GEMS_FALCON_03","GEM"},
{"GEMS_GDXE_ATHENAV2_INTERNAL_USE ","GEM"},
{"GEMS_GDXE_FALCON_04 ","GEM"},
{"GEMS_GENIE_1","GEM"},
{"GEMS_GNHD_01","GEM"},
{"GEMS_HELIOS_01","GEM"},
{"GEMS_IDEN_01","GEM"},
{"GEMS_IMAG_01","GEM"},
{"GEMS_IMPS_01","GEM"},
{"GEMS_IQTB_IDEN_47 ","GEM"},
{"GEMS_PARM_01","GEM"},
{"GEMS_PATI_01","GEM"},
{"GEMS_PETD_01","GEM"},
{"GEMS_RELA_01","GEM"},
{"GEMS_SENO_02","GEM"},
{"GEMS_SERS_01","GEM"},
{"GEMS_STDY_01","GEM"},
{"GEMS_YMHD_01","GEM"},
{"GE_GENESIS_REV3.0 ","GEM"},
{"HOLOGIC ","HOL"},
{"Hologic ","HOL"},
{"ISG shadow","ISG"},
{"ISI ","SSPI"},
{"KINETDX ","KDX"},
{"KINETDX_GRAPHICS","KDX"},
{"LODOX_STATSCAN","LDX"},
{"LORAD Selenia ","LOR"},
{"MERGE TECHNOLOGIES, INC.","MRG"},
{"MITRA LINKED ATTRIBUTES 1.0 ","MIT"},
{"MITRA MARKUP 1.0","MIT"},
{"MITRA OBJECT ATTRIBUTES 1.0 ","MIT"},
{"MITRA OBJECT DOCUMENT 1.0 ","MIT"},
{"MITRA OBJECT UTF8 ATTRIBUTES 1.0","MIT"},
{"MITRA PRESENTATION 1.0","MIT"},
{"MMCPrivate","HIT"},
{"Mayo/IBM Archive Project","MIBM"},
{"MeVis BreastCare","MEV"},
{"Mortara_Inc ","MOR"},
{"PAPYRUS 3.0 ","PA3"},
{"PAPYRUS ","PAP"},
{"PHILIPS MR R5.5/PART","PSPI"},
{"PHILIPS MR R5.6/PART","PSPI"},
{"PHILIPS MR SPECTRO;1","PSPI"},
{"PHILIPS MR","PSPI"},
{"PHILIPS MR/LAST ","PSPI"},
{"PHILIPS MR/PART ","PSPI"},
{"PHILIPS NM -Private ","PMNM"},
{"PHILIPS-MR-1","PDIC"},
{"PMS-THORA-5.1 ","PDIC"},
{"PMTF INFORMATION DATA ","TSH"},
{"Philips Imaging DD 001","PMFE"},
{"Philips Imaging DD 129","PMFE"},
{"Philips MR Imaging DD 001 ","PMFE"},
{"Philips MR Imaging DD 002 ","PMFE"},
{"Philips MR Imaging DD 003 ","PMFE"},
{"Philips MR Imaging DD 004 ","PMFE"},
{"Philips MR Imaging DD 005 ","PMFE"},
{"Philips PET Private Group ","PPET"},
{"Philips X-ray Imaging DD 001","PMFE"},
{"Picker MR Private Group ","PCK"},
{"Picker NM Private Group ","PCK"},
{"SCHICK TECHNOLOGIES - Change Item Creator ID","SCH"},
{"SCHICK TECHNOLOGIES - Change List Creator ID","SCH"},
{"SCHICK TECHNOLOGIES - Image Security Creator ID ","SCH"},
{"SCHICK TECHNOLOGIES - Note List Creator ID","SCH"},
{"SECTRA_Ident_01 ","SEC"},
{"SECTRA_ImageInfo_01 ","SEC"},
{"SECTRA_OverlayInfo_01 ","SEC"},
{"SEGAMI MIML ","MOR"},
{"SHS MagicView 300 ","SSPI"},
{"SIEMENS CM VA0  ACQU","SSPI"},
{"SIEMENS CM VA0  CMS ","SSPI"},
{"SIEMENS CM VA0  LAB ","SSPI"},
{"SIEMENS CSA HEADER","SSPI"},
{"SIEMENS CSA NON-IMAGE ","SSPI"},
{"SIEMENS CSA REPORT","SSPI"},
{"SIEMENS CT VA0  COAD","SSPI"},
{"SIEMENS CT VA0  GEN ","SSPI"},
{"SIEMENS CT VA0  IDE ","SSPI"},
{"SIEMENS CT VA0  ORI ","SSPI"},
{"SIEMENS CT VA0  OST ","SSPI"},
{"SIEMENS CT VA0  RAW ","SSPI"},
{"SIEMENS DFR.01 MANIPULATED","SSPI"},
{"SIEMENS DFR.01 ORIGINAL ","SSPI"},
{"SIEMENS DICOM ","SSPI"},
{"SIEMENS DLR.01","SPI"},
{"SIEMENS DLR.01","SSPI"},
{"SIEMENS ISI ","SSPI"},
{"SIEMENS MED DISPLAY 0000","SSPI"},
{"SIEMENS MED DISPLAY ","SSPI"},
{"SIEMENS MED ECAT FILE INFO","SSPI"},
{"SIEMENS MED HG","SSPI"},
{"SIEMENS MED MAMMO " ,"SSPI"},
{"SIEMENS MED MG","SSPI"},
{"SIEMENS MED NM","SSPI"},
{"SIEMENS MED SP DXMG WH AWS 1","SSPI"},
{"SIEMENS MED ","SSPI"},
{"SIEMENS MEDCOM HEADER ","SSPI"},
{"SIEMENS MEDCOM HEADER2","SSPI"},
{"SIEMENS MEDCOM OOG","SSPI"},
{"SIEMENS MR VA0  COAD","SSPI"},
{"SIEMENS MR VA0  GEN ","SSPI"},
{"SIEMENS MR VA0  RAW ","SSPI"},
{"SIEMENS NUMARIS II","SSPI"},
{"SIEMENS RA GEN","SSPI"},
{"SIEMENS RA PLANE A","SSPI"},
{"SIEMENS RA PLANE B","SSPI"},
{"SIEMENS RIS ","SSPI"},
{"SIEMENS SIENET","SSPI"},
{"SIEMENS SMS-AX  ACQ 1.0 ","SSPI"},
{"SIEMENS SMS-AX  ORIGINAL IMAGE INFO 1.0 ","SSPI"},
{"SIEMENS SMS-AX  QUANT 1.0 ","SSPI"},
{"SIEMENS SMS-AX  VIEW 1.0","SSPI"},
{"SIEMENS Selma ","SSPI"},
{"SIEMENS WH SR 1.0 ","SSPI"},
{"SIEMENS_FLCOMPACT_VA01A_PROC","SSPI"},
{"SIENET","SSPI"},
{"SPI RELEASE 1 ","SPI"},
{"SPI Release 1 ","SPI"},
//{"SPI ","SPI"}, // FIXME ??
{"SPI ","SSPI"},
{"SPI-P Release 1 ","PSPI"},
{"SPI-P Release 1;1 ","PSPI"},
{"SPI-P Release 1;2 ","PSPI"},
{"SPI-P Release 1;3 ","PSPI"},
{"SPI-P Release 2;1 ","PSPI"},
{"SPI-P-CTBE Release 1","PSPI"},
{"SPI-P-CTBE-Private Release 1","PSPI"},
{"SPI-P-GV-CT Release 1 ","PSPI"},
{"SPI-P-PCR Release 2 ","PSPI"},
{"SPI-P-Private-CWS Release 1 ","PSPI"},
{"SPI-P-Private-DCI Release 1 ","PSPI"},
{"SPI-P-Private-DiDi Release 1","PSPI"},
{"SPI-P-Private_CDS Release 1 ","PSPI"},
{"SPI-P-Private_ICS Release 1 ","PSPI"},
{"SPI-P-Private_ICS Release 1;1 ","PSPI"},
{"SPI-P-Private_ICS Release 1;2 ","PSPI"},
{"SPI-P-Private_ICS Release 1;4 ","PSPI"},
{"SPI-P-Private_ICS Release 1;5 ","PSPI"},
{"SPI-P-XSB-DCI Release 1 ","PSPI"},
{"SPI-P-XSB-VISUB Release 1 ","PSPI"},
{"STENTOR ","STE"},
{"Siemens: Thorax/Multix FD Image Stamp ","SSPI"},
{"Siemens: Thorax/Multix FD Lab Settings","SSPI"},
{"Siemens: Thorax/Multix FD Post Processing ","SSPI"},
{"Siemens: Thorax/Multix FD Raw Image Settings","SSPI"},
{"Siemens: Thorax/Multix FD Version ","SSPI"},
{"Silhouette Annot V1.0 ","ISG"},
{"Silhouette Graphics Export V1.0 ","ISG"},
{"Silhouette Line V1.0","ISG"},
{"Silhouette ROI V1.0 ","ISG"},
{"Silhouette Sequence Ids V1.0","ISG"},
{"Silhouette V1.0 ","ISG"},
{"Silhouette VRS 3.0","ISG"},
{"TOSHIBA COMAPL HEADER ","TSH"},
{"TOSHIBA COMAPL OOG","TSH"},
{"TOSHIBA ENCRYPTED SR DATA ","TSH"},
{"TOSHIBA MDW HEADER","TSH"},
{"TOSHIBA MDW NON-IMAGE ","TSH"},
{"TOSHIBA_MEC_1.0 ","TSH"},
{"TOSHIBA_MEC_CT3 ","TSH"},
{"TOSHIBA_MEC_CT_1.0","TSH"},
{"TOSHIBA_MEC_MR3 ","TSH"},
{"TOSHIBA_MEC_OT3 ","TSH"},
{"TOSHIBA_MEC_XA3 ","TSH"},
{"Viewing Protocol",""},
{"http://www.gemedicalsystems.com/it_solutions/rad_pacs/","GEM"},

// Manualy added:
{ "GEMS_Ultrasound_ImageGroup_001", "GEM" },
{ "GEMS_Ultrasound_MovieGroup_001", "GEM" },
{ "SIEMENS MED OCS SITE NAME ", "SSPI" },
{ "PHILIPS IMAGING DD 001", "PMFE" },
{ "PHILIPS MR IMAGING DD 001 ", "PMFE" },
{ "INTELERAD MEDICAL SYSTEMS ", "IMS" },
{ "SIEMENS CT VA1 DUMMY", "SSPI" },
{ "PATRIOT_PRIVATE_IMAGE_DATA", "PPID" },
{ "VEPRO VIF 3.0 DATA", "VV3D" },
{ "VEPRO DICOM TRANSFER 1.0", "VDT1" },
{ "HMC ", "HMC" },
{ "Philips3D ", "PM3D" },
{ "SIEMENS MED SMS USG ANTARES ", "SSPI" },
{ "ACUSON:1.2.840.113680.1.0:7f10", "ACU" },
{ "ACUSON:1.2.840.113680.1.0 ", "ACU" },
{ "SIEMENS MR HEADER", "SSPI" }, // !!
{ "INTEGRIS 1.0", "INT1" },
{ "SYNARC_1.0", "SYN" },
{ "MeVis eatDicom", "MVD" },
{ "PHILIPS MR/PART 6 ", "PMFE" },
{ "PHILIPS MR/PART 7 ", "PMFE" },
{ "PHILIPS MR/PART 12", "PMFE" },
{ "HMC - CT - ID ", "HMC" },
{ "SPI-P-Private_ICS Release 1;8 ", "SPI" },
{ "ACUSON:1.2.840.113680.1.0:7ffe", "ACU" },
{ "SIEMENS MED OCS FIELD NAME", "SSPI" },
{ "SIEMENS MED OCS FIELD ID", "SSPI" },
{ "SIEMENS MED OCS CALIBRATION DATE", "SSPI" },
{ "SIEMENS MED OCS NUMBER OF SUB FRAMES", "SSPI" },
{ "SIEMENS MED OCS AE TITLE", "SSPI" },
{ "GEMS_ADWSoft_DPO1 ", "GEM" },
{ "SPI-P-Private_ICS Release 1;6 ", "SSPI" },
{ "SIEMENS MR HEADER ", "SSPI" },
{ "NO PRIVATE CREATOR", "BLA" },
{ "SFS1.00 ", "SFS" },
{ "ATL PRIVATE TAGS", "ATL" },
{ "Philips EV Imaging DD 022 ", "PMFE" },
{ "Harmony R1.0 C2 ", "HRMY" },
{ "Harmony R1.0", "HRMY" },
{ "Harmony R2.0", "HRMY" },
{ "Harmony R1.0 C3 ", "HRMY" },
{ "Philips US Imaging DD 017 ", "PMFE" },
{ "Philips US Imaging DD 023 ", "PMFE" },
{ "Philips US Imaging DD 033 ", "PMFE" },
{ "Philips US Imaging DD 034 ", "PMFE" },
{ "Philips US Imaging DD 035 ", "PMFE" },
{ "Philips US Imaging DD 036 ", "PMFE" },
{ "Philips US Imaging DD 038 ", "PMFE" },
{ "Philips US Imaging DD 039 ", "PMFE" },
{ "Philips US Imaging DD 040 ", "PMFE" },
{ "Philips US Imaging DD 042 ", "PMFE" },
{ "Philips US Imaging DD 043 ", "PMFE" },
{ "Philips US Imaging DD 081 ", "PMFE" },

{ "Array DICOM private elements version1.0 ", "ARRAY" },
{ "PI Private Block (0781:3000 - 0781:30FF)", "PIP" },






{ "XXXXXXXXXXXXXXXX", "ANO" }, // FIXME !
{ "XXXXXXXXXXXXXX", "ANO"},
{ "XXXXXXXXX_xx", "ANO" }, // FIXME
{ "        MED NM", "ANO" }, // Clearly should be SIEMENS

  { NULL, NULL },
};

std::string GetVersion(std::string const &owner)
{

  const OWNER_VERSION *p = OwnerVersionTable;
  while( p->owner )
    {
//#ifndef NDEBUG
//    if( strlen(p->owner) % 2 )
//      {
//      // HEY !
//      std::cerr << "OWNER= \"" << p->owner << "\"" << std::endl;
//      assert(0);
//      }
//#endif
    //if( owner == p->owner )
    if( strcmp(owner.c_str(), p->owner) == 0 )
      {
      return p->version;
      }
    ++p;
    }
  std::cerr << "OWNER= \"" << owner << "\"" << std::endl;
  return "GDCM:UNKNOWN";
}

// TODO: make it protected:
std::string GetOwner(DataSet const &ds, DataElement const &de)
{
   return ds.GetPrivateCreator(de.GetTag());
}

void DictPrinter::PrintDataElement2(std::ostream& os, const DataSet &ds, const DataElement &de)
{
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();

  //const SequenceOfItems *sqi = de.GetSequenceOfItems();
  //const SequenceOfFragments *sqf = de.GetSequenceOfFragments();

  std::string strowner;
  const char *owner = 0;
  const Tag& t = de.GetTag();
  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    strowner = ds.GetPrivateCreator(t);
    owner = strowner.c_str();
    }
  const DictEntry &entry = dicts.GetDictEntry(t,owner);

  if( de.GetTag().IsPrivate() && de.GetTag().GetElement() >= 0x0100 )
    {
    //owner = GetOwner(ds,de);
    //version = GetVersion(owner);

    const VR &vr = de.GetVR();
    VR pvr = vr;
    if( vr == VR::INVALID ) pvr = VR::UN;
    if( de.GetTag().GetElement() == 0x0 )
      {
      pvr = VR::UL;
      }
    else if( de.GetTag().GetElement() <= 0xFF  )
      {
      pvr = VR::LO;
      owner = "Private Creator";
      }
    VM vm = GuessVMType(de);

    os <<
      "<entry group=\"" << std::hex << std::setw(4) << std::setfill('0') <<
      t.GetGroup() << "\" element=\"" << std::setw(4) << ((uint16_t)(t.GetElement() << 8) >> 8) << "\" ";

    os <<  "vr=\"" << pvr << "\" vm=\"" << vm << "\" ";
    //os <<  "\" retired=\"false\";
    if( de.GetTag().IsPrivate() )
      {
      os << "name=\"?\" owner=\"" << owner
        << /*"\"  version=\"" << version << */ "\"/>\n";
      }
    //os << "\n  <description>?</description>\n";
    //os << "</entry>\n";
    //os << "/>\n";
    //os << "  <description>Unknown ";
    //os << (t.IsPrivate() ? "Private" : "Public");
    //os << " Tag & Data</description>\n";
    //os << "  <representations>\n";
    //os << "    <representation vr=\"" << vr << "\" vm=\"" <<
    //  VM::GetVMString(vm) << "\"/>\n";
    //os << "  </representations>\n";
    //os << "</entry>\n";
    }

  if( entry.GetVR() == VR::SQ || true )
    {
    SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();
    if( sqi )
      {
      SequenceOfItems::ItemVector::const_iterator it = sqi->Items.begin();
      for(; it != sqi->Items.end(); ++it)
        {
        const Item &item = *it;
        const DataSet &nestedds = item.GetNestedDataSet();
        PrintDataSet2(os, nestedds);
        }
      }
    }
}

//-----------------------------------------------------------------------------
void DictPrinter::PrintDataSet2(std::ostream& os, const DataSet &ds)
{
  DataSet::ConstIterator it = ds.Begin();
  for( ; it != ds.End(); ++it )
    {
    const DataElement &de = *it;
    PrintDataElement2(os, ds, de);
    }
}

void DictPrinter::Print(std::ostream& os)
{
  const DataSet &ds = F->GetDataSet();
  PrintDataSet2(os, ds);
  //os << "</dict>\n";
}

}
