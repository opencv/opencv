
// GENERATED FILE DO NOT EDIT

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/


#include "gdcmTypes.h"
#include "gdcmGroupDict.h"

namespace gdcm {

typedef struct
{
  uint16_t group;
  const char *abbreviation;
  const char *name;
} GROUP_ENTRY;

static GROUP_ENTRY groupname[] = {
  {0x0000,"CMD","Command"},
  {0x0002,"META","Meta Element"},
  {0x0004,"DIR","Directory"},
  {0x0004,"FST","File Set"},
  {0x0008,"ID","Identifying"},
  {0x0009,"SPII","SPI Identifying"},
  {0x0010,"PAT","Patient"},
  {0x0012,"CLI","Clinical Trial"},
  {0x0018,"ACQ","Acquisition"},
  {0x0019,"SPIA","SPI Acquisition"},
  {0x0020,"IMG","Image"},
  {0x0021,"SPIIM","SPI Image"},
  {0x0022,"OPHY","Ophtalmology"},
  {0x0028,"IMGP","Image Presentation"},
  {0x0032,"SDY","Study"},
  {0x0038,"VIS","Visit"},
  {0x003a,"WAV","Waveform"},
  {0x0040,"PRC","Procedure"},
  {0x0040,"MOD","Modality Worklist"},
  {0x0042,"EDOC","Encapsulated Document"},
  {0x0050,"XAD","XRay Angio Device"},
  {0x0050,"DEV","Device Information"},
  {0x0054,"NMI","Nuclear Medicine"},
  {0x0060,"HIS","Histogram"},
  {0x0070,"PRS","Presentation State"},
  {0x0072,"HST","Hanging Protocol"},
  {0x0088,"STO","Storage"},
  {0x0088,"MED","Medicine"},
  {0x0100,"AUTH","Authorization"},
  {0x0400,"DSIG","Digital Signature"},
  {0x1000,"COT","Code Table"},
  {0x1010,"ZMAP","Zonal Map"},
  {0x2000,"BFS","Film Session"},
  {0x2010,"BFB","Film Box"},
  {0x2020,"BIB","Image Box"},
  {0x2030,"BAB","Annotation"},
  {0x2040,"IOB","Overlay Box"},
  {0x2050,"PLUT","Presentation LUT"},
  {0x2100,"PJ","Print Job"},
  {0x2110,"PRINTER","Printer"},
  {0x2120,"QUE","Queue"},
  {0x2130,"PCT","Print Content"},
  {0x2200,"MEDIAC","Media Creation"},
  {0x3002,"RTI","RT Image"},
  {0x3004,"RTD","RT Dose"},
  {0x3006,"SSET","RT StructureSet"},
  {0x3008,"RTT","RT Treatment"},
  {0x300a,"RTP","RT Plan"},
  {0x300c,"RTR","RT Relationship"},
  {0x300e,"RTA","RT Approval"},
  {0x4000,"TXT","Text"},
  {0x4008,"RES","Results"},
  {0x4ffe,"MAC","MAC Parameters"},
  {0x5000,"CRV","Curve"},
  {0x5002,"CRV","Curve"},
  {0x5004,"CRV","Curve"},
  {0x5006,"CRV","Curve"},
  {0x5008,"CRV","Curve"},
  {0x500a,"CRV","Curve"},
  {0x500c,"CRV","Curve"},
  {0x500e,"CRV","Curve"},
  {0x5400,"WFM","Waveform Data"},
  {0x6000,"OLY","Overlays"},
  {0x6002,"OLY","Overlays"},
  {0x6004,"OLY","Overlays"},
  {0x6008,"OLY","Overlays"},
  {0x600a,"OLY","Overlays"},
  {0x600c,"OLY","Overlays"},
  {0x600e,"OLY","Overlays"},
  {0xfffc,"GEN","Generic"},
  {0x7fe0,"PXL","Pixel Data"},
  {0xffff,"UNK","Unknown"},
  {0,0,0} // will not be added to the dict
};

void GroupDict::FillDefaultGroupName()
{
  unsigned int i = 0;
  GROUP_ENTRY n = groupname[i];
  while( n.name != 0 )
  {
    Insert( n.group, n.abbreviation, n.name );
    n = groupname[++i];
  }
}

} // namespace gdcm
