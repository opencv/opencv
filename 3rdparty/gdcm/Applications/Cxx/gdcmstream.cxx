/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAttribute.h"
#include "gdcmFile.h"
#include "gdcmFilename.h"
#include "gdcmImageHelper.h"
#include "gdcmItem.h"
#include "gdcmMediaStorage.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmStreamImageReader.h"
#include "gdcmStreamImageWriter.h"
#include "gdcmSystem.h"
#include "gdcmTag.h"
#include "gdcmTrace.h"
#include "gdcmTransferSyntax.h"
#include "gdcmUIDGenerator.h"
#include "gdcmVersion.h"

#ifdef OPENJPEG_MAJOR_VERSION
#if OPENJPEG_MAJOR_VERSION == 1
#include "gdcm_openjpeg.h"
#elif OPENJPEG_MAJOR_VERSION == 2
#define USE_OPJ_DEPRECATED // opj_setup_decoder
#include "gdcm_openjpeg2.h"
#else
#error should not happen
#endif
#else
#error should not happen
#endif

#include <getopt.h>

static void error_callback(const char *msg, void *) {
  (void)msg;
}
static void warning_callback(const char *msg, void *) {
  (void)msg;
}
static void info_callback(const char *msg, void *) {
  (void)msg;
}

template <typename T>
static unsigned int readvector(std::vector<T> &v, const char *str)
{
  if( !str ) return 0;
  std::istringstream os( str );
  T f;
  while( os >> f )
    {
    v.push_back( f );
    os.get(); //  == ","
    }

  return (unsigned int)v.size();
}

static int No_Of_Resolutions(const char *filename)
{
  std::ifstream is;
  is.open( filename, std::ios::binary );
  opj_dparameters_t parameters;  /* decompression parameters */
  opj_event_mgr_t event_mgr;    /* event manager */
  opj_dinfo_t* dinfo;  /* handle to a decompressor */
  opj_cio_t *cio;
  // FIXME: Do some stupid work:
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  char *dummy_buffer = new char[(unsigned int)buf_size];
  is.seekg(0, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  unsigned char *src = (unsigned char*)dummy_buffer;
  uint32_t file_length = (uint32_t)buf_size; // 32bits truncation should be ok since DICOM cannot have larger than 2Gb image


  /* configure the event callbacks (not required) */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = info_callback;

  /* set decoding parameters to default values */
  opj_set_default_decoder_parameters(&parameters);

  // default blindly copied
  parameters.cp_layer=0;
  parameters.cp_reduce= 0;
  //   parameters.decod_format=-1;
  //   parameters.cod_format=-1;

  const char jp2magic[] = "\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A";
  if( memcmp( src, jp2magic, sizeof(jp2magic) ) == 0 )
    {
    /* JPEG-2000 compressed image data ... sigh */
    // gdcmData/ELSCINT1_JP2vsJ2K.dcm
    // gdcmData/MAROTECH_CT_JP2Lossy.dcm
    //gdcmWarningMacro( "J2K start like JPEG-2000 compressed image data instead of codestream" );
    parameters.decod_format = 1; //JP2_CFMT;
    //assert(parameters.decod_format == JP2_CFMT);
    }
  else
    {
    /* JPEG-2000 codestream */
    //parameters.decod_format = J2K_CFMT;
    //assert(parameters.decod_format == J2K_CFMT);
    assert( 0 );
    }
  parameters.cod_format = 11; // PGX_DFMT;
  //assert(parameters.cod_format == PGX_DFMT);

  /* get a decoder handle */
  dinfo = opj_create_decompress(CODEC_JP2);

  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, NULL);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);

  /* open a byte stream */
  cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

  /* decode the stream and fill the image structure */
  if(! opj_decode(dinfo, cio)) {
    opj_destroy_decompress(dinfo);
    opj_cio_close(cio);
    //gdcmErrorMacro( "opj_decode failed" );
    return 1;
  }

       opj_cp_t * cp = ((opj_jp2_t*)dinfo->jp2_handle)->j2k->cp;
       opj_tcp_t *tcp = &cp->tcps[0];
       opj_tccp_t *tccp = &tcp->tccps[0];

  return tccp->numresolutions;
    /*   std::cout << "\n No of Cols In Image" << image->x1;
       std::cout << "\n No of Rows In Image" << image->y1;
       std::cout << "\n No of Components in Image" << image->numcomps;
       std::cout << "\n No of Resolutions"<< tccp->numresolutions << "\n";
*/

}

static bool Write_Resolution(gdcm::StreamImageWriter & theStreamWriter, const char *filename, int res, std::ostream& of, int flag,  gdcm::SequenceOfItems *sq)
{
  (void)of;
  std::ifstream is;
  is.open( filename, std::ios::binary );
  opj_dparameters_t parameters;  /* decompression parameters */
  opj_event_mgr_t event_mgr;    /* event manager */
  opj_dinfo_t* dinfo;  /* handle to a decompressor */
  opj_cio_t *cio;
  opj_image_t *image = NULL;
  // FIXME: Do some stupid work:
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  char *dummy_buffer = new char[buf_size];
  is.seekg(0, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  unsigned char *src = (unsigned char*)dummy_buffer;
  uint32_t file_length = (uint32_t)buf_size; // 32bits truncation should be ok since DICOM cannot have larger than 2Gb image


  /* configure the event callbacks (not required) */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = info_callback;

  /* set decoding parameters to default values */
  opj_set_default_decoder_parameters(&parameters);

  // default blindly copied
  parameters.cp_layer=0;
  parameters.cp_reduce= res;
  //   parameters.decod_format=-1;
  //   parameters.cod_format=-1;

  const char jp2magic[] = "\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A";
  if( memcmp( src, jp2magic, sizeof(jp2magic) ) == 0 )
    {
    /* JPEG-2000 compressed image data ... sigh */
    // gdcmData/ELSCINT1_JP2vsJ2K.dcm
    // gdcmData/MAROTECH_CT_JP2Lossy.dcm
    //gdcmWarningMacro( "J2K start like JPEG-2000 compressed image data instead of codestream" );
    parameters.decod_format = 1; //JP2_CFMT;
    //assert(parameters.decod_format == JP2_CFMT);
    }
  else
    {
    /* JPEG-2000 codestream */
    //parameters.decod_format = J2K_CFMT;
    //assert(parameters.decod_format == J2K_CFMT);
    assert( 0 );
    }
  parameters.cod_format = 11; // PGX_DFMT;
  //assert(parameters.cod_format == PGX_DFMT);

  /* get a decoder handle */
  dinfo = opj_create_decompress(CODEC_JP2);

  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, NULL);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);

  /* open a byte stream */
  cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

  /* decode the stream and fill the image structure */
  image = opj_decode(dinfo, cio);
  if(!image)
    {
    opj_destroy_decompress(dinfo);
    opj_cio_close(cio);
    //gdcmErrorMacro( "opj_decode failed" );
    return 1;
    }

  //opj_cp_t * cp = ((opj_jp2_t*)dinfo->jp2_handle)->j2k->cp;
  //opj_tcp_t *tcp = &cp->tcps[0];
  //opj_tccp_t *tccp = &tcp->tccps[0];
  /*   std::cout << "\n No of Cols In Image" << image->x1;
       std::cout << "\n No of Rows In Image" << image->y1;
       std::cout << "\n No of Components in Image" << image->numcomps;
       std::cout << "\n No of Resolutions"<< tccp->numresolutions << "\n";
   */
  //opj_j2k_t* j2k = NULL;
  //opj_jp2_t* jp2 = NULL;
  //jp2 = (opj_jp2_t*)dinfo->jp2_handle;
  //int reversible = jp2->j2k->cp->tcps->tccps->qmfbid;
  //std:: cout << reversible;
  int Dimensions[2];
{
  int compno = 0;
  opj_image_comp_t *comp = &image->comps[compno];
  Dimensions[0]= comp->w;
  Dimensions[1] = comp->h;
  opj_cio_close(cio);
}
  unsigned long rawlen = Dimensions[0]*Dimensions[1] * image->numcomps;
  //std::cout << "\nTest" <<image->comps[0].factor;
  char *raw = new char[rawlen];

  for (unsigned int compno = 0; compno < (unsigned int)image->numcomps; compno++)
    {
    const opj_image_comp_t *comp = &image->comps[compno];

    int w = comp->w;
    int h = comp->h;
    uint8_t *data8 = (uint8_t*)raw + compno;
    for (int i = 0; i < w * h ; i++)
      {
      int v = image->comps[compno].data[i];
      *data8 = (uint8_t)v;
      data8 += image->numcomps;
      }
    }


  gdcm::Writer w;
  gdcm::File &file = w.GetFile();
  gdcm::DataSet &ds = file.GetDataSet();

  file.GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );

  gdcm::UIDGenerator uid;
  gdcm::DataElement de( gdcm::Tag(0x8,0x18) ); // SOP Instance UID
  de.SetVR( gdcm::VR::UI );
  const char *u = uid.Generate();
  de.SetByteValue( u, (uint32_t)strlen(u) );
  ds.Insert( de );

  gdcm::DataElement de1( gdcm::Tag(0x8,0x16) );
  de1.SetVR( gdcm::VR::UI );
  gdcm::MediaStorage ms( gdcm::MediaStorage::VLWholeSlideMicroscopyImageStorage );
  de1.SetByteValue( ms.GetString(), (uint32_t)strlen(ms.GetString()));
  ds.Insert( de1 );

  gdcm::DataElement de2( gdcm::Tag(0x28,0x04) );
  //de.SetTag(gdcm::Tag(0x28,0x04));
  de2.SetVR( gdcm::VR::CS );

  if(image->numcomps == 1)
    {
    const char mystr[] = "MONOCHROME2";
    de2.SetByteValue(mystr, (uint32_t)strlen(mystr));
    }
  else
    {
    const char mystr1[] = "RGB";
    de2.SetByteValue(mystr1, (uint32_t)strlen(mystr1));
    }

  ds.Insert( de2 );


  gdcm::Attribute<0x0028,0x0100> at = {8};
  ds.Insert( at.GetAsDataElement() );

  gdcm::Attribute<0x0028,0x0002> at1 = { (uint16_t)image->numcomps};
  ds.Insert( at1.GetAsDataElement() );

  gdcm::Attribute<0x0028,0x0101> at2 = {8};
  ds.Insert( at2.GetAsDataElement() );

  gdcm::Attribute<0x0028,0x0102> at3 = {7};
  ds.Insert( at3.GetAsDataElement() );


  if (flag == 1)  //This flag is to write Image Information
    {
    for (int i=0; i <= res; i++)    // Loop to set different dimensions of all resolution
      {
      int a = 1;
      int b =1;
      while(a!=((res+1)-i))
        {
        b = b*2;
        a = a+1;
        }

      uint16_t row = (uint16_t)((image->y1)/b);
      uint16_t col = (uint16_t)((image->x1)/b);

      gdcm::Element<gdcm::VR::IS,gdcm::VM::VM1> el2;
      el2.SetValue(i+1);
      gdcm::DataElement rfn = el2.GetAsDataElement();     //rfn ---> reference frame number
      rfn.SetTag( gdcm::Tag(0x0008,0x1160) );

      gdcm::Element<gdcm::VR::US,gdcm::VM::VM2> el;
      el.SetValue(1,0);
      el.SetValue(1,1);
      gdcm::DataElement ulr = el.GetAsDataElement();     //ulr --> upper left col/row
      ulr.SetTag( gdcm::Tag(0x0048,0x0201) );

      gdcm::Element<gdcm::VR::US,gdcm::VM::VM2> el1;
      el1.SetValue(col,0);
      el1.SetValue(row,1);
      gdcm::DataElement brr = el1.GetAsDataElement();
      brr.SetTag( gdcm::Tag(0x0048,0x0202) );            //brr --> bottom right col/row

      gdcm::Item it;
      gdcm::DataSet &nds = it.GetNestedDataSet();
      nds.Insert( rfn );
      nds.Insert(ulr);
      nds.Insert(brr);

      sq->AddItem(it);
      }//For loop

    gdcm::File &file2 = w.GetFile();
    gdcm::DataSet &ds1 = file2.GetDataSet();

    gdcm::Attribute<0x0048,0x0006> row1 = {(unsigned short)image->y1};
    ds1.Insert( row1.GetAsDataElement() );

    gdcm::Attribute<0x0048,0x0007> col1 = {(unsigned short)image->x1};
    ds1.Insert( col1.GetAsDataElement() );
    gdcm::Attribute<0x0028,0x0008> Number_Of_Frames = {res+1};
    ds1.Insert( Number_Of_Frames.GetAsDataElement() );

    gdcm::DataElement des( gdcm::Tag(0x0048,0x0200) );
    des.SetVR(gdcm::VR::SQ);
    des.SetValue(*sq);
    des.SetVLToUndefined();

    ds1.Insert(des);

    theStreamWriter.SetFile(file2);

    if (!theStreamWriter.WriteImageInformation())
      {
      std::cerr << "unable to write image information" << std::endl;
      return 1; //the CanWrite function should prevent getting here, else,
      //that's a test failure
      }

    ds1.Remove( gdcm::Tag(0x0048,0x0006) );
    ds1.Remove( gdcm::Tag(0x0048,0x0007) );
    ds1.Remove( gdcm::Tag(0x0028,0x0008) );
    }//if (flag == 1)  //This flag is to write Image Information

  gdcm::Attribute<0x0048,0x0006> row = {(unsigned short)image->comps[0].w};
  ds.Insert( row.GetAsDataElement() );

  gdcm::Attribute<0x0048,0x0007> col = {(unsigned short)image->comps[0].h};
  ds.Insert( col.GetAsDataElement() );

  gdcm::Attribute<0x0028,0x0008> Number_Of_Frames = {1};
  ds.Insert( Number_Of_Frames.GetAsDataElement() );

  theStreamWriter.SetFile(file);

  if (!theStreamWriter.CanWriteFile())
    {
    delete [] raw;
    std::cerr << "Not able to write" << std::endl;
    return 0;//this means that the file was unwritable, period.
    //very similar to a ReadImageInformation failure
    }

  // Important to write here
  std::vector<unsigned int> extent = gdcm::ImageHelper::GetDimensionsValue(file);

  unsigned short xmax = (uint16_t)extent[0];
  unsigned short ymax = (uint16_t)extent[1];
  unsigned short theChunkSize = 4;
  unsigned short ychunk = (unsigned short)(extent[1]/theChunkSize); //go in chunk sizes of theChunkSize
  unsigned short zmax = (uint16_t)extent[2];
  //std::cout << "\n"<<xmax << "\n" << ymax<<"\n"<<zmax<<"\n" << image->numcomps<<"\n";

  if (xmax == 0 || ymax == 0)
    {
    std::cerr << "Image has no size, unable to write zero-sized image." << std::endl;
    return 0;
    }

  int z, y, nexty;
  unsigned long prevLen = 0; //when going through the char buffer, make sure to grab
  //the bytes sequentially.  So, store how far you got in the buffer with each iteration.
  for (z = 0; z < zmax; ++z)
    {
    for (y = 0; y < ymax; y += ychunk)
      {
      nexty = y + ychunk;
      if (nexty > ymax) nexty = ymax;
      theStreamWriter.DefinePixelExtent(0, xmax, (uint16_t)y, (uint16_t)nexty, (uint16_t)z, (uint16_t)(z+1));
      unsigned long len = theStreamWriter.DefineProperBufferLength();
      //std::cout << "\n" <<len;
      char* finalBuffer = new char[len];
      memcpy(finalBuffer, &(raw[prevLen]), len);
      //std::cout << "\nable to write";
      if (!theStreamWriter.Write(finalBuffer, len)){
        std::cerr << "writing failure:" << "output.dcm" << " at y = " << y << " and z= " << z << std::endl;
        delete [] raw;
        delete [] finalBuffer;
        return 1;
      }
      delete [] finalBuffer;
      prevLen += len;
      }
    }
  delete raw;
  delete[] src;  //FIXME

  if(dinfo)
    {
    opj_destroy_decompress(dinfo);
    }

  opj_image_destroy(image);

  return true;

}

static bool StreamImageRead_Write(gdcm::StreamImageWriter & theStreamWriter,gdcm::StreamImageReader & reader, int resolution, std::ostream& of, int tile,std::vector<unsigned int> start, std::vector<unsigned int> end)
{
  (void)of;
  gdcm::File file1 = reader.GetFile();
  gdcm::DataSet ds1 = file1.GetDataSet();

  gdcm::Writer w;
  gdcm::File &file = w.GetFile();
  gdcm::DataSet &ds = file.GetDataSet();

  file.GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
  gdcm::DataElement uid = ds1.GetDataElement( gdcm::Tag(0x0008,0x0018) );
  ds.Insert( uid );

  gdcm::DataElement ms = ds1.GetDataElement( gdcm::Tag(0x0008,0x0016) );
  ds.Insert( ms );

  gdcm::DataElement mystr = ds1.GetDataElement( gdcm::Tag(0x0028,0x0004) );
  ds.Insert( mystr );

  std::vector<unsigned int> extent = reader.GetDimensionsValueForResolution(resolution);
  gdcm::Element<gdcm::VR::UL,gdcm::VM::VM1> row;
  if(tile == 1)
    {
    row.SetValue((end[1]-start[1]),0);
    }
  else
    {
    row.SetValue(extent[1],0);
    }
  gdcm::DataElement ulr = row.GetAsDataElement();     //ulr --> upper left col/row
  ulr.SetTag( gdcm::Tag(0x0048,0x0006) );
  ds.Insert( ulr );

  gdcm::Element<gdcm::VR::UL,gdcm::VM::VM1> col;
  if(tile == 1)
    {
    col.SetValue((end[0]-start[0]),0);
    }
  else
    {
    col.SetValue(extent[0],0);
    }
  gdcm::DataElement ulr1 = col.GetAsDataElement();     //ulr --> upper left col/row
  ulr1.SetTag( gdcm::Tag(0x0048,0x0007) );

  ds.Insert( ulr1 );

  gdcm::Attribute<0x0028,0x0008> Number_Of_Frames = {1};
  ds.Insert( Number_Of_Frames.GetAsDataElement() );

  gdcm::DataElement BA = ds1.GetDataElement( gdcm::Tag(0x0028,0x0100) );
  ds.Insert( BA );

  gdcm::DataElement SPP = ds1.GetDataElement( gdcm::Tag(0x0028,0x0002) );
  ds.Insert( SPP );

  gdcm::DataElement BS = ds1.GetDataElement( gdcm::Tag(0x0028,0x0101) );
  ds.Insert( BS );

  gdcm::DataElement HB = ds1.GetDataElement( gdcm::Tag(0x0028,0x0102) );
  ds.Insert( HB );
  theStreamWriter.SetFile(file);

  unsigned short xmin, xmax, ymin, ymax, zmin, zmax, ychunk, theChunkSize;

  if(tile ==1 )
    {
    xmin = (uint16_t)start[0];
    xmax = (uint16_t)end[0];
    ymin = (uint16_t)start[1];
    ymax = (uint16_t)end[1];
    theChunkSize = 4;
    ychunk = (uint16_t)(end[1]-start[1])/theChunkSize; //go in chunk sizes of theChunkSize
    zmin = (unsigned short)(extent[2]-1);
    zmax = (unsigned short)extent[2];
    }

  else
    {
    xmin = 0;
    xmax = (uint16_t)extent[0];
    ymin = 0;
    ymax = (uint16_t)extent[1];
    theChunkSize = 4;
    ychunk = (uint16_t)(extent[1]/theChunkSize); //go in chunk sizes of theChunkSize
    zmin = (uint16_t)(extent[2]-1);
    zmax = (uint16_t)extent[2];
    }

  if (xmax == 0 && ymax == 0)
    {
    std::cerr << "Image has no size, unable to write zero-sized image." << std::endl;
    return 0;
    }

  if (xmax < xmin || ymax < ymin)
    {
    std::cerr << "error in Region Of Interest Information" << std::endl;
    return 0;
    }

  int z, y, nexty;
  //unsigned long prevLen = 0; //when going through the char buffer, make sure to grab
  //the bytes sequentially.  So, store how far you got in the buffer with each iteration.

  for (z = zmin; z < zmax; ++z)
    {
    for (y = ymin; y < ymax; y += ychunk)
      {
      nexty = y + ychunk;
      if (nexty > ymax) nexty = ymax;
      reader.DefinePixelExtent(xmin, xmax, (uint16_t)y, (uint16_t)nexty, (uint16_t)z, (uint16_t)(z+1));
      unsigned long len = reader.DefineProperBufferLength();

      char* finalBuffer = new char[len];
      if (reader.CanReadImage())
        {
        bool result = reader.Read(finalBuffer, len);
        if( !result )
          {
          std::cerr << "res2 failure:"  << std::endl;
          delete [] finalBuffer;
          return 1;
          }
        else
          {
          // std::cout<< "Able to read";
          //delete [] finalBuffer;
          // return 0; //essentially, we're going to skip this file since it can't be read by the streamer
          }
        }
      else
        {
        std::cerr<< "Not able to put in read data buffer"<< std::endl;
        }
      theStreamWriter.DefinePixelExtent(xmin, xmax, (uint16_t)y, (uint16_t)nexty, (uint16_t)z, (uint16_t)(z+1));
      //  unsigned long len = theStreamWriter.DefineProperBufferLength();
      //std::cout << "\n" <<len;
      //char* finalBuffer1 = new char[len];
      //memcpy(finalBuffer1, &(finalBuffer[0]), len);
      //std::cout << "\nable to write";

      if (!theStreamWriter.Write(finalBuffer, len)){
        std::cerr << "writing failure:" << "output.dcm" << " at y = " << y << " and z= " << z << std::endl;
        //delete [] finalBuffer1;
        delete [] finalBuffer;
        return 1;
      }
      delete [] finalBuffer;
      }
    }
  //std::cout << "all is set";
  return true;
}


static bool Different_Resolution_From_DICOM( gdcm::StreamImageWriter & theStreamWriter, const char *filename, std::ostream& of, int res, int tile,std::vector<unsigned int> start, std::vector<unsigned int> end)
{
  //std::vector<std::string>::const_iterator it = filenames.begin();
  gdcm::StreamImageReader reader;
  reader.SetFileName( filename );


  if (!reader.ReadImageInformation())
    {
    std::cerr << "unable to read image information" << std::endl;
    return 1; //unable to read tags as expected.
    }

  gdcm::File file1 = reader.GetFile();
  gdcm::DataSet ds1 = file1.GetDataSet();


  gdcm::Writer w;
  gdcm::File &file = w.GetFile();
  gdcm::DataSet &ds = file.GetDataSet();



  file.GetHeader().SetDataSetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
  gdcm::DataElement uid = ds1.GetDataElement( gdcm::Tag(0x0008,0x0018) );
  ds.Insert( uid );

  gdcm::DataElement ms = ds1.GetDataElement( gdcm::Tag(0x0008,0x0016) );
  ds.Insert( ms );

  gdcm::DataElement mystr = ds1.GetDataElement( gdcm::Tag(0x0028,0x0004) );
  ds.Insert( mystr );

  if(res == 0)
    {
    gdcm::DataElement seq = ds1.GetDataElement( gdcm::Tag(0x0048,0x0200) );
    ds.Insert(seq);
    }
  else
    {
    std::vector<unsigned int> extent = reader.GetDimensionsValueForResolution(res);


    gdcm::SmartPointer<gdcm::SequenceOfItems> sq = new gdcm::SequenceOfItems();
    sq->SetLengthToUndefined();
    gdcm::Element<gdcm::VR::IS,gdcm::VM::VM1> el1;
    el1.SetValue(res);
    gdcm::DataElement rfn = el1.GetAsDataElement();     //rfn ---> reference frame number
    rfn.SetTag( gdcm::Tag(0x0008,0x1160) );

    gdcm::Element<gdcm::VR::US,gdcm::VM::VM2> el2;
    if(tile == 1)
      {
      el2.SetValue((unsigned short)start[0],0);
      el2.SetValue((unsigned short)start[1],1);
      }
    else
      {
      el2.SetValue(1,0);
      el2.SetValue(1,1);
      }
    gdcm::DataElement ulr = el2.GetAsDataElement();     //ulr --> upper left col/row
    ulr.SetTag( gdcm::Tag(0x0048,0x0201) );


    gdcm::Element<gdcm::VR::US,gdcm::VM::VM2> el3;
    if(tile == 1)
      {
      el3.SetValue((unsigned short)end[0],0);
      el3.SetValue((unsigned short)end[1],1);
      }
    else
      {
      el3.SetValue((unsigned short)extent[0],0);
      el3.SetValue((unsigned short)extent[1],1);
      }
    gdcm::DataElement brr = el3.GetAsDataElement();
    brr.SetTag( gdcm::Tag(0x0048,0x0202) );            //brr --> bottom right col/row

    gdcm::Item it;
    gdcm::DataSet &nds = it.GetNestedDataSet();
    nds.Insert( rfn );
    nds.Insert(ulr);
    nds.Insert(brr);

    sq->AddItem(it);

    gdcm::DataElement des( gdcm::Tag(0x0048,0x0200) );
    des.SetVR(gdcm::VR::SQ);
    des.SetValue(*sq);
    des.SetVLToUndefined();

    ds.Insert(des);

    }




  gdcm::DataElement row = ds1.GetDataElement( gdcm::Tag(0x0048,0x0006) );
  assert( row.GetVR() == gdcm::VR::UL );
  ds.Insert(row);

  gdcm::DataElement col = ds1.GetDataElement( gdcm::Tag(0x0048,0x0007) );
  ds.Insert(col);

  gdcm::DataElement Number_Of_Frames = ds1.GetDataElement( gdcm::Tag(0x0028,0x0008) );
  ds.Insert(Number_Of_Frames);

  gdcm::Element<gdcm::VR::IS,gdcm::VM::VM1> el;
  el.SetFromDataElement( Number_Of_Frames );


  uint16_t No_Of_Resolutions  = (uint16_t)el.GetValue(0);
  //std::cout << "HERE NO. "<< No_Of_Resolutions;

  gdcm::DataElement BA = ds1.GetDataElement( gdcm::Tag(0x0028,0x0100) );
  ds.Insert( BA );

  gdcm::DataElement SPP = ds1.GetDataElement( gdcm::Tag(0x0028,0x0002) );
  ds.Insert( SPP );

  gdcm::DataElement BS = ds1.GetDataElement( gdcm::Tag(0x0028,0x0101) );
  ds.Insert( BS );

  gdcm::DataElement HB = ds1.GetDataElement( gdcm::Tag(0x0028,0x0102) );
  ds.Insert( HB );

  theStreamWriter.SetFile(file);

  if (!theStreamWriter.WriteImageInformation())
    {
    std::cerr << "unable to write image information" << std::endl;
    return 1; //the CanWrite function should prevent getting here, else,
    //that's a test failure
    }


  bool b = true;

  if(res == 0)
    {
    for(int i = 1 ; i <= No_Of_Resolutions; ++i)
      {
      b = b && StreamImageRead_Write( theStreamWriter, reader, i, of, tile , start , end );
      }
    }
  else
    b = b && StreamImageRead_Write( theStreamWriter, reader, res, of ,tile, start, end );

  return b;
}



static bool Different_Resolution_From_jp2( gdcm::StreamImageWriter & theStreamWriter, const char *filename, std::ostream& of, int nres)
{
  //std::vector<std::string>::const_iterator it = filenames.begin();
  bool b = true;
  int flag = 1;

  gdcm::SmartPointer<gdcm::SequenceOfItems> sq = new gdcm::SequenceOfItems();
  sq->SetLengthToUndefined();

  int resolutions;

  if(nres==0)
    resolutions = No_Of_Resolutions(filename);
  else
    resolutions = nres;

  for(int i = resolutions-1 ; i>=0; --i)
    {
    b = b && Write_Resolution( theStreamWriter, filename, i, of ,flag,sq);
    flag = 0;
    }
  return b;
}



static void PrintVersion()
{
  std::cout << "gdcmstream: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmstream [OPTION] input.dcm output.dcm" << std::endl;
}

static void end_of_WSIFile(std::ostream& of)
{
  uint16_t firstTag1 =  0xfffe;
  uint16_t secondTag1 = 0xe0dd;
  uint32_t thirdTag1 =  0x00000000;
  //uint16_t fourthTag1 = 0xffff;
  const int theBufferSize1 = 2*sizeof(uint16_t)+sizeof(uint32_t);
  char* tmpBuffer2 = new char[theBufferSize1];
  memcpy(&(tmpBuffer2[0]), &firstTag1, sizeof(uint16_t));
  memcpy(&(tmpBuffer2[sizeof(uint16_t)]), &secondTag1, sizeof(uint16_t));
  memcpy(&(tmpBuffer2[2*sizeof(uint16_t)]), &thirdTag1, sizeof(uint32_t));
  //memcpy(&(tmpBuffer2[3*sizeof(uint16_t)]), &fourthTag1, sizeof(uint16_t));
  assert( of && !of.eof() && of.good() );
  of.write(tmpBuffer2, theBufferSize1);
  of.flush();
  assert( of );
}

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;
  //std::string filename;
  //std::string outfilename;
  gdcm::Filename filename;
  gdcm::Filename outfilename;

  std::string root;
  int rootuid = 0;
  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;
  int nres = 0;
  int res = 0;
  int tile = 0;
  std::vector<unsigned int> start;
  std::vector<unsigned int> end;

  while (1) {

    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},

        // General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {"resolution", 1, &nres, 1},
        {"resolution-only", 1, &res, 1},
        {"roi-start", 1, &tile, 1},
        {"roi-end", 1, &tile, 1},
        {0, 0, 0, 0}
    };
    c = getopt_long (argc, argv, "i:o:VWDEhv:r:n:",long_options, &option_index);

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
            assert( filename.IsEmpty() );
            filename = optarg;
            }
          //printf (" with arg %s, index = %d", optarg, option_index)
          else if( option_index == 1 ) /* input */
            {
            assert( strcmp(s, "output") == 0 );
            assert( outfilename.IsEmpty() );
            outfilename = optarg;
            }
          //printf (" with arg %s, index = %d", optarg, option_index);

          else if( option_index == 8 ) /* number of resolution */
            {
            assert( strcmp(s, "resolution") == 0 );
            nres = atoi(optarg);
            }

          else if( option_index == 9 ) /* number of resolution */
            {
            assert( strcmp(s, "resolution-only") == 0 );
            res = atoi(optarg);
            }

          else if( option_index == 10 ) /* tile */
            {
            assert( strcmp(s, "roi-start") == 0 );
            tile = 1;
            unsigned int n = readvector(start, optarg);
            assert( n == 2 ); (void)n;
            }
          else if( option_index == 11 ) /* tile */
            {
            assert( strcmp(s, "roi-end") == 0 );
            tile = 1;
            unsigned int n = readvector(end, optarg);
            assert( n == 2 ); (void)n;
            }

          //printf ("\n");
          }

        }
      break;

    case 'i':
      assert( filename.IsEmpty() );
      filename = optarg;
      break;

    case 'o':
      assert( outfilename.IsEmpty() );
      outfilename = optarg;
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

    case 'r':
      nres = atoi(optarg);
      break;

    case 'n':
      res = atoi(optarg);
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
    //std::cout << "HERE";
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind]);
      files.push_back( argv[optind] );
      }

    //printf ("\n");
    if( files.size() == 2  && filename.IsEmpty() && outfilename.IsEmpty() )
      {
      filename = files[0].c_str();
      outfilename = files[ files.size() - 1 ].c_str();
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }//if (optind < argc)

  if( version )
    {
    std::cout << "version" << std::endl;
    PrintVersion();
    return 0;
    }

  if( help )
    {
    std::cout << "help here" << std::endl;
    PrintHelp();
    return 0;
    }

  if( filename.IsEmpty())
    {
    std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }

  if( outfilename.IsEmpty() )
    {
    std::cerr << "Need output file (-o)\n";
    PrintHelp();
    return 1;
    }

  // Debug is a little too verbose

  gdcm::Trace::SetDebug( (debug  > 0 ? true : false));
  gdcm::Trace::SetWarning(  (warning  > 0 ? true : false));
  gdcm::Trace::SetError(  (error  > 0 ? true : false));
  // when verbose is true, make sure warningerror are turned on:

  if( verbose )
    {
    gdcm::Trace::SetWarning( (verbose  > 0 ? true : false) );
    gdcm::Trace::SetError( (verbose  > 0 ? true : false) );
    }


  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmstream" );

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

  const char *inputextension  = filename.GetExtension();
  //const char *outputextension = outfilename.GetExtension();

  gdcm::StreamImageWriter theStreamWriter;
  std::ofstream of;
  of.open( outfilename, std::ios::out | std::ios::binary );
  theStreamWriter.SetStream(of);

  if( inputextension )
    {
    if(  gdcm::System::StrCaseCmp(inputextension,".jp2") == 0 )
      {
      if( !Different_Resolution_From_jp2( theStreamWriter, filename,of,nres ) ) return 1;
      end_of_WSIFile(of);
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".dcm") == 0 )
      {
      Different_Resolution_From_DICOM( theStreamWriter, filename, of, res , tile , start , end);
      //if(!StreamImageRead_Write( theStreamWriter, filename, 0)) return 1;
      end_of_WSIFile(of);
      }
    }

  // gdcm::StreamImageReader ...
  return 0;
}
