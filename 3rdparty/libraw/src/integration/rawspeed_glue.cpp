/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/libraw_cxx_defs.h"

#ifdef USE_RAWSPEED
using namespace RawSpeed;

CameraMetaDataLR::CameraMetaDataLR(char *data, int sz) : CameraMetaData()
{
  ctxt = xmlNewParserCtxt();
  if (ctxt == NULL)
  {
    ThrowCME("CameraMetaData:Could not initialize context.");
  }

  xmlResetLastError();
  doc = xmlCtxtReadMemory(ctxt, data, sz, "", NULL, XML_PARSE_DTDVALID);

  if (doc == NULL)
  {
    ThrowCME("CameraMetaData: XML Document could not be parsed successfully. "
             "Error was: %s",
             ctxt->lastError.message);
  }

  if (ctxt->valid == 0)
  {
    if (ctxt->lastError.code == 0x5e)
    {
      // ignore this error code
    }
    else
    {
      ThrowCME("CameraMetaData: XML file does not validate. DTD Error was: %s",
               ctxt->lastError.message);
    }
  }

  xmlNodePtr cur;
  cur = xmlDocGetRootElement(doc);
  if (xmlStrcmp(cur->name, (const xmlChar *)"Cameras"))
  {
    ThrowCME("CameraMetaData: XML document of the wrong type, root node is not "
             "cameras.");
    return;
  }

  cur = cur->xmlChildrenNode;
  while (cur != NULL)
  {
    if ((!xmlStrcmp(cur->name, (const xmlChar *)"Camera")))
    {
      Camera *camera = new Camera(doc, cur);
      addCamera(camera);

      // Create cameras for aliases.
      for (unsigned int i = 0; i < camera->aliases.size(); i++)
      {
        addCamera(new Camera(camera, i));
      }
    }
    cur = cur->next;
  }
  if (doc)
    xmlFreeDoc(doc);
  doc = 0;
  if (ctxt)
    xmlFreeParserCtxt(ctxt);
  ctxt = 0;
}

CameraMetaDataLR *make_camera_metadata()
{
  int len = 0, i;
  for (i = 0; i < RAWSPEED_DATA_COUNT; i++)
    if (_rawspeed_data_xml[i])
    {
      len += int(strlen(_rawspeed_data_xml[i]));
    }
  char *rawspeed_xml =
      (char *)calloc(len + 1, sizeof(_rawspeed_data_xml[0][0]));
  if (!rawspeed_xml)
    return NULL;
  int offt = 0;
  for (i = 0; i < RAWSPEED_DATA_COUNT; i++)
    if (_rawspeed_data_xml[i])
    {
      int ll = int(strlen(_rawspeed_data_xml[i]));
      if (offt + ll > len)
        break;
      memmove(rawspeed_xml + offt, _rawspeed_data_xml[i], ll);
      offt += ll;
    }
  rawspeed_xml[offt] = 0;
  CameraMetaDataLR *ret = NULL;
  try
  {
    ret = new CameraMetaDataLR(rawspeed_xml, offt);
  }
  catch (...)
  {
    // Mask all exceptions
  }
  free(rawspeed_xml);
  return ret;
}

#endif

int LibRaw::set_rawspeed_camerafile(char * filename)
{
#ifdef USE_RAWSPEED
  try
  {
    CameraMetaDataLR *camerameta = new CameraMetaDataLR(filename);
    if (_rawspeed_camerameta)
    {
      CameraMetaDataLR *d =
          static_cast<CameraMetaDataLR *>(_rawspeed_camerameta);
      delete d;
    }
    _rawspeed_camerameta = static_cast<void *>(camerameta);
  }
  catch (...)
  {
    // just return error code
    return -1;
  }
#else
    (void)filename;
#endif
  return 0;
}
#ifdef USE_RAWSPEED
void LibRaw::fix_after_rawspeed(int /*bl*/)
{
  if (load_raw == &LibRaw::lossy_dng_load_raw)
    C.maximum = 0xffff;
  else if (load_raw == &LibRaw::sony_load_raw)
    C.maximum = 0x3ff0;
}
#else
void LibRaw::fix_after_rawspeed(int) {}
#endif

int LibRaw::try_rawspeed()
{
#ifdef USE_RAWSPEED
  int ret = LIBRAW_SUCCESS;

#ifdef USE_RAWSPEED_BITS
  int rawspeed_ignore_errors = (imgdata.rawparams.use_rawspeed & LIBRAW_RAWSPEEDV1_IGNOREERRORS);
#else
  int rawspeed_ignore_errors = 0;
#endif
  if (imgdata.idata.dng_version && imgdata.idata.colors == 3 &&
      !strcasecmp(imgdata.idata.software,
                  "Adobe Photoshop Lightroom 6.1.1 (Windows)"))
    rawspeed_ignore_errors = 1;

  // RawSpeed Supported,
  INT64 spos = ID.input->tell();
  void *_rawspeed_buffer = 0;
  try
  {
    ID.input->seek(0, SEEK_SET);
    INT64 _rawspeed_buffer_sz = ID.input->size() + 32;
    _rawspeed_buffer = malloc(_rawspeed_buffer_sz);
    if (!_rawspeed_buffer)
      throw LIBRAW_EXCEPTION_ALLOC;
    ID.input->read(_rawspeed_buffer, _rawspeed_buffer_sz, 1);
    FileMap map((uchar8 *)_rawspeed_buffer, _rawspeed_buffer_sz);
    RawParser t(&map);
    RawDecoder *d = 0;
    CameraMetaDataLR *meta =
        static_cast<CameraMetaDataLR *>(_rawspeed_camerameta);
    d = t.getDecoder();
    if (!d)
      throw "Unable to find decoder";

#ifdef USE_RAWSPEED_BITS
    if (imgdata.rawparams.use_rawspeed & LIBRAW_RAWSPEEDV1_FAILONUNKNOWN)
      d->failOnUnknown = TRUE;
    else
        d->failOnUnknown = FALSE;
#endif
    d->interpolateBadPixels = FALSE;
    d->applyStage1DngOpcodes = FALSE;

    try
    {
      d->checkSupport(meta);
    }
    catch (const RawDecoderException &e)
    {
      imgdata.process_warnings |= LIBRAW_WARN_RAWSPEED_UNSUPPORTED;
      throw e;
    }
    _rawspeed_decoder = static_cast<void *>(d);
    d->decodeRaw();
    d->decodeMetaData(meta);
    RawImage r = d->mRaw;
    if (r->errors.size() > 0 && !rawspeed_ignore_errors)
    {
      delete d;
      _rawspeed_decoder = 0;
      throw 1;
    }
    if (r->isCFA)
    {
      imgdata.rawdata.raw_image = (ushort *)r->getDataUncropped(0, 0);
    }
    else if (r->getCpp() == 4)
    {
      imgdata.rawdata.color4_image = (ushort(*)[4])r->getDataUncropped(0, 0);
      if (r->whitePoint > 0 && r->whitePoint < 65536)
        C.maximum = r->whitePoint;
    }
    else if (r->getCpp() == 3)
    {
      imgdata.rawdata.color3_image = (ushort(*)[3])r->getDataUncropped(0, 0);
      if (r->whitePoint > 0 && r->whitePoint < 65536)
        C.maximum = r->whitePoint;
    }
    else
    {
      delete d;
      _rawspeed_decoder = 0;
      ret = LIBRAW_UNSPECIFIED_ERROR;
    }
    if (_rawspeed_decoder)
    {
      // set sizes
      iPoint2D rsdim = r->getUncroppedDim();
      S.raw_pitch = r->pitch;
      S.raw_width = rsdim.x;
      S.raw_height = rsdim.y;
      // C.maximum = r->whitePoint;
      fix_after_rawspeed(r->blackLevel);
    }
    free(_rawspeed_buffer);
    _rawspeed_buffer = 0;
    imgdata.process_warnings |= LIBRAW_WARN_RAWSPEED_PROCESSED;
  }
  catch (const RawDecoderException &RDE)
  {
    imgdata.process_warnings |= LIBRAW_WARN_RAWSPEED_PROBLEM;
    if (_rawspeed_buffer)
    {
      free(_rawspeed_buffer);
      _rawspeed_buffer = 0;
    }
    if (!strncmp(RDE.what(), "Decoder canceled", strlen("Decoder canceled")))
      throw LIBRAW_EXCEPTION_CANCELLED_BY_CALLBACK;
    ret = LIBRAW_UNSPECIFIED_ERROR;
  }
  catch (...)
  {
    // We may get here due to cancellation flag
    imgdata.process_warnings |= LIBRAW_WARN_RAWSPEED_PROBLEM;
    if (_rawspeed_buffer)
    {
      free(_rawspeed_buffer);
      _rawspeed_buffer = 0;
    }
    ret = LIBRAW_UNSPECIFIED_ERROR;
  }
  ID.input->seek(spos, SEEK_SET);

  return ret;
#else
  return LIBRAW_NOT_IMPLEMENTED;
#endif
}
