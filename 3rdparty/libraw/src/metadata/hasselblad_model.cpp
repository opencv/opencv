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

#include "../../internal/dcraw_defs.h"
#include "../../internal/libraw_cameraids.h"

  static const struct {
    const int idx;
    const char *FormatName;
  } HassyRawFormat[] = {
    { LIBRAW_HF_Unknown, "Unknown"},
    { LIBRAW_HF_3FR, "-3FR"},
    { LIBRAW_HF_FFF, "-FFF"},
    { LIBRAW_HF_Imacon, "Imacon"},
    { LIBRAW_HF_HasselbladDNG, "hDNG"},
    { LIBRAW_HF_AdobeDNG, "aDNG"},
    { LIBRAW_HF_AdobeDNG_fromPhocusDNG, "a(hDNG)"},
  };

const char* LibRaw::HassyRawFormat_idx2HR(unsigned idx) // HR means "human-readable"
{
    for (int i = 0; i < int(sizeof HassyRawFormat / sizeof *HassyRawFormat); i++)
        if((unsigned)HassyRawFormat[i].idx == idx)
            return HassyRawFormat[i].FormatName;
    return 0;
}

void LibRaw::process_Hassy_Lens (int LensMount) {
// long long unsigned id =
//    mount*100000000ULL + series*10000000ULL +
//    focal1*10000ULL + focal2*10 + version;
  char *ps;
  int c;
  char *q =  strchr(imgdata.lens.Lens, ' ');
  if(!q) return ;
  c = atoi(q+1);
  if (!c)
    return;

  if (LensMount == LIBRAW_MOUNT_Hasselblad_H) {
    if (imgdata.lens.Lens[2] == ' ') // HC lens
      ilm.LensID = LensMount*100000000ULL + 10000000ULL;
    else                             // HCD lens
      ilm.LensID = LensMount*100000000ULL + 20000000ULL;
    ilm.LensFormat = LIBRAW_FORMAT_645;
  } else if (LensMount == LIBRAW_MOUNT_Hasselblad_XCD) {
    ilm.LensFormat = LIBRAW_FORMAT_CROP645;
    ilm.LensID = LensMount*100000000ULL;
  } else
    return;

  ilm.LensMount = LensMount;
  ilm.LensID += c*10000ULL;
  if ((ps=strchr(imgdata.lens.Lens, '-'))) {
    ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
    ilm.LensID += atoi(ps+1)*10ULL;
  } else {
    ilm.FocalType = LIBRAW_FT_PRIME_LENS;
    ilm.LensID += c*10ULL;
  }
  if (strstr(imgdata.lens.Lens, "III"))
    ilm.LensID += 3ULL;
  else if (strstr(imgdata.lens.Lens, "II"))
    ilm.LensID += 2ULL;
}

void LibRaw::parseHassyModel() {

static const char *Hasselblad_Ctrl[] = { // manually selectable options only
  "ELD", "ELX", "Winder CW", "CW", "Pinhole", "Flash Sync",
  "SWC", "200 (Mod)", "200", "500 Mech.", "500", "H Series",
  "H-Series", "H1", "H2", "Black Box", "LENSCONTROL S", "LENSCTRL S", "Generic",
};

static const char *Hasselblad_SensorEnclosures[] = {
  "CFH", "CFV", "CFV", "CFII", "CF", "Ixpress",
};

  char tmp_model[64];
  const char *ps;
  char *eos;
  int c;
  int nPix = raw_width*raw_height;
  int add_MP_toName = 1;
  int norm_model_isSet = 0;

  if (model[0] == ' ')
    memmove(model, model+1, MIN(sizeof(model)-1,strlen(model)));

  imHassy.HostBody[0] = 0;
  if ((ps = strrchr(model, '/')))
    strcpy(imHassy.HostBody, ps+1);
  else if ((ps = strrchr(imgdata.color.LocalizedCameraModel, '/')))
    strcpy(imHassy.HostBody, ps+1);
  else if ((ps = strrchr(imgdata.color.UniqueCameraModel, '/')))
    strcpy(imHassy.HostBody, ps+1);
  else if ((ps = strrchr(imHassy.SensorUnitConnector, '/')))
    strcpy(imHassy.HostBody, ps+1);
  if (imHassy.HostBody[0]) {
  	if ((eos = strrchr(imHassy.HostBody, '-')))
  	  *eos = 0;
  }

  if (!imHassy.format) {
    if (dng_version) {
      if (!strncmp(software, "Adobe", 5)) {
        if (!imgdata.color.OriginalRawFileName[0] ||
            !imgdata.color.LocalizedCameraModel[0] ||
            !strcasestr(imgdata.color.UniqueCameraModel, "coated"))
          imHassy.format = LIBRAW_HF_AdobeDNG_fromPhocusDNG;
        else
          imHassy.format = LIBRAW_HF_AdobeDNG;
      } else imHassy.format = LIBRAW_HF_HasselbladDNG;
    } else if ((imHassy.nIFD_CM[0] != -1) &&
               (imHassy.nIFD_CM[1] == -1) &&
               !imHassy.mnColorMatrix[0][0]) {
      imHassy.format = LIBRAW_HF_3FR;
    } else imHassy.format = LIBRAW_HF_FFF;
  }

  if (imHassy.SensorUnitConnector[0]) {
    char buf[64];
    if (!strncmp(imHassy.SensorUnitConnector, "Hasselblad ", 11))
      memmove(imHassy.SensorUnitConnector, imHassy.SensorUnitConnector+11, 64-11);
    strcpy(buf, imHassy.SensorUnitConnector);
    if ((eos = strrchr(buf, '/'))) {
      *eos = 0;
      if ((eos = strrchr(buf, ' '))) {
        *eos = 0;
        strcpy (imHassy.SensorUnitConnector, buf);
      }
    }
  }

  if (imHassy.format == LIBRAW_HF_AdobeDNG) { // Adobe DNG, use LocalizedCameraModel
      imgdata.color.LocalizedCameraModel[63] = 0; // make sure it's 0-terminated
    if ((ps = strrchr(imgdata.color.LocalizedCameraModel, '-')))
      c = ps-imgdata.color.LocalizedCameraModel;
    else c = int(strlen(imgdata.color.LocalizedCameraModel));
    int cc = MIN(c, (int)sizeof(tmp_model)-1);
    memcpy(tmp_model, imgdata.color.LocalizedCameraModel,cc);
    tmp_model[cc] = 0;
    if (strcasestr(imgdata.color.UniqueCameraModel, "coated")) {
      strncpy(normalized_model, imgdata.color.UniqueCameraModel,sizeof(imgdata.color.UniqueCameraModel)-1);
      normalized_model[sizeof(imgdata.color.UniqueCameraModel) - 1] = 0;
      norm_model_isSet = 1;
    }
      if (!strncmp(normalized_model, "Hasselblad ", 11))
        memmove(normalized_model, normalized_model+11, 64-11);
  } else {
    if ((ps = strrchr(imgdata.color.UniqueCameraModel, '/'))) {
      c = ps-imgdata.color.UniqueCameraModel;
    }
    else c = int(strlen(imgdata.color.UniqueCameraModel));
    int cc = MIN(c, (int)sizeof(tmp_model)-1);
    memcpy(tmp_model, imgdata.color.UniqueCameraModel,cc);
    tmp_model[cc] = 0;
  }
  if (!strncasecmp(tmp_model, "Hasselblad ", 11))
    memmove(tmp_model, tmp_model+11, 64-11);

  strncpy(imHassy.CaptureSequenceInitiator, model,31);
  imHassy.CaptureSequenceInitiator[31] = 0;
  if ((eos = strrchr(imHassy.CaptureSequenceInitiator, '/'))) {
    *eos = 0;
  }
// check if model tag contains manual CaptureSequenceInitiator info:
  FORC(int(sizeof Hasselblad_Ctrl / sizeof *Hasselblad_Ctrl)) {
    if (strcasestr(model, Hasselblad_Ctrl[c])) {
// yes, fill 'model' with sensor unit data
      strncpy(model, tmp_model,63);
      model[63] = 0;
      break;
    }
  }

  if (!imHassy.HostBody[0]) {
    ps = strchr(model, '-');
    if (ps) {                  // check if model contains both host body and sensor version, resolution, MS info
      strncpy(imHassy.SensorUnit, model,63);
      memcpy(imHassy.HostBody, model, ps-model);
      imHassy.HostBody[ps-model] = 0;
      if (!strncmp(ps-2, "II-", 3))
        ps -=2;
      strncpy(imHassy.Sensor, ps,7);
      imHassy.Sensor[7] = 0;
      add_MP_toName = 0;
    } else { // model contains host body only
      strncpy(imHassy.HostBody, model,63);
      imHassy.HostBody[63] = 0;
  // fill 'model' with sensor unit data
      strncpy(model, tmp_model,63);
      model[63] = 0;
    }
  }

  if (strstr(model, "503CWD")) {
    strncpy(imHassy.HostBody, model,63);
    imHassy.HostBody[63] = 0;
    ilm.CameraFormat = LIBRAW_FORMAT_66;
    ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_V;
    if (model[6] == 'I' && model[7] == 'I')
      strcpy(model, "CFVII");
    else strcpy(model, "CFV");
  } else if (strstr(model, "Hasselblad") &&
             (model[10] != ' ')) {
    strcpy(model, "CFV");
    ilm.CameraMount = LIBRAW_MOUNT_DigitalBack;
  } else {
    FORC(int(sizeof Hasselblad_SensorEnclosures / sizeof *Hasselblad_SensorEnclosures)) {
      if (strcasestr(model, Hasselblad_SensorEnclosures[c])) {
        if (add_MP_toName) strcpy(model, Hasselblad_SensorEnclosures[c]);
        ilm.CameraMount = LIBRAW_MOUNT_DigitalBack;
        break;
      }
    }
  }

#define cpynorm(str)                \
  if (!norm_model_isSet) {          \
    strcpy(normalized_model, str);  \
    norm_model_isSet = 1;           \
  }

  if ((imHassy.SensorCode == 4) &&
      (imHassy.CoatingCode < 2)) {
    strcpy(imHassy.Sensor, "-16");
    cpynorm("16-Uncoated");

  } else if ((imHassy.SensorCode == 6) &&
             (imHassy.CoatingCode < 2)) {
    strcpy(imHassy.Sensor, "-22");
    cpynorm("22-Uncoated");

  } else if ((imHassy.SensorCode == 8) &&
             (imHassy.CoatingCode == 1)) {
    strcpy(imHassy.Sensor, "-31");
    cpynorm("31-Uncoated");

  } else if ((imHassy.SensorCode == 9) &&
             (imHassy.CoatingCode < 2)) {
    strcpy(imHassy.Sensor, "-39");
    cpynorm("39-Uncoated");

  } else if ((imHassy.SensorCode == 9) &&
             (imHassy.CoatingCode == 4)) {
    strcpy(imHassy.Sensor, "-39");
    strcpy(model, "H3DII");
    add_MP_toName = 1;
    cpynorm("39-Coated");

  } else if ((imHassy.SensorCode == 13) &&
             (imHassy.CoatingCode == 4)) {
    strcpy(imHassy.Sensor, "-40");
    cpynorm("40-Coated");

  } else if ((imHassy.SensorCode == 13) &&
             (imHassy.CoatingCode == 5)) {
    strcpy(imHassy.Sensor, "-40");
    cpynorm("40-Coated5");

  } else if ((imHassy.SensorCode == 11) &&
             (imHassy.CoatingCode == 4)) {
    if (!strncmp(model, "H3D", 3))
      strcpy(model, "H3DII-50");
    else strcpy(imHassy.Sensor, "-50");
    cpynorm("50-Coated");

  } else if ((imHassy.SensorCode == 11) &&
             (imHassy.CoatingCode == 5)) {
    strcpy(imHassy.Sensor, "-50");
    cpynorm("50-Coated5");

  } else if ((imHassy.SensorCode == 15) &&
             (imHassy.CoatingCode == 5)) {
    strcpy(imHassy.Sensor, "-50c");
    cpynorm("50-15-Coated5");
    if (!strncmp(imHassy.CaptureSequenceInitiator, "CFV II 50C", 10)) {
      imHassy.SensorSubCode = 2;
      add_MP_toName = 0;
      strcat(imHassy.Sensor, " II");
      strcpy(model, "CFV II 50C");
      strcat(normalized_model, "-II");
    } else if (!strncmp(imHassy.CaptureSequenceInitiator, "X1D", 3)) {
      imHassy.SensorSubCode = 2;
      add_MP_toName = 0;
      strcat(imHassy.Sensor, " II");
      if (!strncasecmp(imHassy.CaptureSequenceInitiator, "X1D II 50C", 10)) {
        strcpy(model, "X1D II 50C");
        strcat(normalized_model, "-II");
      } else {
        strcpy(model, "X1D-50c");
      }
    }

  } else if ((imHassy.SensorCode == 12) &&
             (imHassy.CoatingCode == 4)) {
    strcpy(imHassy.Sensor, "-60");
    cpynorm("60-Coated");

  } else if ((imHassy.SensorCode == 17) &&
             (imHassy.CoatingCode == 5)) {
    strcpy(imHassy.Sensor, "-100c");
    cpynorm("100-17-Coated5");

  } else if ((raw_width == 4090) || // V96C
             ((raw_width == 4096) && (raw_height == 4096)) ||
             ((raw_width == 4088) && (raw_height == 4088)) || // Adobe crop
             ((raw_width == 4080) && (raw_height == 4080))) { // Phocus crop
    strcpy(imHassy.Sensor, "-16");
    cpynorm("16-Uncoated");
    if (!imHassy.SensorCode) imHassy.SensorCode = 4;

  } else if ((raw_width == 5568) && (raw_height == 3648)) {
    strcpy(imHassy.Sensor, "-20c");

  } else if (((raw_width == 4096) && (raw_height == 5456)) ||
             ((raw_width == 4088) && (raw_height == 5448)) ||  // Adobe crop
             ((raw_width == 4080) && (raw_height == 5440))) {  // Phocus crop
    strcpy(imHassy.Sensor, "-22");
    cpynorm("22-Uncoated");
    if (!imHassy.SensorCode) imHassy.SensorCode = 6;

  } else if (((raw_width == 6542) && (raw_height == 4916)) ||
             ((raw_width == 6504) && (raw_height == 4880)) || // Adobe crop
             ((raw_width == 6496) && (raw_height == 4872))) { // Phocus crop
    strcpy(imHassy.Sensor, "-31");
    cpynorm("31-Uncoated");
    if (!imHassy.SensorCode) imHassy.SensorCode = 8;

  } else if (((raw_width == 7262) && (raw_height == 5456)) || //
             ((raw_width == 7224) && (raw_height == 5420)) || // Adobe crop
             ((raw_width == 7216) && (raw_height == 5412)) || // Phocus crop
             ((raw_width == 7212) && (raw_height == 5412)) || // CF-39, CFV-39, possibly v.II; Phocus crop
// uncropped, when the exact size is unknown, should be:
// - greater or equal to the smallest Phocus crop for the current size
// - smaller than the smallest Phocus crop for the next size
           ((nPix >= 7212*5412) && (nPix < 7304*5478))) {
    strcpy(imHassy.Sensor, "-39");
    if (!imHassy.SensorCode) imHassy.SensorCode = 9;
    if (!strncmp(model, "H3D", 3)) {
      if (((imHassy.format == LIBRAW_HF_Imacon) ||
          strstr(imgdata.color.UniqueCameraModel, "H3D-39") ||
          strstr(imgdata.color.LocalizedCameraModel, "H3D-39") ||
          strstr(model, "H3D-39")) &&
          !strstr(imgdata.color.UniqueCameraModel, "II") &&
          !strstr(imgdata.color.LocalizedCameraModel, "II") &&
          !strstr(model, "II")) {
        strcpy(model, "H3D-39");
        add_MP_toName = 0;
        cpynorm("39-Uncoated");

      } else {
        strcpy(model, "H3DII-39");
        add_MP_toName = 0;
        cpynorm("39-Coated");
        if (!imHassy.CoatingCode) imHassy.CoatingCode = 4;
      }

    } else
      cpynorm("39-Uncoated");

  } else if (((raw_width == 7410) && (raw_height == 5586)) || // (H4D-40, H5D-40)
             ((raw_width == 7312) && (raw_height == 5486)) || // Adobe crop
             ((raw_width == 7304) && (raw_height == 5478))) { // Phocus crop
    strcpy(imHassy.Sensor, "-40");
    if (!strncmp(model, "H4D", 3)) {
      cpynorm("40-Coated");
      if (!imHassy.SensorCode) imHassy.SensorCode = 13;
      if (!imHassy.CoatingCode) imHassy.CoatingCode = 4;
    } else {
      cpynorm("40-Coated5");
      if (!imHassy.SensorCode) imHassy.SensorCode = 13;
      if (!imHassy.CoatingCode) imHassy.CoatingCode = 5;
    }

  } else if (((raw_width == 8282) && (raw_height == 6240)) || // (CFV-50, H3DII-50, H5D-50)
             ((raw_width == 8184) && (raw_height == 6140)) || // Adobe crop
             ((raw_width == 8176) && (raw_height == 6132))) { // Phocus crop
    strcpy(imHassy.Sensor, "-50");
    if (!strncmp(model, "H5D", 3)) {
      cpynorm("50-Coated5");
      if (!imHassy.SensorCode) imHassy.SensorCode = 11;
      if (!imHassy.CoatingCode) imHassy.CoatingCode = 5;
    } else {
      cpynorm("50-Coated"); // CFV-50, H3DII-50,
      if (!strncmp(model, "H3D", 3)) {
        strcpy(model, "H3DII-50");
      if (!imHassy.SensorCode) imHassy.SensorCode = 11;
      if (!imHassy.CoatingCode) imHassy.CoatingCode = 4;
        add_MP_toName = 0;
      }
    }

  } else if (((raw_width == 8374) && (raw_height == 6304)) ||  // (H5D-50c)
             ((raw_width == 8384) && (raw_height == 6304)) ||  // (X1D-50c, "X1D II 50C", "CFV II 50C")
             ((raw_width == 8280) && (raw_height == 6208)) ||  // Adobe crop
             ((raw_width == 8272) && (raw_height == 6200))) {  // Phocus crop
    cpynorm("50-15-Coated5");
    if (!imHassy.SensorCode) imHassy.SensorCode = 15;
    if (!imHassy.CoatingCode) imHassy.CoatingCode = 5;
    strcpy(imHassy.Sensor, "-50c");
    if ((raw_width == 8384) ||
        !strncmp(imHassy.CaptureSequenceInitiator, "X1D", 3) ||
        !strncmp(imHassy.CaptureSequenceInitiator, "CFV II", 6)) {
      imHassy.SensorSubCode = 2;
      add_MP_toName = 0;
      strcat(imHassy.Sensor, " II");
      if (strstr(imHassy.CaptureSequenceInitiator, " II ")) {
          strcat(normalized_model, "-II");
        if (!strncasecmp(imHassy.CaptureSequenceInitiator, "X1D II 50C", 10)) {
          strcpy(model, "X1D II 50C");
        } else if (!strncasecmp(imHassy.CaptureSequenceInitiator, "CFV II 50C", 10)) {
          strcpy(model, "CFV II 50C");
        }
      } else {
        strcpy(model, "X1D-50c");
      }
    }

  } else if (((raw_width == 9044) && (raw_height == 6732)) ||
             ((raw_width == 8964) && (raw_height == 6716)) || // Adobe crop
             ((raw_width == 8956) && (raw_height == 6708))) { // Phocus crop
    strcpy(imHassy.Sensor, "-60");
    cpynorm("60-Coated");
    if (!imHassy.SensorCode) imHassy.SensorCode = 12;
    if (!imHassy.CoatingCode) imHassy.CoatingCode = 4;


  } else if (((raw_width == 10320) && (raw_height == 7752)) || // Phocus crop, A5D-80
             ((nPix >= 10320*7752) && (nPix < 10520*8000))) {
    strcpy(imHassy.Sensor, "-80");
    cpynorm("80-Coated");

  } else if (((raw_width == 12000) && (raw_height == 8816)) ||
             ((raw_width == 11608) && (raw_height == 8708)) || // Adobe crop
             ((raw_width == 11600) && (raw_height == 8700))) {  // Phocus crop
    strcpy(imHassy.Sensor, "-100c");
    cpynorm("100-17-Coated5");
    if (!imHassy.SensorCode) imHassy.SensorCode = 17;
    if (!imHassy.CoatingCode) imHassy.CoatingCode = 5;

  }

  if (raw_width == 4090)
    strcpy(model, "V96C");

  if (
    (raw_width == 4090) ||
		((raw_width ==  4096) && (raw_height ==  4096)) ||
		((raw_width ==  5568) && (raw_height ==  3648)) ||
		((raw_width ==  4096) && (raw_height ==  5456)) ||
		((raw_width ==  6542) && (raw_height ==  4916)) ||
		((raw_width ==  7262) && (raw_height ==  5456)) ||
		((raw_width ==  7410) && (raw_height ==  5586)) ||
		((raw_width ==  8282) && (raw_height ==  6240)) ||
		((raw_width ==  8374) && (raw_height ==  6304)) ||
		((raw_width ==  8384) && (raw_height ==  6304)) ||
		((raw_width ==  9044) && (raw_height ==  6732)) ||
		((raw_width == 10320) && (raw_height ==  7752)) ||
		((raw_width == 12000) && (raw_height ==  8816))
	)
	imHassy.uncropped = 1;


  if (model[0] && add_MP_toName)
    strcat(model, imHassy.Sensor);
  if (imHassy.Sensor[0] == '-')
    memmove(imHassy.Sensor, imHassy.Sensor+1, strlen(imHassy.Sensor));

  if (dng_version &&
      (imHassy.SensorCode == 13) &&
      (imHassy.CoatingCode == 4)) {
    c = LIBRAW_HF_AdobeDNG;
  } else if ((imHassy.format == LIBRAW_HF_HasselbladDNG) ||
             (imHassy.format == LIBRAW_HF_AdobeDNG_fromPhocusDNG)) {
    c = LIBRAW_HF_FFF;
  } else if (imHassy.format == LIBRAW_HF_Imacon) {
    c = LIBRAW_HF_3FR;
  } else {
    c = imHassy.format;
  }
  ps = HassyRawFormat_idx2HR(c);
  if ((c == LIBRAW_HF_3FR) ||
      (c == LIBRAW_HF_FFF))
    strcat(normalized_model, ps);

  if (((imHassy.CaptureSequenceInitiator[0] == 'H') &&
       (imHassy.CaptureSequenceInitiator[1] != 'a')) ||
      ((imHassy.CaptureSequenceInitiator[0] == 'A') &&
       isdigit(imHassy.CaptureSequenceInitiator[1]))) {
    ilm.CameraFormat = LIBRAW_FORMAT_645;
    ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_H;
    if (imgdata.lens.Lens[0] == 'H')
      process_Hassy_Lens(LIBRAW_MOUNT_Hasselblad_H);
  } else if (((imHassy.CaptureSequenceInitiator[0] == 'X') &&
              isdigit(imHassy.CaptureSequenceInitiator[1])) ||
             !strncmp(imHassy.HostBody, "907", 3)) {
    ilm.CameraFormat = LIBRAW_FORMAT_CROP645;
    ilm.CameraMount = LIBRAW_MOUNT_Hasselblad_XCD;
    if (imgdata.lens.Lens[0] == 'H') {
      process_Hassy_Lens(LIBRAW_MOUNT_Hasselblad_H);
      strcpy(ilm.Adapter, "XH");
    } else {
      if (imgdata.lens.Lens[0] == 'X') {
        process_Hassy_Lens(LIBRAW_MOUNT_Hasselblad_XCD);
      } else if (!imgdata.lens.Lens[0] &&
                 (aperture > 1.0f)   &&
                 (focal_len > 10.0f)) {
        ilm.LensID = focal_len;
        if (ilm.LensID == 35) {
          ilm.FocalType = LIBRAW_FT_ZOOM_LENS;
          ilm.LensID = LIBRAW_MOUNT_Hasselblad_XCD*100000000ULL +
                       35*10000ULL + 75*10;
        }
        else {
          ilm.FocalType = LIBRAW_FT_PRIME_LENS;
          ilm.LensID = LIBRAW_MOUNT_Hasselblad_XCD*100000000ULL +
                       ilm.LensID*10000ULL + ilm.LensID*10;
        }
      }
    }
  }
  if (normalized_model[0]  && !CM_found)
    CM_found = adobe_coeff(maker_index, normalized_model);
}
