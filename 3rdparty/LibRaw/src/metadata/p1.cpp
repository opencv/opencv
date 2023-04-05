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

void LibRaw::setPhaseOneFeatures(unsigned long long id)
{

  ushort i;
  static const struct
  {
    unsigned long long id;
    char t_model[32];
    int CamMnt;
    int CamFmt;
  } p1_unique[] = {
      // Phase One section:
      {0x001ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x00aULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x00cULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x010ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x011ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x012ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x013ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x014ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x015ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x016ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x017ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x018ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x019ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x020ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x022ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x023ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x024ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x025ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x026ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x027ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x028ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x029ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x02aULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x02cULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x02dULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x02eULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x02fULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x030ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x031ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x032ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x033ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x034ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x035ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x036ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x037ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x043ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x044ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x045ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x046ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x047ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x048ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x049ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x04aULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x04cULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x04dULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x04eULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x04fULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x050ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x051ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x052ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x053ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x054ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x055ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x056ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x057ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x063ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x064ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x065ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x066ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x067ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x068ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x069ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x06aULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x070ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x071ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x072ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x073ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x083ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x084ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x085ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x086ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x087ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x088ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x089ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x08aULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x08cULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x08dULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x08eULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x08fULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x094ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x095ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x096ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x097ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x0a0ULL, "A-250", LIBRAW_MOUNT_Alpa, LIBRAW_FORMAT_69},
      {0x0a1ULL, "A-260", LIBRAW_MOUNT_Alpa, LIBRAW_FORMAT_69},
      {0x0a2ULL, "A-280", LIBRAW_MOUNT_Alpa, LIBRAW_FORMAT_69},
      {0x0a7ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x0a8ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x0a9ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x0aaULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x0acULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x0adULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x0aeULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x0afULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x0b0ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x0b1ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x0b2ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x0b3ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x0b4ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x0b5ULL, "Hasselblad H", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x0b6ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x0b7ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x0d0ULL, "Hasselblad V", LIBRAW_MOUNT_Hasselblad_V, LIBRAW_FORMAT_66},
      {0x0d3ULL, "PhaseOne/Mamiya", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x1c0ULL, "Phase One 645AF", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x1c9ULL, "Phase One 645DF", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x1d7ULL, "Phase One 645DF+", LIBRAW_MOUNT_Mamiya645, LIBRAW_FORMAT_645},
      {0x2c0ULL, "Phase One iXA", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x2c1ULL, "Phase One iXA - R", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x2c2ULL, "Phase One iXU 150", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x2c3ULL, "Phase One iXU 150 - NIR", LIBRAW_MOUNT_Unknown,LIBRAW_FORMAT_Unknown},
      {0x2c4ULL, "Phase One iXU 180", LIBRAW_MOUNT_Unknown,LIBRAW_FORMAT_Unknown},
      {0x2d1ULL, "Phase One iXR", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      // Leaf section:
      {0x140ULL, "Universal", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x141ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x142ULL, "Hasselblad H1/H2", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x143ULL, "Mamiya", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x144ULL, "Universal", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x145ULL, "Hasselblad H1/H2", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x146ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x147ULL, "Mamiya", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x149ULL, "Universal", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x14aULL, "Hasselblad H1/H2", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x14cULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x14dULL, "Mamiya", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x14eULL, "AFi", LIBRAW_MOUNT_Rollei_bayonet, LIBRAW_FORMAT_66},
      {0x14fULL, "AFi", LIBRAW_MOUNT_Rollei_bayonet, LIBRAW_FORMAT_66},
      {0x150ULL, "AFi", LIBRAW_MOUNT_Rollei_bayonet, LIBRAW_FORMAT_66},
      {0x151ULL, "Universal", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x152ULL, "Hasselblad H1/H2", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x153ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x154ULL, "Mamiya", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},

      {0x16cULL, "Phase One iXM-RS150F", LIBRAW_MOUNT_PhaseOne_iXM_RS, LIBRAW_FORMAT_645},

      {0x171ULL, "Universal", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x172ULL, "Mamiya", LIBRAW_MOUNT_Unknown, LIBRAW_FORMAT_Unknown},
      {0x173ULL, "Hasselblad H1/H2", LIBRAW_MOUNT_Hasselblad_H, LIBRAW_FORMAT_645},
      {0x174ULL, "Contax 645", LIBRAW_MOUNT_Contax645, LIBRAW_FORMAT_645},
      {0x175ULL, "AFi", LIBRAW_MOUNT_Rollei_bayonet, LIBRAW_FORMAT_66},
  };
  ilm.CamID = id;
  if (id && !ilm.body[0])
  {
    for (i = 0; i < sizeof p1_unique / sizeof *p1_unique; i++)
      if (id == p1_unique[i].id)
      {
        strcpy(ilm.body, p1_unique[i].t_model);
        ilm.CameraFormat = p1_unique[i].CamFmt;
        ilm.CameraMount  = p1_unique[i].CamMnt;
        if ((ilm.CameraMount == LIBRAW_MOUNT_PhaseOne_iXM_RS) ||
            (ilm.CameraMount == LIBRAW_MOUNT_PhaseOne_iXM)) {
          ilm.FocalType = LIBRAW_FT_PRIME_LENS;
          ilm.LensMount = ilm.CameraMount;
        } else if (ilm.CameraMount == LIBRAW_MOUNT_PhaseOne_iXM_MV) {
          ilm.LensMount = ilm.CameraMount;
        }
        break;
      }
  }
  return;
}
