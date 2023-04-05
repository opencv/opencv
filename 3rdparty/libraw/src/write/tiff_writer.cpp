/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw uses code from dcraw.c -- Dave Coffin's raw photo decoder,
 dcraw.c is copyright 1997-2018 by Dave Coffin, dcoffin a cybercom o net.
 LibRaw do not use RESTRICTED code from dcraw.c

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/libraw_cxx_defs.h"

int LibRaw::dcraw_ppm_tiff_writer(const char *filename)
{
  CHECK_ORDER_LOW(LIBRAW_PROGRESS_LOAD_RAW);

  if (!imgdata.image)
    return LIBRAW_OUT_OF_ORDER_CALL;

  if (!filename)
    return ENOENT;
  FILE *f = NULL;
  if (!strcmp(filename, "-"))
  {
#ifdef LIBRAW_WIN32_CALLS
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    f = stdout;
  }
  else
    f = fopen(filename, "wb");

  if (!f)
    return errno;

  try
  {
    if (!libraw_internal_data.output_data.histogram)
    {
      libraw_internal_data.output_data.histogram =
          (int(*)[LIBRAW_HISTOGRAM_SIZE])malloc(
              sizeof(*libraw_internal_data.output_data.histogram) * 4);
    }
    libraw_internal_data.internal_data.output = f;
    write_ppm_tiff();
    SET_PROC_FLAG(LIBRAW_PROGRESS_FLIP);
    libraw_internal_data.internal_data.output = NULL;
    if (strcmp(filename, "-"))
      fclose(f);
    return 0;
  }
  catch (const LibRaw_exceptions& err)
  {
    if (strcmp(filename, "-"))
      fclose(f);
    EXCEPTION_HANDLER(err);
  }
  catch (const std::bad_alloc&)
  {
      if (strcmp(filename, "-"))
          fclose(f);
      EXCEPTION_HANDLER(LIBRAW_EXCEPTION_ALLOC);
  }

}
