
#include "tiffiop.h"

#if defined(JPEG_DUAL_MODE_8_12)

#  define TIFFInitJPEG TIFFInitJPEG_12

#  include LIBJPEG_12_PATH

#  include "tif_jpeg.c"

int TIFFReInitJPEG_12( TIFF *tif, int scheme, int is_encode )

{
    JPEGState* sp;

    assert(scheme == COMPRESSION_JPEG);

    sp = JState(tif);
    sp->tif = tif;				/* back link */

    /*
     * Override parent get/set field methods.
     */
    tif->tif_tagmethods.vgetfield = JPEGVGetField; /* hook for codec tags */
    tif->tif_tagmethods.vsetfield = JPEGVSetField; /* hook for codec tags */
    tif->tif_tagmethods.printdir = JPEGPrintDir;   /* hook for codec tags */

    /*
     * Install codec methods.
     */
    tif->tif_fixuptags = JPEGFixupTags;
    tif->tif_setupdecode = JPEGSetupDecode;
    tif->tif_predecode = JPEGPreDecode;
    tif->tif_decoderow = JPEGDecode;
    tif->tif_decodestrip = JPEGDecode;
    tif->tif_decodetile = JPEGDecode;
    tif->tif_setupencode = JPEGSetupEncode;
    tif->tif_preencode = JPEGPreEncode;
    tif->tif_postencode = JPEGPostEncode;
    tif->tif_encoderow = JPEGEncode;
    tif->tif_encodestrip = JPEGEncode;
    tif->tif_encodetile = JPEGEncode;  
    tif->tif_cleanup = JPEGCleanup;
    tif->tif_defstripsize = JPEGDefaultStripSize;
    tif->tif_deftilesize = JPEGDefaultTileSize;
    tif->tif_flags |= TIFF_NOBITREV;	/* no bit reversal, please */

    sp->cinfo_initialized = FALSE;

    if( is_encode )
        return JPEGSetupEncode(tif);
    else
        return JPEGSetupDecode(tif);
}

#endif /* defined(JPEG_DUAL_MODE_8_12) */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
