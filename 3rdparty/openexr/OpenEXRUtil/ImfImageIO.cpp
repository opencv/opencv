//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//----------------------------------------------------------------------------
//
//      OpenEXR file I/O for deep images.
//
//----------------------------------------------------------------------------

#include "ImfImageIO.h"
#include "ImfDeepImageIO.h"
#include "ImfFlatImageIO.h"
#include <Iex.h>
#include <ImfHeader.h>
#include <ImfMultiPartInputFile.h>
#include <ImfPartType.h>
#include <ImfTestFile.h>

using namespace IMATH_NAMESPACE;
using namespace IEX_NAMESPACE;
using namespace std;

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

void
saveImage (
    const string&    fileName,
    const Header&    hdr,
    const Image&     img,
    DataWindowSource dws)
{
    if (const FlatImage* fimg = dynamic_cast<const FlatImage*> (&img))
    {
        if (fimg->levelMode () != ONE_LEVEL || hdr.hasTileDescription ())
            saveFlatTiledImage (fileName, hdr, *fimg, dws);
        else
            saveFlatScanLineImage (fileName, hdr, *fimg, dws);
    }

    if (const DeepImage* dimg = dynamic_cast<const DeepImage*> (&img))
    {
        if (dimg->levelMode () != ONE_LEVEL || hdr.hasTileDescription ())
            saveDeepTiledImage (fileName, hdr, *dimg, dws);
        else
            saveDeepScanLineImage (fileName, hdr, *dimg, dws);
    }
}

void
saveImage (const string& fileName, const Image& img)
{
    Header hdr;
    hdr.displayWindow () = img.dataWindow ();
    saveImage (fileName, hdr, img);
}

Image*
loadImage (const string& fileName, Header& hdr)
{
    bool tiled, deep, multiPart;

    if (!isOpenExrFile (fileName.c_str (), tiled, deep, multiPart))
    {
        THROW (
            ArgExc,
            "Cannot load image file " << fileName
                                      << ".  "
                                         "The file is not an OpenEXR file.");
    }

    if (multiPart)
    {
        THROW (
            ArgExc,
            "Cannot load image file "
                << fileName
                << ".  "
                   "Multi-part file loading is not supported.");
    }

    //XXX TODO: the tiled flag obtained above is unreliable;
    // open the file as a multi-part file and inspect the header.
    // Can the OpenEXR library be fixed?

    {
        MultiPartInputFile mpi (fileName.c_str ());

        tiled =
            (mpi.parts () > 0 && mpi.header (0).hasType () &&
             isTiled (mpi.header (0).type ()));
    }

    Image* img = 0;

    try
    {
        if (deep)
        {
            DeepImage* dimg = new DeepImage;
            img             = dimg;

            if (tiled)
                loadDeepTiledImage (fileName, hdr, *dimg);
            else
                loadDeepScanLineImage (fileName, hdr, *dimg);
        }
        else
        {
            FlatImage* fimg = new FlatImage;
            img             = fimg;

            if (tiled)
                loadFlatTiledImage (fileName, hdr, *fimg);
            else
                loadFlatScanLineImage (fileName, hdr, *fimg);
        }
    }
    catch (...)
    {
        delete img;
        throw;
    }

    return img;
}

Image*
loadImage (const string& fileName)
{
    Header hdr;
    return loadImage (fileName, hdr);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
