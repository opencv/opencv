// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <stdint.h>

#ifndef RT_IMAGE
#define RT_IMAGE
#ifdef HAVE_RTX64_GIGE

#define NODE_ALIGNMENT_SIZE SYSTEM_CACHE_ALIGNMENT_SIZE
#define ROUNDUP(size, unit) (unit * ((size + unit - 1) / unit))

/************************************** RTX_Image ***************************************/
/** Image format for use with RTX64 */
typedef struct _RTX_Image
{
    uint32_t        bitsPerPixel;               //  Pixel depth in bits:
    uint32_t        pixelFormat;                //  format from camera
    uint32_t        numberOfChannels;           //  i.e. RGB is typically 3 channels
    uint32_t        CvArrayType;                //  Array Type (for example, CV_8UC1, which means 8-bit, Unsigned Char, 1-channel)
    uint32_t        offset_X;                   //  Offset in pixels from image origin. Used for ROI support.
    uint32_t        offset_Y;                   //  Offset in lines from image origin. Used for ROI support.
    uint32_t        Width;                      //  Width of the image
    uint32_t        Height;                     //  Height of the image
    uint32_t        padding_X;                  //  Horizontal padding expressed in bytes. Number of extra bytes transmitted at
                                                //  the end of each line to facilitate image alignment in buffers.
    uint32_t        padding_Y;                  //  Vertical padding expressed in bytes. Number of extra bytes transmitted at the
                                                //  end of the image to facilitate image alignment in buffers.
    uint64_t        maxImage_Size;              //  The allocated image buffer should be maxImage_Size bytes
    uint64_t        Image_Size;                 //  Size of the Image in bytes
    uint32_t        gigE_BlockID;               //
    LARGE_INTEGER   cameraTimeStamp;            //  TimeStamp of the image
    LARGE_INTEGER   leaderFrameRxTimeQPC;       //
    LARGE_INTEGER   trailerFrameRxTimeQPC;      //
    uint64_t        offsetTo_Image_Data;        //  RTX_Image and Image_Data can be allocated in the same block of memmory where Image_Data will be after RTX_Image
                                                //  in order to share between different virtual address spaces *Image_Data can be .. Image_Data = pRTX_Image + pRTX_Image->offsetTo_Image_Data;
    void            *pCustomMetaData;           //  Pointer to the custom meta data
    char            *Image_Data;                //  Pointer to the image data

}RTX_Image;
#endif
#endif