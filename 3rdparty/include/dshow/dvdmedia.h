/*
 * Copyright (C) 2008 Maarten Lankhorst
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#ifndef __DVDMEDIA_H__
#define __DVDMEDIA_H__

#define AMCONTROL_USED 0x00000001
#define AMCONTROL_PAD_TO_4x3 0x00000002
#define AMCONTROL_PAD_TO_16x9 0x00000004

enum AM_MPEG2Level {
    AM_MPEG2Level_Low = 1,
    AM_MPEG2Level_Main,
    AM_MPEG2Level_High1440,
    AM_MPEG2Level_High
};
enum AM_MPEG2Profile {
    AM_MPEG2Profile_Simple = 1,
    AM_MPEG2Profile_Main,
    AM_MPEG2Profile_SNRScalable,
    AM_MPEG2Profile_SpatiallyScalable,
    AM_MPEG2Profile_High
};
typedef enum {
    AM_RATE_ChangeRate = 1,
    AM_RATE_FullDataRateMax = 2,
    AM_RATE_ReverseDecode = 3,
    AM_RATE_DecoderPosition = 4,
    AM_RATE_DecoderVersion = 5
} AM_PROPERTY_DVD_RATE_CHANGE;

typedef struct tagVIDEOINFOHEADER2 {
    RECT rcSource;
    RECT rcTarget;
    DWORD dwBitRate;
    DWORD dwBitErrorRate;
    REFERENCE_TIME AvgTimePerFrame;
    DWORD dwInterlaceFlags;
    DWORD dwCopyProtectFlags;
    DWORD dwPictAspectRatioX;
    DWORD dwPictAspectRatioY;
    union {
        DWORD dwControlFlags;
        DWORD dwReserved1;
    } DUMMYUNIONNAME;
    DWORD dwReserved2;
    BITMAPINFOHEADER bmiHeader;
} VIDEOINFOHEADER2;

typedef struct tagMPEG2VIDEOINFO {
    VIDEOINFOHEADER2 hdr;
    DWORD dwStartTimeCode;
    DWORD cbSequenceHeader;
    DWORD dwProfile;
    DWORD dwLevel;
    DWORD dwFlags;
    DWORD dwSequenceHeader[1];
} MPEG2VIDEOINFO;

#endif /* __DVDMEDIA_H__ */
