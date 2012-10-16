/*
 * Copyright (C) the Wine project
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

#ifndef __DDRAW_INCLUDED__
#define __DDRAW_INCLUDED__

#define COM_NO_WINDOWS_H
#include <objbase.h>
#include <dshow/_mingw_dxhelper.h>

#ifdef __cplusplus
extern "C" {
#endif /* defined(__cplusplus) */

#ifndef	DIRECTDRAW_VERSION
#define	DIRECTDRAW_VERSION	0x0700
#endif /* DIRECTDRAW_VERSION */

/*****************************************************************************
 * Predeclare the interfaces
 */
#ifndef __DDRAW_GUID_DEFINED__
DEFINE_GUID( CLSID_DirectDraw,		0xD7B70EE0,0x4340,0x11CF,0xB0,0x63,0x00,0x20,0xAF,0xC2,0xCD,0x35 );
DEFINE_GUID( CLSID_DirectDraw7,         0x3C305196,0x50DB,0x11D3,0x9C,0xFE,0x00,0xC0,0x4F,0xD9,0x30,0xC5 );
DEFINE_GUID( CLSID_DirectDrawClipper,	0x593817A0,0x7DB3,0x11CF,0xA2,0xDE,0x00,0xAA,0x00,0xb9,0x33,0x56 );
DEFINE_GUID( IID_IDirectDraw,		0x6C14DB80,0xA733,0x11CE,0xA5,0x21,0x00,0x20,0xAF,0x0B,0xE5,0x60 );
DEFINE_GUID( IID_IDirectDraw2,		0xB3A6F3E0,0x2B43,0x11CF,0xA2,0xDE,0x00,0xAA,0x00,0xB9,0x33,0x56 );
DEFINE_GUID( IID_IDirectDraw3,		0x618f8ad4,0x8b7a,0x11d0,0x8f,0xcc,0x0,0xc0,0x4f,0xd9,0x18,0x9d );
DEFINE_GUID( IID_IDirectDraw4,          0x9c59509a,0x39bd,0x11d1,0x8c,0x4a,0x00,0xc0,0x4f,0xd9,0x30,0xc5 );
DEFINE_GUID( IID_IDirectDraw7,          0x15e65ec0,0x3b9c,0x11d2,0xb9,0x2f,0x00,0x60,0x97,0x97,0xea,0x5b );
DEFINE_GUID( IID_IDirectDrawSurface,	0x6C14DB81,0xA733,0x11CE,0xA5,0x21,0x00,0x20,0xAF,0x0B,0xE5,0x60 );
DEFINE_GUID( IID_IDirectDrawSurface2,	0x57805885,0x6eec,0x11cf,0x94,0x41,0xa8,0x23,0x03,0xc1,0x0e,0x27 );
DEFINE_GUID( IID_IDirectDrawSurface3,	0xDA044E00,0x69B2,0x11D0,0xA1,0xD5,0x00,0xAA,0x00,0xB8,0xDF,0xBB );
DEFINE_GUID( IID_IDirectDrawSurface4,   0x0B2B8630,0xAD35,0x11D0,0x8E,0xA6,0x00,0x60,0x97,0x97,0xEA,0x5B );
DEFINE_GUID( IID_IDirectDrawSurface7,   0x06675a80,0x3b9b,0x11d2,0xb9,0x2f,0x00,0x60,0x97,0x97,0xea,0x5b );
DEFINE_GUID( IID_IDirectDrawPalette,	0x6C14DB84,0xA733,0x11CE,0xA5,0x21,0x00,0x20,0xAF,0x0B,0xE5,0x60 );
DEFINE_GUID( IID_IDirectDrawClipper,	0x6C14DB85,0xA733,0x11CE,0xA5,0x21,0x00,0x20,0xAF,0x0B,0xE5,0x60 );
DEFINE_GUID( IID_IDirectDrawColorControl,0x4B9F0EE0,0x0D7E,0x11D0,0x9B,0x06,0x00,0xA0,0xC9,0x03,0xA3,0xB8 );
DEFINE_GUID( IID_IDirectDrawGammaControl,0x69C11C3E,0xB46B,0x11D1,0xAD,0x7A,0x00,0xC0,0x4F,0xC2,0x9B,0x4E );
#endif

typedef struct IDirectDraw *LPDIRECTDRAW;
typedef struct IDirectDraw2 *LPDIRECTDRAW2;
typedef struct IDirectDraw3 *LPDIRECTDRAW3;
typedef struct IDirectDraw4 *LPDIRECTDRAW4;
typedef struct IDirectDraw7 *LPDIRECTDRAW7;
typedef struct IDirectDrawClipper *LPDIRECTDRAWCLIPPER;
typedef struct IDirectDrawPalette *LPDIRECTDRAWPALETTE;
typedef struct IDirectDrawSurface *LPDIRECTDRAWSURFACE;
typedef struct IDirectDrawSurface2 *LPDIRECTDRAWSURFACE2;
typedef struct IDirectDrawSurface3 *LPDIRECTDRAWSURFACE3;
typedef struct IDirectDrawSurface4 *LPDIRECTDRAWSURFACE4;
typedef struct IDirectDrawSurface7 *LPDIRECTDRAWSURFACE7;
typedef struct IDirectDrawColorControl *LPDIRECTDRAWCOLORCONTROL;
typedef struct IDirectDrawGammaControl *LPDIRECTDRAWGAMMACONTROL;


#define DDENUMRET_CANCEL	0
#define DDENUMRET_OK		1

#define DD_OK			S_OK
#define DD_FALSE		S_FALSE


#define _FACDD		0x876
#define MAKE_DDHRESULT( code )  MAKE_HRESULT( 1, _FACDD, code )

#define DDERR_ALREADYINITIALIZED		MAKE_DDHRESULT( 5 )
#define DDERR_CANNOTATTACHSURFACE		MAKE_DDHRESULT( 10 )
#define DDERR_CANNOTDETACHSURFACE		MAKE_DDHRESULT( 20 )
#define DDERR_CURRENTLYNOTAVAIL			MAKE_DDHRESULT( 40 )
#define DDERR_EXCEPTION				MAKE_DDHRESULT( 55 )
#define DDERR_GENERIC				E_FAIL
#define DDERR_HEIGHTALIGN			MAKE_DDHRESULT( 90 )
#define DDERR_INCOMPATIBLEPRIMARY		MAKE_DDHRESULT( 95 )
#define DDERR_INVALIDCAPS			MAKE_DDHRESULT( 100 )
#define DDERR_INVALIDCLIPLIST			MAKE_DDHRESULT( 110 )
#define DDERR_INVALIDMODE			MAKE_DDHRESULT( 120 )
#define DDERR_INVALIDOBJECT			MAKE_DDHRESULT( 130 )
#define DDERR_INVALIDPARAMS			E_INVALIDARG
#define DDERR_INVALIDPIXELFORMAT		MAKE_DDHRESULT( 145 )
#define DDERR_INVALIDRECT			MAKE_DDHRESULT( 150 )
#define DDERR_LOCKEDSURFACES			MAKE_DDHRESULT( 160 )
#define DDERR_NO3D				MAKE_DDHRESULT( 170 )
#define DDERR_NOALPHAHW				MAKE_DDHRESULT( 180 )
#define DDERR_NOSTEREOHARDWARE          	MAKE_DDHRESULT( 181 )
#define DDERR_NOSURFACELEFT                     MAKE_DDHRESULT( 182 )
#define DDERR_NOCLIPLIST			MAKE_DDHRESULT( 205 )
#define DDERR_NOCOLORCONVHW			MAKE_DDHRESULT( 210 )
#define DDERR_NOCOOPERATIVELEVELSET		MAKE_DDHRESULT( 212 )
#define DDERR_NOCOLORKEY			MAKE_DDHRESULT( 215 )
#define DDERR_NOCOLORKEYHW			MAKE_DDHRESULT( 220 )
#define DDERR_NODIRECTDRAWSUPPORT		MAKE_DDHRESULT( 222 )
#define DDERR_NOEXCLUSIVEMODE			MAKE_DDHRESULT( 225 )
#define DDERR_NOFLIPHW				MAKE_DDHRESULT( 230 )
#define DDERR_NOGDI				MAKE_DDHRESULT( 240 )
#define DDERR_NOMIRRORHW			MAKE_DDHRESULT( 250 )
#define DDERR_NOTFOUND				MAKE_DDHRESULT( 255 )
#define DDERR_NOOVERLAYHW			MAKE_DDHRESULT( 260 )
#define DDERR_OVERLAPPINGRECTS                  MAKE_DDHRESULT( 270 )
#define DDERR_NORASTEROPHW			MAKE_DDHRESULT( 280 )
#define DDERR_NOROTATIONHW			MAKE_DDHRESULT( 290 )
#define DDERR_NOSTRETCHHW			MAKE_DDHRESULT( 310 )
#define DDERR_NOT4BITCOLOR			MAKE_DDHRESULT( 316 )
#define DDERR_NOT4BITCOLORINDEX			MAKE_DDHRESULT( 317 )
#define DDERR_NOT8BITCOLOR			MAKE_DDHRESULT( 320 )
#define DDERR_NOTEXTUREHW			MAKE_DDHRESULT( 330 )
#define DDERR_NOVSYNCHW				MAKE_DDHRESULT( 335 )
#define DDERR_NOZBUFFERHW			MAKE_DDHRESULT( 340 )
#define DDERR_NOZOVERLAYHW			MAKE_DDHRESULT( 350 )
#define DDERR_OUTOFCAPS				MAKE_DDHRESULT( 360 )
#define DDERR_OUTOFMEMORY			E_OUTOFMEMORY
#define DDERR_OUTOFVIDEOMEMORY			MAKE_DDHRESULT( 380 )
#define DDERR_OVERLAYCANTCLIP			MAKE_DDHRESULT( 382 )
#define DDERR_OVERLAYCOLORKEYONLYONEACTIVE	MAKE_DDHRESULT( 384 )
#define DDERR_PALETTEBUSY			MAKE_DDHRESULT( 387 )
#define DDERR_COLORKEYNOTSET			MAKE_DDHRESULT( 400 )
#define DDERR_SURFACEALREADYATTACHED		MAKE_DDHRESULT( 410 )
#define DDERR_SURFACEALREADYDEPENDENT		MAKE_DDHRESULT( 420 )
#define DDERR_SURFACEBUSY			MAKE_DDHRESULT( 430 )
#define DDERR_CANTLOCKSURFACE			MAKE_DDHRESULT( 435 )
#define DDERR_SURFACEISOBSCURED			MAKE_DDHRESULT( 440 )
#define DDERR_SURFACELOST			MAKE_DDHRESULT( 450 )
#define DDERR_SURFACENOTATTACHED		MAKE_DDHRESULT( 460 )
#define DDERR_TOOBIGHEIGHT			MAKE_DDHRESULT( 470 )
#define DDERR_TOOBIGSIZE			MAKE_DDHRESULT( 480 )
#define DDERR_TOOBIGWIDTH			MAKE_DDHRESULT( 490 )
#define DDERR_UNSUPPORTED			E_NOTIMPL
#define DDERR_UNSUPPORTEDFORMAT			MAKE_DDHRESULT( 510 )
#define DDERR_UNSUPPORTEDMASK			MAKE_DDHRESULT( 520 )
#define DDERR_INVALIDSTREAM                     MAKE_DDHRESULT( 521 )
#define DDERR_VERTICALBLANKINPROGRESS		MAKE_DDHRESULT( 537 )
#define DDERR_WASSTILLDRAWING			MAKE_DDHRESULT( 540 )
#define DDERR_DDSCAPSCOMPLEXREQUIRED            MAKE_DDHRESULT( 542 )
#define DDERR_XALIGN				MAKE_DDHRESULT( 560 )
#define DDERR_INVALIDDIRECTDRAWGUID		MAKE_DDHRESULT( 561 )
#define DDERR_DIRECTDRAWALREADYCREATED		MAKE_DDHRESULT( 562 )
#define DDERR_NODIRECTDRAWHW			MAKE_DDHRESULT( 563 )
#define DDERR_PRIMARYSURFACEALREADYEXISTS	MAKE_DDHRESULT( 564 )
#define DDERR_NOEMULATION			MAKE_DDHRESULT( 565 )
#define DDERR_REGIONTOOSMALL			MAKE_DDHRESULT( 566 )
#define DDERR_CLIPPERISUSINGHWND		MAKE_DDHRESULT( 567 )
#define DDERR_NOCLIPPERATTACHED			MAKE_DDHRESULT( 568 )
#define DDERR_NOHWND				MAKE_DDHRESULT( 569 )
#define DDERR_HWNDSUBCLASSED			MAKE_DDHRESULT( 570 )
#define DDERR_HWNDALREADYSET			MAKE_DDHRESULT( 571 )
#define DDERR_NOPALETTEATTACHED			MAKE_DDHRESULT( 572 )
#define DDERR_NOPALETTEHW			MAKE_DDHRESULT( 573 )
#define DDERR_BLTFASTCANTCLIP			MAKE_DDHRESULT( 574 )
#define DDERR_NOBLTHW				MAKE_DDHRESULT( 575 )
#define DDERR_NODDROPSHW			MAKE_DDHRESULT( 576 )
#define DDERR_OVERLAYNOTVISIBLE			MAKE_DDHRESULT( 577 )
#define DDERR_NOOVERLAYDEST			MAKE_DDHRESULT( 578 )
#define DDERR_INVALIDPOSITION			MAKE_DDHRESULT( 579 )
#define DDERR_NOTAOVERLAYSURFACE		MAKE_DDHRESULT( 580 )
#define DDERR_EXCLUSIVEMODEALREADYSET		MAKE_DDHRESULT( 581 )
#define DDERR_NOTFLIPPABLE			MAKE_DDHRESULT( 582 )
#define DDERR_CANTDUPLICATE			MAKE_DDHRESULT( 583 )
#define DDERR_NOTLOCKED				MAKE_DDHRESULT( 584 )
#define DDERR_CANTCREATEDC			MAKE_DDHRESULT( 585 )
#define DDERR_NODC				MAKE_DDHRESULT( 586 )
#define DDERR_WRONGMODE				MAKE_DDHRESULT( 587 )
#define DDERR_IMPLICITLYCREATED			MAKE_DDHRESULT( 588 )
#define DDERR_NOTPALETTIZED			MAKE_DDHRESULT( 589 )
#define DDERR_UNSUPPORTEDMODE			MAKE_DDHRESULT( 590 )
#define DDERR_NOMIPMAPHW			MAKE_DDHRESULT( 591 )
#define DDERR_INVALIDSURFACETYPE		MAKE_DDHRESULT( 592 )
#define DDERR_NOOPTIMIZEHW			MAKE_DDHRESULT( 600 )
#define DDERR_NOTLOADED				MAKE_DDHRESULT( 601 )
#define DDERR_NOFOCUSWINDOW			MAKE_DDHRESULT( 602 )
#define DDERR_NOTONMIPMAPSUBLEVEL               MAKE_DDHRESULT( 603 )
#define DDERR_DCALREADYCREATED			MAKE_DDHRESULT( 620 )
#define DDERR_NONONLOCALVIDMEM			MAKE_DDHRESULT( 630 )
#define DDERR_CANTPAGELOCK			MAKE_DDHRESULT( 640 )
#define DDERR_CANTPAGEUNLOCK			MAKE_DDHRESULT( 660 )
#define DDERR_NOTPAGELOCKED			MAKE_DDHRESULT( 680 )
#define DDERR_MOREDATA				MAKE_DDHRESULT( 690 )
#define DDERR_EXPIRED                           MAKE_DDHRESULT( 691 )
#define DDERR_TESTFINISHED                      MAKE_DDHRESULT( 692 )
#define DDERR_NEWMODE                           MAKE_DDHRESULT( 693 )
#define DDERR_D3DNOTINITIALIZED                 MAKE_DDHRESULT( 694 )
#define DDERR_VIDEONOTACTIVE			MAKE_DDHRESULT( 695 )
#define DDERR_NOMONITORINFORMATION              MAKE_DDHRESULT( 696 )
#define DDERR_NODRIVERSUPPORT                   MAKE_DDHRESULT( 697 )
#define DDERR_DEVICEDOESNTOWNSURFACE		MAKE_DDHRESULT( 699 )
#define DDERR_NOTINITIALIZED			CO_E_NOTINITIALIZED

/* dwFlags for Blt* */
#define DDBLT_ALPHADEST				0x00000001
#define DDBLT_ALPHADESTCONSTOVERRIDE		0x00000002
#define DDBLT_ALPHADESTNEG			0x00000004
#define DDBLT_ALPHADESTSURFACEOVERRIDE		0x00000008
#define DDBLT_ALPHAEDGEBLEND			0x00000010
#define DDBLT_ALPHASRC				0x00000020
#define DDBLT_ALPHASRCCONSTOVERRIDE		0x00000040
#define DDBLT_ALPHASRCNEG			0x00000080
#define DDBLT_ALPHASRCSURFACEOVERRIDE		0x00000100
#define DDBLT_ASYNC				0x00000200
#define DDBLT_COLORFILL				0x00000400
#define DDBLT_DDFX				0x00000800
#define DDBLT_DDROPS				0x00001000
#define DDBLT_KEYDEST				0x00002000
#define DDBLT_KEYDESTOVERRIDE			0x00004000
#define DDBLT_KEYSRC				0x00008000
#define DDBLT_KEYSRCOVERRIDE			0x00010000
#define DDBLT_ROP				0x00020000
#define DDBLT_ROTATIONANGLE			0x00040000
#define DDBLT_ZBUFFER				0x00080000
#define DDBLT_ZBUFFERDESTCONSTOVERRIDE		0x00100000
#define DDBLT_ZBUFFERDESTOVERRIDE		0x00200000
#define DDBLT_ZBUFFERSRCCONSTOVERRIDE		0x00400000
#define DDBLT_ZBUFFERSRCOVERRIDE		0x00800000
#define DDBLT_WAIT				0x01000000
#define DDBLT_DEPTHFILL				0x02000000
#define DDBLT_DONOTWAIT                         0x08000000

/* dwTrans for BltFast */
#define DDBLTFAST_NOCOLORKEY			0x00000000
#define DDBLTFAST_SRCCOLORKEY			0x00000001
#define DDBLTFAST_DESTCOLORKEY			0x00000002
#define DDBLTFAST_WAIT				0x00000010
#define DDBLTFAST_DONOTWAIT                     0x00000020

/* dwFlags for Flip */
#define DDFLIP_WAIT		0x00000001
#define DDFLIP_EVEN		0x00000002 /* only valid for overlay */
#define DDFLIP_ODD		0x00000004 /* only valid for overlay */
#define DDFLIP_NOVSYNC		0x00000008
#define DDFLIP_STEREO		0x00000010
#define DDFLIP_DONOTWAIT	0x00000020
#define DDFLIP_INTERVAL2	0x02000000
#define DDFLIP_INTERVAL3	0x03000000
#define DDFLIP_INTERVAL4	0x04000000


/* dwFlags for GetBltStatus */
#define DDGBS_CANBLT				0x00000001
#define DDGBS_ISBLTDONE				0x00000002

/* dwFlags for IDirectDrawSurface7::GetFlipStatus */
#define DDGFS_CANFLIP		1L
#define DDGFS_ISFLIPDONE	2L

/* dwFlags for IDirectDrawSurface7::SetPrivateData */
#define DDSPD_IUNKNOWNPOINTER	1L
#define DDSPD_VOLATILE		2L

/* DDSCAPS.dwCaps */
/* reserved1, was 3d capable */
#define DDSCAPS_RESERVED1		0x00000001
/* surface contains alpha information */
#define DDSCAPS_ALPHA			0x00000002
/* this surface is a backbuffer */
#define DDSCAPS_BACKBUFFER		0x00000004
/* complex surface structure */
#define DDSCAPS_COMPLEX			0x00000008
/* part of surface flipping structure */
#define DDSCAPS_FLIP			0x00000010
/* this surface is the frontbuffer surface */
#define DDSCAPS_FRONTBUFFER		0x00000020
/* this is a plain offscreen surface */
#define DDSCAPS_OFFSCREENPLAIN		0x00000040
/* overlay */
#define DDSCAPS_OVERLAY			0x00000080
/* palette objects can be created and attached to us */
#define DDSCAPS_PALETTE			0x00000100
/* primary surface (the one the user looks at currently)(right eye)*/
#define DDSCAPS_PRIMARYSURFACE		0x00000200
/* primary surface for left eye */
#define DDSCAPS_PRIMARYSURFACELEFT	0x00000400
/* surface exists in systemmemory */
#define DDSCAPS_SYSTEMMEMORY		0x00000800
/* surface can be used as a texture */
#define DDSCAPS_TEXTURE		        0x00001000
/* surface may be destination for 3d rendering */
#define DDSCAPS_3DDEVICE		0x00002000
/* surface exists in videomemory */
#define DDSCAPS_VIDEOMEMORY		0x00004000
/* surface changes immediately visible */
#define DDSCAPS_VISIBLE			0x00008000
/* write only surface */
#define DDSCAPS_WRITEONLY		0x00010000
/* zbuffer surface */
#define DDSCAPS_ZBUFFER			0x00020000
/* has its own DC */
#define DDSCAPS_OWNDC			0x00040000
/* surface should be able to receive live video */
#define DDSCAPS_LIVEVIDEO		0x00080000
/* should be able to have a hw codec decompress stuff into it */
#define DDSCAPS_HWCODEC			0x00100000
/* mode X (320x200 or 320x240) surface */
#define DDSCAPS_MODEX			0x00200000
/* one mipmap surface (1 level) */
#define DDSCAPS_MIPMAP			0x00400000
#define DDSCAPS_RESERVED2		0x00800000
/* memory allocation delayed until Load() */
#define DDSCAPS_ALLOCONLOAD		0x04000000
/* Indicates that the surface will receive data from a video port */
#define DDSCAPS_VIDEOPORT		0x08000000
/* surface is in local videomemory */
#define DDSCAPS_LOCALVIDMEM		0x10000000
/* surface is in nonlocal videomemory */
#define DDSCAPS_NONLOCALVIDMEM		0x20000000
/* surface is a standard VGA mode surface (NOT ModeX) */
#define DDSCAPS_STANDARDVGAMODE		0x40000000
/* optimized? surface */
#define DDSCAPS_OPTIMIZED		0x80000000

typedef struct _DDSCAPS {
    DWORD	dwCaps;	/* capabilities of surface wanted */
} DDSCAPS,*LPDDSCAPS;

/* DDSCAPS2.dwCaps2 */
/* indicates the surface will receive data from a video port using
   deinterlacing hardware. */
#define DDSCAPS2_HARDWAREDEINTERLACE	0x00000002
/* indicates the surface will be locked very frequently. */
#define DDSCAPS2_HINTDYNAMIC		0x00000004
/* indicates surface can be re-ordered or retiled on load() */
#define DDSCAPS2_HINTSTATIC             0x00000008
/* indicates surface to be managed by directdraw/direct3D */
#define DDSCAPS2_TEXTUREMANAGE          0x00000010
/* reserved bits */
#define DDSCAPS2_RESERVED1              0x00000020
#define DDSCAPS2_RESERVED2              0x00000040
/* indicates surface will never be locked again */
#define DDSCAPS2_OPAQUE                 0x00000080
/* set at CreateSurface() time to indicate antialiasing will be used */
#define DDSCAPS2_HINTANTIALIASING       0x00000100
/* set at CreateSurface() time to indicate cubic environment map */
#define DDSCAPS2_CUBEMAP                0x00000200
/* face flags for cube maps */
#define DDSCAPS2_CUBEMAP_POSITIVEX      0x00000400
#define DDSCAPS2_CUBEMAP_NEGATIVEX      0x00000800
#define DDSCAPS2_CUBEMAP_POSITIVEY      0x00001000
#define DDSCAPS2_CUBEMAP_NEGATIVEY      0x00002000
#define DDSCAPS2_CUBEMAP_POSITIVEZ      0x00004000
#define DDSCAPS2_CUBEMAP_NEGATIVEZ      0x00008000
/* specifies all faces of a cube for CreateSurface() */
#define DDSCAPS2_CUBEMAP_ALLFACES ( DDSCAPS2_CUBEMAP_POSITIVEX |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEX |\
                                    DDSCAPS2_CUBEMAP_POSITIVEY |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEY |\
                                    DDSCAPS2_CUBEMAP_POSITIVEZ |\
                                    DDSCAPS2_CUBEMAP_NEGATIVEZ )
/* set for mipmap sublevels on DirectX7 and later.  ignored by CreateSurface() */
#define DDSCAPS2_MIPMAPSUBLEVEL         0x00010000
/* indicates texture surface to be managed by Direct3D *only* */
#define DDSCAPS2_D3DTEXTUREMANAGE       0x00020000
/* indicates managed surface that can safely be lost */
#define DDSCAPS2_DONOTPERSIST           0x00040000
/* indicates surface is part of a stereo flipping chain */
#define DDSCAPS2_STEREOSURFACELEFT      0x00080000

typedef struct _DDSCAPS2 {
    DWORD	dwCaps;	/* capabilities of surface wanted */
    DWORD   dwCaps2; /* additional capabilities */
    DWORD   dwCaps3; /* reserved capabilities */
    __extension__ union {
      DWORD dwCaps4; /* low word is the depth for a volume texture */
      DWORD dwVolumeDepth;
    } DUMMYUNIONNAME1;
} DDSCAPS2,*LPDDSCAPS2;

typedef struct _DDSCAPSEX {
    DWORD	dwCaps2;
    DWORD	dwCaps3;
    __extension__ union {
    DWORD	dwCaps4;
    DWORD	dwVolumeDepth;
    } DUMMYUNIONNAME1;
} DDSCAPSEX,*LPDDSCAPSEX;

#define	DD_ROP_SPACE	(256/32)	/* space required to store ROP array */

typedef struct _DDCAPS_DX7		/* DirectX 7 version of caps struct */
{
    DWORD	dwSize;                 /* size of the DDDRIVERCAPS structure */
    DWORD	dwCaps;                 /* driver specific capabilities */
    DWORD	dwCaps2;                /* more driver specific capabilities */
    DWORD	dwCKeyCaps;             /* color key capabilities of the surface */
    DWORD	dwFXCaps;               /* driver specific stretching and effects capabilities */
    DWORD	dwFXAlphaCaps;          /* alpha driver specific capabilities */
    DWORD	dwPalCaps;              /* palette capabilities */
    DWORD	dwSVCaps;               /* stereo vision capabilities */
    DWORD	dwAlphaBltConstBitDepths;       /* DDBD_2,4,8 */
    DWORD	dwAlphaBltPixelBitDepths;       /* DDBD_1,2,4,8 */
    DWORD	dwAlphaBltSurfaceBitDepths;     /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlayConstBitDepths;   /* DDBD_2,4,8 */
    DWORD	dwAlphaOverlayPixelBitDepths;   /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlaySurfaceBitDepths; /* DDBD_1,2,4,8 */
    DWORD	dwZBufferBitDepths;             /* DDBD_8,16,24,32 */
    DWORD	dwVidMemTotal;          /* total amount of video memory */
    DWORD	dwVidMemFree;           /* amount of free video memory */
    DWORD	dwMaxVisibleOverlays;   /* maximum number of visible overlays */
    DWORD	dwCurrVisibleOverlays;  /* current number of visible overlays */
    DWORD	dwNumFourCCCodes;       /* number of four cc codes */
    DWORD	dwAlignBoundarySrc;     /* source rectangle alignment */
    DWORD	dwAlignSizeSrc;         /* source rectangle byte size */
    DWORD	dwAlignBoundaryDest;    /* dest rectangle alignment */
    DWORD	dwAlignSizeDest;        /* dest rectangle byte size */
    DWORD	dwAlignStrideAlign;     /* stride alignment */
    DWORD	dwRops[DD_ROP_SPACE];   /* ROPs supported */
    DDSCAPS	ddsOldCaps;             /* old DDSCAPS - superseded for DirectX6+ */
    DWORD	dwMinOverlayStretch;    /* minimum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxOverlayStretch;    /* maximum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinLiveVideoStretch;  /* minimum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxLiveVideoStretch;  /* maximum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinHwCodecStretch;    /* minimum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxHwCodecStretch;    /* maximum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwReserved1;
    DWORD	dwReserved2;
    DWORD	dwReserved3;
    DWORD	dwSVBCaps;              /* driver specific capabilities for System->Vmem blts */
    DWORD	dwSVBCKeyCaps;          /* driver color key capabilities for System->Vmem blts */
    DWORD	dwSVBFXCaps;            /* driver FX capabilities for System->Vmem blts */
    DWORD	dwSVBRops[DD_ROP_SPACE];/* ROPs supported for System->Vmem blts */
    DWORD	dwVSBCaps;              /* driver specific capabilities for Vmem->System blts */
    DWORD	dwVSBCKeyCaps;          /* driver color key capabilities for Vmem->System blts */
    DWORD	dwVSBFXCaps;            /* driver FX capabilities for Vmem->System blts */
    DWORD	dwVSBRops[DD_ROP_SPACE];/* ROPs supported for Vmem->System blts */
    DWORD	dwSSBCaps;              /* driver specific capabilities for System->System blts */
    DWORD	dwSSBCKeyCaps;          /* driver color key capabilities for System->System blts */
    DWORD	dwSSBFXCaps;            /* driver FX capabilities for System->System blts */
    DWORD	dwSSBRops[DD_ROP_SPACE];/* ROPs supported for System->System blts */
    DWORD       dwMaxVideoPorts;        /* maximum number of usable video ports */
    DWORD   	dwCurrVideoPorts;       /* current number of video ports used */
    DWORD   	dwSVBCaps2;             /* more driver specific capabilities for System->Vmem blts */
    DWORD   	dwNLVBCaps;             /* driver specific capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBCaps2;            /* more driver specific capabilities non-local->local vidmem blts */
    DWORD   	dwNLVBCKeyCaps;         /* driver color key capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBFXCaps;           /* driver FX capabilities for non-local->local blts */
    DWORD   	dwNLVBRops[DD_ROP_SPACE]; /* ROPs supported for non-local->local blts */
    DDSCAPS2    ddsCaps;		/* surface capabilities */
} DDCAPS_DX7,*LPDDCAPS_DX7;

typedef struct _DDCAPS_DX6		/* DirectX 6 version of caps struct */
{
    DWORD	dwSize;                 /* size of the DDDRIVERCAPS structure */
    DWORD	dwCaps;                 /* driver specific capabilities */
    DWORD	dwCaps2;                /* more driver specific capabilities */
    DWORD	dwCKeyCaps;             /* color key capabilities of the surface */
    DWORD	dwFXCaps;               /* driver specific stretching and effects capabilities */
    DWORD	dwFXAlphaCaps;          /* alpha driver specific capabilities */
    DWORD	dwPalCaps;              /* palette capabilities */
    DWORD	dwSVCaps;               /* stereo vision capabilities */
    DWORD	dwAlphaBltConstBitDepths;       /* DDBD_2,4,8 */
    DWORD	dwAlphaBltPixelBitDepths;       /* DDBD_1,2,4,8 */
    DWORD	dwAlphaBltSurfaceBitDepths;     /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlayConstBitDepths;   /* DDBD_2,4,8 */
    DWORD	dwAlphaOverlayPixelBitDepths;   /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlaySurfaceBitDepths; /* DDBD_1,2,4,8 */
    DWORD	dwZBufferBitDepths;             /* DDBD_8,16,24,32 */
    DWORD	dwVidMemTotal;          /* total amount of video memory */
    DWORD	dwVidMemFree;           /* amount of free video memory */
    DWORD	dwMaxVisibleOverlays;   /* maximum number of visible overlays */
    DWORD	dwCurrVisibleOverlays;  /* current number of visible overlays */
    DWORD	dwNumFourCCCodes;       /* number of four cc codes */
    DWORD	dwAlignBoundarySrc;     /* source rectangle alignment */
    DWORD	dwAlignSizeSrc;         /* source rectangle byte size */
    DWORD	dwAlignBoundaryDest;    /* dest rectangle alignment */
    DWORD	dwAlignSizeDest;        /* dest rectangle byte size */
    DWORD	dwAlignStrideAlign;     /* stride alignment */
    DWORD	dwRops[DD_ROP_SPACE];   /* ROPs supported */
    DDSCAPS	ddsOldCaps;             /* old DDSCAPS - superseded for DirectX6+ */
    DWORD	dwMinOverlayStretch;    /* minimum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxOverlayStretch;    /* maximum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinLiveVideoStretch;  /* minimum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxLiveVideoStretch;  /* maximum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinHwCodecStretch;    /* minimum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxHwCodecStretch;    /* maximum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwReserved1;
    DWORD	dwReserved2;
    DWORD	dwReserved3;
    DWORD	dwSVBCaps;              /* driver specific capabilities for System->Vmem blts */
    DWORD	dwSVBCKeyCaps;          /* driver color key capabilities for System->Vmem blts */
    DWORD	dwSVBFXCaps;            /* driver FX capabilities for System->Vmem blts */
    DWORD	dwSVBRops[DD_ROP_SPACE];/* ROPs supported for System->Vmem blts */
    DWORD	dwVSBCaps;              /* driver specific capabilities for Vmem->System blts */
    DWORD	dwVSBCKeyCaps;          /* driver color key capabilities for Vmem->System blts */
    DWORD	dwVSBFXCaps;            /* driver FX capabilities for Vmem->System blts */
    DWORD	dwVSBRops[DD_ROP_SPACE];/* ROPs supported for Vmem->System blts */
    DWORD	dwSSBCaps;              /* driver specific capabilities for System->System blts */
    DWORD	dwSSBCKeyCaps;          /* driver color key capabilities for System->System blts */
    DWORD	dwSSBFXCaps;            /* driver FX capabilities for System->System blts */
    DWORD	dwSSBRops[DD_ROP_SPACE];/* ROPs supported for System->System blts */
    DWORD       dwMaxVideoPorts;        /* maximum number of usable video ports */
    DWORD   	dwCurrVideoPorts;       /* current number of video ports used */
    DWORD   	dwSVBCaps2;             /* more driver specific capabilities for System->Vmem blts */
    DWORD   	dwNLVBCaps;             /* driver specific capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBCaps2;            /* more driver specific capabilities non-local->local vidmem blts */
    DWORD   	dwNLVBCKeyCaps;         /* driver color key capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBFXCaps;           /* driver FX capabilities for non-local->local blts */
    DWORD   	dwNLVBRops[DD_ROP_SPACE]; /* ROPs supported for non-local->local blts */
    /* and one new member for DirectX 6 */
    DDSCAPS2    ddsCaps;		/* surface capabilities */
} DDCAPS_DX6,*LPDDCAPS_DX6;

typedef struct _DDCAPS_DX5		/* DirectX5 version of caps struct */
{
    DWORD	dwSize;                 /* size of the DDDRIVERCAPS structure */
    DWORD	dwCaps;                 /* driver specific capabilities */
    DWORD	dwCaps2;                /* more driver specific capabilities */
    DWORD	dwCKeyCaps;             /* color key capabilities of the surface */
    DWORD	dwFXCaps;               /* driver specific stretching and effects capabilities */
    DWORD	dwFXAlphaCaps;          /* alpha driver specific capabilities */
    DWORD	dwPalCaps;              /* palette capabilities */
    DWORD	dwSVCaps;               /* stereo vision capabilities */
    DWORD	dwAlphaBltConstBitDepths;       /* DDBD_2,4,8 */
    DWORD	dwAlphaBltPixelBitDepths;       /* DDBD_1,2,4,8 */
    DWORD	dwAlphaBltSurfaceBitDepths;     /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlayConstBitDepths;   /* DDBD_2,4,8 */
    DWORD	dwAlphaOverlayPixelBitDepths;   /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlaySurfaceBitDepths; /* DDBD_1,2,4,8 */
    DWORD	dwZBufferBitDepths;             /* DDBD_8,16,24,32 */
    DWORD	dwVidMemTotal;          /* total amount of video memory */
    DWORD	dwVidMemFree;           /* amount of free video memory */
    DWORD	dwMaxVisibleOverlays;   /* maximum number of visible overlays */
    DWORD	dwCurrVisibleOverlays;  /* current number of visible overlays */
    DWORD	dwNumFourCCCodes;       /* number of four cc codes */
    DWORD	dwAlignBoundarySrc;     /* source rectangle alignment */
    DWORD	dwAlignSizeSrc;         /* source rectangle byte size */
    DWORD	dwAlignBoundaryDest;    /* dest rectangle alignment */
    DWORD	dwAlignSizeDest;        /* dest rectangle byte size */
    DWORD	dwAlignStrideAlign;     /* stride alignment */
    DWORD	dwRops[DD_ROP_SPACE];   /* ROPs supported */
    DDSCAPS	ddsCaps;                /* DDSCAPS structure has all the general capabilities */
    DWORD	dwMinOverlayStretch;    /* minimum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxOverlayStretch;    /* maximum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinLiveVideoStretch;  /* minimum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxLiveVideoStretch;  /* maximum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinHwCodecStretch;    /* minimum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxHwCodecStretch;    /* maximum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwReserved1;
    DWORD	dwReserved2;
    DWORD	dwReserved3;
    DWORD	dwSVBCaps;              /* driver specific capabilities for System->Vmem blts */
    DWORD	dwSVBCKeyCaps;          /* driver color key capabilities for System->Vmem blts */
    DWORD	dwSVBFXCaps;            /* driver FX capabilities for System->Vmem blts */
    DWORD	dwSVBRops[DD_ROP_SPACE];/* ROPs supported for System->Vmem blts */
    DWORD	dwVSBCaps;              /* driver specific capabilities for Vmem->System blts */
    DWORD	dwVSBCKeyCaps;          /* driver color key capabilities for Vmem->System blts */
    DWORD	dwVSBFXCaps;            /* driver FX capabilities for Vmem->System blts */
    DWORD	dwVSBRops[DD_ROP_SPACE];/* ROPs supported for Vmem->System blts */
    DWORD	dwSSBCaps;              /* driver specific capabilities for System->System blts */
    DWORD	dwSSBCKeyCaps;          /* driver color key capabilities for System->System blts */
    DWORD	dwSSBFXCaps;            /* driver FX capabilities for System->System blts */
    DWORD	dwSSBRops[DD_ROP_SPACE];/* ROPs supported for System->System blts */
    /* the following are the new DirectX 5 members */
    DWORD       dwMaxVideoPorts;        /* maximum number of usable video ports */
    DWORD   	dwCurrVideoPorts;       /* current number of video ports used */
    DWORD   	dwSVBCaps2;             /* more driver specific capabilities for System->Vmem blts */
    DWORD   	dwNLVBCaps;             /* driver specific capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBCaps2;            /* more driver specific capabilities non-local->local vidmem blts */
    DWORD   	dwNLVBCKeyCaps;         /* driver color key capabilities for non-local->local vidmem blts */
    DWORD   	dwNLVBFXCaps;           /* driver FX capabilities for non-local->local blts */
    DWORD   	dwNLVBRops[DD_ROP_SPACE]; /* ROPs supported for non-local->local blts */
} DDCAPS_DX5,*LPDDCAPS_DX5;

typedef struct _DDCAPS_DX3		/* DirectX3 version of caps struct */
{
    DWORD	dwSize;                 /* size of the DDDRIVERCAPS structure */
    DWORD	dwCaps;                 /* driver specific capabilities */
    DWORD	dwCaps2;                /* more driver specific capabilities */
    DWORD	dwCKeyCaps;             /* color key capabilities of the surface */
    DWORD	dwFXCaps;               /* driver specific stretching and effects capabilities */
    DWORD	dwFXAlphaCaps;          /* alpha driver specific capabilities */
    DWORD	dwPalCaps;              /* palette capabilities */
    DWORD	dwSVCaps;               /* stereo vision capabilities */
    DWORD	dwAlphaBltConstBitDepths;       /* DDBD_2,4,8 */
    DWORD	dwAlphaBltPixelBitDepths;       /* DDBD_1,2,4,8 */
    DWORD	dwAlphaBltSurfaceBitDepths;     /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlayConstBitDepths;   /* DDBD_2,4,8 */
    DWORD	dwAlphaOverlayPixelBitDepths;   /* DDBD_1,2,4,8 */
    DWORD	dwAlphaOverlaySurfaceBitDepths; /* DDBD_1,2,4,8 */
    DWORD	dwZBufferBitDepths;             /* DDBD_8,16,24,32 */
    DWORD	dwVidMemTotal;          /* total amount of video memory */
    DWORD	dwVidMemFree;           /* amount of free video memory */
    DWORD	dwMaxVisibleOverlays;   /* maximum number of visible overlays */
    DWORD	dwCurrVisibleOverlays;  /* current number of visible overlays */
    DWORD	dwNumFourCCCodes;       /* number of four cc codes */
    DWORD	dwAlignBoundarySrc;     /* source rectangle alignment */
    DWORD	dwAlignSizeSrc;         /* source rectangle byte size */
    DWORD	dwAlignBoundaryDest;    /* dest rectangle alignment */
    DWORD	dwAlignSizeDest;        /* dest rectangle byte size */
    DWORD	dwAlignStrideAlign;     /* stride alignment */
    DWORD	dwRops[DD_ROP_SPACE];   /* ROPs supported */
    DDSCAPS	ddsCaps;                /* DDSCAPS structure has all the general capabilities */
    DWORD	dwMinOverlayStretch;    /* minimum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxOverlayStretch;    /* maximum overlay stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinLiveVideoStretch;  /* minimum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxLiveVideoStretch;  /* maximum live video stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMinHwCodecStretch;    /* minimum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwMaxHwCodecStretch;    /* maximum hardware codec stretch factor multiplied by 1000, eg 1000 == 1.0, 1300 == 1.3 */
    DWORD	dwReserved1;
    DWORD	dwReserved2;
    DWORD	dwReserved3;
    DWORD	dwSVBCaps;              /* driver specific capabilities for System->Vmem blts */
    DWORD	dwSVBCKeyCaps;          /* driver color key capabilities for System->Vmem blts */
    DWORD	dwSVBFXCaps;            /* driver FX capabilities for System->Vmem blts */
    DWORD	dwSVBRops[DD_ROP_SPACE];/* ROPs supported for System->Vmem blts */
    DWORD	dwVSBCaps;              /* driver specific capabilities for Vmem->System blts */
    DWORD	dwVSBCKeyCaps;          /* driver color key capabilities for Vmem->System blts */
    DWORD	dwVSBFXCaps;            /* driver FX capabilities for Vmem->System blts */
    DWORD	dwVSBRops[DD_ROP_SPACE];/* ROPs supported for Vmem->System blts */
    DWORD	dwSSBCaps;              /* driver specific capabilities for System->System blts */
    DWORD	dwSSBCKeyCaps;          /* driver color key capabilities for System->System blts */
    DWORD	dwSSBFXCaps;            /* driver FX capabilities for System->System blts */
    DWORD	dwSSBRops[DD_ROP_SPACE];/* ROPs supported for System->System blts */
    DWORD	dwReserved4;
    DWORD	dwReserved5;
    DWORD	dwReserved6;
} DDCAPS_DX3,*LPDDCAPS_DX3;

/* set caps struct according to DIRECTDRAW_VERSION */

#if DIRECTDRAW_VERSION <= 0x300
typedef DDCAPS_DX3 DDCAPS;
#elif DIRECTDRAW_VERSION <= 0x500
typedef DDCAPS_DX5 DDCAPS;
#elif DIRECTDRAW_VERSION <= 0x600
typedef DDCAPS_DX6 DDCAPS;
#else
typedef DDCAPS_DX7 DDCAPS;
#endif

typedef DDCAPS *LPDDCAPS;

/* DDCAPS.dwCaps */
#define DDCAPS_3D			0x00000001
#define DDCAPS_ALIGNBOUNDARYDEST	0x00000002
#define DDCAPS_ALIGNSIZEDEST		0x00000004
#define DDCAPS_ALIGNBOUNDARYSRC		0x00000008
#define DDCAPS_ALIGNSIZESRC		0x00000010
#define DDCAPS_ALIGNSTRIDE		0x00000020
#define DDCAPS_BLT			0x00000040
#define DDCAPS_BLTQUEUE			0x00000080
#define DDCAPS_BLTFOURCC		0x00000100
#define DDCAPS_BLTSTRETCH		0x00000200
#define DDCAPS_GDI			0x00000400
#define DDCAPS_OVERLAY			0x00000800
#define DDCAPS_OVERLAYCANTCLIP		0x00001000
#define DDCAPS_OVERLAYFOURCC		0x00002000
#define DDCAPS_OVERLAYSTRETCH		0x00004000
#define DDCAPS_PALETTE			0x00008000
#define DDCAPS_PALETTEVSYNC		0x00010000
#define DDCAPS_READSCANLINE		0x00020000
#define DDCAPS_STEREOVIEW		0x00040000
#define DDCAPS_VBI			0x00080000
#define DDCAPS_ZBLTS			0x00100000
#define DDCAPS_ZOVERLAYS		0x00200000
#define DDCAPS_COLORKEY			0x00400000
#define DDCAPS_ALPHA			0x00800000
#define DDCAPS_COLORKEYHWASSIST		0x01000000
#define DDCAPS_NOHARDWARE		0x02000000
#define DDCAPS_BLTCOLORFILL		0x04000000
#define DDCAPS_BANKSWITCHED		0x08000000
#define DDCAPS_BLTDEPTHFILL		0x10000000
#define DDCAPS_CANCLIP			0x20000000
#define DDCAPS_CANCLIPSTRETCHED		0x40000000
#define DDCAPS_CANBLTSYSMEM		0x80000000

/* DDCAPS.dwCaps2 */
#define DDCAPS2_CERTIFIED		0x00000001
#define DDCAPS2_NO2DDURING3DSCENE       0x00000002
#define DDCAPS2_VIDEOPORT		0x00000004
#define DDCAPS2_AUTOFLIPOVERLAY		0x00000008
#define DDCAPS2_CANBOBINTERLEAVED	0x00000010
#define DDCAPS2_CANBOBNONINTERLEAVED	0x00000020
#define DDCAPS2_COLORCONTROLOVERLAY	0x00000040
#define DDCAPS2_COLORCONTROLPRIMARY	0x00000080
#define DDCAPS2_CANDROPZ16BIT		0x00000100
#define DDCAPS2_NONLOCALVIDMEM		0x00000200
#define DDCAPS2_NONLOCALVIDMEMCAPS	0x00000400
#define DDCAPS2_NOPAGELOCKREQUIRED	0x00000800
#define DDCAPS2_WIDESURFACES		0x00001000
#define DDCAPS2_CANFLIPODDEVEN		0x00002000
#define DDCAPS2_CANBOBHARDWARE		0x00004000
#define DDCAPS2_COPYFOURCC              0x00008000
#define DDCAPS2_PRIMARYGAMMA            0x00020000
#define DDCAPS2_CANRENDERWINDOWED       0x00080000
#define DDCAPS2_CANCALIBRATEGAMMA       0x00100000
#define DDCAPS2_FLIPINTERVAL            0x00200000
#define DDCAPS2_FLIPNOVSYNC             0x00400000
#define DDCAPS2_CANMANAGETEXTURE        0x00800000
#define DDCAPS2_TEXMANINNONLOCALVIDMEM  0x01000000
#define DDCAPS2_STEREO                  0x02000000
#define DDCAPS2_SYSTONONLOCAL_AS_SYSTOLOCAL   0x04000000


/* Set/Get Colour Key Flags */
#define DDCKEY_COLORSPACE  0x00000001  /* Struct is single colour space */
#define DDCKEY_DESTBLT     0x00000002  /* To be used as dest for blt */
#define DDCKEY_DESTOVERLAY 0x00000004  /* To be used as dest for CK overlays */
#define DDCKEY_SRCBLT      0x00000008  /* To be used as src for blt */
#define DDCKEY_SRCOVERLAY  0x00000010  /* To be used as src for CK overlays */

typedef struct _DDCOLORKEY
{
    DWORD	dwColorSpaceLowValue;/* low boundary of color space that is to
                                      * be treated as Color Key, inclusive
                      */
    DWORD	dwColorSpaceHighValue;/* high boundary of color space that is
                                       * to be treated as Color Key, inclusive
                       */
} DDCOLORKEY,*LPDDCOLORKEY;

/* ddCKEYCAPS bits */
#define DDCKEYCAPS_DESTBLT			0x00000001
#define DDCKEYCAPS_DESTBLTCLRSPACE		0x00000002
#define DDCKEYCAPS_DESTBLTCLRSPACEYUV		0x00000004
#define DDCKEYCAPS_DESTBLTYUV			0x00000008
#define DDCKEYCAPS_DESTOVERLAY			0x00000010
#define DDCKEYCAPS_DESTOVERLAYCLRSPACE		0x00000020
#define DDCKEYCAPS_DESTOVERLAYCLRSPACEYUV	0x00000040
#define DDCKEYCAPS_DESTOVERLAYONEACTIVE		0x00000080
#define DDCKEYCAPS_DESTOVERLAYYUV		0x00000100
#define DDCKEYCAPS_SRCBLT			0x00000200
#define DDCKEYCAPS_SRCBLTCLRSPACE		0x00000400
#define DDCKEYCAPS_SRCBLTCLRSPACEYUV		0x00000800
#define DDCKEYCAPS_SRCBLTYUV			0x00001000
#define DDCKEYCAPS_SRCOVERLAY			0x00002000
#define DDCKEYCAPS_SRCOVERLAYCLRSPACE		0x00004000
#define DDCKEYCAPS_SRCOVERLAYCLRSPACEYUV	0x00008000
#define DDCKEYCAPS_SRCOVERLAYONEACTIVE		0x00010000
#define DDCKEYCAPS_SRCOVERLAYYUV		0x00020000
#define DDCKEYCAPS_NOCOSTOVERLAY		0x00040000

typedef struct _DDPIXELFORMAT {
    DWORD	dwSize;                 /* 0: size of structure */
    DWORD	dwFlags;                /* 4: pixel format flags */
    DWORD	dwFourCC;               /* 8: (FOURCC code) */
    __extension__ union {
    DWORD	dwRGBBitCount;          /* C: how many bits per pixel */
    DWORD	dwYUVBitCount;          /* C: how many bits per pixel */
    DWORD	dwZBufferBitDepth;      /* C: how many bits for z buffers */
    DWORD	dwAlphaBitDepth;        /* C: how many bits for alpha channels*/
    DWORD	dwLuminanceBitCount;
    DWORD	dwBumpBitCount;
    } DUMMYUNIONNAME1;
    __extension__ union {
    DWORD	dwRBitMask;             /* 10: mask for red bit*/
    DWORD	dwYBitMask;             /* 10: mask for Y bits*/
    DWORD	dwStencilBitDepth;
    DWORD	dwLuminanceBitMask;
    DWORD	dwBumpDuBitMask;
    } DUMMYUNIONNAME2;
    __extension__ union {
    DWORD	dwGBitMask;             /* 14: mask for green bits*/
    DWORD	dwUBitMask;             /* 14: mask for U bits*/
    DWORD	dwZBitMask;
    DWORD	dwBumpDvBitMask;
    } DUMMYUNIONNAME3;
    __extension__ union {
    DWORD   dwBBitMask;             /* 18: mask for blue bits*/
    DWORD   dwVBitMask;             /* 18: mask for V bits*/
    DWORD	dwStencilBitMask;
    DWORD	dwBumpLuminanceBitMask;
    } DUMMYUNIONNAME4;
    __extension__ union {
        DWORD	dwRGBAlphaBitMask;	/* 1C: mask for alpha channel */
        DWORD	dwYUVAlphaBitMask;	/* 1C: mask for alpha channel */
    DWORD	dwLuminanceAlphaBitMask;
    DWORD	dwRGBZBitMask;		/* 1C: mask for Z channel */
    DWORD	dwYUVZBitMask;		/* 1C: mask for Z channel */
    } DUMMYUNIONNAME5;
                        /* 20: next structure */
} DDPIXELFORMAT,*LPDDPIXELFORMAT;

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3)  \
    ((DWORD)(BYTE)(ch0) | ((DWORD)(BYTE)(ch1) << 8) |  \
    ((DWORD)(BYTE)(ch2) << 16) | ((DWORD)(BYTE)(ch3) << 24 ))
#endif /* MAKEFOURCC */

/* DDCAPS.dwFXCaps */
#define DDFXCAPS_BLTALPHA               0x00000001
#define DDFXCAPS_OVERLAYALPHA           0x00000004
#define DDFXCAPS_BLTARITHSTRETCHYN	0x00000010
#define DDFXCAPS_BLTARITHSTRETCHY	0x00000020
#define DDFXCAPS_BLTMIRRORLEFTRIGHT	0x00000040
#define DDFXCAPS_BLTMIRRORUPDOWN	0x00000080
#define DDFXCAPS_BLTROTATION		0x00000100
#define DDFXCAPS_BLTROTATION90		0x00000200
#define DDFXCAPS_BLTSHRINKX		0x00000400
#define DDFXCAPS_BLTSHRINKXN		0x00000800
#define DDFXCAPS_BLTSHRINKY		0x00001000
#define DDFXCAPS_BLTSHRINKYN		0x00002000
#define DDFXCAPS_BLTSTRETCHX		0x00004000
#define DDFXCAPS_BLTSTRETCHXN		0x00008000
#define DDFXCAPS_BLTSTRETCHY		0x00010000
#define DDFXCAPS_BLTSTRETCHYN		0x00020000
#define DDFXCAPS_OVERLAYARITHSTRETCHY	0x00040000
#define DDFXCAPS_OVERLAYARITHSTRETCHYN	0x00000008
#define DDFXCAPS_OVERLAYSHRINKX		0x00080000
#define DDFXCAPS_OVERLAYSHRINKXN	0x00100000
#define DDFXCAPS_OVERLAYSHRINKY		0x00200000
#define DDFXCAPS_OVERLAYSHRINKYN	0x00400000
#define DDFXCAPS_OVERLAYSTRETCHX	0x00800000
#define DDFXCAPS_OVERLAYSTRETCHXN	0x01000000
#define DDFXCAPS_OVERLAYSTRETCHY	0x02000000
#define DDFXCAPS_OVERLAYSTRETCHYN	0x04000000
#define DDFXCAPS_OVERLAYMIRRORLEFTRIGHT	0x08000000
#define DDFXCAPS_OVERLAYMIRRORUPDOWN	0x10000000

#define DDFXCAPS_OVERLAYFILTER          DDFXCAPS_OVERLAYARITHSTRETCHY

/* DDCAPS.dwFXAlphaCaps */
#define DDFXALPHACAPS_BLTALPHAEDGEBLEND		0x00000001
#define DDFXALPHACAPS_BLTALPHAPIXELS		0x00000002
#define DDFXALPHACAPS_BLTALPHAPIXELSNEG		0x00000004
#define DDFXALPHACAPS_BLTALPHASURFACES		0x00000008
#define DDFXALPHACAPS_BLTALPHASURFACESNEG	0x00000010
#define DDFXALPHACAPS_OVERLAYALPHAEDGEBLEND	0x00000020
#define DDFXALPHACAPS_OVERLAYALPHAPIXELS	0x00000040
#define DDFXALPHACAPS_OVERLAYALPHAPIXELSNEG	0x00000080
#define DDFXALPHACAPS_OVERLAYALPHASURFACES	0x00000100
#define DDFXALPHACAPS_OVERLAYALPHASURFACESNEG	0x00000200

/* DDCAPS.dwPalCaps */
#define DDPCAPS_4BIT			0x00000001
#define DDPCAPS_8BITENTRIES		0x00000002
#define DDPCAPS_8BIT			0x00000004
#define DDPCAPS_INITIALIZE		0x00000008
#define DDPCAPS_PRIMARYSURFACE		0x00000010
#define DDPCAPS_PRIMARYSURFACELEFT	0x00000020
#define DDPCAPS_ALLOW256		0x00000040
#define DDPCAPS_VSYNC			0x00000080
#define DDPCAPS_1BIT			0x00000100
#define DDPCAPS_2BIT			0x00000200
#define DDPCAPS_ALPHA                   0x00000400

/* DDCAPS.dwSVCaps */
/* the first 4 of these are now obsolete */
#if DIRECTDRAW_VERSION >= 0x700	/* FIXME: I'm not sure when this switch occurred */
#define DDSVCAPS_RESERVED1		0x00000001
#define DDSVCAPS_RESERVED2		0x00000002
#define DDSVCAPS_RESERVED3		0x00000004
#define DDSVCAPS_RESERVED4		0x00000008
#else
#define DDSVCAPS_ENIGMA			0x00000001
#define DDSVCAPS_FLICKER		0x00000002
#define DDSVCAPS_REDBLUE		0x00000004
#define DDSVCAPS_SPLIT			0x00000008
#endif
#define DDSVCAPS_STEREOSEQUENTIAL       0x00000010

/* BitDepths */
#define DDBD_1				0x00004000
#define DDBD_2				0x00002000
#define DDBD_4				0x00001000
#define DDBD_8				0x00000800
#define DDBD_16				0x00000400
#define DDBD_24				0x00000200
#define DDBD_32				0x00000100

/* DDOVERLAYFX.dwDDFX */
#define DDOVERFX_ARITHSTRETCHY		0x00000001
#define DDOVERFX_MIRRORLEFTRIGHT	0x00000002
#define DDOVERFX_MIRRORUPDOWN		0x00000004

/* UpdateOverlay flags */
#define DDOVER_ALPHADEST                        0x00000001
#define DDOVER_ALPHADESTCONSTOVERRIDE           0x00000002
#define DDOVER_ALPHADESTNEG                     0x00000004
#define DDOVER_ALPHADESTSURFACEOVERRIDE         0x00000008
#define DDOVER_ALPHAEDGEBLEND                   0x00000010
#define DDOVER_ALPHASRC                         0x00000020
#define DDOVER_ALPHASRCCONSTOVERRIDE            0x00000040
#define DDOVER_ALPHASRCNEG                      0x00000080
#define DDOVER_ALPHASRCSURFACEOVERRIDE          0x00000100
#define DDOVER_HIDE                             0x00000200
#define DDOVER_KEYDEST                          0x00000400
#define DDOVER_KEYDESTOVERRIDE                  0x00000800
#define DDOVER_KEYSRC                           0x00001000
#define DDOVER_KEYSRCOVERRIDE                   0x00002000
#define DDOVER_SHOW                             0x00004000
#define DDOVER_ADDDIRTYRECT                     0x00008000
#define DDOVER_REFRESHDIRTYRECTS                0x00010000
#define DDOVER_REFRESHALL                       0x00020000
#define DDOVER_DDFX                             0x00080000
#define DDOVER_AUTOFLIP                         0x00100000
#define DDOVER_BOB                              0x00200000
#define DDOVER_OVERRIDEBOBWEAVE                 0x00400000
#define DDOVER_INTERLEAVED                      0x00800000

/* DDCOLORKEY.dwFlags */
#define DDPF_ALPHAPIXELS		0x00000001
#define DDPF_ALPHA			0x00000002
#define DDPF_FOURCC			0x00000004
#define DDPF_PALETTEINDEXED4		0x00000008
#define DDPF_PALETTEINDEXEDTO8		0x00000010
#define DDPF_PALETTEINDEXED8		0x00000020
#define DDPF_RGB			0x00000040
#define DDPF_COMPRESSED			0x00000080
#define DDPF_RGBTOYUV			0x00000100
#define DDPF_YUV			0x00000200
#define DDPF_ZBUFFER			0x00000400
#define DDPF_PALETTEINDEXED1		0x00000800
#define DDPF_PALETTEINDEXED2		0x00001000
#define DDPF_ZPIXELS			0x00002000
#define DDPF_STENCILBUFFER              0x00004000
#define DDPF_ALPHAPREMULT               0x00008000
#define DDPF_LUMINANCE                  0x00020000
#define DDPF_BUMPLUMINANCE              0x00040000
#define DDPF_BUMPDUDV                   0x00080000

/* SetCooperativeLevel dwFlags */
#define DDSCL_FULLSCREEN		0x00000001
#define DDSCL_ALLOWREBOOT		0x00000002
#define DDSCL_NOWINDOWCHANGES		0x00000004
#define DDSCL_NORMAL			0x00000008
#define DDSCL_EXCLUSIVE			0x00000010
#define DDSCL_ALLOWMODEX		0x00000040
#define DDSCL_SETFOCUSWINDOW		0x00000080
#define DDSCL_SETDEVICEWINDOW		0x00000100
#define DDSCL_CREATEDEVICEWINDOW	0x00000200
#define DDSCL_MULTITHREADED             0x00000400
#define DDSCL_FPUSETUP                  0x00000800
#define DDSCL_FPUPRESERVE               0x00001000


/* DDSURFACEDESC.dwFlags */
#define	DDSD_CAPS		0x00000001
#define	DDSD_HEIGHT		0x00000002
#define	DDSD_WIDTH		0x00000004
#define	DDSD_PITCH		0x00000008
#define	DDSD_BACKBUFFERCOUNT	0x00000020
#define	DDSD_ZBUFFERBITDEPTH	0x00000040
#define	DDSD_ALPHABITDEPTH	0x00000080
#define	DDSD_LPSURFACE		0x00000800
#define	DDSD_PIXELFORMAT	0x00001000
#define	DDSD_CKDESTOVERLAY	0x00002000
#define	DDSD_CKDESTBLT		0x00004000
#define	DDSD_CKSRCOVERLAY	0x00008000
#define	DDSD_CKSRCBLT		0x00010000
#define	DDSD_MIPMAPCOUNT	0x00020000
#define	DDSD_REFRESHRATE	0x00040000
#define	DDSD_LINEARSIZE		0x00080000
#define DDSD_TEXTURESTAGE       0x00100000
#define DDSD_FVF                0x00200000
#define DDSD_SRCVBHANDLE        0x00400000
#define DDSD_DEPTH              0x00800000
#define DDSD_ALL                0x00fff9ee

/* EnumSurfaces flags */
#define DDENUMSURFACES_ALL          0x00000001
#define DDENUMSURFACES_MATCH        0x00000002
#define DDENUMSURFACES_NOMATCH      0x00000004
#define DDENUMSURFACES_CANBECREATED 0x00000008
#define DDENUMSURFACES_DOESEXIST    0x00000010

/* SetDisplayMode flags */
#define DDSDM_STANDARDVGAMODE	0x00000001

/* EnumDisplayModes flags */
#define DDEDM_REFRESHRATES	0x00000001
#define DDEDM_STANDARDVGAMODES	0x00000002

/* WaitForVerticalDisplay flags */

#define DDWAITVB_BLOCKBEGIN		0x00000001
#define DDWAITVB_BLOCKBEGINEVENT	0x00000002
#define DDWAITVB_BLOCKEND		0x00000004

typedef struct _DDSURFACEDESC
{
    DWORD	dwSize;		/* 0: size of the DDSURFACEDESC structure*/
    DWORD	dwFlags;	/* 4: determines what fields are valid*/
    DWORD	dwHeight;	/* 8: height of surface to be created*/
    DWORD	dwWidth;	/* C: width of input surface*/
    __extension__ union {
        LONG	lPitch;	/* 10: distance to start of next line (return value only)*/
        DWORD	dwLinearSize;
    } DUMMYUNIONNAME1;
    DWORD	dwBackBufferCount;/* 14: number of back buffers requested*/
    __extension__ union {
        DWORD	dwMipMapCount;/* 18:number of mip-map levels requested*/
        DWORD	dwZBufferBitDepth;/*18: depth of Z buffer requested*/
        DWORD	dwRefreshRate;/* 18:refresh rate (used when display mode is described)*/
    } DUMMYUNIONNAME2;
    DWORD	dwAlphaBitDepth;/* 1C:depth of alpha buffer requested*/
    DWORD	dwReserved;	/* 20:reserved*/
    LPVOID	lpSurface;	/* 24:pointer to the associated surface memory*/
    DDCOLORKEY	ddckCKDestOverlay;/* 28: CK for dest overlay use*/
    DDCOLORKEY	ddckCKDestBlt;	/* 30: CK for destination blt use*/
    DDCOLORKEY	ddckCKSrcOverlay;/* 38: CK for source overlay use*/
    DDCOLORKEY	ddckCKSrcBlt;	/* 40: CK for source blt use*/
    DDPIXELFORMAT	ddpfPixelFormat;/* 48: pixel format description of the surface*/
    DDSCAPS		ddsCaps;	/* 68: direct draw surface caps */
} DDSURFACEDESC,*LPDDSURFACEDESC;

typedef struct _DDSURFACEDESC2
{
    DWORD	dwSize;		/* 0: size of the DDSURFACEDESC2 structure*/
    DWORD	dwFlags;	/* 4: determines what fields are valid*/
    DWORD	dwHeight;	/* 8: height of surface to be created*/
    DWORD	dwWidth;	/* C: width of input surface*/
    __extension__ union {
        LONG	lPitch;	      /*10: distance to start of next line (return value only)*/
        DWORD   dwLinearSize; /*10: formless late-allocated optimized surface size */
    } DUMMYUNIONNAME1;
    DWORD	dwBackBufferCount;/* 14: number of back buffers requested*/
    __extension__ union {
        DWORD	dwMipMapCount;/* 18:number of mip-map levels requested*/
        DWORD	dwRefreshRate;/* 18:refresh rate (used when display mode is described)*/
        DWORD   dwSrcVBHandle;/* 18:source used in VB::Optimize */
    } DUMMYUNIONNAME2;
    DWORD	dwAlphaBitDepth;/* 1C:depth of alpha buffer requested*/
    DWORD	dwReserved;	/* 20:reserved*/
    LPVOID	lpSurface;	/* 24:pointer to the associated surface memory*/
    __extension__ union {
        DDCOLORKEY	ddckCKDestOverlay; /* 28: CK for dest overlay use*/
        DWORD 		dwEmptyFaceColor;  /* 28: color for empty cubemap faces */
    } DUMMYUNIONNAME3;
    DDCOLORKEY	ddckCKDestBlt;	/* 30: CK for destination blt use*/
    DDCOLORKEY	ddckCKSrcOverlay;/* 38: CK for source overlay use*/
    DDCOLORKEY	ddckCKSrcBlt;	/* 40: CK for source blt use*/

    __extension__ union {
        DDPIXELFORMAT	ddpfPixelFormat;/* 48: pixel format description of the surface*/
        DWORD 		dwFVF;	/* 48: vertex format description of vertex buffers */
    } DUMMYUNIONNAME4;
    DDSCAPS2	ddsCaps;  /* 68: DDraw surface caps */
    DWORD		dwTextureStage; /* 78: stage in multitexture cascade */
} DDSURFACEDESC2,*LPDDSURFACEDESC2;


typedef struct _DDARGB {
    BYTE	blue;
    BYTE	green;
    BYTE	red;
    BYTE	alpha;
} DDARGB, *LPDDARGB;

typedef struct _DDRGBA {
    BYTE	red;
    BYTE	green;
    BYTE	blue;
    BYTE	alpha;
} DDRGBA, *LPDDRGBA;


/* DDCOLORCONTROL.dwFlags */
#define DDCOLOR_BRIGHTNESS	0x00000001
#define DDCOLOR_CONTRAST	0x00000002
#define DDCOLOR_HUE		0x00000004
#define DDCOLOR_SATURATION	0x00000008
#define DDCOLOR_SHARPNESS	0x00000010
#define DDCOLOR_GAMMA		0x00000020
#define DDCOLOR_COLORENABLE	0x00000040

typedef struct {
    DWORD	dwSize;
    DWORD	dwFlags;
    LONG	lBrightness;
    LONG	lContrast;
    LONG	lHue;
    LONG	lSaturation;
    LONG	lSharpness;
    LONG	lGamma;
    LONG	lColorEnable;
    DWORD	dwReserved1;
} DDCOLORCONTROL,*LPDDCOLORCONTROL;

typedef struct {
    WORD	red[256];
    WORD	green[256];
    WORD	blue[256];
} DDGAMMARAMP,*LPDDGAMMARAMP;

typedef BOOL (CALLBACK *LPDDENUMCALLBACKA)(GUID *, LPSTR, LPSTR, LPVOID);
typedef BOOL (CALLBACK *LPDDENUMCALLBACKW)(GUID *, LPWSTR, LPWSTR, LPVOID);
DECL_WINELIB_TYPE_AW(LPDDENUMCALLBACK)

typedef HRESULT (CALLBACK *LPDDENUMMODESCALLBACK)(LPDDSURFACEDESC, LPVOID);
typedef HRESULT (CALLBACK *LPDDENUMMODESCALLBACK2)(LPDDSURFACEDESC2, LPVOID);
typedef HRESULT (CALLBACK *LPDDENUMSURFACESCALLBACK)(LPDIRECTDRAWSURFACE, LPDDSURFACEDESC, LPVOID);
typedef HRESULT (CALLBACK *LPDDENUMSURFACESCALLBACK2)(LPDIRECTDRAWSURFACE4, LPDDSURFACEDESC2, LPVOID);
typedef HRESULT (CALLBACK *LPDDENUMSURFACESCALLBACK7)(LPDIRECTDRAWSURFACE7, LPDDSURFACEDESC2, LPVOID);

typedef BOOL (CALLBACK *LPDDENUMCALLBACKEXA)(GUID *, LPSTR, LPSTR, LPVOID, HMONITOR);
typedef BOOL (CALLBACK *LPDDENUMCALLBACKEXW)(GUID *, LPWSTR, LPWSTR, LPVOID, HMONITOR);
DECL_WINELIB_TYPE_AW(LPDDENUMCALLBACKEX)

HRESULT WINAPI DirectDrawEnumerateA(LPDDENUMCALLBACKA,LPVOID);
HRESULT WINAPI DirectDrawEnumerateW(LPDDENUMCALLBACKW,LPVOID);
#define DirectDrawEnumerate WINELIB_NAME_AW(DirectDrawEnumerate)

HRESULT WINAPI DirectDrawEnumerateExA( LPDDENUMCALLBACKEXA lpCallback, LPVOID lpContext, DWORD dwFlags);
HRESULT WINAPI DirectDrawEnumerateExW( LPDDENUMCALLBACKEXW lpCallback, LPVOID lpContext, DWORD dwFlags);
#define DirectDrawEnumerateEx WINELIB_NAME_AW(DirectDrawEnumerateEx)

typedef HRESULT (WINAPI * LPDIRECTDRAWENUMERATEEXA)( LPDDENUMCALLBACKEXA lpCallback, LPVOID lpContext, DWORD dwFlags);
typedef HRESULT (WINAPI * LPDIRECTDRAWENUMERATEEXW)( LPDDENUMCALLBACKEXW lpCallback, LPVOID lpContext, DWORD dwFlags);
DECL_WINELIB_TYPE_AW(LPDIRECTDRAWENUMERATEEX)

/* flags for DirectDrawEnumerateEx */
#define DDENUM_ATTACHEDSECONDARYDEVICES	0x00000001
#define DDENUM_DETACHEDSECONDARYDEVICES	0x00000002
#define DDENUM_NONDISPLAYDEVICES	0x00000004

/* flags for DirectDrawCreate or IDirectDraw::Initialize */
#define DDCREATE_HARDWAREONLY	1L
#define DDCREATE_EMULATIONONLY	2L

typedef struct _DDBLTFX
{
    DWORD       dwSize;                         /* size of structure */
    DWORD       dwDDFX;                         /* FX operations */
    DWORD       dwROP;                          /* Win32 raster operations */
    DWORD       dwDDROP;                        /* Raster operations new for DirectDraw */
    DWORD       dwRotationAngle;                /* Rotation angle for blt */
    DWORD       dwZBufferOpCode;                /* ZBuffer compares */
    DWORD       dwZBufferLow;                   /* Low limit of Z buffer */
    DWORD       dwZBufferHigh;                  /* High limit of Z buffer */
    DWORD       dwZBufferBaseDest;              /* Destination base value */
    DWORD       dwZDestConstBitDepth;           /* Bit depth used to specify Z constant for destination */
    __extension__ union
    {
        DWORD   dwZDestConst;                   /* Constant to use as Z buffer for dest */
        LPDIRECTDRAWSURFACE lpDDSZBufferDest;   /* Surface to use as Z buffer for dest */
    } DUMMYUNIONNAME1;
    DWORD       dwZSrcConstBitDepth;            /* Bit depth used to specify Z constant for source */
    __extension__ union
    {
        DWORD   dwZSrcConst;                    /* Constant to use as Z buffer for src */
        LPDIRECTDRAWSURFACE lpDDSZBufferSrc;    /* Surface to use as Z buffer for src */
    } DUMMYUNIONNAME2;
    DWORD       dwAlphaEdgeBlendBitDepth;       /* Bit depth used to specify constant for alpha edge blend */
    DWORD       dwAlphaEdgeBlend;               /* Alpha for edge blending */
    DWORD       dwReserved;
    DWORD       dwAlphaDestConstBitDepth;       /* Bit depth used to specify alpha constant for destination */
    __extension__ union
    {
        DWORD   dwAlphaDestConst;               /* Constant to use as Alpha Channel */
        LPDIRECTDRAWSURFACE lpDDSAlphaDest;     /* Surface to use as Alpha Channel */
    } DUMMYUNIONNAME3;
    DWORD       dwAlphaSrcConstBitDepth;        /* Bit depth used to specify alpha constant for source */
    __extension__ union
    {
        DWORD   dwAlphaSrcConst;                /* Constant to use as Alpha Channel */
        LPDIRECTDRAWSURFACE lpDDSAlphaSrc;      /* Surface to use as Alpha Channel */
    } DUMMYUNIONNAME4;
    __extension__ union
    {
        DWORD   dwFillColor;                    /* color in RGB or Palettized */
        DWORD   dwFillDepth;                    /* depth value for z-buffer */
    DWORD   dwFillPixel;			/* pixel val for RGBA or RGBZ */
        LPDIRECTDRAWSURFACE lpDDSPattern;       /* Surface to use as pattern */
    } DUMMYUNIONNAME5;
    DDCOLORKEY  ddckDestColorkey;               /* DestColorkey override */
    DDCOLORKEY  ddckSrcColorkey;                /* SrcColorkey override */
} DDBLTFX,*LPDDBLTFX;

/* dwDDFX */
/* arithmetic stretching along y axis */
#define DDBLTFX_ARITHSTRETCHY			0x00000001
/* mirror on y axis */
#define DDBLTFX_MIRRORLEFTRIGHT			0x00000002
/* mirror on x axis */
#define DDBLTFX_MIRRORUPDOWN			0x00000004
/* do not tear */
#define DDBLTFX_NOTEARING			0x00000008
/* 180 degrees clockwise rotation */
#define DDBLTFX_ROTATE180			0x00000010
/* 270 degrees clockwise rotation */
#define DDBLTFX_ROTATE270			0x00000020
/* 90 degrees clockwise rotation */
#define DDBLTFX_ROTATE90			0x00000040
/* dwZBufferLow and dwZBufferHigh specify limits to the copied Z values */
#define DDBLTFX_ZBUFFERRANGE			0x00000080
/* add dwZBufferBaseDest to every source z value before compare */
#define DDBLTFX_ZBUFFERBASEDEST			0x00000100

typedef struct _DDOVERLAYFX
{
    DWORD       dwSize;                         /* size of structure */
    DWORD       dwAlphaEdgeBlendBitDepth;       /* Bit depth used to specify constant for alpha edge blend */
    DWORD       dwAlphaEdgeBlend;               /* Constant to use as alpha for edge blend */
    DWORD       dwReserved;
    DWORD       dwAlphaDestConstBitDepth;       /* Bit depth used to specify alpha constant for destination */
    __extension__ union
    {
        DWORD   dwAlphaDestConst;               /* Constant to use as alpha channel for dest */
        LPDIRECTDRAWSURFACE lpDDSAlphaDest;     /* Surface to use as alpha channel for dest */
    } DUMMYUNIONNAME1;
    DWORD       dwAlphaSrcConstBitDepth;        /* Bit depth used to specify alpha constant for source */
    __extension__ union
    {
        DWORD   dwAlphaSrcConst;                /* Constant to use as alpha channel for src */
        LPDIRECTDRAWSURFACE lpDDSAlphaSrc;      /* Surface to use as alpha channel for src */
    } DUMMYUNIONNAME2;
    DDCOLORKEY  dckDestColorkey;                /* DestColorkey override */
    DDCOLORKEY  dckSrcColorkey;                 /* DestColorkey override */
    DWORD       dwDDFX;                         /* Overlay FX */
    DWORD       dwFlags;                        /* flags */
} DDOVERLAYFX,*LPDDOVERLAYFX;

typedef struct _DDBLTBATCH
{
    LPRECT		lprDest;
    LPDIRECTDRAWSURFACE	lpDDSSrc;
    LPRECT		lprSrc;
    DWORD		dwFlags;
    LPDDBLTFX		lpDDBltFx;
} DDBLTBATCH,*LPDDBLTBATCH;

#define MAX_DDDEVICEID_STRING          512

#define DDGDI_GETHOSTIDENTIFIER 1

typedef struct tagDDDEVICEIDENTIFIER {
  char    szDriver[MAX_DDDEVICEID_STRING];
  char    szDescription[MAX_DDDEVICEID_STRING];
  LARGE_INTEGER  liDriverVersion;
  DWORD   dwVendorId;
  DWORD   dwDeviceId;
  DWORD   dwSubSysId;
  DWORD   dwRevision;
  GUID    guidDeviceIdentifier;
} DDDEVICEIDENTIFIER, * LPDDDEVICEIDENTIFIER;

typedef struct tagDDDEVICEIDENTIFIER2 {
  char    szDriver[MAX_DDDEVICEID_STRING];	/* user readable driver name */
  char    szDescription[MAX_DDDEVICEID_STRING]; /* user readable description */
  LARGE_INTEGER  liDriverVersion;		/* driver version */
  DWORD   dwVendorId;				/* vendor ID, zero if unknown */
  DWORD   dwDeviceId;				/* chipset ID, zero if unknown */
  DWORD   dwSubSysId;				/* board ID, zero if unknown */
  DWORD   dwRevision;				/* chipset version, zero if unknown */
  GUID    guidDeviceIdentifier;			/* unique ID for this driver/chipset combination */
  DWORD   dwWHQLLevel;				/* Windows Hardware Quality Lab certification level */
} DDDEVICEIDENTIFIER2, * LPDDDEVICEIDENTIFIER2;

/*****************************************************************************
 * IDirectDrawPalette interface
 */
#undef INTERFACE
#define INTERFACE IDirectDrawPalette
DECLARE_INTERFACE_(IDirectDrawPalette,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawPalette methods ***/
    STDMETHOD(GetCaps)(THIS_ LPDWORD lpdwCaps) PURE;
    STDMETHOD(GetEntries)(THIS_ DWORD dwFlags, DWORD dwBase, DWORD dwNumEntries, LPPALETTEENTRY lpEntries) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, DWORD dwFlags, LPPALETTEENTRY lpDDColorTable) PURE;
    STDMETHOD(SetEntries)(THIS_ DWORD dwFlags, DWORD dwStartingEntry, DWORD dwCount, LPPALETTEENTRY lpEntries) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawPalette_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawPalette_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawPalette_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawPalette methods ***/
#define IDirectDrawPalette_GetCaps(p,a)          (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawPalette_GetEntries(p,a,b,c,d) (p)->lpVtbl->GetEntries(p,a,b,c,d)
#define IDirectDrawPalette_Initialize(p,a,b,c)   (p)->lpVtbl->Initialize(p,a,b,c)
#define IDirectDrawPalette_SetEntries(p,a,b,c,d) (p)->lpVtbl->SetEntries(p,a,b,c,d)
#else
/*** IUnknown methods ***/
#define IDirectDrawPalette_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawPalette_AddRef(p)             (p)->AddRef()
#define IDirectDrawPalette_Release(p)            (p)->Release()
/*** IDirectDrawPalette methods ***/
#define IDirectDrawPalette_GetCaps(p,a)          (p)->GetCaps(a)
#define IDirectDrawPalette_GetEntries(p,a,b,c,d) (p)->GetEntries(a,b,c,d)
#define IDirectDrawPalette_Initialize(p,a,b,c)   (p)->Initialize(a,b,c)
#define IDirectDrawPalette_SetEntries(p,a,b,c,d) (p)->SetEntries(a,b,c,d)
#endif


/*****************************************************************************
 * IDirectDrawClipper interface
 */
#define INTERFACE IDirectDrawClipper
DECLARE_INTERFACE_(IDirectDrawClipper,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawClipper methods ***/
    STDMETHOD(GetClipList)(THIS_ LPRECT lpRect, LPRGNDATA lpClipList, LPDWORD lpdwSize) PURE;
    STDMETHOD(GetHWnd)(THIS_ HWND *lphWnd) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, DWORD dwFlags) PURE;
    STDMETHOD(IsClipListChanged)(THIS_ BOOL *lpbChanged) PURE;
    STDMETHOD(SetClipList)(THIS_ LPRGNDATA lpClipList, DWORD dwFlags) PURE;
    STDMETHOD(SetHWnd)(THIS_ DWORD dwFlags, HWND hWnd) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawClipper_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawClipper_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawClipper_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawClipper methods ***/
#define IDirectDrawClipper_GetClipList(p,a,b,c)   (p)->lpVtbl->GetClipList(p,a,b,c)
#define IDirectDrawClipper_GetHWnd(p,a)           (p)->lpVtbl->GetHWnd(p,a)
#define IDirectDrawClipper_Initialize(p,a,b)      (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawClipper_IsClipListChanged(p,a) (p)->lpVtbl->IsClipListChanged(p,a)
#define IDirectDrawClipper_SetClipList(p,a,b)     (p)->lpVtbl->SetClipList(p,a,b)
#define IDirectDrawClipper_SetHWnd(p,a,b)         (p)->lpVtbl->SetHWnd(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDrawClipper_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawClipper_AddRef(p)             (p)->AddRef()
#define IDirectDrawClipper_Release(p)            (p)->Release()
/*** IDirectDrawClipper methods ***/
#define IDirectDrawClipper_GetClipList(p,a,b,c)   (p)->GetClipList(a,b,c)
#define IDirectDrawClipper_GetHWnd(p,a)           (p)->GetHWnd(a)
#define IDirectDrawClipper_Initialize(p,a,b)      (p)->Initialize(a,b)
#define IDirectDrawClipper_IsClipListChanged(p,a) (p)->IsClipListChanged(a)
#define IDirectDrawClipper_SetClipList(p,a,b)     (p)->SetClipList(a,b)
#define IDirectDrawClipper_SetHWnd(p,a,b)         (p)->SetHWnd(a,b)
#endif


/*****************************************************************************
 * IDirectDraw interface
 */
#define INTERFACE IDirectDraw
DECLARE_INTERFACE_(IDirectDraw,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDraw methods ***/
    STDMETHOD(Compact)(THIS) PURE;
    STDMETHOD(CreateClipper)(THIS_ DWORD dwFlags, LPDIRECTDRAWCLIPPER *lplpDDClipper, IUnknown *pUnkOuter) PURE;
    STDMETHOD(CreatePalette)(THIS_ DWORD dwFlags, LPPALETTEENTRY lpColorTable, LPDIRECTDRAWPALETTE *lplpDDPalette, IUnknown *pUnkOuter) PURE;
    STDMETHOD(CreateSurface)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc, LPDIRECTDRAWSURFACE *lplpDDSurface, IUnknown *pUnkOuter) PURE;
    STDMETHOD(DuplicateSurface)(THIS_ LPDIRECTDRAWSURFACE lpDDSurface, LPDIRECTDRAWSURFACE *lplpDupDDSurface) PURE;
    STDMETHOD(EnumDisplayModes)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSurfaceDesc, LPVOID lpContext, LPDDENUMMODESCALLBACK lpEnumModesCallback) PURE;
    STDMETHOD(EnumSurfaces)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSD, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
    STDMETHOD(FlipToGDISurface)(THIS) PURE;
    STDMETHOD(GetCaps)(THIS_ LPDDCAPS lpDDDriverCaps, LPDDCAPS lpDDHELCaps) PURE;
    STDMETHOD(GetDisplayMode)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
    STDMETHOD(GetFourCCCodes)(THIS_ LPDWORD lpNumCodes, LPDWORD lpCodes) PURE;
    STDMETHOD(GetGDISurface)(THIS_ LPDIRECTDRAWSURFACE *lplpGDIDDSurface) PURE;
    STDMETHOD(GetMonitorFrequency)(THIS_ LPDWORD lpdwFrequency) PURE;
    STDMETHOD(GetScanLine)(THIS_ LPDWORD lpdwScanLine) PURE;
    STDMETHOD(GetVerticalBlankStatus)(THIS_ BOOL *lpbIsInVB) PURE;
    STDMETHOD(Initialize)(THIS_ GUID *lpGUID) PURE;
    STDMETHOD(RestoreDisplayMode)(THIS) PURE;
    STDMETHOD(SetCooperativeLevel)(THIS_ HWND hWnd, DWORD dwFlags) PURE;
    STDMETHOD(SetDisplayMode)(THIS_ DWORD dwWidth, DWORD dwHeight, DWORD dwBPP) PURE;
    STDMETHOD(WaitForVerticalBlank)(THIS_ DWORD dwFlags, HANDLE hEvent) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDraw_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDraw_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDraw_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDraw methods ***/
#define IDirectDraw_Compact(p)                  (p)->lpVtbl->Compact(p)
#define IDirectDraw_CreateClipper(p,a,b,c)      (p)->lpVtbl->CreateClipper(p,a,b,c)
#define IDirectDraw_CreatePalette(p,a,b,c,d)    (p)->lpVtbl->CreatePalette(p,a,b,c,d)
#define IDirectDraw_CreateSurface(p,a,b,c)      (p)->lpVtbl->CreateSurface(p,a,b,c)
#define IDirectDraw_DuplicateSurface(p,a,b)     (p)->lpVtbl->DuplicateSurface(p,a,b)
#define IDirectDraw_EnumDisplayModes(p,a,b,c,d) (p)->lpVtbl->EnumDisplayModes(p,a,b,c,d)
#define IDirectDraw_EnumSurfaces(p,a,b,c,d)     (p)->lpVtbl->EnumSurfaces(p,a,b,c,d)
#define IDirectDraw_FlipToGDISurface(p)         (p)->lpVtbl->FlipToGDISurface(p)
#define IDirectDraw_GetCaps(p,a,b)              (p)->lpVtbl->GetCaps(p,a,b)
#define IDirectDraw_GetDisplayMode(p,a)         (p)->lpVtbl->GetDisplayMode(p,a)
#define IDirectDraw_GetFourCCCodes(p,a,b)       (p)->lpVtbl->GetFourCCCodes(p,a,b)
#define IDirectDraw_GetGDISurface(p,a)          (p)->lpVtbl->GetGDISurface(p,a)
#define IDirectDraw_GetMonitorFrequency(p,a)    (p)->lpVtbl->GetMonitorFrequency(p,a)
#define IDirectDraw_GetScanLine(p,a)            (p)->lpVtbl->GetScanLine(p,a)
#define IDirectDraw_GetVerticalBlankStatus(p,a) (p)->lpVtbl->GetVerticalBlankStatus(p,a)
#define IDirectDraw_Initialize(p,a)             (p)->lpVtbl->Initialize(p,a)
#define IDirectDraw_RestoreDisplayMode(p)       (p)->lpVtbl->RestoreDisplayMode(p)
#define IDirectDraw_SetCooperativeLevel(p,a,b)  (p)->lpVtbl->SetCooperativeLevel(p,a,b)
#define IDirectDraw_SetDisplayMode(p,a,b,c)     (p)->lpVtbl->SetDisplayMode(p,a,b,c)
#define IDirectDraw_WaitForVerticalBlank(p,a,b) (p)->lpVtbl->WaitForVerticalBlank(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDraw_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDraw_AddRef(p)             (p)->AddRef()
#define IDirectDraw_Release(p)            (p)->Release()
/*** IDirectDraw methods ***/
#define IDirectDraw_Compact(p)                  (p)->Compact()
#define IDirectDraw_CreateClipper(p,a,b,c)      (p)->CreateClipper(a,b,c)
#define IDirectDraw_CreatePalette(p,a,b,c,d)    (p)->CreatePalette(a,b,c,d)
#define IDirectDraw_CreateSurface(p,a,b,c)      (p)->CreateSurface(a,b,c)
#define IDirectDraw_DuplicateSurface(p,a,b)     (p)->DuplicateSurface(a,b)
#define IDirectDraw_EnumDisplayModes(p,a,b,c,d) (p)->EnumDisplayModes(a,b,c,d)
#define IDirectDraw_EnumSurfaces(p,a,b,c,d)     (p)->EnumSurfaces(a,b,c,d)
#define IDirectDraw_FlipToGDISurface(p)         (p)->FlipToGDISurface()
#define IDirectDraw_GetCaps(p,a,b)              (p)->GetCaps(a,b)
#define IDirectDraw_GetDisplayMode(p,a)         (p)->GetDisplayMode(a)
#define IDirectDraw_GetFourCCCodes(p,a,b)       (p)->GetFourCCCodes(a,b)
#define IDirectDraw_GetGDISurface(p,a)          (p)->GetGDISurface(a)
#define IDirectDraw_GetMonitorFrequency(p,a)    (p)->GetMonitorFrequency(a)
#define IDirectDraw_GetScanLine(p,a)            (p)->GetScanLine(a)
#define IDirectDraw_GetVerticalBlankStatus(p,a) (p)->GetVerticalBlankStatus(a)
#define IDirectDraw_Initialize(p,a)             (p)->Initialize(a)
#define IDirectDraw_RestoreDisplayMode(p)       (p)->RestoreDisplayMode()
#define IDirectDraw_SetCooperativeLevel(p,a,b)  (p)->SetCooperativeLevel(a,b)
#define IDirectDraw_SetDisplayMode(p,a,b,c)     (p)->SetDisplayMode(a,b,c)
#define IDirectDraw_WaitForVerticalBlank(p,a,b) (p)->WaitForVerticalBlank(a,b)
#endif


/* flags for Lock() */
#define DDLOCK_SURFACEMEMORYPTR	0x00000000
#define DDLOCK_WAIT		0x00000001
#define DDLOCK_EVENT		0x00000002
#define DDLOCK_READONLY		0x00000010
#define DDLOCK_WRITEONLY	0x00000020
#define DDLOCK_NOSYSLOCK	0x00000800
#define DDLOCK_NOOVERWRITE      0x00001000
#define DDLOCK_DISCARDCONTENTS  0x00002000


/*****************************************************************************
 * IDirectDraw2 interface
 */
/* Note: IDirectDraw2 cannot derive from IDirectDraw because the number of
 * arguments of SetDisplayMode has changed !
 */
#define INTERFACE IDirectDraw2
DECLARE_INTERFACE_(IDirectDraw2,IUnknown)
{
          /*** IUnknown methods ***/
/*00*/    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
/*04*/    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
/*08*/    STDMETHOD_(ULONG,Release)(THIS) PURE;
          /*** IDirectDraw2 methods ***/
/*0c*/    STDMETHOD(Compact)(THIS) PURE;
/*10*/    STDMETHOD(CreateClipper)(THIS_ DWORD dwFlags, LPDIRECTDRAWCLIPPER *lplpDDClipper, IUnknown *pUnkOuter) PURE;
/*14*/    STDMETHOD(CreatePalette)(THIS_ DWORD dwFlags, LPPALETTEENTRY lpColorTable, LPDIRECTDRAWPALETTE *lplpDDPalette, IUnknown *pUnkOuter) PURE;
/*18*/    STDMETHOD(CreateSurface)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc, LPDIRECTDRAWSURFACE *lplpDDSurface, IUnknown *pUnkOuter) PURE;
/*1c*/    STDMETHOD(DuplicateSurface)(THIS_ LPDIRECTDRAWSURFACE lpDDSurface, LPDIRECTDRAWSURFACE *lplpDupDDSurface) PURE;
/*20*/    STDMETHOD(EnumDisplayModes)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSurfaceDesc, LPVOID lpContext, LPDDENUMMODESCALLBACK lpEnumModesCallback) PURE;
/*24*/    STDMETHOD(EnumSurfaces)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSD, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
/*28*/    STDMETHOD(FlipToGDISurface)(THIS) PURE;
/*2c*/    STDMETHOD(GetCaps)(THIS_ LPDDCAPS lpDDDriverCaps, LPDDCAPS lpDDHELCaps) PURE;
/*30*/    STDMETHOD(GetDisplayMode)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
/*34*/    STDMETHOD(GetFourCCCodes)(THIS_ LPDWORD lpNumCodes, LPDWORD lpCodes) PURE;
/*38*/    STDMETHOD(GetGDISurface)(THIS_ LPDIRECTDRAWSURFACE *lplpGDIDDSurface) PURE;
/*3c*/    STDMETHOD(GetMonitorFrequency)(THIS_ LPDWORD lpdwFrequency) PURE;
/*40*/    STDMETHOD(GetScanLine)(THIS_ LPDWORD lpdwScanLine) PURE;
/*44*/    STDMETHOD(GetVerticalBlankStatus)(THIS_ BOOL *lpbIsInVB) PURE;
/*48*/    STDMETHOD(Initialize)(THIS_ GUID *lpGUID) PURE;
/*4c*/    STDMETHOD(RestoreDisplayMode)(THIS) PURE;
/*50*/    STDMETHOD(SetCooperativeLevel)(THIS_ HWND hWnd, DWORD dwFlags) PURE;
/*54*/    STDMETHOD(SetDisplayMode)(THIS_ DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwRefreshRate, DWORD dwFlags) PURE;
/*58*/    STDMETHOD(WaitForVerticalBlank)(THIS_ DWORD dwFlags, HANDLE hEvent) PURE;
          /* added in v2 */
/*5c*/    STDMETHOD(GetAvailableVidMem)(THIS_ LPDDSCAPS lpDDCaps, LPDWORD lpdwTotal, LPDWORD lpdwFree) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDraw2_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDraw2_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDraw2_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDraw methods ***/
#define IDirectDraw2_Compact(p)                  (p)->lpVtbl->Compact(p)
#define IDirectDraw2_CreateClipper(p,a,b,c)      (p)->lpVtbl->CreateClipper(p,a,b,c)
#define IDirectDraw2_CreatePalette(p,a,b,c,d)    (p)->lpVtbl->CreatePalette(p,a,b,c,d)
#define IDirectDraw2_CreateSurface(p,a,b,c)      (p)->lpVtbl->CreateSurface(p,a,b,c)
#define IDirectDraw2_DuplicateSurface(p,a,b)     (p)->lpVtbl->DuplicateSurface(p,a,b)
#define IDirectDraw2_EnumDisplayModes(p,a,b,c,d) (p)->lpVtbl->EnumDisplayModes(p,a,b,c,d)
#define IDirectDraw2_EnumSurfaces(p,a,b,c,d)     (p)->lpVtbl->EnumSurfaces(p,a,b,c,d)
#define IDirectDraw2_FlipToGDISurface(p)         (p)->lpVtbl->FlipToGDISurface(p)
#define IDirectDraw2_GetCaps(p,a,b)              (p)->lpVtbl->GetCaps(p,a,b)
#define IDirectDraw2_GetDisplayMode(p,a)         (p)->lpVtbl->GetDisplayMode(p,a)
#define IDirectDraw2_GetFourCCCodes(p,a,b)       (p)->lpVtbl->GetFourCCCodes(p,a,b)
#define IDirectDraw2_GetGDISurface(p,a)          (p)->lpVtbl->GetGDISurface(p,a)
#define IDirectDraw2_GetMonitorFrequency(p,a)    (p)->lpVtbl->GetMonitorFrequency(p,a)
#define IDirectDraw2_GetScanLine(p,a)            (p)->lpVtbl->GetScanLine(p,a)
#define IDirectDraw2_GetVerticalBlankStatus(p,a) (p)->lpVtbl->GetVerticalBlankStatus(p,a)
#define IDirectDraw2_Initialize(p,a)             (p)->lpVtbl->Initialize(p,a)
#define IDirectDraw2_RestoreDisplayMode(p)       (p)->lpVtbl->RestoreDisplayMode(p)
#define IDirectDraw2_SetCooperativeLevel(p,a,b)  (p)->lpVtbl->SetCooperativeLevel(p,a,b)
#define IDirectDraw2_SetDisplayMode(p,a,b,c,d,e) (p)->lpVtbl->SetDisplayMode(p,a,b,c,d,e)
#define IDirectDraw2_WaitForVerticalBlank(p,a,b) (p)->lpVtbl->WaitForVerticalBlank(p,a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw2_GetAvailableVidMem(p,a,b,c) (p)->lpVtbl->GetAvailableVidMem(p,a,b,c)
#else
/*** IUnknown methods ***/
#define IDirectDraw2_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDraw2_AddRef(p)             (p)->AddRef()
#define IDirectDraw2_Release(p)            (p)->Release()
/*** IDirectDraw methods ***/
#define IDirectDraw2_Compact(p)                  (p)->Compact()
#define IDirectDraw2_CreateClipper(p,a,b,c)      (p)->CreateClipper(a,b,c)
#define IDirectDraw2_CreatePalette(p,a,b,c,d)    (p)->CreatePalette(a,b,c,d)
#define IDirectDraw2_CreateSurface(p,a,b,c)      (p)->CreateSurface(a,b,c)
#define IDirectDraw2_DuplicateSurface(p,a,b)     (p)->DuplicateSurface(a,b)
#define IDirectDraw2_EnumDisplayModes(p,a,b,c,d) (p)->EnumDisplayModes(a,b,c,d)
#define IDirectDraw2_EnumSurfaces(p,a,b,c,d)     (p)->EnumSurfaces(a,b,c,d)
#define IDirectDraw2_FlipToGDISurface(p)         (p)->FlipToGDISurface()
#define IDirectDraw2_GetCaps(p,a,b)              (p)->GetCaps(a,b)
#define IDirectDraw2_GetDisplayMode(p,a)         (p)->GetDisplayMode(a)
#define IDirectDraw2_GetFourCCCodes(p,a,b)       (p)->GetFourCCCodes(a,b)
#define IDirectDraw2_GetGDISurface(p,a)          (p)->GetGDISurface(a)
#define IDirectDraw2_GetMonitorFrequency(p,a)    (p)->GetMonitorFrequency(a)
#define IDirectDraw2_GetScanLine(p,a)            (p)->GetScanLine(a)
#define IDirectDraw2_GetVerticalBlankStatus(p,a) (p)->GetVerticalBlankStatus(a)
#define IDirectDraw2_Initialize(p,a)             (p)->Initialize(a)
#define IDirectDraw2_RestoreDisplayMode(p)       (p)->RestoreDisplayMode()
#define IDirectDraw2_SetCooperativeLevel(p,a,b)  (p)->SetCooperativeLevel(a,b)
#define IDirectDraw2_SetDisplayMode(p,a,b,c,d,e) (p)->SetDisplayMode(a,b,c,d,e)
#define IDirectDraw2_WaitForVerticalBlank(p,a,b) (p)->WaitForVerticalBlank(a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw2_GetAvailableVidMem(p,a,b,c) (p)->GetAvailableVidMem(a,b,c)
#endif


/*****************************************************************************
 * IDirectDraw3 interface
 */
#define INTERFACE IDirectDraw3
DECLARE_INTERFACE_(IDirectDraw3,IUnknown)
{
          /*** IUnknown methods ***/
/*00*/    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
/*04*/    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
/*08*/    STDMETHOD_(ULONG,Release)(THIS) PURE;
          /*** IDirectDraw2 methods ***/
/*0c*/    STDMETHOD(Compact)(THIS) PURE;
/*10*/    STDMETHOD(CreateClipper)(THIS_ DWORD dwFlags, LPDIRECTDRAWCLIPPER *lplpDDClipper, IUnknown *pUnkOuter) PURE;
/*14*/    STDMETHOD(CreatePalette)(THIS_ DWORD dwFlags, LPPALETTEENTRY lpColorTable, LPDIRECTDRAWPALETTE *lplpDDPalette, IUnknown *pUnkOuter) PURE;
/*18*/    STDMETHOD(CreateSurface)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc, LPDIRECTDRAWSURFACE *lplpDDSurface, IUnknown *pUnkOuter) PURE;
/*1c*/    STDMETHOD(DuplicateSurface)(THIS_ LPDIRECTDRAWSURFACE lpDDSurface, LPDIRECTDRAWSURFACE *lplpDupDDSurface) PURE;
/*20*/    STDMETHOD(EnumDisplayModes)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSurfaceDesc, LPVOID lpContext, LPDDENUMMODESCALLBACK lpEnumModesCallback) PURE;
/*24*/    STDMETHOD(EnumSurfaces)(THIS_ DWORD dwFlags, LPDDSURFACEDESC lpDDSD, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
/*28*/    STDMETHOD(FlipToGDISurface)(THIS) PURE;
/*2c*/    STDMETHOD(GetCaps)(THIS_ LPDDCAPS lpDDDriverCaps, LPDDCAPS lpDDHELCaps) PURE;
/*30*/    STDMETHOD(GetDisplayMode)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
/*34*/    STDMETHOD(GetFourCCCodes)(THIS_ LPDWORD lpNumCodes, LPDWORD lpCodes) PURE;
/*38*/    STDMETHOD(GetGDISurface)(THIS_ LPDIRECTDRAWSURFACE *lplpGDIDDSurface) PURE;
/*3c*/    STDMETHOD(GetMonitorFrequency)(THIS_ LPDWORD lpdwFrequency) PURE;
/*40*/    STDMETHOD(GetScanLine)(THIS_ LPDWORD lpdwScanLine) PURE;
/*44*/    STDMETHOD(GetVerticalBlankStatus)(THIS_ BOOL *lpbIsInVB) PURE;
/*48*/    STDMETHOD(Initialize)(THIS_ GUID *lpGUID) PURE;
/*4c*/    STDMETHOD(RestoreDisplayMode)(THIS) PURE;
/*50*/    STDMETHOD(SetCooperativeLevel)(THIS_ HWND hWnd, DWORD dwFlags) PURE;
/*54*/    STDMETHOD(SetDisplayMode)(THIS_ DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwRefreshRate, DWORD dwFlags) PURE;
/*58*/    STDMETHOD(WaitForVerticalBlank)(THIS_ DWORD dwFlags, HANDLE hEvent) PURE;
          /* added in v2 */
/*5c*/    STDMETHOD(GetAvailableVidMem)(THIS_ LPDDSCAPS lpDDCaps, LPDWORD lpdwTotal, LPDWORD lpdwFree) PURE;
          /* added in v3 */
/*60*/    STDMETHOD(GetSurfaceFromDC)(THIS_ HDC hdc, LPDIRECTDRAWSURFACE *pSurf) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDraw3_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDraw3_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDraw3_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDraw methods ***/
#define IDirectDraw3_Compact(p)                  (p)->lpVtbl->Compact(p)
#define IDirectDraw3_CreateClipper(p,a,b,c)      (p)->lpVtbl->CreateClipper(p,a,b,c)
#define IDirectDraw3_CreatePalette(p,a,b,c,d)    (p)->lpVtbl->CreatePalette(p,a,b,c,d)
#define IDirectDraw3_CreateSurface(p,a,b,c)      (p)->lpVtbl->CreateSurface(p,a,b,c)
#define IDirectDraw3_DuplicateSurface(p,a,b)     (p)->lpVtbl->DuplicateSurface(p,a,b)
#define IDirectDraw3_EnumDisplayModes(p,a,b,c,d) (p)->lpVtbl->EnumDisplayModes(p,a,b,c,d)
#define IDirectDraw3_EnumSurfaces(p,a,b,c,d)     (p)->lpVtbl->EnumSurfaces(p,a,b,c,d)
#define IDirectDraw3_FlipToGDISurface(p)         (p)->lpVtbl->FlipToGDISurface(p)
#define IDirectDraw3_GetCaps(p,a,b)              (p)->lpVtbl->GetCaps(p,a,b)
#define IDirectDraw3_GetDisplayMode(p,a)         (p)->lpVtbl->GetDisplayMode(p,a)
#define IDirectDraw3_GetFourCCCodes(p,a,b)       (p)->lpVtbl->GetFourCCCodes(p,a,b)
#define IDirectDraw3_GetGDISurface(p,a)          (p)->lpVtbl->GetGDISurface(p,a)
#define IDirectDraw3_GetMonitorFrequency(p,a)    (p)->lpVtbl->GetMonitorFrequency(p,a)
#define IDirectDraw3_GetScanLine(p,a)            (p)->lpVtbl->GetScanLine(p,a)
#define IDirectDraw3_GetVerticalBlankStatus(p,a) (p)->lpVtbl->GetVerticalBlankStatus(p,a)
#define IDirectDraw3_Initialize(p,a)             (p)->lpVtbl->Initialize(p,a)
#define IDirectDraw3_RestoreDisplayMode(p)       (p)->lpVtbl->RestoreDisplayMode(p)
#define IDirectDraw3_SetCooperativeLevel(p,a,b)  (p)->lpVtbl->SetCooperativeLevel(p,a,b)
#define IDirectDraw3_SetDisplayMode(p,a,b,c,d,e) (p)->lpVtbl->SetDisplayMode(p,a,b,c,d,e)
#define IDirectDraw3_WaitForVerticalBlank(p,a,b) (p)->lpVtbl->WaitForVerticalBlank(p,a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw3_GetAvailableVidMem(p,a,b,c) (p)->lpVtbl->GetAvailableVidMem(p,a,b,c)
/*** IDirectDraw3 methods ***/
#define IDirectDraw3_GetSurfaceFromDC(p,a,b)    (p)->lpVtbl->GetSurfaceFromDC(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDraw3_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDraw3_AddRef(p)             (p)->AddRef()
#define IDirectDraw3_Release(p)            (p)->Release()
/*** IDirectDraw methods ***/
#define IDirectDraw3_Compact(p)                  (p)->Compact()
#define IDirectDraw3_CreateClipper(p,a,b,c)      (p)->CreateClipper(a,b,c)
#define IDirectDraw3_CreatePalette(p,a,b,c,d)    (p)->CreatePalette(a,b,c,d)
#define IDirectDraw3_CreateSurface(p,a,b,c)      (p)->CreateSurface(a,b,c)
#define IDirectDraw3_DuplicateSurface(p,a,b)     (p)->DuplicateSurface(a,b)
#define IDirectDraw3_EnumDisplayModes(p,a,b,c,d) (p)->EnumDisplayModes(a,b,c,d)
#define IDirectDraw3_EnumSurfaces(p,a,b,c,d)     (p)->EnumSurfaces(a,b,c,d)
#define IDirectDraw3_FlipToGDISurface(p)         (p)->FlipToGDISurface()
#define IDirectDraw3_GetCaps(p,a,b)              (p)->GetCaps(a,b)
#define IDirectDraw3_GetDisplayMode(p,a)         (p)->GetDisplayMode(a)
#define IDirectDraw3_GetFourCCCodes(p,a,b)       (p)->GetFourCCCodes(a,b)
#define IDirectDraw3_GetGDISurface(p,a)          (p)->GetGDISurface(a)
#define IDirectDraw3_GetMonitorFrequency(p,a)    (p)->GetMonitorFrequency(a)
#define IDirectDraw3_GetScanLine(p,a)            (p)->GetScanLine(a)
#define IDirectDraw3_GetVerticalBlankStatus(p,a) (p)->GetVerticalBlankStatus(a)
#define IDirectDraw3_Initialize(p,a)             (p)->Initialize(a)
#define IDirectDraw3_RestoreDisplayMode(p)       (p)->RestoreDisplayMode()
#define IDirectDraw3_SetCooperativeLevel(p,a,b)  (p)->SetCooperativeLevel(a,b)
#define IDirectDraw3_SetDisplayMode(p,a,b,c,d,e) (p)->SetDisplayMode(a,b,c,d,e)
#define IDirectDraw3_WaitForVerticalBlank(p,a,b) (p)->WaitForVerticalBlank(a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw3_GetAvailableVidMem(p,a,b,c) (p)->GetAvailableVidMem(a,b,c)
/*** IDirectDraw3 methods ***/
#define IDirectDraw3_GetSurfaceFromDC(p,a,b)    (p)->GetSurfaceFromDC(a,b)
#endif


/*****************************************************************************
 * IDirectDraw4 interface
 */
#define INTERFACE IDirectDraw4
DECLARE_INTERFACE_(IDirectDraw4,IUnknown)
{
          /*** IUnknown methods ***/
/*00*/    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
/*04*/    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
/*08*/    STDMETHOD_(ULONG,Release)(THIS) PURE;
          /*** IDirectDraw4 methods ***/
/*0c*/    STDMETHOD(Compact)(THIS) PURE;
/*10*/    STDMETHOD(CreateClipper)(THIS_ DWORD dwFlags, LPDIRECTDRAWCLIPPER *lplpDDClipper, IUnknown *pUnkOuter) PURE;
/*14*/    STDMETHOD(CreatePalette)(THIS_ DWORD dwFlags, LPPALETTEENTRY lpColorTable, LPDIRECTDRAWPALETTE *lplpDDPalette, IUnknown *pUnkOuter) PURE;
/*18*/    STDMETHOD(CreateSurface)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc, LPDIRECTDRAWSURFACE4 *lplpDDSurface, IUnknown *pUnkOuter) PURE;
/*1c*/    STDMETHOD(DuplicateSurface)(THIS_ LPDIRECTDRAWSURFACE4 lpDDSurface, LPDIRECTDRAWSURFACE4 *lplpDupDDSurface) PURE;
/*20*/    STDMETHOD(EnumDisplayModes)(THIS_ DWORD dwFlags, LPDDSURFACEDESC2 lpDDSurfaceDesc, LPVOID lpContext, LPDDENUMMODESCALLBACK2 lpEnumModesCallback) PURE;
/*24*/    STDMETHOD(EnumSurfaces)(THIS_ DWORD dwFlags, LPDDSURFACEDESC2 lpDDSD, LPVOID lpContext, LPDDENUMSURFACESCALLBACK2 lpEnumSurfacesCallback) PURE;
/*28*/    STDMETHOD(FlipToGDISurface)(THIS) PURE;
/*2c*/    STDMETHOD(GetCaps)(THIS_ LPDDCAPS lpDDDriverCaps, LPDDCAPS lpDDHELCaps) PURE;
/*30*/    STDMETHOD(GetDisplayMode)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
/*34*/    STDMETHOD(GetFourCCCodes)(THIS_ LPDWORD lpNumCodes, LPDWORD lpCodes) PURE;
/*38*/    STDMETHOD(GetGDISurface)(THIS_ LPDIRECTDRAWSURFACE4 *lplpGDIDDSurface) PURE;
/*3c*/    STDMETHOD(GetMonitorFrequency)(THIS_ LPDWORD lpdwFrequency) PURE;
/*40*/    STDMETHOD(GetScanLine)(THIS_ LPDWORD lpdwScanLine) PURE;
/*44*/    STDMETHOD(GetVerticalBlankStatus)(THIS_ BOOL *lpbIsInVB) PURE;
/*48*/    STDMETHOD(Initialize)(THIS_ GUID *lpGUID) PURE;
/*4c*/    STDMETHOD(RestoreDisplayMode)(THIS) PURE;
/*50*/    STDMETHOD(SetCooperativeLevel)(THIS_ HWND hWnd, DWORD dwFlags) PURE;
/*54*/    STDMETHOD(SetDisplayMode)(THIS_ DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwRefreshRate, DWORD dwFlags) PURE;
/*58*/    STDMETHOD(WaitForVerticalBlank)(THIS_ DWORD dwFlags, HANDLE hEvent) PURE;
          /* added in v2 */
/*5c*/    STDMETHOD(GetAvailableVidMem)(THIS_ LPDDSCAPS2 lpDDCaps, LPDWORD lpdwTotal, LPDWORD lpdwFree) PURE;
          /* added in v4 */
/*60*/    STDMETHOD(GetSurfaceFromDC)(THIS_ HDC hdc, LPDIRECTDRAWSURFACE4 *pSurf) PURE;
/*64*/    STDMETHOD(RestoreAllSurfaces)(THIS) PURE;
/*68*/    STDMETHOD(TestCooperativeLevel)(THIS) PURE;
/*6c*/    STDMETHOD(GetDeviceIdentifier)(THIS_ LPDDDEVICEIDENTIFIER pDDDI, DWORD dwFlags) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDraw4_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDraw4_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDraw4_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDraw methods ***/
#define IDirectDraw4_Compact(p)                  (p)->lpVtbl->Compact(p)
#define IDirectDraw4_CreateClipper(p,a,b,c)      (p)->lpVtbl->CreateClipper(p,a,b,c)
#define IDirectDraw4_CreatePalette(p,a,b,c,d)    (p)->lpVtbl->CreatePalette(p,a,b,c,d)
#define IDirectDraw4_CreateSurface(p,a,b,c)      (p)->lpVtbl->CreateSurface(p,a,b,c)
#define IDirectDraw4_DuplicateSurface(p,a,b)     (p)->lpVtbl->DuplicateSurface(p,a,b)
#define IDirectDraw4_EnumDisplayModes(p,a,b,c,d) (p)->lpVtbl->EnumDisplayModes(p,a,b,c,d)
#define IDirectDraw4_EnumSurfaces(p,a,b,c,d)     (p)->lpVtbl->EnumSurfaces(p,a,b,c,d)
#define IDirectDraw4_FlipToGDISurface(p)         (p)->lpVtbl->FlipToGDISurface(p)
#define IDirectDraw4_GetCaps(p,a,b)              (p)->lpVtbl->GetCaps(p,a,b)
#define IDirectDraw4_GetDisplayMode(p,a)         (p)->lpVtbl->GetDisplayMode(p,a)
#define IDirectDraw4_GetFourCCCodes(p,a,b)       (p)->lpVtbl->GetFourCCCodes(p,a,b)
#define IDirectDraw4_GetGDISurface(p,a)          (p)->lpVtbl->GetGDISurface(p,a)
#define IDirectDraw4_GetMonitorFrequency(p,a)    (p)->lpVtbl->GetMonitorFrequency(p,a)
#define IDirectDraw4_GetScanLine(p,a)            (p)->lpVtbl->GetScanLine(p,a)
#define IDirectDraw4_GetVerticalBlankStatus(p,a) (p)->lpVtbl->GetVerticalBlankStatus(p,a)
#define IDirectDraw4_Initialize(p,a)             (p)->lpVtbl->Initialize(p,a)
#define IDirectDraw4_RestoreDisplayMode(p)       (p)->lpVtbl->RestoreDisplayMode(p)
#define IDirectDraw4_SetCooperativeLevel(p,a,b)  (p)->lpVtbl->SetCooperativeLevel(p,a,b)
#define IDirectDraw4_SetDisplayMode(p,a,b,c,d,e) (p)->lpVtbl->SetDisplayMode(p,a,b,c,d,e)
#define IDirectDraw4_WaitForVerticalBlank(p,a,b) (p)->lpVtbl->WaitForVerticalBlank(p,a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw4_GetAvailableVidMem(p,a,b,c) (p)->lpVtbl->GetAvailableVidMem(p,a,b,c)
/*** IDirectDraw4 methods ***/
#define IDirectDraw4_GetSurfaceFromDC(p,a,b)    (p)->lpVtbl->GetSurfaceFromDC(p,a,b)
#define IDirectDraw4_RestoreAllSurfaces(p)      (p)->lpVtbl->RestoreAllSurfaces(p)
#define IDirectDraw4_TestCooperativeLevel(p)    (p)->lpVtbl->TestCooperativeLevel(p)
#define IDirectDraw4_GetDeviceIdentifier(p,a,b) (p)->lpVtbl->GetDeviceIdentifier(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDraw4_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDraw4_AddRef(p)             (p)->AddRef()
#define IDirectDraw4_Release(p)            (p)->Release()
/*** IDirectDraw methods ***/
#define IDirectDraw4_Compact(p)                  (p)->Compact()
#define IDirectDraw4_CreateClipper(p,a,b,c)      (p)->CreateClipper(a,b,c)
#define IDirectDraw4_CreatePalette(p,a,b,c,d)    (p)->CreatePalette(a,b,c,d)
#define IDirectDraw4_CreateSurface(p,a,b,c)      (p)->CreateSurface(a,b,c)
#define IDirectDraw4_DuplicateSurface(p,a,b)     (p)->DuplicateSurface(a,b)
#define IDirectDraw4_EnumDisplayModes(p,a,b,c,d) (p)->EnumDisplayModes(a,b,c,d)
#define IDirectDraw4_EnumSurfaces(p,a,b,c,d)     (p)->EnumSurfaces(a,b,c,d)
#define IDirectDraw4_FlipToGDISurface(p)         (p)->FlipToGDISurface()
#define IDirectDraw4_GetCaps(p,a,b)              (p)->GetCaps(a,b)
#define IDirectDraw4_GetDisplayMode(p,a)         (p)->GetDisplayMode(a)
#define IDirectDraw4_GetFourCCCodes(p,a,b)       (p)->GetFourCCCodes(a,b)
#define IDirectDraw4_GetGDISurface(p,a)          (p)->GetGDISurface(a)
#define IDirectDraw4_GetMonitorFrequency(p,a)    (p)->GetMonitorFrequency(a)
#define IDirectDraw4_GetScanLine(p,a)            (p)->GetScanLine(a)
#define IDirectDraw4_GetVerticalBlankStatus(p,a) (p)->GetVerticalBlankStatus(a)
#define IDirectDraw4_Initialize(p,a)             (p)->Initialize(a)
#define IDirectDraw4_RestoreDisplayMode(p)       (p)->RestoreDisplayMode()
#define IDirectDraw4_SetCooperativeLevel(p,a,b)  (p)->SetCooperativeLevel(a,b)
#define IDirectDraw4_SetDisplayMode(p,a,b,c,d,e) (p)->SetDisplayMode(a,b,c,d,e)
#define IDirectDraw4_WaitForVerticalBlank(p,a,b) (p)->WaitForVerticalBlank(a,b)
/*** IDirectDraw2 methods ***/
#define IDirectDraw4_GetAvailableVidMem(p,a,b,c) (p)->GetAvailableVidMem(a,b,c)
/*** IDirectDraw4 methods ***/
#define IDirectDraw4_GetSurfaceFromDC(p,a,b)    (p)->GetSurfaceFromDC(a,b)
#define IDirectDraw4_RestoreAllSurfaces(pc)     (p)->RestoreAllSurfaces()
#define IDirectDraw4_TestCooperativeLevel(p)    (p)->TestCooperativeLevel()
#define IDirectDraw4_GetDeviceIdentifier(p,a,b) (p)->GetDeviceIdentifier(a,b)
#endif


/*****************************************************************************
 * IDirectDraw7 interface
 */
/* Note: IDirectDraw7 cannot derive from IDirectDraw4; it is even documented
 * as not interchangeable with earlier DirectDraw interfaces.
 */
#define INTERFACE IDirectDraw7
DECLARE_INTERFACE_(IDirectDraw7,IUnknown)
{
          /*** IUnknown methods ***/
/*00*/    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
/*04*/    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
/*08*/    STDMETHOD_(ULONG,Release)(THIS) PURE;
          /*** IDirectDraw7 methods ***/
/*0c*/    STDMETHOD(Compact)(THIS) PURE;
/*10*/    STDMETHOD(CreateClipper)(THIS_ DWORD dwFlags, LPDIRECTDRAWCLIPPER *lplpDDClipper, IUnknown *pUnkOuter) PURE;
/*14*/    STDMETHOD(CreatePalette)(THIS_ DWORD dwFlags, LPPALETTEENTRY lpColorTable, LPDIRECTDRAWPALETTE *lplpDDPalette, IUnknown *pUnkOuter) PURE;
/*18*/    STDMETHOD(CreateSurface)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc, LPDIRECTDRAWSURFACE7 *lplpDDSurface, IUnknown *pUnkOuter) PURE;
/*1c*/    STDMETHOD(DuplicateSurface)(THIS_ LPDIRECTDRAWSURFACE7 lpDDSurface, LPDIRECTDRAWSURFACE7 *lplpDupDDSurface) PURE;
/*20*/    STDMETHOD(EnumDisplayModes)(THIS_ DWORD dwFlags, LPDDSURFACEDESC2 lpDDSurfaceDesc, LPVOID lpContext, LPDDENUMMODESCALLBACK2 lpEnumModesCallback) PURE;
/*24*/    STDMETHOD(EnumSurfaces)(THIS_ DWORD dwFlags, LPDDSURFACEDESC2 lpDDSD, LPVOID lpContext, LPDDENUMSURFACESCALLBACK7 lpEnumSurfacesCallback) PURE;
/*28*/    STDMETHOD(FlipToGDISurface)(THIS) PURE;
/*2c*/    STDMETHOD(GetCaps)(THIS_ LPDDCAPS lpDDDriverCaps, LPDDCAPS lpDDHELCaps) PURE;
/*30*/    STDMETHOD(GetDisplayMode)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
/*34*/    STDMETHOD(GetFourCCCodes)(THIS_ LPDWORD lpNumCodes, LPDWORD lpCodes) PURE;
/*38*/    STDMETHOD(GetGDISurface)(THIS_ LPDIRECTDRAWSURFACE7 *lplpGDIDDSurface) PURE;
/*3c*/    STDMETHOD(GetMonitorFrequency)(THIS_ LPDWORD lpdwFrequency) PURE;
/*40*/    STDMETHOD(GetScanLine)(THIS_ LPDWORD lpdwScanLine) PURE;
/*44*/    STDMETHOD(GetVerticalBlankStatus)(THIS_ BOOL *lpbIsInVB) PURE;
/*48*/    STDMETHOD(Initialize)(THIS_ GUID *lpGUID) PURE;
/*4c*/    STDMETHOD(RestoreDisplayMode)(THIS) PURE;
/*50*/    STDMETHOD(SetCooperativeLevel)(THIS_ HWND hWnd, DWORD dwFlags) PURE;
/*54*/    STDMETHOD(SetDisplayMode)(THIS_ DWORD dwWidth, DWORD dwHeight, DWORD dwBPP, DWORD dwRefreshRate, DWORD dwFlags) PURE;
/*58*/    STDMETHOD(WaitForVerticalBlank)(THIS_ DWORD dwFlags, HANDLE hEvent) PURE;
          /* added in v2 */
/*5c*/    STDMETHOD(GetAvailableVidMem)(THIS_ LPDDSCAPS2 lpDDCaps, LPDWORD lpdwTotal, LPDWORD lpdwFree) PURE;
          /* added in v4 */
/*60*/    STDMETHOD(GetSurfaceFromDC)(THIS_ HDC hdc, LPDIRECTDRAWSURFACE7 *pSurf) PURE;
/*64*/    STDMETHOD(RestoreAllSurfaces)(THIS) PURE;
/*68*/    STDMETHOD(TestCooperativeLevel)(THIS) PURE;
/*6c*/    STDMETHOD(GetDeviceIdentifier)(THIS_ LPDDDEVICEIDENTIFIER2 pDDDI, DWORD dwFlags) PURE;
          /* added in v7 */
/*70*/    STDMETHOD(StartModeTest)(THIS_ LPSIZE pModes, DWORD dwNumModes, DWORD dwFlags) PURE;
/*74*/    STDMETHOD(EvaluateMode)(THIS_ DWORD dwFlags, DWORD  *pTimeout) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDraw7_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDraw7_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDraw7_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDraw methods ***/
#define IDirectDraw7_Compact(p)                  (p)->lpVtbl->Compact(p)
#define IDirectDraw7_CreateClipper(p,a,b,c)      (p)->lpVtbl->CreateClipper(p,a,b,c)
#define IDirectDraw7_CreatePalette(p,a,b,c,d)    (p)->lpVtbl->CreatePalette(p,a,b,c,d)
#define IDirectDraw7_CreateSurface(p,a,b,c)      (p)->lpVtbl->CreateSurface(p,a,b,c)
#define IDirectDraw7_DuplicateSurface(p,a,b)     (p)->lpVtbl->DuplicateSurface(p,a,b)
#define IDirectDraw7_EnumDisplayModes(p,a,b,c,d) (p)->lpVtbl->EnumDisplayModes(p,a,b,c,d)
#define IDirectDraw7_EnumSurfaces(p,a,b,c,d)     (p)->lpVtbl->EnumSurfaces(p,a,b,c,d)
#define IDirectDraw7_FlipToGDISurface(p)         (p)->lpVtbl->FlipToGDISurface(p)
#define IDirectDraw7_GetCaps(p,a,b)              (p)->lpVtbl->GetCaps(p,a,b)
#define IDirectDraw7_GetDisplayMode(p,a)         (p)->lpVtbl->GetDisplayMode(p,a)
#define IDirectDraw7_GetFourCCCodes(p,a,b)       (p)->lpVtbl->GetFourCCCodes(p,a,b)
#define IDirectDraw7_GetGDISurface(p,a)          (p)->lpVtbl->GetGDISurface(p,a)
#define IDirectDraw7_GetMonitorFrequency(p,a)    (p)->lpVtbl->GetMonitorFrequency(p,a)
#define IDirectDraw7_GetScanLine(p,a)            (p)->lpVtbl->GetScanLine(p,a)
#define IDirectDraw7_GetVerticalBlankStatus(p,a) (p)->lpVtbl->GetVerticalBlankStatus(p,a)
#define IDirectDraw7_Initialize(p,a)             (p)->lpVtbl->Initialize(p,a)
#define IDirectDraw7_RestoreDisplayMode(p)       (p)->lpVtbl->RestoreDisplayMode(p)
#define IDirectDraw7_SetCooperativeLevel(p,a,b)  (p)->lpVtbl->SetCooperativeLevel(p,a,b)
#define IDirectDraw7_SetDisplayMode(p,a,b,c,d,e) (p)->lpVtbl->SetDisplayMode(p,a,b,c,d,e)
#define IDirectDraw7_WaitForVerticalBlank(p,a,b) (p)->lpVtbl->WaitForVerticalBlank(p,a,b)
/*** added in IDirectDraw2 ***/
#define IDirectDraw7_GetAvailableVidMem(p,a,b,c) (p)->lpVtbl->GetAvailableVidMem(p,a,b,c)
/*** added in IDirectDraw4 ***/
#define IDirectDraw7_GetSurfaceFromDC(p,a,b)    (p)->lpVtbl->GetSurfaceFromDC(p,a,b)
#define IDirectDraw7_RestoreAllSurfaces(p)     (p)->lpVtbl->RestoreAllSurfaces(p)
#define IDirectDraw7_TestCooperativeLevel(p)    (p)->lpVtbl->TestCooperativeLevel(p)
#define IDirectDraw7_GetDeviceIdentifier(p,a,b) (p)->lpVtbl->GetDeviceIdentifier(p,a,b)
/*** added in IDirectDraw 7 ***/
#define IDirectDraw7_StartModeTest(p,a,b,c)     (p)->lpVtbl->StartModeTest(p,a,b,c)
#define IDirectDraw7_EvaluateMode(p,a,b)        (p)->lpVtbl->EvaluateMode(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDraw7_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDraw7_AddRef(p)             (p)->AddRef()
#define IDirectDraw7_Release(p)            (p)->Release()
/*** IDirectDraw methods ***/
#define IDirectDraw7_Compact(p)                  (p)->Compact()
#define IDirectDraw7_CreateClipper(p,a,b,c)      (p)->CreateClipper(a,b,c)
#define IDirectDraw7_CreatePalette(p,a,b,c,d)    (p)->CreatePalette(a,b,c,d)
#define IDirectDraw7_CreateSurface(p,a,b,c)      (p)->CreateSurface(a,b,c)
#define IDirectDraw7_DuplicateSurface(p,a,b)     (p)->DuplicateSurface(a,b)
#define IDirectDraw7_EnumDisplayModes(p,a,b,c,d) (p)->EnumDisplayModes(a,b,c,d)
#define IDirectDraw7_EnumSurfaces(p,a,b,c,d)     (p)->EnumSurfaces(a,b,c,d)
#define IDirectDraw7_FlipToGDISurface(p)         (p)->FlipToGDISurface()
#define IDirectDraw7_GetCaps(p,a,b)              (p)->GetCaps(a,b)
#define IDirectDraw7_GetDisplayMode(p,a)         (p)->GetDisplayMode(a)
#define IDirectDraw7_GetFourCCCodes(p,a,b)       (p)->GetFourCCCodes(a,b)
#define IDirectDraw7_GetGDISurface(p,a)          (p)->GetGDISurface(a)
#define IDirectDraw7_GetMonitorFrequency(p,a)    (p)->GetMonitorFrequency(a)
#define IDirectDraw7_GetScanLine(p,a)            (p)->GetScanLine(a)
#define IDirectDraw7_GetVerticalBlankStatus(p,a) (p)->GetVerticalBlankStatus(a)
#define IDirectDraw7_Initialize(p,a)             (p)->Initialize(a)
#define IDirectDraw7_RestoreDisplayMode(p)       (p)->RestoreDisplayMode()
#define IDirectDraw7_SetCooperativeLevel(p,a,b)  (p)->SetCooperativeLevel(a,b)
#define IDirectDraw7_SetDisplayMode(p,a,b,c,d,e) (p)->SetDisplayMode(a,b,c,d,e)
#define IDirectDraw7_WaitForVerticalBlank(p,a,b) (p)->WaitForVerticalBlank(a,b)
/*** added in IDirectDraw2 ***/
#define IDirectDraw7_GetAvailableVidMem(p,a,b,c) (p)->GetAvailableVidMem(a,b,c)
/*** added in IDirectDraw4 ***/
#define IDirectDraw7_GetSurfaceFromDC(p,a,b)    (p)->GetSurfaceFromDC(a,b)
#define IDirectDraw7_RestoreAllSurfaces(p)     (p)->RestoreAllSurfaces()
#define IDirectDraw7_TestCooperativeLevel(p)    (p)->TestCooperativeLevel()
#define IDirectDraw7_GetDeviceIdentifier(p,a,b) (p)->GetDeviceIdentifier(a,b)
/*** added in IDirectDraw 7 ***/
#define IDirectDraw7_StartModeTest(p,a,b,c)     (p)->StartModeTest(a,b,c)
#define IDirectDraw7_EvaluateMode(p,a,b)        (p)->EvaluateMode(a,b)
#endif


/*****************************************************************************
 * IDirectDrawSurface interface
 */
#define INTERFACE IDirectDrawSurface
DECLARE_INTERFACE_(IDirectDrawSurface,IUnknown)
{
          /*** IUnknown methods ***/
/*00*/    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
/*04*/    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
/*08*/    STDMETHOD_(ULONG,Release)(THIS) PURE;
          /*** IDirectDrawSurface methods ***/
/*0c*/    STDMETHOD(AddAttachedSurface)(THIS_ LPDIRECTDRAWSURFACE lpDDSAttachedSurface) PURE;
/*10*/    STDMETHOD(AddOverlayDirtyRect)(THIS_ LPRECT lpRect) PURE;
/*14*/    STDMETHOD(Blt)(THIS_ LPRECT lpDestRect, LPDIRECTDRAWSURFACE lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) PURE;
/*18*/    STDMETHOD(BltBatch)(THIS_ LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags) PURE;
/*1c*/    STDMETHOD(BltFast)(THIS_ DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwTrans) PURE;
/*20*/    STDMETHOD(DeleteAttachedSurface)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE lpDDSAttachedSurface) PURE;
/*24*/    STDMETHOD(EnumAttachedSurfaces)(THIS_ LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
/*28*/    STDMETHOD(EnumOverlayZOrders)(THIS_ DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpfnCallback) PURE;
/*2c*/    STDMETHOD(Flip)(THIS_ LPDIRECTDRAWSURFACE lpDDSurfaceTargetOverride, DWORD dwFlags) PURE;
/*30*/    STDMETHOD(GetAttachedSurface)(THIS_ LPDDSCAPS lpDDSCaps, LPDIRECTDRAWSURFACE *lplpDDAttachedSurface) PURE;
/*34*/    STDMETHOD(GetBltStatus)(THIS_ DWORD dwFlags) PURE;
/*38*/    STDMETHOD(GetCaps)(THIS_ LPDDSCAPS lpDDSCaps) PURE;
/*3c*/    STDMETHOD(GetClipper)(THIS_ LPDIRECTDRAWCLIPPER *lplpDDClipper) PURE;
/*40*/    STDMETHOD(GetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
/*44*/    STDMETHOD(GetDC)(THIS_ HDC *lphDC) PURE;
/*48*/    STDMETHOD(GetFlipStatus)(THIS_ DWORD dwFlags) PURE;
/*4c*/    STDMETHOD(GetOverlayPosition)(THIS_ LPLONG lplX, LPLONG lplY) PURE;
/*50*/    STDMETHOD(GetPalette)(THIS_ LPDIRECTDRAWPALETTE *lplpDDPalette) PURE;
/*54*/    STDMETHOD(GetPixelFormat)(THIS_ LPDDPIXELFORMAT lpDDPixelFormat) PURE;
/*58*/    STDMETHOD(GetSurfaceDesc)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
/*5c*/    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
/*60*/    STDMETHOD(IsLost)(THIS) PURE;
/*64*/    STDMETHOD(Lock)(THIS_ LPRECT lpDestRect, LPDDSURFACEDESC lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent) PURE;
/*68*/    STDMETHOD(ReleaseDC)(THIS_ HDC hDC) PURE;
/*6c*/    STDMETHOD(Restore)(THIS) PURE;
/*70*/    STDMETHOD(SetClipper)(THIS_ LPDIRECTDRAWCLIPPER lpDDClipper) PURE;
/*74*/    STDMETHOD(SetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
/*78*/    STDMETHOD(SetOverlayPosition)(THIS_ LONG lX, LONG lY) PURE;
/*7c*/    STDMETHOD(SetPalette)(THIS_ LPDIRECTDRAWPALETTE lpDDPalette) PURE;
/*80*/    STDMETHOD(Unlock)(THIS_ LPVOID lpSurfaceData) PURE;
/*84*/    STDMETHOD(UpdateOverlay)(THIS_ LPRECT lpSrcRect, LPDIRECTDRAWSURFACE lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx) PURE;
/*88*/    STDMETHOD(UpdateOverlayDisplay)(THIS_ DWORD dwFlags) PURE;
/*8c*/    STDMETHOD(UpdateOverlayZOrder)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE lpDDSReference) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawSurface_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawSurface_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawSurface_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawSurface methods ***/
#define IDirectDrawSurface_AddAttachedSurface(p,a)      (p)->lpVtbl->AddAttachedSurface(p,a)
#define IDirectDrawSurface_AddOverlayDirtyRect(p,a)     (p)->lpVtbl->AddOverlayDirtyRect(p,a)
#define IDirectDrawSurface_Blt(p,a,b,c,d,e)             (p)->lpVtbl->Blt(p,a,b,c,d,e)
#define IDirectDrawSurface_BltBatch(p,a,b,c)            (p)->lpVtbl->BltBatch(p,a,b,c)
#define IDirectDrawSurface_BltFast(p,a,b,c,d,e)         (p)->lpVtbl->BltFast(p,a,b,c,d,e)
#define IDirectDrawSurface_DeleteAttachedSurface(p,a,b) (p)->lpVtbl->DeleteAttachedSurface(p,a,b)
#define IDirectDrawSurface_EnumAttachedSurfaces(p,a,b)  (p)->lpVtbl->EnumAttachedSurfaces(p,a,b)
#define IDirectDrawSurface_EnumOverlayZOrders(p,a,b,c)  (p)->lpVtbl->EnumOverlayZOrders(p,a,b,c)
#define IDirectDrawSurface_Flip(p,a,b)                  (p)->lpVtbl->Flip(p,a,b)
#define IDirectDrawSurface_GetAttachedSurface(p,a,b)    (p)->lpVtbl->GetAttachedSurface(p,a,b)
#define IDirectDrawSurface_GetBltStatus(p,a)            (p)->lpVtbl->GetBltStatus(p,a)
#define IDirectDrawSurface_GetCaps(p,a)                 (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawSurface_GetClipper(p,a)              (p)->lpVtbl->GetClipper(p,a)
#define IDirectDrawSurface_GetColorKey(p,a,b)           (p)->lpVtbl->GetColorKey(p,a,b)
#define IDirectDrawSurface_GetDC(p,a)                   (p)->lpVtbl->GetDC(p,a)
#define IDirectDrawSurface_GetFlipStatus(p,a)           (p)->lpVtbl->GetFlipStatus(p,a)
#define IDirectDrawSurface_GetOverlayPosition(p,a,b)    (p)->lpVtbl->GetOverlayPosition(p,a,b)
#define IDirectDrawSurface_GetPalette(p,a)              (p)->lpVtbl->GetPalette(p,a)
#define IDirectDrawSurface_GetPixelFormat(p,a)          (p)->lpVtbl->GetPixelFormat(p,a)
#define IDirectDrawSurface_GetSurfaceDesc(p,a)          (p)->lpVtbl->GetSurfaceDesc(p,a)
#define IDirectDrawSurface_Initialize(p,a,b)            (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawSurface_IsLost(p)                    (p)->lpVtbl->IsLost(p)
#define IDirectDrawSurface_Lock(p,a,b,c,d)              (p)->lpVtbl->Lock(p,a,b,c,d)
#define IDirectDrawSurface_ReleaseDC(p,a)               (p)->lpVtbl->ReleaseDC(p,a)
#define IDirectDrawSurface_Restore(p)                   (p)->lpVtbl->Restore(p)
#define IDirectDrawSurface_SetClipper(p,a)              (p)->lpVtbl->SetClipper(p,a)
#define IDirectDrawSurface_SetColorKey(p,a,b)           (p)->lpVtbl->SetColorKey(p,a,b)
#define IDirectDrawSurface_SetOverlayPosition(p,a,b)    (p)->lpVtbl->SetOverlayPosition(p,a,b)
#define IDirectDrawSurface_SetPalette(p,a)              (p)->lpVtbl->SetPalette(p,a)
#define IDirectDrawSurface_Unlock(p,a)                  (p)->lpVtbl->Unlock(p,a)
#define IDirectDrawSurface_UpdateOverlay(p,a,b,c,d,e)   (p)->lpVtbl->UpdateOverlay(p,a,b,c,d,e)
#define IDirectDrawSurface_UpdateOverlayDisplay(p,a)    (p)->lpVtbl->UpdateOverlayDisplay(p,a)
#define IDirectDrawSurface_UpdateOverlayZOrder(p,a,b)   (p)->lpVtbl->UpdateOverlayZOrder(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDrawSurface_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawSurface_AddRef(p)             (p)->AddRef()
#define IDirectDrawSurface_Release(p)            (p)->Release()
/*** IDirectDrawSurface methods ***/
#define IDirectDrawSurface_AddAttachedSurface(p,a)      (p)->AddAttachedSurface(a)
#define IDirectDrawSurface_AddOverlayDirtyRect(p,a)     (p)->AddOverlayDirtyRect(a)
#define IDirectDrawSurface_Blt(p,a,b,c,d,e)             (p)->Blt(a,b,c,d,e)
#define IDirectDrawSurface_BltBatch(p,a,b,c)            (p)->BltBatch(a,b,c)
#define IDirectDrawSurface_BltFast(p,a,b,c,d,e)         (p)->BltFast(a,b,c,d,e)
#define IDirectDrawSurface_DeleteAttachedSurface(p,a,b) (p)->DeleteAttachedSurface(a,b)
#define IDirectDrawSurface_EnumAttachedSurfaces(p,a,b)  (p)->EnumAttachedSurfaces(a,b)
#define IDirectDrawSurface_EnumOverlayZOrders(p,a,b,c)  (p)->EnumOverlayZOrders(a,b,c)
#define IDirectDrawSurface_Flip(p,a,b)                  (p)->Flip(a,b)
#define IDirectDrawSurface_GetAttachedSurface(p,a,b)    (p)->GetAttachedSurface(a,b)
#define IDirectDrawSurface_GetBltStatus(p,a)            (p)->GetBltStatus(a)
#define IDirectDrawSurface_GetCaps(p,a)                 (p)->GetCaps(a)
#define IDirectDrawSurface_GetClipper(p,a)              (p)->GetClipper(a)
#define IDirectDrawSurface_GetColorKey(p,a,b)           (p)->GetColorKey(a,b)
#define IDirectDrawSurface_GetDC(p,a)                   (p)->GetDC(a)
#define IDirectDrawSurface_GetFlipStatus(p,a)           (p)->GetFlipStatus(a)
#define IDirectDrawSurface_GetOverlayPosition(p,a,b)    (p)->GetOverlayPosition(a,b)
#define IDirectDrawSurface_GetPalette(p,a)              (p)->GetPalette(a)
#define IDirectDrawSurface_GetPixelFormat(p,a)          (p)->GetPixelFormat(a)
#define IDirectDrawSurface_GetSurfaceDesc(p,a)          (p)->GetSurfaceDesc(a)
#define IDirectDrawSurface_Initialize(p,a,b)            (p)->Initialize(a,b)
#define IDirectDrawSurface_IsLost(p)                    (p)->IsLost()
#define IDirectDrawSurface_Lock(p,a,b,c,d)              (p)->Lock(a,b,c,d)
#define IDirectDrawSurface_ReleaseDC(p,a)               (p)->ReleaseDC(a)
#define IDirectDrawSurface_Restore(p)                   (p)->Restore()
#define IDirectDrawSurface_SetClipper(p,a)              (p)->SetClipper(a)
#define IDirectDrawSurface_SetColorKey(p,a,b)           (p)->SetColorKey(a,b)
#define IDirectDrawSurface_SetOverlayPosition(p,a,b)    (p)->SetOverlayPosition(a,b)
#define IDirectDrawSurface_SetPalette(p,a)              (p)->SetPalette(a)
#define IDirectDrawSurface_Unlock(p,a)                  (p)->Unlock(a)
#define IDirectDrawSurface_UpdateOverlay(p,a,b,c,d,e)   (p)->UpdateOverlay(a,b,c,d,e)
#define IDirectDrawSurface_UpdateOverlayDisplay(p,a)    (p)->UpdateOverlayDisplay(a)
#define IDirectDrawSurface_UpdateOverlayZOrder(p,a,b)   (p)->UpdateOverlayZOrder(a,b)
#endif


/*****************************************************************************
 * IDirectDrawSurface2 interface
 */
/* Cannot inherit from IDirectDrawSurface because the LPDIRECTDRAWSURFACE parameters
 * have been converted to LPDIRECTDRAWSURFACE2.
 */
#define INTERFACE IDirectDrawSurface2
DECLARE_INTERFACE_(IDirectDrawSurface2,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawSurface2 methods ***/
    STDMETHOD(AddAttachedSurface)(THIS_ LPDIRECTDRAWSURFACE2 lpDDSAttachedSurface) PURE;
    STDMETHOD(AddOverlayDirtyRect)(THIS_ LPRECT lpRect) PURE;
    STDMETHOD(Blt)(THIS_ LPRECT lpDestRect, LPDIRECTDRAWSURFACE2 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) PURE;
    STDMETHOD(BltBatch)(THIS_ LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags) PURE;
    STDMETHOD(BltFast)(THIS_ DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE2 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwTrans) PURE;
    STDMETHOD(DeleteAttachedSurface)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE2 lpDDSAttachedSurface) PURE;
    STDMETHOD(EnumAttachedSurfaces)(THIS_ LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
    STDMETHOD(EnumOverlayZOrders)(THIS_ DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpfnCallback) PURE;
    STDMETHOD(Flip)(THIS_ LPDIRECTDRAWSURFACE2 lpDDSurfaceTargetOverride, DWORD dwFlags) PURE;
    STDMETHOD(GetAttachedSurface)(THIS_ LPDDSCAPS lpDDSCaps, LPDIRECTDRAWSURFACE2 *lplpDDAttachedSurface) PURE;
    STDMETHOD(GetBltStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetCaps)(THIS_ LPDDSCAPS lpDDSCaps) PURE;
    STDMETHOD(GetClipper)(THIS_ LPDIRECTDRAWCLIPPER *lplpDDClipper) PURE;
    STDMETHOD(GetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(GetDC)(THIS_ HDC *lphDC) PURE;
    STDMETHOD(GetFlipStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetOverlayPosition)(THIS_ LPLONG lplX, LPLONG lplY) PURE;
    STDMETHOD(GetPalette)(THIS_ LPDIRECTDRAWPALETTE *lplpDDPalette) PURE;
    STDMETHOD(GetPixelFormat)(THIS_ LPDDPIXELFORMAT lpDDPixelFormat) PURE;
    STDMETHOD(GetSurfaceDesc)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
    STDMETHOD(IsLost)(THIS) PURE;
    STDMETHOD(Lock)(THIS_ LPRECT lpDestRect, LPDDSURFACEDESC lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent) PURE;
    STDMETHOD(ReleaseDC)(THIS_ HDC hDC) PURE;
    STDMETHOD(Restore)(THIS) PURE;
    STDMETHOD(SetClipper)(THIS_ LPDIRECTDRAWCLIPPER lpDDClipper) PURE;
    STDMETHOD(SetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(SetOverlayPosition)(THIS_ LONG lX, LONG lY) PURE;
    STDMETHOD(SetPalette)(THIS_ LPDIRECTDRAWPALETTE lpDDPalette) PURE;
    STDMETHOD(Unlock)(THIS_ LPVOID lpSurfaceData) PURE;
    STDMETHOD(UpdateOverlay)(THIS_ LPRECT lpSrcRect, LPDIRECTDRAWSURFACE2 lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx) PURE;
    STDMETHOD(UpdateOverlayDisplay)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(UpdateOverlayZOrder)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE2 lpDDSReference) PURE;
    /* added in v2 */
    STDMETHOD(GetDDInterface)(THIS_ LPVOID *lplpDD) PURE;
    STDMETHOD(PageLock)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(PageUnlock)(THIS_ DWORD dwFlags) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawSurface2_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawSurface2_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawSurface2_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawSurface methods (almost) ***/
#define IDirectDrawSurface2_AddAttachedSurface(p,a)      (p)->lpVtbl->AddAttachedSurface(p,a)
#define IDirectDrawSurface2_AddOverlayDirtyRect(p,a)     (p)->lpVtbl->AddOverlayDirtyRect(p,a)
#define IDirectDrawSurface2_Blt(p,a,b,c,d,e)             (p)->lpVtbl->Blt(p,a,b,c,d,e)
#define IDirectDrawSurface2_BltBatch(p,a,b,c)            (p)->lpVtbl->BltBatch(p,a,b,c)
#define IDirectDrawSurface2_BltFast(p,a,b,c,d,e)         (p)->lpVtbl->BltFast(p,a,b,c,d,e)
#define IDirectDrawSurface2_DeleteAttachedSurface(p,a,b) (p)->lpVtbl->DeleteAttachedSurface(p,a,b)
#define IDirectDrawSurface2_EnumAttachedSurfaces(p,a,b)  (p)->lpVtbl->EnumAttachedSurfaces(p,a,b)
#define IDirectDrawSurface2_EnumOverlayZOrders(p,a,b,c)  (p)->lpVtbl->EnumOverlayZOrders(p,a,b,c)
#define IDirectDrawSurface2_Flip(p,a,b)                  (p)->lpVtbl->Flip(p,a,b)
#define IDirectDrawSurface2_GetAttachedSurface(p,a,b)    (p)->lpVtbl->GetAttachedSurface(p,a,b)
#define IDirectDrawSurface2_GetBltStatus(p,a)            (p)->lpVtbl->GetBltStatus(p,a)
#define IDirectDrawSurface2_GetCaps(p,a)                 (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawSurface2_GetClipper(p,a)              (p)->lpVtbl->GetClipper(p,a)
#define IDirectDrawSurface2_GetColorKey(p,a,b)           (p)->lpVtbl->GetColorKey(p,a,b)
#define IDirectDrawSurface2_GetDC(p,a)                   (p)->lpVtbl->GetDC(p,a)
#define IDirectDrawSurface2_GetFlipStatus(p,a)           (p)->lpVtbl->GetFlipStatus(p,a)
#define IDirectDrawSurface2_GetOverlayPosition(p,a,b)    (p)->lpVtbl->GetOverlayPosition(p,a,b)
#define IDirectDrawSurface2_GetPalette(p,a)              (p)->lpVtbl->GetPalette(p,a)
#define IDirectDrawSurface2_GetPixelFormat(p,a)          (p)->lpVtbl->GetPixelFormat(p,a)
#define IDirectDrawSurface2_GetSurfaceDesc(p,a)          (p)->lpVtbl->GetSurfaceDesc(p,a)
#define IDirectDrawSurface2_Initialize(p,a,b)            (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawSurface2_IsLost(p)                    (p)->lpVtbl->IsLost(p)
#define IDirectDrawSurface2_Lock(p,a,b,c,d)              (p)->lpVtbl->Lock(p,a,b,c,d)
#define IDirectDrawSurface2_ReleaseDC(p,a)               (p)->lpVtbl->ReleaseDC(p,a)
#define IDirectDrawSurface2_Restore(p)                   (p)->lpVtbl->Restore(p)
#define IDirectDrawSurface2_SetClipper(p,a)              (p)->lpVtbl->SetClipper(p,a)
#define IDirectDrawSurface2_SetColorKey(p,a,b)           (p)->lpVtbl->SetColorKey(p,a,b)
#define IDirectDrawSurface2_SetOverlayPosition(p,a,b)    (p)->lpVtbl->SetOverlayPosition(p,a,b)
#define IDirectDrawSurface2_SetPalette(p,a)              (p)->lpVtbl->SetPalette(p,a)
#define IDirectDrawSurface2_Unlock(p,a)                  (p)->lpVtbl->Unlock(p,a)
#define IDirectDrawSurface2_UpdateOverlay(p,a,b,c,d,e)   (p)->lpVtbl->UpdateOverlay(p,a,b,c,d,e)
#define IDirectDrawSurface2_UpdateOverlayDisplay(p,a)    (p)->lpVtbl->UpdateOverlayDisplay(p,a)
#define IDirectDrawSurface2_UpdateOverlayZOrder(p,a,b)   (p)->lpVtbl->UpdateOverlayZOrder(p,a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface2_GetDDInterface(p,a) (p)->lpVtbl->GetDDInterface(p,a)
#define IDirectDrawSurface2_PageLock(p,a)       (p)->lpVtbl->PageLock(p,a)
#define IDirectDrawSurface2_PageUnlock(p,a)     (p)->lpVtbl->PageUnlock(p,a)
#else
/*** IUnknown methods ***/
#define IDirectDrawSurface2_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawSurface2_AddRef(p)             (p)->AddRef()
#define IDirectDrawSurface2_Release(p)            (p)->Release()
/*** IDirectDrawSurface methods (almost) ***/
#define IDirectDrawSurface2_AddAttachedSurface(p,a)      (p)->AddAttachedSurface(a)
#define IDirectDrawSurface2_AddOverlayDirtyRect(p,a)     (p)->AddOverlayDirtyRect(a)
#define IDirectDrawSurface2_Blt(p,a,b,c,d,e)             (p)->Blt(a,b,c,d,e)
#define IDirectDrawSurface2_BltBatch(p,a,b,c)            (p)->BltBatch(a,b,c)
#define IDirectDrawSurface2_BltFast(p,a,b,c,d,e)         (p)->BltFast(a,b,c,d,e)
#define IDirectDrawSurface2_DeleteAttachedSurface(p,a,b) (p)->DeleteAttachedSurface(a,b)
#define IDirectDrawSurface2_EnumAttachedSurfaces(p,a,b)  (p)->EnumAttachedSurfaces(a,b)
#define IDirectDrawSurface2_EnumOverlayZOrders(p,a,b,c)  (p)->EnumOverlayZOrders(a,b,c)
#define IDirectDrawSurface2_Flip(p,a,b)                  (p)->Flip(a,b)
#define IDirectDrawSurface2_GetAttachedSurface(p,a,b)    (p)->GetAttachedSurface(a,b)
#define IDirectDrawSurface2_GetBltStatus(p,a)            (p)->GetBltStatus(a)
#define IDirectDrawSurface2_GetCaps(p,a)                 (p)->GetCaps(a)
#define IDirectDrawSurface2_GetClipper(p,a)              (p)->GetClipper(a)
#define IDirectDrawSurface2_GetColorKey(p,a,b)           (p)->GetColorKey(a,b)
#define IDirectDrawSurface2_GetDC(p,a)                   (p)->GetDC(a)
#define IDirectDrawSurface2_GetFlipStatus(p,a)           (p)->GetFlipStatus(a)
#define IDirectDrawSurface2_GetOverlayPosition(p,a,b)    (p)->GetOverlayPosition(a,b)
#define IDirectDrawSurface2_GetPalette(p,a)              (p)->GetPalette(a)
#define IDirectDrawSurface2_GetPixelFormat(p,a)          (p)->GetPixelFormat(a)
#define IDirectDrawSurface2_GetSurfaceDesc(p,a)          (p)->GetSurfaceDesc(a)
#define IDirectDrawSurface2_Initialize(p,a,b)            (p)->Initialize(a,b)
#define IDirectDrawSurface2_IsLost(p)                    (p)->IsLost()
#define IDirectDrawSurface2_Lock(p,a,b,c,d)              (p)->Lock(a,b,c,d)
#define IDirectDrawSurface2_ReleaseDC(p,a)               (p)->ReleaseDC(a)
#define IDirectDrawSurface2_Restore(p)                   (p)->Restore()
#define IDirectDrawSurface2_SetClipper(p,a)              (p)->SetClipper(a)
#define IDirectDrawSurface2_SetColorKey(p,a,b)           (p)->SetColorKey(a,b)
#define IDirectDrawSurface2_SetOverlayPosition(p,a,b)    (p)->SetOverlayPosition(a,b)
#define IDirectDrawSurface2_SetPalette(p,a)              (p)->SetPalette(a)
#define IDirectDrawSurface2_Unlock(p,a)                  (p)->Unlock(a)
#define IDirectDrawSurface2_UpdateOverlay(p,a,b,c,d,e)   (p)->UpdateOverlay(a,b,c,d,e)
#define IDirectDrawSurface2_UpdateOverlayDisplay(p,a)    (p)->UpdateOverlayDisplay(a)
#define IDirectDrawSurface2_UpdateOverlayZOrder(p,a,b)   (p)->UpdateOverlayZOrder(a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface2_GetDDInterface(p,a) (p)->GetDDInterface(a)
#define IDirectDrawSurface2_PageLock(p,a)       (p)->PageLock(a)
#define IDirectDrawSurface2_PageUnlock(p,a)     (p)->PageUnlock(a)
#endif


/*****************************************************************************
 * IDirectDrawSurface3 interface
 */
/* Cannot inherit from IDirectDrawSurface2 because the LPDIRECTDRAWSURFACE2 parameters
 * have been converted to LPDIRECTDRAWSURFACE3.
 */
#define INTERFACE IDirectDrawSurface3
DECLARE_INTERFACE_(IDirectDrawSurface3,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawSurface3 methods ***/
    STDMETHOD(AddAttachedSurface)(THIS_ LPDIRECTDRAWSURFACE3 lpDDSAttachedSurface) PURE;
    STDMETHOD(AddOverlayDirtyRect)(THIS_ LPRECT lpRect) PURE;
    STDMETHOD(Blt)(THIS_ LPRECT lpDestRect, LPDIRECTDRAWSURFACE3 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) PURE;
    STDMETHOD(BltBatch)(THIS_ LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags) PURE;
    STDMETHOD(BltFast)(THIS_ DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE3 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwTrans) PURE;
    STDMETHOD(DeleteAttachedSurface)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE3 lpDDSAttachedSurface) PURE;
    STDMETHOD(EnumAttachedSurfaces)(THIS_ LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpEnumSurfacesCallback) PURE;
    STDMETHOD(EnumOverlayZOrders)(THIS_ DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK lpfnCallback) PURE;
    STDMETHOD(Flip)(THIS_ LPDIRECTDRAWSURFACE3 lpDDSurfaceTargetOverride, DWORD dwFlags) PURE;
    STDMETHOD(GetAttachedSurface)(THIS_ LPDDSCAPS lpDDSCaps, LPDIRECTDRAWSURFACE3 *lplpDDAttachedSurface) PURE;
    STDMETHOD(GetBltStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetCaps)(THIS_ LPDDSCAPS lpDDSCaps) PURE;
    STDMETHOD(GetClipper)(THIS_ LPDIRECTDRAWCLIPPER *lplpDDClipper) PURE;
    STDMETHOD(GetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(GetDC)(THIS_ HDC *lphDC) PURE;
    STDMETHOD(GetFlipStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetOverlayPosition)(THIS_ LPLONG lplX, LPLONG lplY) PURE;
    STDMETHOD(GetPalette)(THIS_ LPDIRECTDRAWPALETTE *lplpDDPalette) PURE;
    STDMETHOD(GetPixelFormat)(THIS_ LPDDPIXELFORMAT lpDDPixelFormat) PURE;
    STDMETHOD(GetSurfaceDesc)(THIS_ LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, LPDDSURFACEDESC lpDDSurfaceDesc) PURE;
    STDMETHOD(IsLost)(THIS) PURE;
    STDMETHOD(Lock)(THIS_ LPRECT lpDestRect, LPDDSURFACEDESC lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent) PURE;
    STDMETHOD(ReleaseDC)(THIS_ HDC hDC) PURE;
    STDMETHOD(Restore)(THIS) PURE;
    STDMETHOD(SetClipper)(THIS_ LPDIRECTDRAWCLIPPER lpDDClipper) PURE;
    STDMETHOD(SetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(SetOverlayPosition)(THIS_ LONG lX, LONG lY) PURE;
    STDMETHOD(SetPalette)(THIS_ LPDIRECTDRAWPALETTE lpDDPalette) PURE;
    STDMETHOD(Unlock)(THIS_ LPVOID lpSurfaceData) PURE;
    STDMETHOD(UpdateOverlay)(THIS_ LPRECT lpSrcRect, LPDIRECTDRAWSURFACE3 lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx) PURE;
    STDMETHOD(UpdateOverlayDisplay)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(UpdateOverlayZOrder)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE3 lpDDSReference) PURE;
    /* added in v2 */
    STDMETHOD(GetDDInterface)(THIS_ LPVOID *lplpDD) PURE;
    STDMETHOD(PageLock)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(PageUnlock)(THIS_ DWORD dwFlags) PURE;
    /* added in v3 */
    STDMETHOD(SetSurfaceDesc)(THIS_ LPDDSURFACEDESC lpDDSD, DWORD dwFlags) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawSurface3_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawSurface3_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawSurface3_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawSurface methods (almost) ***/
#define IDirectDrawSurface3_AddAttachedSurface(p,a)      (p)->lpVtbl->AddAttachedSurface(p,a)
#define IDirectDrawSurface3_AddOverlayDirtyRect(p,a)     (p)->lpVtbl->AddOverlayDirtyRect(p,a)
#define IDirectDrawSurface3_Blt(p,a,b,c,d,e)             (p)->lpVtbl->Blt(p,a,b,c,d,e)
#define IDirectDrawSurface3_BltBatch(p,a,b,c)            (p)->lpVtbl->BltBatch(p,a,b,c)
#define IDirectDrawSurface3_BltFast(p,a,b,c,d,e)         (p)->lpVtbl->BltFast(p,a,b,c,d,e)
#define IDirectDrawSurface3_DeleteAttachedSurface(p,a,b) (p)->lpVtbl->DeleteAttachedSurface(p,a,b)
#define IDirectDrawSurface3_EnumAttachedSurfaces(p,a,b)  (p)->lpVtbl->EnumAttachedSurfaces(p,a,b)
#define IDirectDrawSurface3_EnumOverlayZOrders(p,a,b,c)  (p)->lpVtbl->EnumOverlayZOrders(p,a,b,c)
#define IDirectDrawSurface3_Flip(p,a,b)                  (p)->lpVtbl->Flip(p,a,b)
#define IDirectDrawSurface3_GetAttachedSurface(p,a,b)    (p)->lpVtbl->GetAttachedSurface(p,a,b)
#define IDirectDrawSurface3_GetBltStatus(p,a)            (p)->lpVtbl->GetBltStatus(p,a)
#define IDirectDrawSurface3_GetCaps(p,a)                 (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawSurface3_GetClipper(p,a)              (p)->lpVtbl->GetClipper(p,a)
#define IDirectDrawSurface3_GetColorKey(p,a,b)           (p)->lpVtbl->GetColorKey(p,a,b)
#define IDirectDrawSurface3_GetDC(p,a)                   (p)->lpVtbl->GetDC(p,a)
#define IDirectDrawSurface3_GetFlipStatus(p,a)           (p)->lpVtbl->GetFlipStatus(p,a)
#define IDirectDrawSurface3_GetOverlayPosition(p,a,b)    (p)->lpVtbl->GetOverlayPosition(p,a,b)
#define IDirectDrawSurface3_GetPalette(p,a)              (p)->lpVtbl->GetPalette(p,a)
#define IDirectDrawSurface3_GetPixelFormat(p,a)          (p)->lpVtbl->GetPixelFormat(p,a)
#define IDirectDrawSurface3_GetSurfaceDesc(p,a)          (p)->lpVtbl->GetSurfaceDesc(p,a)
#define IDirectDrawSurface3_Initialize(p,a,b)            (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawSurface3_IsLost(p)                    (p)->lpVtbl->IsLost(p)
#define IDirectDrawSurface3_Lock(p,a,b,c,d)              (p)->lpVtbl->Lock(p,a,b,c,d)
#define IDirectDrawSurface3_ReleaseDC(p,a)               (p)->lpVtbl->ReleaseDC(p,a)
#define IDirectDrawSurface3_Restore(p)                   (p)->lpVtbl->Restore(p)
#define IDirectDrawSurface3_SetClipper(p,a)              (p)->lpVtbl->SetClipper(p,a)
#define IDirectDrawSurface3_SetColorKey(p,a,b)           (p)->lpVtbl->SetColorKey(p,a,b)
#define IDirectDrawSurface3_SetOverlayPosition(p,a,b)    (p)->lpVtbl->SetOverlayPosition(p,a,b)
#define IDirectDrawSurface3_SetPalette(p,a)              (p)->lpVtbl->SetPalette(p,a)
#define IDirectDrawSurface3_Unlock(p,a)                  (p)->lpVtbl->Unlock(p,a)
#define IDirectDrawSurface3_UpdateOverlay(p,a,b,c,d,e)   (p)->lpVtbl->UpdateOverlay(p,a,b,c,d,e)
#define IDirectDrawSurface3_UpdateOverlayDisplay(p,a)    (p)->lpVtbl->UpdateOverlayDisplay(p,a)
#define IDirectDrawSurface3_UpdateOverlayZOrder(p,a,b)   (p)->lpVtbl->UpdateOverlayZOrder(p,a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface3_GetDDInterface(p,a) (p)->lpVtbl->GetDDInterface(p,a)
#define IDirectDrawSurface3_PageLock(p,a)       (p)->lpVtbl->PageLock(p,a)
#define IDirectDrawSurface3_PageUnlock(p,a)     (p)->lpVtbl->PageUnlock(p,a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface3_SetSurfaceDesc(p,a,b) (p)->lpVtbl->SetSurfaceDesc(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDrawSurface3_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawSurface3_AddRef(p)             (p)->AddRef()
#define IDirectDrawSurface3_Release(p)            (p)->Release()
/*** IDirectDrawSurface methods (almost) ***/
#define IDirectDrawSurface3_AddAttachedSurface(p,a)      (p)->AddAttachedSurface(a)
#define IDirectDrawSurface3_AddOverlayDirtyRect(p,a)     (p)->AddOverlayDirtyRect(a)
#define IDirectDrawSurface3_Blt(p,a,b,c,d,e)             (p)->Blt(a,b,c,d,e)
#define IDirectDrawSurface3_BltBatch(p,a,b,c)            (p)->BltBatch(a,b,c)
#define IDirectDrawSurface3_BltFast(p,a,b,c,d,e)         (p)->BltFast(a,b,c,d,e)
#define IDirectDrawSurface3_DeleteAttachedSurface(p,a,b) (p)->DeleteAttachedSurface(a,b)
#define IDirectDrawSurface3_EnumAttachedSurfaces(p,a,b)  (p)->EnumAttachedSurfaces(a,b)
#define IDirectDrawSurface3_EnumOverlayZOrders(p,a,b,c)  (p)->EnumOverlayZOrders(a,b,c)
#define IDirectDrawSurface3_Flip(p,a,b)                  (p)->Flip(a,b)
#define IDirectDrawSurface3_GetAttachedSurface(p,a,b)    (p)->GetAttachedSurface(a,b)
#define IDirectDrawSurface3_GetBltStatus(p,a)            (p)->GetBltStatus(a)
#define IDirectDrawSurface3_GetCaps(p,a)                 (p)->GetCaps(a)
#define IDirectDrawSurface3_GetClipper(p,a)              (p)->GetClipper(a)
#define IDirectDrawSurface3_GetColorKey(p,a,b)           (p)->GetColorKey(a,b)
#define IDirectDrawSurface3_GetDC(p,a)                   (p)->GetDC(a)
#define IDirectDrawSurface3_GetFlipStatus(p,a)           (p)->GetFlipStatus(a)
#define IDirectDrawSurface3_GetOverlayPosition(p,a,b)    (p)->GetOverlayPosition(a,b)
#define IDirectDrawSurface3_GetPalette(p,a)              (p)->GetPalette(a)
#define IDirectDrawSurface3_GetPixelFormat(p,a)          (p)->GetPixelFormat(a)
#define IDirectDrawSurface3_GetSurfaceDesc(p,a)          (p)->GetSurfaceDesc(a)
#define IDirectDrawSurface3_Initialize(p,a,b)            (p)->Initialize(a,b)
#define IDirectDrawSurface3_IsLost(p)                    (p)->IsLost()
#define IDirectDrawSurface3_Lock(p,a,b,c,d)              (p)->Lock(a,b,c,d)
#define IDirectDrawSurface3_ReleaseDC(p,a)               (p)->ReleaseDC(a)
#define IDirectDrawSurface3_Restore(p)                   (p)->Restore()
#define IDirectDrawSurface3_SetClipper(p,a)              (p)->SetClipper(a)
#define IDirectDrawSurface3_SetColorKey(p,a,b)           (p)->SetColorKey(a,b)
#define IDirectDrawSurface3_SetOverlayPosition(p,a,b)    (p)->SetOverlayPosition(a,b)
#define IDirectDrawSurface3_SetPalette(p,a)              (p)->SetPalette(a)
#define IDirectDrawSurface3_Unlock(p,a)                  (p)->Unlock(a)
#define IDirectDrawSurface3_UpdateOverlay(p,a,b,c,d,e)   (p)->UpdateOverlay(a,b,c,d,e)
#define IDirectDrawSurface3_UpdateOverlayDisplay(p,a)    (p)->UpdateOverlayDisplay(a)
#define IDirectDrawSurface3_UpdateOverlayZOrder(p,a,b)   (p)->UpdateOverlayZOrder(a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface3_GetDDInterface(p,a) (p)->GetDDInterface(a)
#define IDirectDrawSurface3_PageLock(p,a)       (p)->PageLock(a)
#define IDirectDrawSurface3_PageUnlock(p,a)     (p)->PageUnlock(a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface3_SetSurfaceDesc(p,a,b) (p)->SetSurfaceDesc(a,b)
#endif


/*****************************************************************************
 * IDirectDrawSurface4 interface
 */
/* Cannot inherit from IDirectDrawSurface2 because DDSCAPS changed to DDSCAPS2.
 */
#define INTERFACE IDirectDrawSurface4
DECLARE_INTERFACE_(IDirectDrawSurface4,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawSurface4 methods ***/
    STDMETHOD(AddAttachedSurface)(THIS_ LPDIRECTDRAWSURFACE4 lpDDSAttachedSurface) PURE;
    STDMETHOD(AddOverlayDirtyRect)(THIS_ LPRECT lpRect) PURE;
    STDMETHOD(Blt)(THIS_ LPRECT lpDestRect, LPDIRECTDRAWSURFACE4 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) PURE;
    STDMETHOD(BltBatch)(THIS_ LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags) PURE;
    STDMETHOD(BltFast)(THIS_ DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE4 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwTrans) PURE;
    STDMETHOD(DeleteAttachedSurface)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE4 lpDDSAttachedSurface) PURE;
    STDMETHOD(EnumAttachedSurfaces)(THIS_ LPVOID lpContext, LPDDENUMSURFACESCALLBACK2 lpEnumSurfacesCallback) PURE;
    STDMETHOD(EnumOverlayZOrders)(THIS_ DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK2 lpfnCallback) PURE;
    STDMETHOD(Flip)(THIS_ LPDIRECTDRAWSURFACE4 lpDDSurfaceTargetOverride, DWORD dwFlags) PURE;
    STDMETHOD(GetAttachedSurface)(THIS_ LPDDSCAPS2 lpDDSCaps, LPDIRECTDRAWSURFACE4 *lplpDDAttachedSurface) PURE;
    STDMETHOD(GetBltStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetCaps)(THIS_ LPDDSCAPS2 lpDDSCaps) PURE;
    STDMETHOD(GetClipper)(THIS_ LPDIRECTDRAWCLIPPER *lplpDDClipper) PURE;
    STDMETHOD(GetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(GetDC)(THIS_ HDC *lphDC) PURE;
    STDMETHOD(GetFlipStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetOverlayPosition)(THIS_ LPLONG lplX, LPLONG lplY) PURE;
    STDMETHOD(GetPalette)(THIS_ LPDIRECTDRAWPALETTE *lplpDDPalette) PURE;
    STDMETHOD(GetPixelFormat)(THIS_ LPDDPIXELFORMAT lpDDPixelFormat) PURE;
    STDMETHOD(GetSurfaceDesc)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
    STDMETHOD(IsLost)(THIS) PURE;
    STDMETHOD(Lock)(THIS_ LPRECT lpDestRect, LPDDSURFACEDESC2 lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent) PURE;
    STDMETHOD(ReleaseDC)(THIS_ HDC hDC) PURE;
    STDMETHOD(Restore)(THIS) PURE;
    STDMETHOD(SetClipper)(THIS_ LPDIRECTDRAWCLIPPER lpDDClipper) PURE;
    STDMETHOD(SetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(SetOverlayPosition)(THIS_ LONG lX, LONG lY) PURE;
    STDMETHOD(SetPalette)(THIS_ LPDIRECTDRAWPALETTE lpDDPalette) PURE;
    STDMETHOD(Unlock)(THIS_ LPRECT lpSurfaceData) PURE;
    STDMETHOD(UpdateOverlay)(THIS_ LPRECT lpSrcRect, LPDIRECTDRAWSURFACE4 lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx) PURE;
    STDMETHOD(UpdateOverlayDisplay)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(UpdateOverlayZOrder)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE4 lpDDSReference) PURE;
    /* added in v2 */
    STDMETHOD(GetDDInterface)(THIS_ LPVOID *lplpDD) PURE;
    STDMETHOD(PageLock)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(PageUnlock)(THIS_ DWORD dwFlags) PURE;
    /* added in v3 */
    STDMETHOD(SetSurfaceDesc)(THIS_ LPDDSURFACEDESC2 lpDDSD, DWORD dwFlags) PURE;
    /* added in v4 */
    STDMETHOD(SetPrivateData)(THIS_ REFGUID tag, LPVOID pData, DWORD cbSize, DWORD dwFlags) PURE;
    STDMETHOD(GetPrivateData)(THIS_ REFGUID tag, LPVOID pBuffer, LPDWORD pcbBufferSize) PURE;
    STDMETHOD(FreePrivateData)(THIS_ REFGUID tag) PURE;
    STDMETHOD(GetUniquenessValue)(THIS_ LPDWORD pValue) PURE;
    STDMETHOD(ChangeUniquenessValue)(THIS) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawSurface4_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawSurface4_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawSurface4_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawSurface (almost) methods ***/
#define IDirectDrawSurface4_AddAttachedSurface(p,a)      (p)->lpVtbl->AddAttachedSurface(p,a)
#define IDirectDrawSurface4_AddOverlayDirtyRect(p,a)     (p)->lpVtbl->AddOverlayDirtyRect(p,a)
#define IDirectDrawSurface4_Blt(p,a,b,c,d,e)             (p)->lpVtbl->Blt(p,a,b,c,d,e)
#define IDirectDrawSurface4_BltBatch(p,a,b,c)            (p)->lpVtbl->BltBatch(p,a,b,c)
#define IDirectDrawSurface4_BltFast(p,a,b,c,d,e)         (p)->lpVtbl->BltFast(p,a,b,c,d,e)
#define IDirectDrawSurface4_DeleteAttachedSurface(p,a,b) (p)->lpVtbl->DeleteAttachedSurface(p,a,b)
#define IDirectDrawSurface4_EnumAttachedSurfaces(p,a,b)  (p)->lpVtbl->EnumAttachedSurfaces(p,a,b)
#define IDirectDrawSurface4_EnumOverlayZOrders(p,a,b,c)  (p)->lpVtbl->EnumOverlayZOrders(p,a,b,c)
#define IDirectDrawSurface4_Flip(p,a,b)                  (p)->lpVtbl->Flip(p,a,b)
#define IDirectDrawSurface4_GetAttachedSurface(p,a,b)    (p)->lpVtbl->GetAttachedSurface(p,a,b)
#define IDirectDrawSurface4_GetBltStatus(p,a)            (p)->lpVtbl->GetBltStatus(p,a)
#define IDirectDrawSurface4_GetCaps(p,a)                 (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawSurface4_GetClipper(p,a)              (p)->lpVtbl->GetClipper(p,a)
#define IDirectDrawSurface4_GetColorKey(p,a,b)           (p)->lpVtbl->GetColorKey(p,a,b)
#define IDirectDrawSurface4_GetDC(p,a)                   (p)->lpVtbl->GetDC(p,a)
#define IDirectDrawSurface4_GetFlipStatus(p,a)           (p)->lpVtbl->GetFlipStatus(p,a)
#define IDirectDrawSurface4_GetOverlayPosition(p,a,b)    (p)->lpVtbl->GetOverlayPosition(p,a,b)
#define IDirectDrawSurface4_GetPalette(p,a)              (p)->lpVtbl->GetPalette(p,a)
#define IDirectDrawSurface4_GetPixelFormat(p,a)          (p)->lpVtbl->GetPixelFormat(p,a)
#define IDirectDrawSurface4_GetSurfaceDesc(p,a)          (p)->lpVtbl->GetSurfaceDesc(p,a)
#define IDirectDrawSurface4_Initialize(p,a,b)            (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawSurface4_IsLost(p)                    (p)->lpVtbl->IsLost(p)
#define IDirectDrawSurface4_Lock(p,a,b,c,d)              (p)->lpVtbl->Lock(p,a,b,c,d)
#define IDirectDrawSurface4_ReleaseDC(p,a)               (p)->lpVtbl->ReleaseDC(p,a)
#define IDirectDrawSurface4_Restore(p)                   (p)->lpVtbl->Restore(p)
#define IDirectDrawSurface4_SetClipper(p,a)              (p)->lpVtbl->SetClipper(p,a)
#define IDirectDrawSurface4_SetColorKey(p,a,b)           (p)->lpVtbl->SetColorKey(p,a,b)
#define IDirectDrawSurface4_SetOverlayPosition(p,a,b)    (p)->lpVtbl->SetOverlayPosition(p,a,b)
#define IDirectDrawSurface4_SetPalette(p,a)              (p)->lpVtbl->SetPalette(p,a)
#define IDirectDrawSurface4_Unlock(p,a)                  (p)->lpVtbl->Unlock(p,a)
#define IDirectDrawSurface4_UpdateOverlay(p,a,b,c,d,e)   (p)->lpVtbl->UpdateOverlay(p,a,b,c,d,e)
#define IDirectDrawSurface4_UpdateOverlayDisplay(p,a)    (p)->lpVtbl->UpdateOverlayDisplay(p,a)
#define IDirectDrawSurface4_UpdateOverlayZOrder(p,a,b)   (p)->lpVtbl->UpdateOverlayZOrder(p,a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface4_GetDDInterface(p,a) (p)->lpVtbl->GetDDInterface(p,a)
#define IDirectDrawSurface4_PageLock(p,a)       (p)->lpVtbl->PageLock(p,a)
#define IDirectDrawSurface4_PageUnlock(p,a)     (p)->lpVtbl->PageUnlock(p,a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface4_SetSurfaceDesc(p,a,b) (p)->lpVtbl->SetSurfaceDesc(p,a,b)
/*** IDirectDrawSurface4 methods ***/
#define IDirectDrawSurface4_SetPrivateData(p,a,b,c,d) (p)->lpVtbl->SetPrivateData(p,a,b,c,d)
#define IDirectDrawSurface4_GetPrivateData(p,a,b,c)   (p)->lpVtbl->GetPrivateData(p,a,b,c)
#define IDirectDrawSurface4_FreePrivateData(p,a)      (p)->lpVtbl->FreePrivateData(p,a)
#define IDirectDrawSurface4_GetUniquenessValue(p,a)   (p)->lpVtbl->GetUniquenessValue(p,a)
#define IDirectDrawSurface4_ChangeUniquenessValue(p)  (p)->lpVtbl->ChangeUniquenessValue(p)
#else
/*** IUnknown methods ***/
#define IDirectDrawSurface4_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawSurface4_AddRef(p)             (p)->AddRef()
#define IDirectDrawSurface4_Release(p)            (p)->Release()
/*** IDirectDrawSurface (almost) methods ***/
#define IDirectDrawSurface4_AddAttachedSurface(p,a)      (p)->AddAttachedSurface(a)
#define IDirectDrawSurface4_AddOverlayDirtyRect(p,a)     (p)->AddOverlayDirtyRect(a)
#define IDirectDrawSurface4_Blt(p,a,b,c,d,e)             (p)->Blt(a,b,c,d,e)
#define IDirectDrawSurface4_BltBatch(p,a,b,c)            (p)->BltBatch(a,b,c)
#define IDirectDrawSurface4_BltFast(p,a,b,c,d,e)         (p)->BltFast(a,b,c,d,e)
#define IDirectDrawSurface4_DeleteAttachedSurface(p,a,b) (p)->DeleteAttachedSurface(a,b)
#define IDirectDrawSurface4_EnumAttachedSurfaces(p,a,b)  (p)->EnumAttachedSurfaces(a,b)
#define IDirectDrawSurface4_EnumOverlayZOrders(p,a,b,c)  (p)->EnumOverlayZOrders(a,b,c)
#define IDirectDrawSurface4_Flip(p,a,b)                  (p)->Flip(a,b)
#define IDirectDrawSurface4_GetAttachedSurface(p,a,b)    (p)->GetAttachedSurface(a,b)
#define IDirectDrawSurface4_GetBltStatus(p,a)            (p)->GetBltStatus(a)
#define IDirectDrawSurface4_GetCaps(p,a)                 (p)->GetCaps(a)
#define IDirectDrawSurface4_GetClipper(p,a)              (p)->GetClipper(a)
#define IDirectDrawSurface4_GetColorKey(p,a,b)           (p)->GetColorKey(a,b)
#define IDirectDrawSurface4_GetDC(p,a)                   (p)->GetDC(a)
#define IDirectDrawSurface4_GetFlipStatus(p,a)           (p)->GetFlipStatus(a)
#define IDirectDrawSurface4_GetOverlayPosition(p,a,b)    (p)->GetOverlayPosition(a,b)
#define IDirectDrawSurface4_GetPalette(p,a)              (p)->GetPalette(a)
#define IDirectDrawSurface4_GetPixelFormat(p,a)          (p)->GetPixelFormat(a)
#define IDirectDrawSurface4_GetSurfaceDesc(p,a)          (p)->GetSurfaceDesc(a)
#define IDirectDrawSurface4_Initialize(p,a,b)            (p)->Initialize(a,b)
#define IDirectDrawSurface4_IsLost(p)                    (p)->IsLost()
#define IDirectDrawSurface4_Lock(p,a,b,c,d)              (p)->Lock(a,b,c,d)
#define IDirectDrawSurface4_ReleaseDC(p,a)               (p)->ReleaseDC(a)
#define IDirectDrawSurface4_Restore(p)                   (p)->Restore()
#define IDirectDrawSurface4_SetClipper(p,a)              (p)->SetClipper(a)
#define IDirectDrawSurface4_SetColorKey(p,a,b)           (p)->SetColorKey(a,b)
#define IDirectDrawSurface4_SetOverlayPosition(p,a,b)    (p)->SetOverlayPosition(a,b)
#define IDirectDrawSurface4_SetPalette(p,a)              (p)->SetPalette(a)
#define IDirectDrawSurface4_Unlock(p,a)                  (p)->Unlock(a)
#define IDirectDrawSurface4_UpdateOverlay(p,a,b,c,d,e)   (p)->UpdateOverlay(a,b,c,d,e)
#define IDirectDrawSurface4_UpdateOverlayDisplay(p,a)    (p)->UpdateOverlayDisplay(a)
#define IDirectDrawSurface4_UpdateOverlayZOrder(p,a,b)   (p)->UpdateOverlayZOrder(a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface4_GetDDInterface(p,a) (p)->GetDDInterface(a)
#define IDirectDrawSurface4_PageLock(p,a)       (p)->PageLock(a)
#define IDirectDrawSurface4_PageUnlock(p,a)     (p)->PageUnlock(a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface4_SetSurfaceDesc(p,a,b) (p)->SetSurfaceDesc(a,b)
/*** IDirectDrawSurface4 methods ***/
#define IDirectDrawSurface4_SetPrivateData(p,a,b,c,d) (p)->SetPrivateData(a,b,c,d)
#define IDirectDrawSurface4_GetPrivateData(p,a,b,c)   (p)->GetPrivateData(a,b,c)
#define IDirectDrawSurface4_FreePrivateData(p,a)      (p)->FreePrivateData(a)
#define IDirectDrawSurface4_GetUniquenessValue(p,a)   (p)->GetUniquenessValue(a)
#define IDirectDrawSurface4_ChangeUniquenessValue(p)  (p)->ChangeUniquenessValue()
#endif


/*****************************************************************************
 * IDirectDrawSurface7 interface
 */
#define INTERFACE IDirectDrawSurface7
DECLARE_INTERFACE_(IDirectDrawSurface7,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawSurface7 methods ***/
    STDMETHOD(AddAttachedSurface)(THIS_ LPDIRECTDRAWSURFACE7 lpDDSAttachedSurface) PURE;
    STDMETHOD(AddOverlayDirtyRect)(THIS_ LPRECT lpRect) PURE;
    STDMETHOD(Blt)(THIS_ LPRECT lpDestRect, LPDIRECTDRAWSURFACE7 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwFlags, LPDDBLTFX lpDDBltFx) PURE;
    STDMETHOD(BltBatch)(THIS_ LPDDBLTBATCH lpDDBltBatch, DWORD dwCount, DWORD dwFlags) PURE;
    STDMETHOD(BltFast)(THIS_ DWORD dwX, DWORD dwY, LPDIRECTDRAWSURFACE7 lpDDSrcSurface, LPRECT lpSrcRect, DWORD dwTrans) PURE;
    STDMETHOD(DeleteAttachedSurface)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE7 lpDDSAttachedSurface) PURE;
    STDMETHOD(EnumAttachedSurfaces)(THIS_ LPVOID lpContext, LPDDENUMSURFACESCALLBACK7 lpEnumSurfacesCallback) PURE;
    STDMETHOD(EnumOverlayZOrders)(THIS_ DWORD dwFlags, LPVOID lpContext, LPDDENUMSURFACESCALLBACK7 lpfnCallback) PURE;
    STDMETHOD(Flip)(THIS_ LPDIRECTDRAWSURFACE7 lpDDSurfaceTargetOverride, DWORD dwFlags) PURE;
    STDMETHOD(GetAttachedSurface)(THIS_ LPDDSCAPS2 lpDDSCaps, LPDIRECTDRAWSURFACE7 *lplpDDAttachedSurface) PURE;
    STDMETHOD(GetBltStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetCaps)(THIS_ LPDDSCAPS2 lpDDSCaps) PURE;
    STDMETHOD(GetClipper)(THIS_ LPDIRECTDRAWCLIPPER *lplpDDClipper) PURE;
    STDMETHOD(GetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(GetDC)(THIS_ HDC *lphDC) PURE;
    STDMETHOD(GetFlipStatus)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(GetOverlayPosition)(THIS_ LPLONG lplX, LPLONG lplY) PURE;
    STDMETHOD(GetPalette)(THIS_ LPDIRECTDRAWPALETTE *lplpDDPalette) PURE;
    STDMETHOD(GetPixelFormat)(THIS_ LPDDPIXELFORMAT lpDDPixelFormat) PURE;
    STDMETHOD(GetSurfaceDesc)(THIS_ LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
    STDMETHOD(Initialize)(THIS_ LPDIRECTDRAW lpDD, LPDDSURFACEDESC2 lpDDSurfaceDesc) PURE;
    STDMETHOD(IsLost)(THIS) PURE;
    STDMETHOD(Lock)(THIS_ LPRECT lpDestRect, LPDDSURFACEDESC2 lpDDSurfaceDesc, DWORD dwFlags, HANDLE hEvent) PURE;
    STDMETHOD(ReleaseDC)(THIS_ HDC hDC) PURE;
    STDMETHOD(Restore)(THIS) PURE;
    STDMETHOD(SetClipper)(THIS_ LPDIRECTDRAWCLIPPER lpDDClipper) PURE;
    STDMETHOD(SetColorKey)(THIS_ DWORD dwFlags, LPDDCOLORKEY lpDDColorKey) PURE;
    STDMETHOD(SetOverlayPosition)(THIS_ LONG lX, LONG lY) PURE;
    STDMETHOD(SetPalette)(THIS_ LPDIRECTDRAWPALETTE lpDDPalette) PURE;
    STDMETHOD(Unlock)(THIS_ LPRECT lpSurfaceData) PURE;
    STDMETHOD(UpdateOverlay)(THIS_ LPRECT lpSrcRect, LPDIRECTDRAWSURFACE7 lpDDDestSurface, LPRECT lpDestRect, DWORD dwFlags, LPDDOVERLAYFX lpDDOverlayFx) PURE;
    STDMETHOD(UpdateOverlayDisplay)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(UpdateOverlayZOrder)(THIS_ DWORD dwFlags, LPDIRECTDRAWSURFACE7 lpDDSReference) PURE;
    /* added in v2 */
    STDMETHOD(GetDDInterface)(THIS_ LPVOID *lplpDD) PURE;
    STDMETHOD(PageLock)(THIS_ DWORD dwFlags) PURE;
    STDMETHOD(PageUnlock)(THIS_ DWORD dwFlags) PURE;
    /* added in v3 */
    STDMETHOD(SetSurfaceDesc)(THIS_ LPDDSURFACEDESC2 lpDDSD, DWORD dwFlags) PURE;
    /* added in v4 */
    STDMETHOD(SetPrivateData)(THIS_ REFGUID tag, LPVOID pData, DWORD cbSize, DWORD dwFlags) PURE;
    STDMETHOD(GetPrivateData)(THIS_ REFGUID tag, LPVOID pBuffer, LPDWORD pcbBufferSize) PURE;
    STDMETHOD(FreePrivateData)(THIS_ REFGUID tag) PURE;
    STDMETHOD(GetUniquenessValue)(THIS_ LPDWORD pValue) PURE;
    STDMETHOD(ChangeUniquenessValue)(THIS) PURE;
    /* added in v7 */
    STDMETHOD(SetPriority)(THIS_ DWORD prio) PURE;
    STDMETHOD(GetPriority)(THIS_ LPDWORD prio) PURE;
    STDMETHOD(SetLOD)(THIS_ DWORD lod) PURE;
    STDMETHOD(GetLOD)(THIS_ LPDWORD lod) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawSurface7_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawSurface7_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawSurface7_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawSurface (almost) methods ***/
#define IDirectDrawSurface7_AddAttachedSurface(p,a)      (p)->lpVtbl->AddAttachedSurface(p,a)
#define IDirectDrawSurface7_AddOverlayDirtyRect(p,a)     (p)->lpVtbl->AddOverlayDirtyRect(p,a)
#define IDirectDrawSurface7_Blt(p,a,b,c,d,e)             (p)->lpVtbl->Blt(p,a,b,c,d,e)
#define IDirectDrawSurface7_BltBatch(p,a,b,c)            (p)->lpVtbl->BltBatch(p,a,b,c)
#define IDirectDrawSurface7_BltFast(p,a,b,c,d,e)         (p)->lpVtbl->BltFast(p,a,b,c,d,e)
#define IDirectDrawSurface7_DeleteAttachedSurface(p,a,b) (p)->lpVtbl->DeleteAttachedSurface(p,a,b)
#define IDirectDrawSurface7_EnumAttachedSurfaces(p,a,b)  (p)->lpVtbl->EnumAttachedSurfaces(p,a,b)
#define IDirectDrawSurface7_EnumOverlayZOrders(p,a,b,c)  (p)->lpVtbl->EnumOverlayZOrders(p,a,b,c)
#define IDirectDrawSurface7_Flip(p,a,b)                  (p)->lpVtbl->Flip(p,a,b)
#define IDirectDrawSurface7_GetAttachedSurface(p,a,b)    (p)->lpVtbl->GetAttachedSurface(p,a,b)
#define IDirectDrawSurface7_GetBltStatus(p,a)            (p)->lpVtbl->GetBltStatus(p,a)
#define IDirectDrawSurface7_GetCaps(p,a)                 (p)->lpVtbl->GetCaps(p,a)
#define IDirectDrawSurface7_GetClipper(p,a)              (p)->lpVtbl->GetClipper(p,a)
#define IDirectDrawSurface7_GetColorKey(p,a,b)           (p)->lpVtbl->GetColorKey(p,a,b)
#define IDirectDrawSurface7_GetDC(p,a)                   (p)->lpVtbl->GetDC(p,a)
#define IDirectDrawSurface7_GetFlipStatus(p,a)           (p)->lpVtbl->GetFlipStatus(p,a)
#define IDirectDrawSurface7_GetOverlayPosition(p,a,b)    (p)->lpVtbl->GetOverlayPosition(p,a,b)
#define IDirectDrawSurface7_GetPalette(p,a)              (p)->lpVtbl->GetPalette(p,a)
#define IDirectDrawSurface7_GetPixelFormat(p,a)          (p)->lpVtbl->GetPixelFormat(p,a)
#define IDirectDrawSurface7_GetSurfaceDesc(p,a)          (p)->lpVtbl->GetSurfaceDesc(p,a)
#define IDirectDrawSurface7_Initialize(p,a,b)            (p)->lpVtbl->Initialize(p,a,b)
#define IDirectDrawSurface7_IsLost(p)                    (p)->lpVtbl->IsLost(p)
#define IDirectDrawSurface7_Lock(p,a,b,c,d)              (p)->lpVtbl->Lock(p,a,b,c,d)
#define IDirectDrawSurface7_ReleaseDC(p,a)               (p)->lpVtbl->ReleaseDC(p,a)
#define IDirectDrawSurface7_Restore(p)                   (p)->lpVtbl->Restore(p)
#define IDirectDrawSurface7_SetClipper(p,a)              (p)->lpVtbl->SetClipper(p,a)
#define IDirectDrawSurface7_SetColorKey(p,a,b)           (p)->lpVtbl->SetColorKey(p,a,b)
#define IDirectDrawSurface7_SetOverlayPosition(p,a,b)    (p)->lpVtbl->SetOverlayPosition(p,a,b)
#define IDirectDrawSurface7_SetPalette(p,a)              (p)->lpVtbl->SetPalette(p,a)
#define IDirectDrawSurface7_Unlock(p,a)                  (p)->lpVtbl->Unlock(p,a)
#define IDirectDrawSurface7_UpdateOverlay(p,a,b,c,d,e)   (p)->lpVtbl->UpdateOverlay(p,a,b,c,d,e)
#define IDirectDrawSurface7_UpdateOverlayDisplay(p,a)    (p)->lpVtbl->UpdateOverlayDisplay(p,a)
#define IDirectDrawSurface7_UpdateOverlayZOrder(p,a,b)   (p)->lpVtbl->UpdateOverlayZOrder(p,a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface7_GetDDInterface(p,a) (p)->lpVtbl->GetDDInterface(p,a)
#define IDirectDrawSurface7_PageLock(p,a)       (p)->lpVtbl->PageLock(p,a)
#define IDirectDrawSurface7_PageUnlock(p,a)     (p)->lpVtbl->PageUnlock(p,a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface7_SetSurfaceDesc(p,a,b) (p)->lpVtbl->SetSurfaceDesc(p,a,b)
/*** IDirectDrawSurface4 methods ***/
#define IDirectDrawSurface7_SetPrivateData(p,a,b,c,d) (p)->lpVtbl->SetPrivateData(p,a,b,c,d)
#define IDirectDrawSurface7_GetPrivateData(p,a,b,c)   (p)->lpVtbl->GetPrivateData(p,a,b,c)
#define IDirectDrawSurface7_FreePrivateData(p,a)      (p)->lpVtbl->FreePrivateData(p,a)
#define IDirectDrawSurface7_GetUniquenessValue(p,a)   (p)->lpVtbl->GetUniquenessValue(p,a)
#define IDirectDrawSurface7_ChangeUniquenessValue(p)  (p)->lpVtbl->ChangeUniquenessValue(p)
/*** IDirectDrawSurface7 methods ***/
#define IDirectDrawSurface7_SetPriority(p,a)          (p)->lpVtbl->SetPriority(p,a)
#define IDirectDrawSurface7_GetPriority(p,a)          (p)->lpVtbl->GetPriority(p,a)
#define IDirectDrawSurface7_SetLOD(p,a)               (p)->lpVtbl->SetLOD(p,a)
#define IDirectDrawSurface7_GetLOD(p,a)               (p)->lpVtbl->GetLOD(p,a)
#else
/*** IUnknown methods ***/
#define IDirectDrawSurface7_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawSurface7_AddRef(p)             (p)->AddRef()
#define IDirectDrawSurface7_Release(p)            (p)->Release()
/*** IDirectDrawSurface (almost) methods ***/
#define IDirectDrawSurface7_AddAttachedSurface(p,a)      (p)->AddAttachedSurface(a)
#define IDirectDrawSurface7_AddOverlayDirtyRect(p,a)     (p)->AddOverlayDirtyRect(a)
#define IDirectDrawSurface7_Blt(p,a,b,c,d,e)             (p)->Blt(a,b,c,d,e)
#define IDirectDrawSurface7_BltBatch(p,a,b,c)            (p)->BltBatch(a,b,c)
#define IDirectDrawSurface7_BltFast(p,a,b,c,d,e)         (p)->BltFast(a,b,c,d,e)
#define IDirectDrawSurface7_DeleteAttachedSurface(p,a,b) (p)->DeleteAttachedSurface(a,b)
#define IDirectDrawSurface7_EnumAttachedSurfaces(p,a,b)  (p)->EnumAttachedSurfaces(a,b)
#define IDirectDrawSurface7_EnumOverlayZOrders(p,a,b,c)  (p)->EnumOverlayZOrders(a,b,c)
#define IDirectDrawSurface7_Flip(p,a,b)                  (p)->Flip(a,b)
#define IDirectDrawSurface7_GetAttachedSurface(p,a,b)    (p)->GetAttachedSurface(a,b)
#define IDirectDrawSurface7_GetBltStatus(p,a)            (p)->GetBltStatus(a)
#define IDirectDrawSurface7_GetCaps(p,a)                 (p)->GetCaps(a)
#define IDirectDrawSurface7_GetClipper(p,a)              (p)->GetClipper(a)
#define IDirectDrawSurface7_GetColorKey(p,a,b)           (p)->GetColorKey(a,b)
#define IDirectDrawSurface7_GetDC(p,a)                   (p)->GetDC(a)
#define IDirectDrawSurface7_GetFlipStatus(p,a)           (p)->GetFlipStatus(a)
#define IDirectDrawSurface7_GetOverlayPosition(p,a,b)    (p)->GetOverlayPosition(a,b)
#define IDirectDrawSurface7_GetPalette(p,a)              (p)->GetPalette(a)
#define IDirectDrawSurface7_GetPixelFormat(p,a)          (p)->GetPixelFormat(a)
#define IDirectDrawSurface7_GetSurfaceDesc(p,a)          (p)->GetSurfaceDesc(a)
#define IDirectDrawSurface7_Initialize(p,a,b)            (p)->Initialize(a,b)
#define IDirectDrawSurface7_IsLost(p)                    (p)->IsLost()
#define IDirectDrawSurface7_Lock(p,a,b,c,d)              (p)->Lock(a,b,c,d)
#define IDirectDrawSurface7_ReleaseDC(p,a)               (p)->ReleaseDC(a)
#define IDirectDrawSurface7_Restore(p)                   (p)->Restore()
#define IDirectDrawSurface7_SetClipper(p,a)              (p)->SetClipper(a)
#define IDirectDrawSurface7_SetColorKey(p,a,b)           (p)->SetColorKey(a,b)
#define IDirectDrawSurface7_SetOverlayPosition(p,a,b)    (p)->SetOverlayPosition(a,b)
#define IDirectDrawSurface7_SetPalette(p,a)              (p)->SetPalette(a)
#define IDirectDrawSurface7_Unlock(p,a)                  (p)->Unlock(a)
#define IDirectDrawSurface7_UpdateOverlay(p,a,b,c,d,e)   (p)->UpdateOverlay(a,b,c,d,e)
#define IDirectDrawSurface7_UpdateOverlayDisplay(p,a)    (p)->UpdateOverlayDisplay(a)
#define IDirectDrawSurface7_UpdateOverlayZOrder(p,a,b)   (p)->UpdateOverlayZOrder(a,b)
/*** IDirectDrawSurface2 methods ***/
#define IDirectDrawSurface7_GetDDInterface(p,a) (p)->GetDDInterface(a)
#define IDirectDrawSurface7_PageLock(p,a)       (p)->PageLock(a)
#define IDirectDrawSurface7_PageUnlock(p,a)     (p)->PageUnlock(a)
/*** IDirectDrawSurface3 methods ***/
#define IDirectDrawSurface7_SetSurfaceDesc(p,a,b) (p)->SetSurfaceDesc(a,b)
/*** IDirectDrawSurface4 methods ***/
#define IDirectDrawSurface7_SetPrivateData(p,a,b,c,d) (p)->SetPrivateData(a,b,c,d)
#define IDirectDrawSurface7_GetPrivateData(p,a,b,c)   (p)->GetPrivateData(a,b,c)
#define IDirectDrawSurface7_FreePrivateData(p,a)      (p)->FreePrivateData(a)
#define IDirectDrawSurface7_GetUniquenessValue(p,a)   (p)->GetUniquenessValue(a)
#define IDirectDrawSurface7_ChangeUniquenessValue(p)  (p)->ChangeUniquenessValue()
/*** IDirectDrawSurface7 methods ***/
#define IDirectDrawSurface7_SetPriority(p,a)          (p)->SetPriority(a)
#define IDirectDrawSurface7_GetPriority(p,a)          (p)->GetPriority(a)
#define IDirectDrawSurface7_SetLOD(p,a)               (p)->SetLOD(a)
#define IDirectDrawSurface7_GetLOD(p,a)               (p)->GetLOD(a)
#endif

/*****************************************************************************
 * IDirectDrawColorControl interface
 */
#define INTERFACE IDirectDrawColorControl
DECLARE_INTERFACE_(IDirectDrawColorControl,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawColorControl methods ***/
    STDMETHOD(GetColorControls)(THIS_ LPDDCOLORCONTROL lpColorControl) PURE;
    STDMETHOD(SetColorControls)(THIS_ LPDDCOLORCONTROL lpColorControl) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawColorControl_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawColorControl_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawColorControl_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawColorControl methods ***/
#define IDirectDrawColorControl_GetColorControls(p,a) (p)->lpVtbl->GetColorControls(p,a)
#define IDirectDrawColorControl_SetColorControls(p,a) (p)->lpVtbl->SetColorControls(p,a)
#else
/*** IUnknown methods ***/
#define IDirectDrawColorControl_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawColorControl_AddRef(p)             (p)->AddRef()
#define IDirectDrawColorControl_Release(p)            (p)->Release()
/*** IDirectDrawColorControl methods ***/
#define IDirectDrawColorControl_GetColorControls(p,a) (p)->GetColorControls(a)
#define IDirectDrawColorControl_SetColorControls(p,a) (p)->SetColorControls(a)
#endif

/*****************************************************************************
 * IDirectDrawGammaControl interface
 */
#define INTERFACE IDirectDrawGammaControl
DECLARE_INTERFACE_(IDirectDrawGammaControl,IUnknown)
{
    /*** IUnknown methods ***/
    STDMETHOD_(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObject) PURE;
    STDMETHOD_(ULONG,AddRef)(THIS) PURE;
    STDMETHOD_(ULONG,Release)(THIS) PURE;
    /*** IDirectDrawGammaControl methods ***/
    STDMETHOD(GetGammaRamp)(THIS_ DWORD dwFlags, LPDDGAMMARAMP lpGammaRamp) PURE;
    STDMETHOD(SetGammaRamp)(THIS_ DWORD dwFlags, LPDDGAMMARAMP lpGammaRamp) PURE;
};
#undef INTERFACE

#if !defined(__cplusplus) || defined(CINTERFACE)
/*** IUnknown methods ***/
#define IDirectDrawGammaControl_QueryInterface(p,a,b) (p)->lpVtbl->QueryInterface(p,a,b)
#define IDirectDrawGammaControl_AddRef(p)             (p)->lpVtbl->AddRef(p)
#define IDirectDrawGammaControl_Release(p)            (p)->lpVtbl->Release(p)
/*** IDirectDrawGammaControl methods ***/
#define IDirectDrawGammaControl_GetGammaRamp(p,a,b)   (p)->lpVtbl->GetGammaRamp(p,a,b)
#define IDirectDrawGammaControl_SetGammaRamp(p,a,b)   (p)->lpVtbl->SetGammaRamp(p,a,b)
#else
/*** IUnknown methods ***/
#define IDirectDrawGammaControl_QueryInterface(p,a,b) (p)->QueryInterface(a,b)
#define IDirectDrawGammaControl_AddRef(p)             (p)->AddRef()
#define IDirectDrawGammaControl_Release(p)            (p)->Release()
/*** IDirectDrawGammaControl methods ***/
#define IDirectDrawGammaControl_GetGammaRamp(p,a,b)   (p)->GetGammaRamp(a,b)
#define IDirectDrawGammaControl_SetGammaRamp(p,a,b)   (p)->SetGammaRamp(a,b)
#endif


HRESULT WINAPI DirectDrawCreate(GUID*,LPDIRECTDRAW*,IUnknown*);
HRESULT WINAPI DirectDrawCreateEx(GUID*,LPVOID*,REFIID,IUnknown*);
HRESULT WINAPI DirectDrawCreateClipper(DWORD,LPDIRECTDRAWCLIPPER*,IUnknown*);

#ifdef __cplusplus
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif /* __DDRAW_INCLUDED__ */
