//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_STANDARD_ATTRIBUTES_H
#define INCLUDED_IMF_STANDARD_ATTRIBUTES_H

//-----------------------------------------------------------------------------
//
//	Optional Standard Attributes -- these attributes are "optional"
//	because not every image file header has them, but they define a
//	"standard" way to represent commonly used data in the file header.
//
//	For each attribute, with name "foo", and type "T", the following
//	functions are automatically generated via macros:
//
//	void			   addFoo (Header &header, const T &value);
//	bool			   hasFoo (const Header &header);
//	const TypedAttribute<T> &  fooAttribute (const Header &header);
//	TypedAttribute<T> &	   fooAttribute (Header &header);
//	const T &		   foo (const Header &Header);
//	T &			   foo (Header &Header);
//
//-----------------------------------------------------------------------------

#include "ImfHeader.h"
#include "ImfBoxAttribute.h"
#include "ImfChromaticitiesAttribute.h"
#include "ImfEnvmapAttribute.h"
#include "ImfDeepImageStateAttribute.h"
#include "ImfFloatAttribute.h"
#include "ImfKeyCodeAttribute.h"
#include "ImfMatrixAttribute.h"
#include "ImfRationalAttribute.h"
#include "ImfStringAttribute.h"
#include "ImfStringVectorAttribute.h"
#include "ImfTimeCodeAttribute.h"
#include "ImfVecAttribute.h"
#include "ImfNamespace.h"
#include "ImfExport.h"
#include "ImfIDManifestAttribute.h"

#define IMF_ADD_SUFFIX(suffix) add##suffix
#define IMF_HAS_SUFFIX(suffix) has##suffix
#define IMF_NAME_ATTRIBUTE(name) name##Attribute

#define IMF_STD_ATTRIBUTE_DEF(name, suffix, object)                            \
                                                                               \
    OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER                                \
    IMF_EXPORT void IMF_ADD_SUFFIX (suffix) (                                  \
        Header & header, const object& v);                                     \
    IMF_EXPORT bool  IMF_HAS_SUFFIX (suffix) (const Header& header);           \
    IMF_EXPORT const TypedAttribute<object>& IMF_NAME_ATTRIBUTE (name) (       \
        const Header& header);                                                 \
    IMF_EXPORT TypedAttribute<object>& IMF_NAME_ATTRIBUTE (name) (             \
        Header & header);                                                      \
    IMF_EXPORT const object& name (const Header& header);                      \
    IMF_EXPORT object& name (Header& header);                                  \
    OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#define IMF_STD_ATTRIBUTE_DEF_DEPRECATED(name, suffix, object, msg)            \
                                                                               \
    OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER                                \
    OPENEXR_DEPRECATED (msg)                                                   \
    IMF_EXPORT void IMF_ADD_SUFFIX (suffix) (                                  \
        Header & header, const object& v);                                     \
    OPENEXR_DEPRECATED (msg)                                                   \
    IMF_EXPORT bool IMF_HAS_SUFFIX (suffix) (const Header& header);            \
    OPENEXR_DEPRECATED (msg)                                                   \
    IMF_EXPORT const TypedAttribute<object>& IMF_NAME_ATTRIBUTE (name) (       \
        const Header& header);                                                 \
    OPENEXR_DEPRECATED (msg)                                                   \
    IMF_EXPORT TypedAttribute<object>& IMF_NAME_ATTRIBUTE (name) (             \
        Header & header);                                                      \
    OPENEXR_DEPRECATED (msg)                                                   \
    IMF_EXPORT const object&            name (const Header& header);           \
    OPENEXR_DEPRECATED (msg) IMF_EXPORT object& name (Header& header);         \
    OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

//
// chromaticities -- for RGB images, specifies the CIE (x,y)
// chromaticities of the primaries and the white point
//

IMF_STD_ATTRIBUTE_DEF (chromaticities, Chromaticities, Chromaticities)


//
// whiteLuminance -- for RGB images, defines the luminance, in Nits
// (candelas per square meter) of the RGB value (1.0, 1.0, 1.0).
//
// If the chromaticities and the whiteLuminance of an RGB image are
// known, then it is possible to convert the image's pixels from RGB
// to CIE XYZ tristimulus values (see function RGBtoXYZ() in header
// file ImfChromaticities.h).
// 
//

IMF_STD_ATTRIBUTE_DEF (whiteLuminance, WhiteLuminance, float)


//
// adoptedNeutral -- specifies the CIE (x,y) coordinates that should
// be considered neutral during color rendering.  Pixels in the image
// file whose (x,y) coordinates match the adoptedNeutral value should
// be mapped to neutral values on the display.
//

IMF_STD_ATTRIBUTE_DEF (adoptedNeutral, AdoptedNeutral, IMATH_NAMESPACE::V2f)


//
// renderingTransform, lookModTransform -- specify the names of the
// CTL functions that implements the intended color rendering and look
// modification transforms for this image.
// 

IMF_STD_ATTRIBUTE_DEF (renderingTransform, RenderingTransform, std::string)
IMF_STD_ATTRIBUTE_DEF (lookModTransform, LookModTransform, std::string)


//
// xDensity -- horizontal output density, in pixels per inch.
// The image's vertical output density is xDensity * pixelAspectRatio.
//

IMF_STD_ATTRIBUTE_DEF (xDensity, XDensity, float)


//
// owner -- name of the owner of the image
//

IMF_STD_ATTRIBUTE_DEF (owner, Owner, std::string)
   

//
// comments -- additional image information in human-readable
// form, for example a verbal description of the image
//

IMF_STD_ATTRIBUTE_DEF (comments, Comments, std::string)


//
// capDate -- the date when the image was created or captured,
// in local time, and formatted as
//
//    YYYY:MM:DD hh:mm:ss
//
// where YYYY is the year (4 digits, e.g. 2003), MM is the month
// (2 digits, 01, 02, ... 12), DD is the day of the month (2 digits,
// 01, 02, ... 31), hh is the hour (2 digits, 00, 01, ... 23), mm
// is the minute, and ss is the second (2 digits, 00, 01, ... 59).
//
//

IMF_STD_ATTRIBUTE_DEF (capDate, CapDate, std::string)


//
// utcOffset -- offset of local time at capDate from
// Universal Coordinated Time (UTC), in seconds:
//
//    UTC == local time + utcOffset
//

IMF_STD_ATTRIBUTE_DEF (utcOffset, UtcOffset, float)


//
// longitude, latitude, altitude -- for images of real objects, the
// location where the image was recorded.  Longitude and latitude are
// in degrees east of Greenwich and north of the equator.  Altitude
// is in meters above sea level.  For example, Kathmandu, Nepal is
// at longitude 85.317, latitude 27.717, altitude 1305.
//

IMF_STD_ATTRIBUTE_DEF (longitude, Longitude, float)
IMF_STD_ATTRIBUTE_DEF (latitude, Latitude, float)
IMF_STD_ATTRIBUTE_DEF (altitude, Altitude, float)


//
// focus -- the camera's focus distance, in meters
//

IMF_STD_ATTRIBUTE_DEF (focus, Focus, float)


//
// exposure -- exposure time, in seconds
//

IMF_STD_ATTRIBUTE_DEF (expTime, ExpTime, float)


//
// aperture -- the camera's lens aperture, in f-stops (focal length
// of the lens divided by the diameter of the iris opening)
//

IMF_STD_ATTRIBUTE_DEF (aperture, Aperture, float)


//
// isoSpeed -- the ISO speed of the film or image sensor
// that was used to record the image
//

IMF_STD_ATTRIBUTE_DEF (isoSpeed, IsoSpeed, float)


//
// envmap -- if this attribute is present, the image represents
// an environment map.  The attribute's value defines how 3D
// directions are mapped to 2D pixel locations.  For details
// see header file ImfEnvmap.h
//

IMF_STD_ATTRIBUTE_DEF (envmap, Envmap, Envmap)


//
// keyCode -- for motion picture film frames.  Identifies film
// manufacturer, film type, film roll and frame position within
// the roll.
//

IMF_STD_ATTRIBUTE_DEF (keyCode, KeyCode, KeyCode)


//
// timeCode -- time and control code
//

IMF_STD_ATTRIBUTE_DEF (timeCode, TimeCode, TimeCode)


//
// wrapmodes -- determines how texture map images are extrapolated.
// If an OpenEXR file is used as a texture map for 3D rendering,
// texture coordinates (0.0, 0.0) and (1.0, 1.0) correspond to
// the upper left and lower right corners of the data window.
// If the image is mapped onto a surface with texture coordinates
// outside the zero-to-one range, then the image must be extrapolated.
// This attribute tells the renderer how to do this extrapolation.
// The attribute contains either a pair of comma-separated keywords,
// to specify separate extrapolation modes for the horizontal and
// vertical directions; or a single keyword, to specify extrapolation
// in both directions (e.g. "clamp,periodic" or "clamp").  Extra white
// space surrounding the keywords is allowed, but should be ignored
// by the renderer ("clamp, black " is equivalent to "clamp,black").
// The keywords listed below are predefined; some renderers may support
// additional extrapolation modes:
//
//	black		pixels outside the zero-to-one range are black
//
//	clamp		texture coordinates less than 0.0 and greater
//			than 1.0 are clamped to 0.0 and 1.0 respectively
//
//	periodic	the texture image repeats periodically
//
//	mirror		the texture image repeats periodically, but
//			every other instance is mirrored
//

IMF_STD_ATTRIBUTE_DEF (wrapmodes, Wrapmodes, std::string)


//
// framesPerSecond -- defines the nominal playback frame rate for image
// sequences, in frames per second.  Every image in a sequence should
// have a framesPerSecond attribute, and the attribute value should be
// the same for all images in the sequence.  If an image sequence has
// no framesPerSecond attribute, playback software should assume that
// the frame rate for the sequence is 24 frames per second.
//
// In order to allow exact representation of NTSC frame and field rates,
// framesPerSecond is stored as a rational number.  A rational number is
// a pair of integers, n and d, that represents the value n/d.
//
// For the exact values of commonly used frame rates, please see header
// file ImfFramesPerSecond.h.
//

IMF_STD_ATTRIBUTE_DEF (framesPerSecond, FramesPerSecond, Rational)


//
// multiView -- defines the view names for multi-view image files.
// A multi-view image contains two or more views of the same scene,
// as seen from different viewpoints, for example a left-eye and
// a right-eye view for stereo displays.  The multiView attribute
// lists the names of the views in an image, and a naming convention
// identifies the channels that belong to each view.
//
// For details, please see header file ImfMultiView.h
//

IMF_STD_ATTRIBUTE_DEF (multiView , MultiView, StringVector)


// 
// worldToCamera -- for images generated by 3D computer graphics rendering,
// a matrix that transforms 3D points from the world to the camera coordinate
// space of the renderer.
// 
// The camera coordinate space is left-handed.  Its origin indicates the
// location of the camera.  The positive x and y axes correspond to the
// "right" and "up" directions in the rendered image.  The positive z
// axis indicates the camera's viewing direction.  (Objects in front of
// the camera have positive z coordinates.)
// 
// Camera coordinate space in OpenEXR is the same as in Pixar's Renderman.
// 

IMF_STD_ATTRIBUTE_DEF (worldToCamera, WorldToCamera, IMATH_NAMESPACE::M44f)


// 
// worldToNDC -- for images generated by 3D computer graphics rendering, a
// matrix that transforms 3D points from the world to the Normalized Device
// Coordinate (NDC) space of the renderer.
// 
// NDC is a 2D coordinate space that corresponds to the image plane, with
// positive x and pointing to the right and y positive pointing down.  The
// coordinates (0, 0) and (1, 1) correspond to the upper left and lower right
// corners of the OpenEXR display window.
// 
// To transform a 3D point in word space into a 2D point in NDC space,
// multiply the 3D point by the worldToNDC matrix and discard the z
// coordinate.
// 
// NDC space in OpenEXR is the same as in Pixar's Renderman.
// 

IMF_STD_ATTRIBUTE_DEF (worldToNDC, WorldToNDC, IMATH_NAMESPACE::M44f)


//
// deepImageState -- specifies whether the pixels in a deep image are
// sorted and non-overlapping.
//
// Note: this attribute can be set by application code that writes a file
// in order to tell applications that read the file whether the pixel data
// must be cleaned up prior to image processing operations such as flattening. 
// The OpenEXR library does not verify that the attribute is consistent with
// the actual state of the pixels.  Application software may assume that the
// attribute is valid, as long as the software will not crash or lock up if
// any pixels are inconsistent with the deepImageState attribute.
//

IMF_STD_ATTRIBUTE_DEF (deepImageState, DeepImageState, DeepImageState)


//
// originalDataWindow -- if application software crops an image, then it
// should save the data window of the original, un-cropped image in the
// originalDataWindow attribute.
//

IMF_STD_ATTRIBUTE_DEF
    (originalDataWindow, OriginalDataWindow, IMATH_NAMESPACE::Box2i)


//
// dwaCompressionLevel -- sets the quality level for images compressed
// with the DWAA or DWAB method.
//
// DEPRECATED: use the methods directly in the header
IMF_STD_ATTRIBUTE_DEF_DEPRECATED (
    dwaCompressionLevel,
    DwaCompressionLevel,
    float,
    "use compression method in ImfHeader")

//
// ID Manifest
//

IMF_STD_ATTRIBUTE_DEF( idManifest,IDManifest,CompressedIDManifest)

#endif
