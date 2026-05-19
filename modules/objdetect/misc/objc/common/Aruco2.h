//
//  Aruco2.h
//
//  Hand-written ArUco2 utility class: enums + all static API functions.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/objdetect/aruco2.hpp>
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import "Mat.h"
#import "Scalar.h"
#import "Size2i.h"
#import "Aruco2DetectionParameters.h"
#import "Aruco2FiducialMarker.h"
#import "Aruco2GridBoard.h"
#import "Aruco2Diamond.h"
#import "Aruco2FractalMarker.h"

// ---------------------------------------------------------------------------
// DictionaryType enum
// ---------------------------------------------------------------------------

typedef NS_ENUM(int, Aruco2DictionaryType) {
    Aruco2_DICT_4X4_50         = 0,
    Aruco2_DICT_4X4_100        = 1,
    Aruco2_DICT_4X4_250        = 2,
    Aruco2_DICT_4X4_1000       = 3,
    Aruco2_DICT_5X5_50         = 4,
    Aruco2_DICT_5X5_100        = 5,
    Aruco2_DICT_5X5_250        = 6,
    Aruco2_DICT_5X5_1000       = 7,
    Aruco2_DICT_6X6_50         = 8,
    Aruco2_DICT_6X6_100        = 9,
    Aruco2_DICT_6X6_250        = 10,
    Aruco2_DICT_6X6_1000       = 11,
    Aruco2_DICT_7X7_50         = 12,
    Aruco2_DICT_7X7_100        = 13,
    Aruco2_DICT_7X7_250        = 14,
    Aruco2_DICT_7X7_1000       = 15,
    Aruco2_DICT_ARUCO_ORIGINAL = 16,
    Aruco2_DICT_APRILTAG_16h5  = 17,
    Aruco2_DICT_APRILTAG_25h9  = 18,
    Aruco2_DICT_APRILTAG_36h10 = 19,
    Aruco2_DICT_APRILTAG_36h11 = 20,
    Aruco2_DICT_ARUCO_MIP_36h12= 21
};

// ---------------------------------------------------------------------------
// FractalType enum
// ---------------------------------------------------------------------------

typedef NS_ENUM(int, Aruco2FractalType) {
    Aruco2_FRACTAL_2L_6 = 0,
    Aruco2_FRACTAL_3L_6 = 1,
    Aruco2_FRACTAL_4L_6 = 2,
    Aruco2_FRACTAL_5L_6 = 3
};

// ---------------------------------------------------------------------------
// Aruco2 utility class
// ---------------------------------------------------------------------------

NS_ASSUME_NONNULL_BEGIN

/**
 * ArUco2 — fiducial marker detection, board detection, fractal markers.
 *
 * All methods are static. Use the Aruco2DictionaryType and Aruco2FractalType
 * enumerations for the `dictionary` and `ftype` parameters.
 */
CV_EXPORTS @interface Aruco2 : NSObject

// ---- getFiducialMarker -------------------------------------------------------

/**
 * Generate a canonical marker image ready for printing.
 * @param img        Output grayscale image (CV_8UC1).
 * @param dictionary Predefined dictionary (Aruco2DictionaryType).
 * @param markerId   Marker identifier; must be valid for the chosen dictionary.
 * @param bitSize    Output size in pixels of each marker bit. Default 20.
 * @param externalBorder Whether to add a white external border. Default YES.
 */
+ (void)getFiducialMarker:(Mat*)img
               dictionary:(int)dictionary
                       id:(int)markerId
                  bitSize:(int)bitSize
           externalBorder:(BOOL)externalBorder
NS_SWIFT_NAME(getFiducialMarker(img:dictionary:id:bitSize:externalBorder:));

+ (void)getFiducialMarker:(Mat*)img
               dictionary:(int)dictionary
                       id:(int)markerId
NS_SWIFT_NAME(getFiducialMarker(img:dictionary:id:));

// ---- detectFiducialMarkers (single dict) ------------------------------------

/**
 * Detect ArUco fiducial markers in an image using a single dictionary.
 * @param image    Input image (grayscale or BGR).
 * @param dictionary Dictionary to search (Aruco2DictionaryType).
 * @param params   Detection parameters. Pass nil for defaults.
 * @return Array of detected FiducialMarker objects; empty if none found.
 */
+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkers:(Mat*)image
                                               dictionary:(int)dictionary
                                                   params:(nullable Aruco2DetectionParameters*)params
NS_SWIFT_NAME(detectFiducialMarkers(image:dictionary:params:));

+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkers:(Mat*)image
                                               dictionary:(int)dictionary
NS_SWIFT_NAME(detectFiducialMarkers(image:dictionary:));

// ---- detectFiducialMarkers (multiple dicts) ---------------------------------

/**
 * Detect ArUco fiducial markers searching across multiple dictionaries.
 * @param image    Input image (grayscale or BGR).
 * @param dicts    Array of DictionaryType values (as NSNumber ints).
 * @param params   Detection parameters. Pass nil for defaults.
 * @return Array of detected FiducialMarker objects; each carries its dictionary.
 */
+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkersMulti:(Mat*)image
                                                          dicts:(NSArray<NSNumber*>*)dicts
                                                         params:(nullable Aruco2DetectionParameters*)params
NS_SWIFT_NAME(detectFiducialMarkersMulti(image:dicts:params:));

// ---- drawFiducialMarkers ----------------------------------------------------

/**
 * Draw detected fiducial markers onto an image.
 * @param image   Input/output image (1 or 3 channels); modified in place.
 * @param markers Markers returned by detectFiducialMarkers.
 * @param borderColor Color used for the marker outline.
 */
+ (void)drawFiducialMarkers:(Mat*)image
                    markers:(NSArray<Aruco2FiducialMarker*>*)markers
                borderColor:(Scalar*)borderColor
NS_SWIFT_NAME(drawFiducialMarkers(image:markers:borderColor:));

+ (void)drawFiducialMarkers:(Mat*)image
                    markers:(NSArray<Aruco2FiducialMarker*>*)markers
NS_SWIFT_NAME(drawFiducialMarkers(image:markers:));

// ---- drawAxis ---------------------------------------------------------------

/**
 * Draw the XYZ coordinate frame of a pose estimate onto an image.
 * @param image        Input/output BGR image.
 * @param cameraMatrix 3×3 camera intrinsic matrix.
 * @param distCoeffs   Distortion coefficients.
 * @param rvec         Rotation vector (from solvePnP).
 * @param tvec         Translation vector (from solvePnP).
 * @param length       Axis length in the same unit as tvec.
 */
+ (void)drawAxis:(Mat*)image
    cameraMatrix:(Mat*)cameraMatrix
      distCoeffs:(Mat*)distCoeffs
            rvec:(Mat*)rvec
            tvec:(Mat*)tvec
          length:(float)length
NS_SWIFT_NAME(drawAxis(image:cameraMatrix:distCoeffs:rvec:tvec:length:));

// ---- getSolvePnpPoints (FiducialMarker) -------------------------------------

/**
 * Compute object and image points for a single fiducial marker.
 * @param marker     A detected fiducial marker.
 * @param objPoints  Output 4×1 array of 3-D object points (CV_32FC3).
 * @param imgPoints  Output 4×1 array of 2-D image points (CV_32FC2).
 * @param markerSize Physical side length of the marker. Default 1.0.
 */
+ (void)getSolvePnpPointsForFiducialMarker:(Aruco2FiducialMarker*)marker
                                 objPoints:(Mat*)objPoints
                                 imgPoints:(Mat*)imgPoints
                                markerSize:(float)markerSize
NS_SWIFT_NAME(getSolvePnpPoints(marker:objPoints:imgPoints:markerSize:));

+ (void)getSolvePnpPointsForFiducialMarker:(Aruco2FiducialMarker*)marker
                                 objPoints:(Mat*)objPoints
                                 imgPoints:(Mat*)imgPoints
NS_SWIFT_NAME(getSolvePnpPoints(marker:objPoints:imgPoints:));

// ---- getGridBoard -----------------------------------------------------------

/**
 * Generate a grid board image ready for printing.
 * @param img       Output grayscale image (CV_8UC1).
 * @param boardSize Board layout as columns × rows.
 * @param dictionary Dictionary (Aruco2DictionaryType).
 * @param bitSize   Size of each marker bit in pixels. Default 25.
 * @param ids       Optional custom marker id list (Mat of ints); pass nil for 0…N-1.
 */
+ (void)getGridBoard:(Mat*)img
           boardSize:(Size2i*)boardSize
          dictionary:(int)dictionary
             bitSize:(int)bitSize
                 ids:(nullable Mat*)ids
NS_SWIFT_NAME(getGridBoard(img:boardSize:dictionary:bitSize:ids:));

+ (void)getGridBoard:(Mat*)img
           boardSize:(Size2i*)boardSize
          dictionary:(int)dictionary
NS_SWIFT_NAME(getGridBoard(img:boardSize:dictionary:));

// ---- detectGridBoard --------------------------------------------------------

/**
 * Detect a rectangular grid board of ArUco markers.
 * @param image    Input image (grayscale or BGR).
 * @param gridSize Board layout as columns × rows.
 * @param dictionary Dictionary (Aruco2DictionaryType).
 * @param board    Output GridBoard populated with the detected markers.
 * @param ids      Optional custom marker id list (Mat of ints); pass nil for 0…N-1.
 * @return YES if at least one board marker was detected.
 */
+ (BOOL)detectGridBoard:(Mat*)image
               gridSize:(Size2i*)gridSize
             dictionary:(int)dictionary
                  board:(Aruco2GridBoard*)board
                    ids:(nullable Mat*)ids
NS_SWIFT_NAME(detectGridBoard(image:gridSize:dictionary:board:ids:));

+ (BOOL)detectGridBoard:(Mat*)image
               gridSize:(Size2i*)gridSize
             dictionary:(int)dictionary
                  board:(Aruco2GridBoard*)board
NS_SWIFT_NAME(detectGridBoard(image:gridSize:dictionary:board:));

// ---- drawGridBoard ----------------------------------------------------------

/**
 * Draw detected board corners and optionally marker ids onto an image.
 * @param image        Input/output image; modified in place.
 * @param board        Board returned by detectGridBoard.
 * @param color        Color for corners and text.
 * @param drawMarkerIds If YES, draw marker ids at each centroid.
 */
+ (void)drawGridBoard:(Mat*)image
                board:(Aruco2GridBoard*)board
                color:(Scalar*)color
        drawMarkerIds:(BOOL)drawMarkerIds
NS_SWIFT_NAME(drawGridBoard(image:board:color:drawMarkerIds:));

+ (void)drawGridBoard:(Mat*)image
                board:(Aruco2GridBoard*)board
NS_SWIFT_NAME(drawGridBoard(image:board:));

// ---- getSolvePnpPoints (GridBoard) ------------------------------------------

/**
 * Compute object and image points for a detected grid board.
 * @param board      A detected board from detectGridBoard.
 * @param objPoints  Output array of 3-D object points (CV_32FC3).
 * @param imgPoints  Output array of 2-D image points (CV_32FC2).
 * @param markerSize Physical side length of one marker. Default 1.0.
 */
+ (void)getSolvePnpPointsForGridBoard:(Aruco2GridBoard*)board
                             objPoints:(Mat*)objPoints
                             imgPoints:(Mat*)imgPoints
                            markerSize:(float)markerSize
NS_SWIFT_NAME(getSolvePnpPoints(board:objPoints:imgPoints:markerSize:));

+ (void)getSolvePnpPointsForGridBoard:(Aruco2GridBoard*)board
                             objPoints:(Mat*)objPoints
                             imgPoints:(Mat*)imgPoints
NS_SWIFT_NAME(getSolvePnpPoints(board:objPoints:imgPoints:));

// ---- getDiamondImage --------------------------------------------------------

/**
 * Generate a ChArUco2-style diamond image ready for printing.
 * @param img        Output grayscale image (CV_8UC1).
 * @param dictionary Predefined dictionary (Aruco2DictionaryType).
 * @param ids        Ids of the 4 constituent markers in clockwise order from top-left.
 * @param bitSize    Size of each marker bit in pixels. Default 20.
 */
+ (void)getDiamondImage:(Mat*)img
             dictionary:(int)dictionary
                    ids:(Int4*)ids
                bitSize:(int)bitSize
NS_SWIFT_NAME(getDiamondImage(img:dictionary:ids:bitSize:));

+ (void)getDiamondImage:(Mat*)img
             dictionary:(int)dictionary
                    ids:(Int4*)ids
NS_SWIFT_NAME(getDiamondImage(img:dictionary:ids:));

// ---- detectDiamonds ---------------------------------------------------------

/**
 * Detect ChArUco2-style diamond markers in an image.
 * @param image Input image (grayscale or BGR).
 * @param dictionary Dictionary (Aruco2DictionaryType).
 * @return Array of detected Diamond objects; empty if none found.
 */
+ (NSArray<Aruco2Diamond*>*)detectDiamonds:(Mat*)image
                                 dictionary:(int)dictionary
NS_SWIFT_NAME(detectDiamonds(image:dictionary:));

// ---- drawDiamonds -----------------------------------------------------------

/**
 * Draw detected diamond outlines and optionally marker ids onto an image.
 * @param image        Input/output image; modified in place.
 * @param diamonds     Diamonds from detectDiamonds.
 * @param color        Color for the diamond outline.
 * @param drawMarkerIds If YES, draw constituent marker ids.
 */
+ (void)drawDiamonds:(Mat*)image
            diamonds:(NSArray<Aruco2Diamond*>*)diamonds
               color:(Scalar*)color
       drawMarkerIds:(BOOL)drawMarkerIds
NS_SWIFT_NAME(drawDiamonds(image:diamonds:color:drawMarkerIds:));

+ (void)drawDiamonds:(Mat*)image
            diamonds:(NSArray<Aruco2Diamond*>*)diamonds
NS_SWIFT_NAME(drawDiamonds(image:diamonds:));

// ---- getSolvePnpPoints (Diamond) --------------------------------------------

/**
 * Compute object and image points for a detected diamond (9 grid points).
 * @param diamond    A detected diamond from detectDiamonds.
 * @param objPoints  Output 9×1 array of 3-D object points (CV_32FC3).
 * @param imgPoints  Output 9×1 array of 2-D image points (CV_32FC2).
 * @param markerSize Physical side length of one marker. Default 1.0.
 */
+ (void)getSolvePnpPointsForDiamond:(Aruco2Diamond*)diamond
                           objPoints:(Mat*)objPoints
                           imgPoints:(Mat*)imgPoints
                          markerSize:(float)markerSize
NS_SWIFT_NAME(getSolvePnpPoints(diamond:objPoints:imgPoints:markerSize:));

+ (void)getSolvePnpPointsForDiamond:(Aruco2Diamond*)diamond
                           objPoints:(Mat*)objPoints
                           imgPoints:(Mat*)imgPoints
NS_SWIFT_NAME(getSolvePnpPoints(diamond:objPoints:imgPoints:));

// ---- getFractalImage --------------------------------------------------------

/**
 * Render a fractal marker to a grayscale image.
 * @param img     Output CV_8UC1 image.
 * @param ftype   Fractal configuration (Aruco2FractalType).
 * @param bitSize Side length of one bit cell in pixels. Default 20.
 */
+ (void)getFractalImage:(Mat*)img
                  ftype:(int)ftype
                bitSize:(int)bitSize
NS_SWIFT_NAME(getFractalImage(img:ftype:bitSize:));

+ (void)getFractalImage:(Mat*)img
                  ftype:(int)ftype
NS_SWIFT_NAME(getFractalImage(img:ftype:));

// ---- detectFractals ---------------------------------------------------------

/**
 * Detect fractal markers in an image.
 * @param image Input image (BGR or grayscale).
 * @param ftype Fractal configuration to search for (Aruco2FractalType).
 * @return Array of detected FractalMarker objects; empty if none found.
 */
+ (NSArray<Aruco2FractalMarker*>*)detectFractals:(Mat*)image
                                            ftype:(int)ftype
NS_SWIFT_NAME(detectFractals(image:ftype:));

// ---- drawFractals -----------------------------------------------------------

/**
 * Draw detected fractal markers on an image.
 * @param image              Input/output image.
 * @param fractals           Array from detectFractals.
 * @param color              Border and label colour.
 * @param drawAllImagePoints If YES, draw a circle at every matched image point.
 */
+ (void)drawFractals:(Mat*)image
            fractals:(NSArray<Aruco2FractalMarker*>*)fractals
               color:(Scalar*)color
 drawAllImagePoints:(BOOL)drawAllImagePoints
NS_SWIFT_NAME(drawFractals(image:fractals:color:drawAllImagePoints:));

+ (void)drawFractals:(Mat*)image
            fractals:(NSArray<Aruco2FractalMarker*>*)fractals
NS_SWIFT_NAME(drawFractals(image:fractals:));

// ---- getSolvePnpPoints (FractalMarker) --------------------------------------

/**
 * Extract solvePnP inputs for a detected fractal marker.
 * @param fractal    A detected fractal marker from detectFractals.
 * @param objPoints  Output N×1 array of 3-D object points (CV_32FC3).
 * @param imgPoints  Output N×1 array of 2-D image points (CV_32FC2).
 * @param markerSize Physical side length of the outer marker. Default 1.0.
 */
+ (void)getSolvePnpPointsForFractalMarker:(Aruco2FractalMarker*)fractal
                                 objPoints:(Mat*)objPoints
                                 imgPoints:(Mat*)imgPoints
                                markerSize:(float)markerSize
NS_SWIFT_NAME(getSolvePnpPoints(fractal:objPoints:imgPoints:markerSize:));

+ (void)getSolvePnpPointsForFractalMarker:(Aruco2FractalMarker*)fractal
                                 objPoints:(Mat*)objPoints
                                 imgPoints:(Mat*)imgPoints
NS_SWIFT_NAME(getSolvePnpPoints(fractal:objPoints:imgPoints:));

@end

NS_ASSUME_NONNULL_END
