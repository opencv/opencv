//
//  Aruco2.mm
//

#import "Aruco2.h"

// Helper: NSArray<NSNumber*> → std::vector<cv::aruco2::DictionaryType>
static std::vector<cv::aruco2::DictionaryType> nsArrayToDictTypeVector(NSArray<NSNumber*>* arr) {
    std::vector<cv::aruco2::DictionaryType> v;
    v.reserve([arr count]);
    for (NSNumber* n in arr) {
        v.push_back((cv::aruco2::DictionaryType)[n intValue]);
    }
    return v;
}

// Helper: std::vector<FiducialMarker> → NSArray
static NSArray<Aruco2FiducialMarker*>* fiducialVecToArray(std::vector<cv::aruco2::FiducialMarker>& vec) {
    NSMutableArray<Aruco2FiducialMarker*>* arr = [NSMutableArray arrayWithCapacity:vec.size()];
    for (size_t i = 0; i < vec.size(); i++) {
        [arr addObject:[Aruco2FiducialMarker fromNative:vec[i]]];
    }
    return arr;
}

// Helper: NSArray<Aruco2FiducialMarker*> → std::vector<FiducialMarker>
static std::vector<cv::aruco2::FiducialMarker> fiducialArrayToVec(NSArray<Aruco2FiducialMarker*>* arr) {
    std::vector<cv::aruco2::FiducialMarker> vec;
    vec.reserve([arr count]);
    for (Aruco2FiducialMarker* m in arr) {
        vec.push_back(m.nativeRef);
    }
    return vec;
}

// Helper: std::vector<Diamond> → NSArray
static NSArray<Aruco2Diamond*>* diamondVecToArray(std::vector<cv::aruco2::Diamond>& vec) {
    NSMutableArray<Aruco2Diamond*>* arr = [NSMutableArray arrayWithCapacity:vec.size()];
    for (size_t i = 0; i < vec.size(); i++) {
        [arr addObject:[Aruco2Diamond fromNative:vec[i]]];
    }
    return arr;
}

// Helper: std::vector<FractalMarker> → NSArray
static NSArray<Aruco2FractalMarker*>* fractalVecToArray(std::vector<cv::aruco2::FractalMarker>& vec) {
    NSMutableArray<Aruco2FractalMarker*>* arr = [NSMutableArray arrayWithCapacity:vec.size()];
    for (size_t i = 0; i < vec.size(); i++) {
        [arr addObject:[Aruco2FractalMarker fromNative:vec[i]]];
    }
    return arr;
}

// Helper: NSArray<Aruco2Diamond*> → std::vector<Diamond>
static std::vector<cv::aruco2::Diamond> diamondArrayToVec(NSArray<Aruco2Diamond*>* arr) {
    std::vector<cv::aruco2::Diamond> vec;
    vec.reserve([arr count]);
    for (Aruco2Diamond* d in arr) {
        vec.push_back(d.nativeRef);
    }
    return vec;
}

// Helper: NSArray<Aruco2FractalMarker*> → std::vector<FractalMarker>
static std::vector<cv::aruco2::FractalMarker> fractalArrayToVec(NSArray<Aruco2FractalMarker*>* arr) {
    std::vector<cv::aruco2::FractalMarker> vec;
    vec.reserve([arr count]);
    for (Aruco2FractalMarker* f in arr) {
        vec.push_back(f.nativeRef);
    }
    return vec;
}

@implementation Aruco2

// ---- getFiducialMarkerImage --------------------------------------------------

+ (void)getFiducialMarkerImage:(Mat*)img dictionary:(int)dictionary id:(int)markerId
                       bitSize:(int)bitSize externalBorder:(BOOL)externalBorder {
    cv::aruco2::getFiducialMarkerImage(img.nativeRef, (cv::aruco2::DictionaryType)dictionary,
                                       markerId, bitSize, (bool)externalBorder);
}

+ (void)getFiducialMarkerImage:(Mat*)img dictionary:(int)dictionary id:(int)markerId {
    cv::aruco2::getFiducialMarkerImage(img.nativeRef, (cv::aruco2::DictionaryType)dictionary, markerId);
}

// ---- detectFiducialMarkers (single dict) ------------------------------------

+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkers:(Mat*)image dictionary:(int)dictionary
                                                    params:(nullable Aruco2DetectionParameters*)params {
    cv::aruco2::DetectionParameters cppParams = (params != nil)
        ? params.nativeRef
        : cv::aruco2::DetectionParameters{};
    std::vector<cv::aruco2::FiducialMarker> result =
        cv::aruco2::detectFiducialMarkers(image.nativeRef, (cv::aruco2::DictionaryType)dictionary, cppParams);
    return fiducialVecToArray(result);
}

+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkers:(Mat*)image dictionary:(int)dictionary {
    return [Aruco2 detectFiducialMarkers:image dictionary:dictionary params:nil];
}

// ---- detectFiducialMarkers (multiple dicts) ---------------------------------

+ (NSArray<Aruco2FiducialMarker*>*)detectFiducialMarkersMulti:(Mat*)image
                                                          dicts:(NSArray<NSNumber*>*)dicts
                                                         params:(nullable Aruco2DetectionParameters*)params {
    std::vector<cv::aruco2::DictionaryType> dictVec = nsArrayToDictTypeVector(dicts);
    cv::aruco2::DetectionParameters cppParams = (params != nil)
        ? params.nativeRef
        : cv::aruco2::DetectionParameters{};
    std::vector<cv::aruco2::FiducialMarker> result =
        cv::aruco2::detectFiducialMarkers(image.nativeRef, dictVec, cppParams);
    return fiducialVecToArray(result);
}

// ---- drawFiducialMarkers ----------------------------------------------------

+ (void)drawFiducialMarkers:(Mat*)image markers:(NSArray<Aruco2FiducialMarker*>*)markers
                borderColor:(Scalar*)borderColor {
    std::vector<cv::aruco2::FiducialMarker> vec = fiducialArrayToVec(markers);
    cv::aruco2::drawFiducialMarkers(image.nativeRef, vec, borderColor.nativeRef);
}

+ (void)drawFiducialMarkers:(Mat*)image markers:(NSArray<Aruco2FiducialMarker*>*)markers {
    std::vector<cv::aruco2::FiducialMarker> vec = fiducialArrayToVec(markers);
    cv::aruco2::drawFiducialMarkers(image.nativeRef, vec);
}

// ---- drawAxis ---------------------------------------------------------------

+ (void)drawAxis:(Mat*)image cameraMatrix:(Mat*)cameraMatrix distCoeffs:(Mat*)distCoeffs
            rvec:(Mat*)rvec tvec:(Mat*)tvec length:(float)length {
    cv::aruco2::drawAxis(image.nativeRef, cameraMatrix.nativeRef, distCoeffs.nativeRef,
                         rvec.nativeRef, tvec.nativeRef, length);
}

// ---- getSolvePnpPoints (FiducialMarker) -------------------------------------

+ (void)getSolvePnpPointsForFiducialMarker:(Aruco2FiducialMarker*)marker
                                 objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints
                                markerSize:(float)markerSize {
    cv::aruco2::getSolvePnpPoints(marker.nativeRef, objPoints.nativeRef, imgPoints.nativeRef, markerSize);
}

+ (void)getSolvePnpPointsForFiducialMarker:(Aruco2FiducialMarker*)marker
                                 objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints {
    cv::aruco2::getSolvePnpPoints(marker.nativeRef, objPoints.nativeRef, imgPoints.nativeRef);
}

// ---- getGridBoardImage ------------------------------------------------------

+ (void)getGridBoardImage:(Mat*)img boardSize:(Size2i*)boardSize dictionary:(int)dictionary
                  bitSize:(int)bitSize ids:(nullable Mat*)ids {
    if (ids != nil) {
        cv::aruco2::getGridBoardImage(img.nativeRef, boardSize.nativeRef,
                                      (cv::aruco2::DictionaryType)dictionary, bitSize, ids.nativeRef);
    } else {
        cv::aruco2::getGridBoardImage(img.nativeRef, boardSize.nativeRef,
                                      (cv::aruco2::DictionaryType)dictionary, bitSize);
    }
}

+ (void)getGridBoardImage:(Mat*)img boardSize:(Size2i*)boardSize dictionary:(int)dictionary {
    cv::aruco2::getGridBoardImage(img.nativeRef, boardSize.nativeRef, (cv::aruco2::DictionaryType)dictionary);
}

// ---- detectGridBoard --------------------------------------------------------

+ (BOOL)detectGridBoard:(Mat*)image gridSize:(Size2i*)gridSize dictionary:(int)dictionary
                  board:(Aruco2GridBoard*)board ids:(nullable Mat*)ids {
    bool found;
    if (ids != nil) {
        found = cv::aruco2::detectGridBoard(image.nativeRef, gridSize.nativeRef,
                                            (cv::aruco2::DictionaryType)dictionary,
                                            board.nativeRef, ids.nativeRef);
    } else {
        found = cv::aruco2::detectGridBoard(image.nativeRef, gridSize.nativeRef,
                                            (cv::aruco2::DictionaryType)dictionary,
                                            board.nativeRef);
    }
    return found ? YES : NO;
}

+ (BOOL)detectGridBoard:(Mat*)image gridSize:(Size2i*)gridSize dictionary:(int)dictionary
                  board:(Aruco2GridBoard*)board {
    return [Aruco2 detectGridBoard:image gridSize:gridSize dictionary:dictionary board:board ids:nil];
}

// ---- drawGridBoard ----------------------------------------------------------

+ (void)drawGridBoard:(Mat*)image board:(Aruco2GridBoard*)board
                color:(Scalar*)color drawMarkerIds:(BOOL)drawMarkerIds {
    cv::aruco2::drawGridBoard(image.nativeRef, board.nativeRef, color.nativeRef, (bool)drawMarkerIds);
}

+ (void)drawGridBoard:(Mat*)image board:(Aruco2GridBoard*)board {
    cv::aruco2::drawGridBoard(image.nativeRef, board.nativeRef);
}

// ---- getSolvePnpPoints (GridBoard) ------------------------------------------

+ (void)getSolvePnpPointsForGridBoard:(Aruco2GridBoard*)board
                             objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints
                            markerSize:(float)markerSize {
    cv::aruco2::getSolvePnpPoints(board.nativeRef, objPoints.nativeRef, imgPoints.nativeRef, markerSize);
}

+ (void)getSolvePnpPointsForGridBoard:(Aruco2GridBoard*)board
                             objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints {
    cv::aruco2::getSolvePnpPoints(board.nativeRef, objPoints.nativeRef, imgPoints.nativeRef);
}

// ---- getDiamondImage --------------------------------------------------------

+ (void)getDiamondImage:(Mat*)img dictionary:(int)dictionary ids:(Int4*)ids bitSize:(int)bitSize {
    cv::aruco2::getDiamondImage(img.nativeRef, (cv::aruco2::DictionaryType)dictionary,
                                ids.nativeRef, bitSize);
}

+ (void)getDiamondImage:(Mat*)img dictionary:(int)dictionary ids:(Int4*)ids {
    cv::aruco2::getDiamondImage(img.nativeRef, (cv::aruco2::DictionaryType)dictionary,
                                ids.nativeRef);
}

// ---- detectDiamonds ---------------------------------------------------------

+ (NSArray<Aruco2Diamond*>*)detectDiamonds:(Mat*)image dictionary:(int)dictionary {
    std::vector<cv::aruco2::Diamond> result =
        cv::aruco2::detectDiamonds(image.nativeRef, (cv::aruco2::DictionaryType)dictionary);
    return diamondVecToArray(result);
}

// ---- drawDiamonds -----------------------------------------------------------

+ (void)drawDiamonds:(Mat*)image diamonds:(NSArray<Aruco2Diamond*>*)diamonds
               color:(Scalar*)color drawMarkerIds:(BOOL)drawMarkerIds {
    std::vector<cv::aruco2::Diamond> vec = diamondArrayToVec(diamonds);
    cv::aruco2::drawDiamonds(image.nativeRef, vec, color.nativeRef, (bool)drawMarkerIds);
}

+ (void)drawDiamonds:(Mat*)image diamonds:(NSArray<Aruco2Diamond*>*)diamonds {
    std::vector<cv::aruco2::Diamond> vec = diamondArrayToVec(diamonds);
    cv::aruco2::drawDiamonds(image.nativeRef, vec);
}

// ---- getSolvePnpPoints (Diamond) --------------------------------------------

+ (void)getSolvePnpPointsForDiamond:(Aruco2Diamond*)diamond
                           objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints
                          markerSize:(float)markerSize {
    cv::aruco2::getSolvePnpPoints(diamond.nativeRef, objPoints.nativeRef, imgPoints.nativeRef, markerSize);
}

+ (void)getSolvePnpPointsForDiamond:(Aruco2Diamond*)diamond
                           objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints {
    cv::aruco2::getSolvePnpPoints(diamond.nativeRef, objPoints.nativeRef, imgPoints.nativeRef);
}

// ---- getFractalMarkerImage --------------------------------------------------

+ (void)getFractalMarkerImage:(Mat*)img ftype:(int)ftype bitSize:(int)bitSize {
    cv::aruco2::getFractalMarkerImage(img.nativeRef, (cv::aruco2::FractalType)ftype, bitSize);
}

+ (void)getFractalMarkerImage:(Mat*)img ftype:(int)ftype {
    cv::aruco2::getFractalMarkerImage(img.nativeRef, (cv::aruco2::FractalType)ftype);
}

// ---- detectFractals ---------------------------------------------------------

+ (NSArray<Aruco2FractalMarker*>*)detectFractals:(Mat*)image ftype:(int)ftype {
    std::vector<cv::aruco2::FractalMarker> result =
        cv::aruco2::detectFractals(image.nativeRef, (cv::aruco2::FractalType)ftype);
    return fractalVecToArray(result);
}

// ---- drawFractals -----------------------------------------------------------

+ (void)drawFractals:(Mat*)image fractals:(NSArray<Aruco2FractalMarker*>*)fractals
               color:(Scalar*)color drawAllImagePoints:(BOOL)drawAllImagePoints {
    std::vector<cv::aruco2::FractalMarker> vec = fractalArrayToVec(fractals);
    cv::aruco2::drawFractals(image.nativeRef, vec, color.nativeRef, (bool)drawAllImagePoints);
}

+ (void)drawFractals:(Mat*)image fractals:(NSArray<Aruco2FractalMarker*>*)fractals {
    std::vector<cv::aruco2::FractalMarker> vec = fractalArrayToVec(fractals);
    cv::aruco2::drawFractals(image.nativeRef, vec);
}

// ---- getSolvePnpPoints (FractalMarker) --------------------------------------

+ (void)getSolvePnpPointsForFractalMarker:(Aruco2FractalMarker*)fractal
                                 objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints
                                markerSize:(float)markerSize {
    cv::aruco2::getSolvePnpPoints(fractal.nativeRef, objPoints.nativeRef, imgPoints.nativeRef, markerSize);
}

+ (void)getSolvePnpPointsForFractalMarker:(Aruco2FractalMarker*)fractal
                                 objPoints:(Mat*)objPoints imgPoints:(Mat*)imgPoints {
    cv::aruco2::getSolvePnpPoints(fractal.nativeRef, objPoints.nativeRef, imgPoints.nativeRef);
}

@end
