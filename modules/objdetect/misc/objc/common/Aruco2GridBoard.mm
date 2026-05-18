//
//  Aruco2GridBoard.mm
//

#import "Aruco2GridBoard.h"

@implementation Aruco2GridBoard {
    cv::aruco2::GridBoard native;
}

- (Size2i*)gridSize {
    return [Size2i fromNative:native.gridSize];
}

- (void)setGridSize:(Size2i*)size {
    native.gridSize = size.nativeRef;
}

- (int)dict { return (int)native.dict; }
- (void)setDict:(int)v { native.dict = (cv::aruco2::DictionaryType)v; }

- (NSArray<Aruco2FiducialMarker*>*)markers {
    NSMutableArray<Aruco2FiducialMarker*>* result = [NSMutableArray arrayWithCapacity:native.markers.size()];
    for (size_t i = 0; i < native.markers.size(); i++) {
        [result addObject:[Aruco2FiducialMarker fromNative:native.markers[i]]];
    }
    return result;
}

- (void)setMarkers:(NSArray<Aruco2FiducialMarker*>*)markers {
    native.markers.clear();
    for (Aruco2FiducialMarker* m in markers) {
        native.markers.push_back(m.nativeRef);
    }
}

- (cv::aruco2::GridBoard&)nativeRef { return native; }

- (instancetype)init {
    self = [super init];
    if (self) { native = cv::aruco2::GridBoard(); }
    return self;
}

+ (instancetype)fromNative:(cv::aruco2::GridBoard&)board {
    Aruco2GridBoard* obj = [[Aruco2GridBoard alloc] init];
    obj->native = board;
    return obj;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"Aruco2GridBoard { gridSize: %dx%d, markers: %lu }",
            native.gridSize.width, native.gridSize.height, (unsigned long)native.markers.size()];
}

@end
