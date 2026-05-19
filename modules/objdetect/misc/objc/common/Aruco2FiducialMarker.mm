//
//  Aruco2FiducialMarker.mm
//

#import "Aruco2FiducialMarker.h"

@implementation Aruco2FiducialMarker {
    cv::aruco2::FiducialMarker native;
}

- (NSMutableArray<Point2f*>*)corners {
    NSMutableArray<Point2f*>* result = [NSMutableArray arrayWithCapacity:native.corners.size()];
    for (size_t i = 0; i < native.corners.size(); i++) {
        [result addObject:[Point2f fromNative:native.corners[i]]];
    }
    return result;
}

- (void)setCorners:(NSMutableArray<Point2f*>*)corners {
    native.corners.clear();
    for (Point2f* p in corners) {
        native.corners.push_back(p.nativeRef);
    }
}

- (int)id { return native.id; }
- (void)setId:(int)v { native.id = v; }

- (int)dictionary { return (int)native.dictionary; }
- (void)setDictionary:(int)v { native.dictionary = (cv::aruco2::DictionaryType)v; }

- (cv::aruco2::FiducialMarker&)nativeRef { return native; }

- (instancetype)init {
    self = [super init];
    return self;
}

+ (instancetype)fromNative:(cv::aruco2::FiducialMarker&)marker {
    Aruco2FiducialMarker* obj = [[Aruco2FiducialMarker alloc] init];
    obj->native = marker;
    return obj;
}

- (Point2f*)getCorner:(int)i {
    return [Point2f fromNative:native.corners[i]];
}

- (NSString*)description {
    return [NSString stringWithFormat:@"Aruco2FiducialMarker { id: %d, corners: %lu }",
            native.id, (unsigned long)native.corners.size()];
}

@end
