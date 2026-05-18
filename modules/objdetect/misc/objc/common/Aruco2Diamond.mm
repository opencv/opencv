//
//  Aruco2Diamond.mm
//

#import "Aruco2Diamond.h"

@implementation Aruco2Diamond {
    cv::aruco2::Diamond native;
}

- (Int4*)id {
    return [Int4 fromNative:native.id];
}

- (void)setId:(Int4*)v {
    native.id = v.nativeRef;
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

- (cv::aruco2::Diamond&)nativeRef { return native; }

- (instancetype)init {
    self = [super init];
    if (self) { native = cv::aruco2::Diamond(); }
    return self;
}

+ (instancetype)fromNative:(cv::aruco2::Diamond&)diamond {
    Aruco2Diamond* obj = [[Aruco2Diamond alloc] init];
    obj->native = diamond;
    return obj;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"Aruco2Diamond { id: [%d,%d,%d,%d] }",
            native.id[0], native.id[1], native.id[2], native.id[3]];
}

@end
