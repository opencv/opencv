//
//  Aruco2FractalMarker.mm
//

#import "Aruco2FractalMarker.h"

@implementation Aruco2FractalMarker {
    cv::aruco2::FractalMarker native;
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

- (int)type { return (int)native.type; }
- (void)setType:(int)v { native.type = (cv::aruco2::FractalType)v; }

- (int)id { return native.id; }
- (void)setId:(int)v { native.id = v; }

- (cv::aruco2::FractalMarker&)nativeRef { return native; }

- (instancetype)init {
    self = [super init];
    return self;
}

+ (instancetype)fromNative:(cv::aruco2::FractalMarker&)fractal {
    Aruco2FractalMarker* obj = [[Aruco2FractalMarker alloc] init];
    obj->native = fractal;
    return obj;
}

- (Point2f*)getCorner:(int)i {
    return [Point2f fromNative:native.corners[i]];
}

- (NSString*)description {
    return [NSString stringWithFormat:@"Aruco2FractalMarker { id: %d, type: %d, corners: %lu }",
            native.id, (int)native.type, (unsigned long)native.corners.size()];
}

@end
