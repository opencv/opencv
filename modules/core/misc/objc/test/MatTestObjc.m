//
//  MatTests.m
//
//  Created by Giles Payne on 2020/01/25.
//

#import <XCTest/XCTest.h>
#import <OpenCV/OpenCV.h>

#define CV_8U 0
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)
#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags) & CV_MAT_DEPTH_MASK)
#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)

@interface MatTestsObjc : XCTestCase

@end

@implementation MatTestsObjc

// XCTAssertThrows only works in Objective-C so these tests are separate from the main MatTest.swift
- (void)testBadData {
    Mat* m1 = [[Mat alloc] initWithRows:5 cols:5 type:CV_8UC3];
    Mat* m2 = [[Mat alloc] initWithSizes:@[@5, @5, @5] type:CV_8UC3];
    Mat* m3 = [[Mat alloc] initWithRows:5 cols:5 type:CV_32FC3];
    Mat* m4 = [[Mat alloc] initWithSizes:@[@5, @5, @5] type:CV_32FC3];
    Mat* m5 = [[Mat alloc] initWithRows:5 cols:5 type:CV_32SC3];
    Mat* m6 = [[Mat alloc] initWithSizes:@[@5, @5, @5] type:CV_32SC3];
    Mat* m7 = [[Mat alloc] initWithRows:5 cols:5 type:CV_16SC3];
    Mat* m8 = [[Mat alloc] initWithSizes:@[@5, @5, @5] type:CV_16SC3];
    NSMutableArray<NSNumber*>* badData7 = [NSMutableArray arrayWithArray: @[@0, @0, @0, @0, @0, @0, @0]];
    NSMutableArray<NSNumber*>* badData5 = [NSMutableArray arrayWithArray: @[@0, @0, @0, @0, @0]];

    XCTAssertThrows([m1 get: 2 col: 2 data: badData7]);
    XCTAssertThrows([m1 put: 2 col: 2 data: badData5]);
    XCTAssertThrows([m2 put:(@[@2, @2, @0]) data: badData5]);
    XCTAssertThrows([m3 put: 2 col: 2 data: badData5]);
    XCTAssertThrows([m4 put:(@[@4, @2, @2]) data: badData5]);
    XCTAssertThrows([m5 put: 2 col: 2 data: badData5]);
    XCTAssertThrows([m6 put:(@[@2, @2, @0]) data: badData5]);
    XCTAssertThrows([m7 put: 2 col: 2 data: badData5]);
    XCTAssertThrows([m8 put:(@[@2, @2, @0]) data: badData5]);
}

- (void)testRelease {
    Mat* m = [[Mat alloc] initWithRows:5 cols:5 type:CV_8UC3];
    XCTAssertNoThrow(m = nil);
}
@end
