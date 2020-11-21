// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "OpenCV",
    platforms: [
        .macOS(.v10_12), .iOS(.v9),
    ],
    products: [
        .library(
            name: "OpenCV",
            targets: ["OpenCV"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "OpenCV",
            dependencies: ["opencv2"],
            path: "modules/core/misc/objc/swift-package-manager/Sources",
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("Accelerate"),
                .linkedFramework("OpenCL", .when(platforms: [.macOS]))
            ]
        ),
        // Recompute checksum via `swift package --package-path /path/to/opencv compute-checksum /path/to/opencv2.xcframework.zip`
        .binaryTarget(
            name: "opencv2",
            url: "https://github.com/Rightpoint/opencv/releases/download/4.5.1/opencv2-4.5.1-dynamic.xcframework.zip",
            checksum: "a2088bcdebb64e6126e9b85409083b06d8ea7b3ab206e72daa06b105f5ca6f28"
        ),
        // If you are compiling OpenCV locally, you can uncomment the below block to use a custom copy
        // e.g. `$ python platforms/apple/build_xcframework.py --dynamic build/dynamic-xcframework`
//        .binaryTarget(
//            name: "opencv2",
//            path: "build/dynamic-xcframework/opencv2.xcframework"
//        ),
        .testTarget(
            name: "OpenCVTests",
            dependencies: ["OpenCV"],
            path: "modules/core/misc/objc/test/swift",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "OpenCVObjCTests",
            dependencies: ["OpenCV"],
            path: "modules/core/misc/objc/test/objc"
        ),
    ]
)
