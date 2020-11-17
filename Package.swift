// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
            path: "modules/core/misc/objc/swift-package-manager/Sources"),
        // Recompute checksum via `swift package --package-path /path/to/opencv compute-checksum /path/to/opencv2.xcframework.zip`
//        .binaryTarget(
//            name: "opencv2",
//            url: "https://url/to/some/remote/opencv2.xcframework.zip",
//            checksum: "The checksum of the ZIP archive that contains the XCFramework."
//        ),
        // If you are compiling OpenCV locally, you can uncomment the below block to use a custom copy
        .binaryTarget(
            name: "opencv2",
            path: "platforms/apple/xcframework-build/opencv2.xcframework"
        ),
        .testTarget(
            name: "OpenCVTests",
            dependencies: ["OpenCV"],
            path: "modules/core/misc/objc/swift-package-manager/Tests"
            ),
    ]
)
