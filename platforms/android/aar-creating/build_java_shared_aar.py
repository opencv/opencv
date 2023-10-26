import argparse
from os import path
import os
import shutil
import subprocess

from build_static_aar import fill_template


COPY_FROM_SDK_TO_ANDROID_PROJECT = [
    ["sdk/native/jni/include", "OpenCV/src/main/cpp/include"],
    ["sdk/java/src/org", "OpenCV/src/main/java/org"],
    ["sdk/java/res", "OpenCV/src/main/res"]
]

ABIS = ["arm64-v8a", "armeabi-v7a", "x86", "x86_64"] # if you want to change it, you also need to change Android project template

COPY_FROM_SDK_TO_APK = [
    ["sdk/native/libs/<ABI>/libopencv_java4.so", "jni/<ABI>/libopencv_java4.so"],
    ["sdk/native/libs/<ABI>/libopencv_java4.so", "prefab/modules/opencv_java4/libs/android.<ABI>/libopencv_java4.so"],
]

ANDROID_PROJECT_TEMPLATE_DIR = "OpenCVAndroidProject"
ANDROID_PROJECT_DIR = "build_java_shared/AndroidProject"
COMPILED_AAR_PATH = path.join(ANDROID_PROJECT_DIR, "OpenCV/build/outputs/aar/opencv-release.aar")
AAR_UNZIPPED_DIR = "build_java_shared/aar_unzipped"
FINAL_AAR_PATH = "outputs/opencv_java_shared.aar"
FINAL_REPO_PATH = "outputs/maven_repo"

def main(sdk_dir, opencv_version):
    print("Preparing Android project...")
    # ANDROID_PROJECT_TEMPLATE_DIR contains an Android project template that creates AAR
    shutil.copytree(ANDROID_PROJECT_TEMPLATE_DIR, ANDROID_PROJECT_DIR)

    # Configuring the Android project to Java + shared C++ lib version
    shutil.rmtree(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/include"))

    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle"),
                  {"LIB_NAME": "opencv_java4", "LIB_TYPE": "c++_shared", "PACKAGE_NAME": "opencv", "OPENCV_VERSION": opencv_version})
    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt"),
                  {"LIB_NAME": "opencv_java4", "LIB_TYPE": "SHARED"})

    # Copying Java code and C++ public headers from SDK to the Android project
    for src, dst in COPY_FROM_SDK_TO_ANDROID_PROJECT:
        shutil.copytree(path.join(sdk_dir, src),
                        path.join(ANDROID_PROJECT_DIR, dst))

    print("Running gradle assembleRelease...")
    # Running gradle to build the Android project
    subprocess.run(["gradlew", "assembleRelease"],
                shell=True,
                cwd=ANDROID_PROJECT_DIR)

    print("Adding libs to AAR...")
    # The created AAR package doesn't contain C++ shared libs.
    # We need to add them manually.
    # AAR package is just a zip archive.
    shutil.unpack_archive(COMPILED_AAR_PATH, AAR_UNZIPPED_DIR, "zip")

    for abi in ABIS:
        for src, dst in COPY_FROM_SDK_TO_APK:
            src = src.replace("<ABI>", abi)
            dst = dst.replace("<ABI>", abi)
            shutil.copy(path.join(sdk_dir, src),
                path.join(AAR_UNZIPPED_DIR, dst))

    # Creating final AAR zip archive
    os.makedirs("outputs", exist_ok=True)
    shutil.make_archive(FINAL_AAR_PATH, "zip", AAR_UNZIPPED_DIR, ".")
    os.rename(FINAL_AAR_PATH + ".zip", FINAL_AAR_PATH)

    print("Creating local maven repo...")

    shutil.copy(FINAL_AAR_PATH, path.join(ANDROID_PROJECT_DIR, "OpenCV/opencv-release.aar"))
    subprocess.run(["gradlew", "publishReleasePublicationToMyrepoRepository"],
            shell=True,
            cwd=ANDROID_PROJECT_DIR)

    os.makedirs(path.join(FINAL_REPO_PATH, "org/opencv"), exist_ok=True)
    shutil.move(path.join(ANDROID_PROJECT_DIR, "OpenCV/build/repo/org/opencv/opencv"),
                path.join(FINAL_REPO_PATH, "org/opencv/opencv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds AAR with Java and shared C++ libs from OpenCV SDK")
    parser.add_argument('opencv_sdk_path')
    parser.add_argument('opencv_version')
    args = parser.parse_args()

    main(args.opencv_sdk_path, args.opencv_version)
