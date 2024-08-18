## Scripts for creating an AAR package and a local Maven repository with OpenCV libraries for Android

### How to run the scripts
1. Set JAVA_HOME and ANDROID_HOME environment variables. For example:
```
export JAVA_HOME=~/Android Studio/jbr
export ANDROID_HOME=~/Android/SDK
```
2. Download OpenCV SDK for Android
3. Run build script for version with Java and a shared C++ library:
```
python build_java_shared_aar.py "~/opencv-4.7.0-android-sdk/OpenCV-android-sdk"
```
4. Run build script for version with static C++ libraries:
```
python build_static_aar.py "~/opencv-4.7.0-android-sdk/OpenCV-android-sdk"
```
The AAR libraries and the local Maven repository will be created in the **outputs** directory
### Technical details
The scripts consist of 5 steps:
1. Preparing Android AAR library project template
2. Adding Java code to the project. Adding C++ public headers for shared version to the project.
3. Compiling the project to build an AAR package
4. Adding C++ binary libraries to the AAR package. Adding C++ public headers for static version to the AAR package.
5. Creating Maven repository with the AAR package

There are a few minor limitations:
1. Due to the AAR design the Java + shared C++ AAR package contains duplicates of C++ binary libraries, but the final user's Android application contains only one library instance.
2. The compile definitions from cmake configs are skipped, but it shouldn't affect the library because the script uses precompiled C++ binaries from SDK.
