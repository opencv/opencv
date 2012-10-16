#!/usr/bin/python

import os
import sys

ANDROID_SDK_PATH = "/opt/android-sdk-linux"
ANDROID_NDK_PATH = None
INSTALL_DIRECTORY = None
CLASS_PATH = None
TMP_HEADER_PATH="tmp_include"
HEADER_EXTS = set(['h', 'hpp'])
SYS_INCLUDES = ["platforms/android-8/arch-arm/usr/include", "sources/cxx-stl/gnu-libstdc++/include", "sources/cxx-stl/gnu-libstdc++/libs/armeabi/include"]

PROJECT_NAME = "OpenCV-branch"
TARGET_LIBS = ["libopencv_java.so"]
ARCH = "armeabi"
GCC_OPTIONS = "-fpermissive"
EXCLUDE_HEADERS = set(["hdf5.h", "eigen.hpp", "cxeigen.hpp"]);

def FindClasses(root, prefix):
    classes = []
    if ("" != prefix):
	prefix = prefix + "."
    for path in os.listdir(root):
	currentPath = os.path.join(root, path)
	if (os.path.isdir(currentPath)):
	    classes += FindClasses(currentPath, prefix + path)
	else:
	    name = str.split(path, ".")[0]
	    ext = str.split(path, ".")[1]
	    if (ext == "class"):
		#print("class: %s" % (prefix + name))
		classes.append(prefix+name)
    return classes

def FindHeaders(root):
    headers = []
    for path in os.listdir(root):
	currentPath = os.path.join(root, path)
	if (os.path.isdir(currentPath)):
	    headers += FindHeaders(currentPath)
	else:
	    ext = str.split(path, ".")[-1]
	    #print("%s: \"%s\"" % (currentPath, ext))
	    if (ext in HEADER_EXTS):
		#print("Added as header file")
		if (path not in EXCLUDE_HEADERS):
		    headers.append(currentPath)
    return headers

if (len(sys.argv) < 3):
    print("Error: Invalid command line arguments")
    exit(-1)

INSTALL_DIRECTORY = sys.argv[1]
PROJECT_NAME = sys.argv[2]

CLASS_PATH = os.path.join(INSTALL_DIRECTORY, "sdk/java/bin/classes")
if (not os.path.exists(CLASS_PATH)):
    print("Error: no java classes found in \"%s\"" % CLASS_PATH)
    exit(-2)

if (os.environ.has_key("NDK_ROOT")):
    ANDROID_NDK_PATH = os.environ["NDK_ROOT"];
    print("Using Android NDK from NDK_ROOT (\"%s\")" % ANDROID_NDK_PATH)

if (not ANDROID_NDK_PATH):
    pipe = os.popen("which ndk-build")
    tmp = str.strip(pipe.readline(), "\n")
    while(not tmp):
	tmp = str.strip(pipe.readline(), "\n")
    pipe.close()
    ANDROID_NDK_PATH = os.path.split(tmp)[0]
    print("Using Android NDK from PATH (\"%s\")" % ANDROID_NDK_PATH)

print("Using Android SDK from \"%s\"" % ANDROID_SDK_PATH)

outputFileName = PROJECT_NAME + ".xml"
try:
    outputFile = open(outputFileName, "w")
except:
    print("Error: Cannot open output file \"%s\" for writing" % outputFileName)

allJavaClasses = FindClasses(CLASS_PATH, "")
if (not allJavaClasses):
    print("Error: No Java classes found :(")
    exit(-1)

if (not os.path.exists(TMP_HEADER_PATH)):
    os.makedirs(os.path.join(os.getcwd(), TMP_HEADER_PATH))

print("Generating JNI headers for Java API ...")
AndroidJavaDeps = os.path.join(ANDROID_SDK_PATH, "platforms/android-11/android.jar")
for currentClass in allJavaClasses:
    os.system("javah -d %s -classpath %s:%s %s" % (TMP_HEADER_PATH, CLASS_PATH, AndroidJavaDeps, currentClass))

print("Building JNI headers list ...")
jniHeaders = FindHeaders(TMP_HEADER_PATH)
#print(jniHeaders)

print("Building Native OpenCV header list ...")
cHeaders = FindHeaders(os.path.join(INSTALL_DIRECTORY, "sdk/native/jni/include/opencv"))
cppHeaders = FindHeaders(os.path.join(INSTALL_DIRECTORY, "sdk/native/jni/include/opencv2"))
#print(cHeaders)
#print(cppHeaders)

print("Writing config file ...")
outputFile.write("<descriptor>\n\n<version>\n\t%s\n</version>\n\n<headers>\n" % PROJECT_NAME)
outputFile.write("\t"   + "\n\t".join(cHeaders))
outputFile.write("\n\t" + "\n\t".join(cppHeaders))
outputFile.write("\n\t" + "\n\t".join(jniHeaders))
outputFile.write("\n</headers>\n\n")

includes = [os.path.join(INSTALL_DIRECTORY, "sdk", "native", "jni", "include"),
    os.path.join(INSTALL_DIRECTORY, "sdk", "native", "jni", "include", "opencv"),
    os.path.join(INSTALL_DIRECTORY, "sdk", "native", "jni", "include", "opencv2")]

for inc in SYS_INCLUDES:
    includes.append(os.path.join(ANDROID_NDK_PATH, inc))

outputFile.write("<include_paths>\n\t%s\n</include_paths>\n\n" % "\n\t".join(includes))

libraries = []
for lib in TARGET_LIBS:
    libraries.append(os.path.join(INSTALL_DIRECTORY, "sdk/native/libs", ARCH, lib))

outputFile.write("<libs>\n\t%s\n</libs>\n\n" % "\n\t".join(libraries))
outputFile.write("<gcc_options>\n\t%s\n</gcc_options>\n\n</descriptor>" % GCC_OPTIONS)

print("done!")
