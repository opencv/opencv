#!/usr/bin/python

import os
import sys
import shutil

ScriptHome = os.path.split(sys.argv[0])[0]
ConfFile = open(os.path.join(ScriptHome, "camera_build.conf"), "rt")
HomeDir = os.getcwd()

stub = os.environ.get("ANDROID_STUB_ROOT", "")

if (stub == ""):
    print("Warning: ANDROID_STUB_ROOT environment variable is not set or is empty")

for s in ConfFile.readlines():
    s = s[0:s.find("#")]
    if (not s):
        continue
    keys = s.split(";")
    if (len(keys) < 4):
        print("Error: invalid config line: \"%s\"" % s)
        continue
    MakeTarget = str.strip(keys[0])
    Arch = str.strip(keys[1])
    NativeApiLevel = str.strip(keys[2])
    AndroidTreeRoot = str.strip(keys[3])
    AndroidTreeRoot = str.strip(AndroidTreeRoot, "\n")
    AndroidTreeRoot = os.path.expandvars(AndroidTreeRoot)
    print("Building %s for %s" % (MakeTarget, Arch))
    BuildDir = os.path.join(HomeDir, MakeTarget + "_" + Arch)

    if (os.path.exists(BuildDir)):
        shutil.rmtree(BuildDir)

    try:
        os.mkdir(BuildDir)
    except:
        print("Error: cannot create direcotry \"%s\"" % BuildDir)
        continue

    shutil.rmtree(os.path.join(AndroidTreeRoot, "out", "target", "product", "generic", "system"), ignore_errors=True)

    LinkerLibs = os.path.join(AndroidTreeRoot, "bin_arm", "system")
    if (Arch == "x86"):
        LinkerLibs = os.path.join(AndroidTreeRoot, "bin_x86", "system")
    elif (Arch == "mips"):
        LinkerLibs = os.path.join(AndroidTreeRoot, "bin_mips", "system")

    if (not os.path.exists(LinkerLibs)):
        print("Error: Paltform libs for linker in path \"%s\" not found" % LinkerLibs)
        print("Building %s for %s\t[\033[91mFAILED\033[0m]" % (MakeTarget, Arch))
        continue

    shutil.copytree(LinkerLibs, os.path.join(AndroidTreeRoot, "out", "target", "product", "generic", "system"))

    os.chdir(BuildDir)
    BuildLog = os.path.join(BuildDir, "build.log")
    CmakeCmdLine = "cmake -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake -DANDROID_SOURCE_TREE=\"%s\" -DANDROID_NATIVE_API_LEVEL=\"%s\" -DANDROID_ABI=\"%s\" -DANDROID_STL=stlport_static ../.. > \"%s\" 2>&1" % (AndroidTreeRoot, NativeApiLevel, Arch, BuildLog)
    MakeCmdLine = "make %s >> \"%s\" 2>&1" % (MakeTarget, BuildLog);
    #print(CmakeCmdLine)
    os.system(CmakeCmdLine)
    #print(MakeCmdLine)
    os.system(MakeCmdLine)
    os.chdir(HomeDir)
    CameraLib = os.path.join(BuildDir, "lib", Arch, "lib" + MakeTarget + ".so")
    if (os.path.exists(CameraLib)):
        try:
            shutil.copyfile(CameraLib, os.path.join("..", "3rdparty", "lib", Arch, "lib" + MakeTarget + ".so"))
            print("Building %s for %s\t[\033[92mOK\033[0m]" % (MakeTarget, Arch));
        except:
            print("Building %s for %s\t[\033[91mFAILED\033[0m]" % (MakeTarget, Arch));
    else:
        print("Building %s for %s\t[\033[91mFAILED\033[0m]" % (MakeTarget, Arch));

ConfFile.close()
