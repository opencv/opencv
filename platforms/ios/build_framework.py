#!/usr/bin/env python3
"""
The script builds OpenCV.framework for iOS.
The built framework is universal, it can be used to build app and run it on either iOS simulator or real device.

Usage:
    ./build_framework.py <outputdir>

By cmake conventions (and especially if you work with OpenCV repository),
the output dir should not be a subdirectory of OpenCV source tree.

Script will create <outputdir>, if it's missing, and a few its subdirectories:

    <outputdir>
        build/
            iPhoneOS-*/
               [cmake-generated build tree for an iOS device target]
            iPhoneSimulator-*/
               [cmake-generated build tree for iOS simulator]
        {framework_name}.framework/
            [the framework content]

The script should handle minor OpenCV updates efficiently
- it does not recompile the library from scratch each time.
However, {framework_name}.framework directory is erased and recreated on each run.

Adding --dynamic parameter will build {framework_name}.framework as App Store dynamic framework. Only iOS 8+ versions are supported.
"""

from __future__ import print_function
import glob, os, os.path, shutil, sys, argparse, traceback, multiprocessing
from subprocess import check_output

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'apple')))
from cv_build_utils import execute, print_error, get_xcode_major, get_xcode_setting, get_xcode_version, get_cmake_version

IPHONEOS_DEPLOYMENT_TARGET = '9.0'  # default, can be changed via command line options or environment variable

CURRENT_FILE_DIR = os.path.dirname(__file__)
APPLE_CLANG_FLAGS = '-fno-define-target-os-macros'


class Builder:
    def __init__(self, opencv, contrib, dynamic, bitcodedisabled, exclude, disable, enablenonfree, targets, debug, debug_info, framework_name):
        self.opencv = os.path.abspath(opencv)
        self.contrib = None
        if contrib:
            modpath = os.path.join(contrib, "modules")
            if os.path.isdir(modpath):
                self.contrib = os.path.abspath(modpath)
            else:
                print("Note: contrib repository is bad - modules subfolder not found", file=sys.stderr)
        self.dynamic = dynamic
        self.bitcodedisabled = bitcodedisabled
        self.exclude = exclude
        self.disable = disable
        self.enablenonfree = enablenonfree
        self.targets = targets
        self.debug = debug
        self.debug_info = debug_info
        self.framework_name = framework_name

    def checkCMakeVersion(self):
        if get_xcode_version() >= (12, 2):
            assert get_cmake_version() >= (3, 19), "CMake 3.19 or later is required when building with Xcode 12.2 or greater. Current version is {}".format(get_cmake_version())
        else:
            assert get_cmake_version() >= (3, 17), "CMake 3.17 or later is required. Current version is {}".format(get_cmake_version())

    def getBuildDir(self, parent, target):
        res = os.path.join(parent, 'build-%s-%s' % (target[0].lower(), target[1].lower()))
        if not os.path.isdir(res):
            os.makedirs(res)
        return os.path.abspath(res)

    def _build(self, outdir):
        self.checkCMakeVersion()
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        main_working_dir = os.path.join(outdir, "build")
        dirs = []

        xcode_ver = get_xcode_major()
        xcode_supports_ios_32bit_arch = xcode_ver <= 13

        alltargets = []
        for target_group in self.targets:
            for arch in target_group[0]:
                if arch in ["armv7", "armv7s", "i386"] and not xcode_supports_ios_32bit_arch:
                    print("Skipping unsupported architecture: " + arch)
                    continue
                alltargets.append((arch, target_group[1]))

        for target in alltargets:
            main_build_dir = self.getBuildDir(main_working_dir, target)
            dirs.append(main_build_dir)

            cmake_flags = []
            if self.contrib:
                cmake_flags.append("-DOPENCV_EXTRA_MODULES_PATH=%s" % self.contrib)

            print("::group::Building target", target[0], target[1], flush=True)
            self.buildOne(target[0], target[1], main_build_dir, cmake_flags)
            print("::endgroup::", flush=True)

            if not self.dynamic:
                print("::group::Merge libs", target[0], target[1], flush=True)
                self.mergeLibs(main_build_dir)
                print("::endgroup::", flush=True)
            else:
                print("::group::Make dynamic lib", target[0], target[1], flush=True)
                self.makeDynamicLib(main_build_dir)
                print("::endgroup::", flush=True)

        self.makeFramework(outdir, dirs)

    def build(self, outdir):
        try:
            self._build(outdir)
        except Exception as e:
            print_error(e)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    def getToolchain(self, arch, target):
        return None

    def getConfiguration(self):
        return "Debug" if self.debug else "Release"

    def getCMakeArgs(self, arch, target):
        c_flags = APPLE_CLANG_FLAGS
        cxx_flags = APPLE_CLANG_FLAGS
        if not self.bitcodedisabled and target == 'iPhoneOS':
            c_flags += ' -fembed-bitcode'
            cxx_flags += ' -fembed-bitcode'

        args = [
            "cmake",
            "-GXcode",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DAPPLE_FRAMEWORK=ON",
            "-DCMAKE_INSTALL_PREFIX=install",
            "-DCMAKE_BUILD_TYPE=%s" % self.getConfiguration(),
            "-DOPENCV_INCLUDE_INSTALL_PATH=include",
            "-DOPENCV_3P_LIB_INSTALL_PATH=lib/3rdparty",
            "-DFRAMEWORK_NAME=%s" % self.framework_name,
            "-DCMAKE_C_FLAGS=%s" % c_flags,
            "-DCMAKE_CXX_FLAGS=%s" % cxx_flags,
            "-DWITH_OPENCL=OFF",
        ]
        if self.dynamic:
            args += ["-DDYNAMIC_PLIST=ON"]
        if self.enablenonfree:
            args += ["-DOPENCV_ENABLE_NONFREE=ON"]
        if self.debug_info:
            args += ["-DBUILD_WITH_DEBUG_INFO=ON"]

        if len(self.exclude) > 0:
            args += ["-DBUILD_opencv_%s=OFF" % m for m in self.exclude]

        if len(self.disable) > 0:
            args += ["-DWITH_%s=OFF" % f for f in self.disable]

        return args

    def getBuildCommand(self, arch, target):
        buildcmd = ["xcodebuild"]
        buildcmd += [
            "IPHONEOS_DEPLOYMENT_TARGET=" + os.environ['IPHONEOS_DEPLOYMENT_TARGET'],
            "ARCHS=%s" % arch,
            "-sdk", target.lower(),
            "-configuration", self.getConfiguration(),
            "-parallelizeTargets",
            "-jobs", str(multiprocessing.cpu_count()),
        ]
        if self.dynamic and not self.bitcodedisabled:
            buildcmd.append("BITCODE_GENERATION_MODE=bitcode")
        return buildcmd

    def getInfoPlist(self, builddirs):
        return os.path.join(builddirs[0], "ios", "Info.plist")

    def makeCMakeCmd(self, arch, target, dir, cmakeargs=[]):
        toolchain = self.getToolchain(arch, target)
        cmakecmd = self.getCMakeArgs(arch, target) + \
            (["-DCMAKE_TOOLCHAIN_FILE=%s" % toolchain] if toolchain is not None else [])
        target_lower = target.lower()
        if target_lower.startswith("iphoneos"):
            cmakecmd.append("-DCPU_BASELINE=DETECT")
        if target_lower.startswith("iphonesimulator"):
            build_arch = check_output(["uname", "-m"]).decode('utf-8').rstrip()
            if build_arch != arch:
                print("build_arch (%s) != arch (%s)" % (build_arch, arch))
                cmakecmd.append("-DCMAKE_SYSTEM_PROCESSOR=" + arch)
                cmakecmd.append("-DCMAKE_OSX_ARCHITECTURES=" + arch)
                cmakecmd.append("-DCPU_BASELINE=DETECT")
                cmakecmd.append("-DCMAKE_CROSSCOMPILING=ON")
                cmakecmd.append("-DOPENCV_WORKAROUND_CMAKE_20989=ON")

        cmakecmd.append(dir)
        cmakecmd.extend(cmakeargs)
        return cmakecmd

    def buildOne(self, arch, target, builddir, cmakeargs=[]):
        cmakecmd = self.makeCMakeCmd(arch, target, self.opencv, cmakeargs)
        print("")
        print("=================================")
        print("CMake")
        print("=================================")
        print("")
        execute(cmakecmd, cwd=builddir)
        print("")
        print("=================================")
        print("Xcodebuild")
        print("=================================")
        print("")

        clean_dir = os.path.join(builddir, "install")
        if os.path.isdir(clean_dir):
            shutil.rmtree(clean_dir)
        buildcmd = self.getBuildCommand(arch, target)
        execute(buildcmd + ["-target", "ALL_BUILD", "build"], cwd=builddir)
        execute(["cmake", "-DBUILD_TYPE=%s" % self.getConfiguration(), "-P", "cmake_install.cmake"], cwd=builddir)

    def mergeLibs(self, builddir):
        res = os.path.join(builddir, "lib", self.getConfiguration(), "libopencv_merged.a")
        libs = glob.glob(os.path.join(builddir, "install", "lib", "*.a"))
        libs3 = glob.glob(os.path.join(builddir, "install", "lib", "3rdparty", "*.a"))
        print("Merging libraries:\n\t%s" % "\n\t".join(libs + libs3), file=sys.stderr)
        execute(["libtool", "-static", "-o", res] + libs + libs3)

    def makeDynamicLib(self, builddir):
        target = builddir[(builddir.rfind("build-") + 6):]
        target_platform = target[(target.rfind("-") + 1):]
        framework_dir = os.path.join(builddir, "install", "lib", self.framework_name + ".framework")
        if not os.path.exists(framework_dir):
            os.makedirs(framework_dir)
        res = os.path.join(framework_dir, self.framework_name)
        libs = glob.glob(os.path.join(builddir, "install", "lib", "*.a"))
        libs3 = glob.glob(os.path.join(builddir, "install", "lib", "3rdparty", "*.a"))

        if os.environ.get('IPHONEOS_DEPLOYMENT_TARGET'):
            link_target = target[:target.find("-")] + "-apple-ios" + os.environ['IPHONEOS_DEPLOYMENT_TARGET'] + ("-simulator" if target.endswith("simulator") else "")
        else:
            link_target = "%s-apple-darwin" % target[:target.find("-")]
        sdk_dir = get_xcode_setting("SDK_DIR", builddir)
        framework_options = [
            "-iframework", "%s/System/iOSSupport/System/Library/Frameworks" % sdk_dir,
            "-framework", "AVFoundation", "-framework", "CoreGraphics",
            "-framework", "CoreImage", "-framework", "CoreMedia", "-framework", "QuartzCore",
            "-framework", "Accelerate", "-framework", "UIKit", "-framework", "CoreVideo",
        ]
        execute([
            "clang++",
            "-target", link_target,
            "-isysroot", sdk_dir,
        ] + framework_options + [
            "-install_name", "@rpath/" + self.framework_name + ".framework/" + self.framework_name,
            "-dynamiclib", "-dead_strip", "-fobjc-link-runtime", "-all_load",
            "-o", res
        ] + libs + libs3)

    def makeFramework(self, outdir, builddirs):
        name = self.framework_name

        framework_dir = os.path.join(outdir, "%s.framework" % name)
        if os.path.isdir(framework_dir):
            shutil.rmtree(framework_dir)
        os.makedirs(framework_dir)

        if self.dynamic:
            dstdir = framework_dir
        else:
            dstdir = os.path.join(framework_dir, "Versions", "A")

        shutil.copytree(os.path.join(builddirs[0], "install", "include", "opencv2"), os.path.join(dstdir, "Headers"))
        if name != "opencv2":
            for dirname, dirs, files in os.walk(os.path.join(dstdir, "Headers")):
                for filename in files:
                    filepath = os.path.join(dirname, filename)
                    with open(filepath, "r", encoding="utf-8") as file:
                        body = file.read()
                    body = body.replace("include \"opencv2/", "include \"" + name + "/")
                    body = body.replace("include <opencv2/", "include <" + name + "/")
                    with open(filepath, "w", encoding="utf-8") as file:
                        file.write(body)

        if self.dynamic:
            libs = [os.path.join(d, "install", "lib", name + ".framework", name) for d in builddirs]
        else:
            libs = [os.path.join(d, "lib", self.getConfiguration(), "libopencv_merged.a") for d in builddirs]
        lipocmd = ["lipo", "-create"]
        lipocmd.extend(libs)
        lipocmd.extend(["-o", os.path.join(dstdir, name)])
        print("Creating universal library from:\n\t%s" % "\n\t".join(libs), file=sys.stderr)
        execute(lipocmd)

        if self.dynamic:
            resdir = dstdir
            shutil.copyfile(self.getInfoPlist(builddirs), os.path.join(resdir, "Info.plist"))
        else:
            resdir = os.path.join(dstdir, "Resources")
            os.makedirs(resdir)
            shutil.copyfile(self.getInfoPlist(builddirs), os.path.join(resdir, "Info.plist"))

            links = [
                (["A"], ["Versions", "Current"]),
                (["Versions", "Current", "Headers"], ["Headers"]),
                (["Versions", "Current", "Resources"], ["Resources"]),
                (["Versions", "Current", name], [name])
            ]
            for l in links:
                s = os.path.join(*l[0])
                d = os.path.join(framework_dir, *l[1])
                os.symlink(s, d)

        shutil.copyfile(os.path.join(CURRENT_FILE_DIR, "PrivacyInfo.xcprivacy"),
                        os.path.join(resdir, "PrivacyInfo.xcprivacy"))


class iOSBuilder(Builder):

    def getToolchain(self, arch, target):
        return os.path.join(self.opencv, "platforms", "ios", "cmake", "Toolchains", "Toolchain-%s_Xcode.cmake" % target)

    def getCMakeArgs(self, arch, target):
        args = Builder.getCMakeArgs(self, arch, target)
        args = args + ['-DIOS_ARCH=%s' % arch]
        return args


if __name__ == "__main__":
    folder = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../.."))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for iOS.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built framework')
    parser.add_argument('--opencv', metavar='DIR', default=folder, help='folder with opencv repository (default is "../.." relative to script location)')
    parser.add_argument('--contrib', metavar='DIR', default=None, help='folder with opencv_contrib repository (default is "None" - build only main framework)')
    parser.add_argument('--without', metavar='MODULE', default=[], action='append', help='OpenCV modules to exclude from the framework. To exclude multiple, specify this flag again, e.g. "--without video"')
    parser.add_argument('--disable', metavar='FEATURE', default=[], action='append', help='OpenCV features to disable (add WITH_*=OFF). To disable multiple, specify this flag again, e.g. "--disable tbb --disable openmp"')
    parser.add_argument('--dynamic', default=False, action='store_true', help='build dynamic framework (default is "False" - builds static framework)')
    parser.add_argument('--disable-bitcode', default=False, dest='bitcodedisabled', action='store_true', help='disable bitcode (enabled by default for iPhoneOS)')
    parser.add_argument('--iphoneos_deployment_target', default=os.environ.get('IPHONEOS_DEPLOYMENT_TARGET', IPHONEOS_DEPLOYMENT_TARGET), help='specify IPHONEOS_DEPLOYMENT_TARGET')
    parser.add_argument('--build_only_specified_archs', default=False, action='store_true', help='if enabled, only directly specified archs are built and defaults are ignored')
    parser.add_argument('--iphoneos_archs', default=None, help='select iPhoneOS target ARCHS. Default is "armv7,arm64"')
    parser.add_argument('--iphonesimulator_archs', default=None, help='select iPhoneSimulator target ARCHS. Default is "x86_64,arm64"')
    parser.add_argument('--enable_nonfree', default=False, dest='enablenonfree', action='store_true', help='enable non-free modules (disabled by default)')
    parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='Build "Debug" binaries (disabled by default)')
    parser.add_argument('--debug_info', default=False, dest='debug_info', action='store_true', help='Build with debug information (useful for Release mode: BUILD_WITH_DEBUG_INFO=ON)')
    parser.add_argument('--framework_name', default='opencv2', dest='framework_name', help='Name of OpenCV framework (default: opencv2)')

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("The following args are not recognized and will not be used: %s" % unknown_args)

    os.environ['IPHONEOS_DEPLOYMENT_TARGET'] = args.iphoneos_deployment_target
    print('Using IPHONEOS_DEPLOYMENT_TARGET=' + os.environ['IPHONEOS_DEPLOYMENT_TARGET'])

    iphoneos_archs = None
    if args.iphoneos_archs:
        iphoneos_archs = args.iphoneos_archs.split(',')
    elif not args.build_only_specified_archs:
        iphoneos_archs = ["armv7", "arm64"]
    print('Using iPhoneOS ARCHS=' + str(iphoneos_archs))

    iphonesimulator_archs = None
    if args.iphonesimulator_archs:
        iphonesimulator_archs = args.iphonesimulator_archs.split(',')
    elif not args.build_only_specified_archs:
        iphonesimulator_archs = ["x86_64", "arm64"]
    print('Using iPhoneSimulator ARCHS=' + str(iphonesimulator_archs))

    if iphoneos_archs and iphonesimulator_archs:
        duplicate_archs = set(iphoneos_archs).intersection(iphonesimulator_archs)
        if duplicate_archs:
            print_error("Cannot have the same architecture for multiple platforms in a fat framework! Consider using build_xcframework.py in the apple platform folder instead. Duplicate archs are %s" % duplicate_archs)
            sys.exit(1)

    targets = []
    if os.environ.get('BUILD_PRECOMMIT', None):
        if not iphoneos_archs:
            print_error("--iphoneos_archs must have at least one value")
            sys.exit(1)
        targets.append((iphoneos_archs, "iPhoneOS"))
    else:
        if not iphoneos_archs and not iphonesimulator_archs:
            print_error("--iphoneos_archs and --iphonesimulator_archs are undefined; nothing will be built.")
            sys.exit(1)
        if iphoneos_archs:
            targets.append((iphoneos_archs, "iPhoneOS"))
        if iphonesimulator_archs:
            targets.append((iphonesimulator_archs, "iPhoneSimulator"))

    b = iOSBuilder(args.opencv, args.contrib, args.dynamic, args.bitcodedisabled, args.without, args.disable,
                   args.enablenonfree, targets, args.debug, args.debug_info, args.framework_name)
    b.build(args.out)
