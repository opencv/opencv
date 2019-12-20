#!/usr/bin/env python
import os
import re
import getpass
from run_utils import Err, log, execute, isColorEnabled, hostos
from run_suite import TestSuite


def exe(program):
    return program + ".exe" if hostos == 'nt' else program


class ApkInfo:
    def __init__(self):
        self.pkg_name = None
        self.pkg_target = None
        self.pkg_runner = None

    def forcePackage(self, package):
        if package:
            if package.startswith("."):
                self.pkg_target += package
            else:
                self.pkg_target = package


class Tool:
    def __init__(self):
        self.cmd = []

    def run(self, args=[], silent=False):
        cmd = self.cmd[:]
        cmd.extend(args)
        return execute(self.cmd + args, silent)


class Adb(Tool):
    def __init__(self, sdk_dir):
        Tool.__init__(self)
        exe_path = os.path.join(sdk_dir, exe("platform-tools/adb"))
        if not os.path.isfile(exe_path) or not os.access(exe_path, os.X_OK):
            exe_path = None
        # fix adb tool location
        if not exe_path:
            exe_path = "adb"
        self.cmd = [exe_path]

    def init(self, serial):
        # remember current device serial. Needed if another device is connected while this script runs
        if not serial:
            serial = self.detectSerial()
        if serial:
            self.cmd.extend(["-s", serial])

    def detectSerial(self):
        adb_res = self.run(["devices"], silent=True)
        # assume here that device name may consists of any characters except newline
        connected_devices = re.findall(r"^[^\n]+[ \t]+device\r?$", adb_res, re.MULTILINE)
        if not connected_devices:
            raise Err("Can not find Android device")
        elif len(connected_devices) != 1:
            raise Err("Too many (%s) devices are connected. Please specify single device using --serial option:\n\n%s", len(connected_devices), adb_res)
        else:
            return connected_devices[0].split("\t")[0]

    def getOSIdentifier(self):
        return "Android" + self.run(["shell", "getprop ro.build.version.release"], silent=True).strip()


class Aapt(Tool):
    def __init__(self, sdk_dir):
        Tool.__init__(self)
        aapt_fn = exe("aapt")
        aapt = None
        for r, ds, fs in os.walk(os.path.join(sdk_dir, 'build-tools')):
            if aapt_fn in fs:
                aapt = os.path.join(r, aapt_fn)
                break
        if not aapt:
            raise Err("Can not find aapt tool: %s", aapt_fn)
        self.cmd = [aapt]

    def dump(self, exe):
        res = ApkInfo()
        output = self.run(["dump", "xmltree", exe, "AndroidManifest.xml"], silent=True)
        if not output:
            raise Err("Can not dump manifest from %s", exe)
        tags = re.split(r"[ ]+E: ", output)
        # get package name
        manifest_tag = [t for t in tags if t.startswith("manifest ")]
        if not manifest_tag:
            raise Err("Can not read package name from: %s", exe)
        res.pkg_name = re.search(r"^[ ]+A: package=\"(?P<pkg>.*?)\" \(Raw: \"(?P=pkg)\"\)\r?$", manifest_tag[0], flags=re.MULTILINE).group("pkg")
        # get test instrumentation info
        instrumentation_tag = [t for t in tags if t.startswith("instrumentation ")]
        if not instrumentation_tag:
            raise Err("Can not find instrumentation detials in: %s", exe)
        res.pkg_runner = re.search(r"^[ ]+A: android:name\(0x[0-9a-f]{8}\)=\"(?P<runner>.*?)\" \(Raw: \"(?P=runner)\"\)\r?$", instrumentation_tag[0], flags=re.MULTILINE).group("runner")
        res.pkg_target = re.search(r"^[ ]+A: android:targetPackage\(0x[0-9a-f]{8}\)=\"(?P<pkg>.*?)\" \(Raw: \"(?P=pkg)\"\)\r?$", instrumentation_tag[0], flags=re.MULTILINE).group("pkg")
        if not res.pkg_name or not res.pkg_runner or not res.pkg_target:
            raise Err("Can not find instrumentation detials in: %s", exe)
        return res


class AndroidTestSuite(TestSuite):
    def __init__(self, options, cache, id, android_env={}):
        TestSuite.__init__(self, options, cache, id)
        sdk_dir = options.android_sdk or os.environ.get("ANDROID_SDK", False) or os.path.dirname(os.path.dirname(self.cache.android_executable))
        log.debug("Detecting Android tools in directory: %s", sdk_dir)
        self.adb = Adb(sdk_dir)
        self.aapt = Aapt(sdk_dir)
        self.env = android_env

    def isTest(self, fullpath):
        if os.path.isfile(fullpath):
            if fullpath.endswith(".apk") or os.access(fullpath, os.X_OK):
                return True
        return False

    def getOS(self):
        return self.adb.getOSIdentifier()

    def checkPrerequisites(self):
        self.adb.init(self.options.serial)

    def runTest(self, module, path, logfile, workingDir, args=[]):
        args = args[:]
        exe = os.path.abspath(path)

        if exe.endswith(".apk"):
            info = self.aapt.dump(exe)
            if not info:
                raise Err("Can not read info from test package: %s", exe)
            info.forcePackage(self.options.package)
            self.adb.run(["uninstall", info.pkg_name])

            output = self.adb.run(["install", exe], silent=True)
            if not (output and "Success" in output):
                raise Err("Can not install package: %s", exe)

            params = ["-e package %s" % info.pkg_target]
            ret = self.adb.run(["shell", "am instrument -w %s %s/%s" % (" ".join(params), info.pkg_name, info.pkg_runner)])
            return None, ret
        else:
            device_dir = getpass.getuser().replace(" ", "") + "_" + self.options.mode + "/"
            if isColorEnabled(args):
                args.append("--gtest_color=yes")
            tempdir = "/data/local/tmp/"
            android_dir = tempdir + device_dir
            exename = os.path.basename(exe)
            android_exe = android_dir + exename
            self.adb.run(["push", exe, android_exe])
            self.adb.run(["shell", "chmod 777 " + android_exe])
            env_pieces = ["export %s=%s" % (a, b) for a, b in self.env.items()]
            pieces = ["cd %s" % android_dir, "./%s %s" % (exename, " ".join(args))]
            log.warning("Run: %s" % " && ".join(pieces))
            ret = self.adb.run(["shell", " && ".join(env_pieces + pieces)])
            # try get log
            hostlogpath = os.path.join(workingDir, logfile)
            self.adb.run(["pull", android_dir + logfile, hostlogpath])
            # cleanup
            self.adb.run(["shell", "rm " + android_dir + logfile])
            self.adb.run(["shell", "rm " + tempdir + "__opencv_temp.*"], silent=True)
            if os.path.isfile(hostlogpath):
                return hostlogpath, ret
            return None, ret


if __name__ == "__main__":
    log.error("This is utility file, please execute run.py script")
