#!/usr/bin/env python
import sys
import os
import platform
import re
import tempfile
import glob
import logging
import shutil
from subprocess import check_call, check_output, CalledProcessError, STDOUT


def initLogger():
    logger = logging.getLogger("run.py")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


log = initLogger()
hostos = os.name  # 'nt', 'posix'


class Err(Exception):
    def __init__(self, msg, *args):
        self.msg = msg % args


def execute(cmd, silent=False, cwd=".", env=None):
    try:
        log.debug("Run: %s", cmd)
        if env is not None:
            for k in env:
                log.debug("    Environ: %s=%s", k, env[k])
            new_env = os.environ.copy()
            new_env.update(env)
            env = new_env
        if silent:
            return check_output(cmd, stderr=STDOUT, cwd=cwd, env=env).decode("latin-1")
        else:
            return check_call(cmd, cwd=cwd, env=env)
    except CalledProcessError as e:
        if silent:
            log.debug("Process returned: %d", e.returncode)
            return e.output.decode("latin-1")
        else:
            log.error("Process returned: %d", e.returncode)
            return e.returncode


def isColorEnabled(args):
    usercolor = [a for a in args if a.startswith("--gtest_color=")]
    return len(usercolor) == 0 and sys.stdout.isatty() and hostos != "nt"


def getPlatformVersion():
    mv = platform.mac_ver()
    if mv[0]:
        return "Darwin" + mv[0]
    else:
        wv = platform.win32_ver()
        if wv[0]:
            return "Windows" + wv[0]
        else:
            lv = platform.linux_distribution()
            if lv[0]:
                return lv[0] + lv[1]
    return None


parse_patterns = (
    {'name': "cmake_home",               'default': None,       'pattern': re.compile(r"^CMAKE_HOME_DIRECTORY:\w+=(.+)$")},
    {'name': "opencv_home",              'default': None,       'pattern': re.compile(r"^OpenCV_SOURCE_DIR:\w+=(.+)$")},
    {'name': "opencv_build",             'default': None,       'pattern': re.compile(r"^OpenCV_BINARY_DIR:\w+=(.+)$")},
    {'name': "tests_dir",                'default': None,       'pattern': re.compile(r"^EXECUTABLE_OUTPUT_PATH:\w+=(.+)$")},
    {'name': "build_type",               'default': "Release",  'pattern': re.compile(r"^CMAKE_BUILD_TYPE:\w+=(.*)$")},
    {'name': "android_abi",              'default': None,       'pattern': re.compile(r"^CMAKE_ANDROID_ARCH_ABI:\w+=(.*)$")},
    {'name': "android_executable",       'default': None,       'pattern': re.compile(r"^ANDROID_EXECUTABLE:\w+=(.*android.*)$")},
    {'name': "ant_executable",           'default': None,       'pattern': re.compile(r"^ANT_EXECUTABLE:\w+=(.*ant.*)$")},
    {'name': "java_test_dir",            'default': None,       'pattern': re.compile(r"^OPENCV_JAVA_TEST_DIR:\w+=(.*)$")},
    {'name': "is_x64",                   'default': "OFF",      'pattern': re.compile(r"^CUDA_64_BIT_DEVICE_CODE:\w+=(ON)$")},
    {'name': "cmake_generator",          'default': None,       'pattern': re.compile(r"^CMAKE_GENERATOR:\w+=(.+)$")},
    {'name': "python2",                  'default': None,       'pattern': re.compile(r"^BUILD_opencv_python2:\w+=(.*)$")},
    {'name': "python3",                  'default': None,       'pattern': re.compile(r"^BUILD_opencv_python3:\w+=(.*)$")},
)


class CMakeCache:
    def __init__(self, cfg=None):
        self.setDefaultAttrs()
        self.main_modules = []
        if cfg:
            self.build_type = cfg

    def setDummy(self, path):
        self.tests_dir = os.path.normpath(path)

    def read(self, path, fname):
        rx = re.compile(r'^OPENCV_MODULE_opencv_(\w+)_LOCATION:INTERNAL=(.*)$')
        module_paths = {}  # name -> path
        with open(fname, "rt") as cachefile:
            for l in cachefile.readlines():
                ll = l.strip()
                if not ll or ll.startswith("#"):
                    continue
                for p in parse_patterns:
                    match = p["pattern"].match(ll)
                    if match:
                        value = match.groups()[0]
                        if value and not value.endswith("-NOTFOUND"):
                            setattr(self, p["name"], value)
                            # log.debug("cache value: %s = %s", p["name"], value)

                match = rx.search(ll)
                if match:
                    module_paths[match.group(1)] = match.group(2)

        if not self.tests_dir:
            self.tests_dir = path
        else:
            rel = os.path.relpath(self.tests_dir, self.opencv_build)
            self.tests_dir = os.path.join(path, rel)
        self.tests_dir = os.path.normpath(self.tests_dir)

        # fix VS test binary path (add Debug or Release)
        if "Visual Studio" in self.cmake_generator:
            self.tests_dir = os.path.join(self.tests_dir, self.build_type)

        for module, path in module_paths.items():
            rel = os.path.relpath(path, self.opencv_home)
            if ".." not in rel:
                self.main_modules.append(module)

    def setDefaultAttrs(self):
        for p in parse_patterns:
            setattr(self, p["name"], p["default"])

    def gatherTests(self, mask, isGood=None):
        if self.tests_dir and os.path.isdir(self.tests_dir):
            d = os.path.abspath(self.tests_dir)
            files = glob.glob(os.path.join(d, mask))
            if not self.getOS() == "android" and self.withJava():
                files.append("java")
            if self.withPython2():
                files.append("python2")
            if self.withPython3():
                files.append("python3")
            return [f for f in files if isGood(f)]
        return []

    def isMainModule(self, name):
        return name in self.main_modules + ['python2', 'python3']

    def withJava(self):
        return self.ant_executable and self.java_test_dir and os.path.exists(self.java_test_dir)

    def withPython2(self):
        return self.python2 == 'ON'

    def withPython3(self):
        return self.python3 == 'ON'

    def getOS(self):
        if self.android_executable:
            return "android"
        else:
            return hostos


class TempEnvDir:
    def __init__(self, envname, prefix):
        self.envname = envname
        self.prefix = prefix
        self.saved_name = None
        self.new_name = None

    def init(self):
        self.saved_name = os.environ.get(self.envname)
        self.new_name = tempfile.mkdtemp(prefix=self.prefix, dir=self.saved_name or None)
        os.environ[self.envname] = self.new_name

    def clean(self):
        if self.saved_name:
            os.environ[self.envname] = self.saved_name
        else:
            del os.environ[self.envname]
        try:
            shutil.rmtree(self.new_name)
        except:
            pass


if __name__ == "__main__":
    log.error("This is utility file, please execute run.py script")
