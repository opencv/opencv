#!/usr/bin/env python

import sys, os, platform, re, tempfile, glob, getpass, logging
from subprocess import check_call, check_output, CalledProcessError, STDOUT

hostos = os.name # 'nt', 'posix'
hostmachine = platform.machine() # 'x86', 'AMD64', 'x86_64'

def initLogger():
    l = logging.getLogger("run.py")
    l.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(message)s"))
    l.addHandler(ch)
    return l

log = initLogger()

#===================================================================================================

class Err(Exception):
    def __init__(self, msg, *args):
        self.msg = msg % args

def execute(cmd, silent = False, cwd = ".", env = None):
    try:
        log.debug("Run: %s", cmd)
        if env:
            for k in env:
                log.debug("    Environ: %s=%s", k, env[k])
            env = os.environ.update(env)
        if silent:
            return check_output(cmd, stderr = STDOUT, cwd = cwd, env = env).decode("latin-1")
        else:
            return check_call(cmd, cwd = cwd, env = env)
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

#===================================================================================================

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

def readGitVersion(git, path):
    if not path or not git or not os.path.isdir(os.path.join(path, ".git")):
        return None
    try:
        output = execute([git, "-C", path, "rev-parse", "--short", "HEAD"], silent = True)
        return output.strip()
    except OSError:
        log.warning("Git version read failed")
        return None

SIMD_DETECTION_PROGRAM="""
#if __SSE5__
# error SSE5
#endif
#if __AVX2__
# error AVX2
#endif
#if __AVX__
# error AVX
#endif
#if __SSE4_2__
# error SSE4.2
#endif
#if __SSE4_1__
# error SSE4.1
#endif
#if __SSSE3__
# error SSSE3
#endif
#if __SSE3__
# error SSE3
#endif
#if __AES__
# error AES
#endif
#if __SSE2__
# error SSE2
#endif
#if __SSE__
# error SSE
#endif
#if __3dNOW__
# error 3dNOW
#endif
#if __MMX__
# error MMX
#endif
#if __ARM_NEON__
# error NEON
#endif
#error NOSIMD
"""

def testSIMD(compiler, cxx_flags, compiler_arg = None):
    if not compiler:
        return None
    compiler_output = ""
    try:
        _, tmpfile = tempfile.mkstemp(suffix=".cpp", text = True)
        with open(tmpfile, "w+") as fd:
            fd.write(SIMD_DETECTION_PROGRAM)
        options = [compiler]
        if compiler_arg:
            options.append(compiler_arg)

        prev_option = None
        for opt in " ".join(cxx_flags).split():
            if opt.count('\"') % 2 == 1:
                if prev_option is None:
                     prev_option = opt
                else:
                     options.append(prev_option + " " + opt)
                     prev_option = None
            elif prev_option is None:
                options.append(opt)
            else:
                prev_option = prev_option + " " + opt
        options.append(tmpfile)
        compiler_output = execute(options, silent = True)
        os.remove(tmpfile)
        m = re.search("#error\W+(\w+)", compiler_output)
        if m:
            return m.group(1)
    except OSError:
        pass
    log.debug("SIMD detection failed")
    return None

#==============================================================================

parse_patterns = (
    {'name': "cmake_home",               'default': None,       'pattern': re.compile(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$")},
    {'name': "opencv_home",              'default': None,       'pattern': re.compile(r"^OpenCV_SOURCE_DIR:STATIC=(.+)$")},
    {'name': "opencv_build",             'default': None,       'pattern': re.compile(r"^OpenCV_BINARY_DIR:STATIC=(.+)$")},
    {'name': "tests_dir",                'default': None,       'pattern': re.compile(r"^EXECUTABLE_OUTPUT_PATH:PATH=(.+)$")},
    {'name': "build_type",               'default': "Release",  'pattern': re.compile(r"^CMAKE_BUILD_TYPE:\w+=(.*)$")},
    {'name': "git_executable",           'default': None,       'pattern': re.compile(r"^GIT_EXECUTABLE:FILEPATH=(.*)$")},
    {'name': "cxx_flags",                'default': "",         'pattern': re.compile(r"^CMAKE_CXX_FLAGS:STRING=(.*)$")},
    {'name': "cxx_flags_debug",          'default': "",         'pattern': re.compile(r"^CMAKE_CXX_FLAGS_DEBUG:STRING=(.*)$")},
    {'name': "cxx_flags_release",        'default': "",         'pattern': re.compile(r"^CMAKE_CXX_FLAGS_RELEASE:STRING=(.*)$")},
    {'name': "opencv_cxx_flags",         'default': "",         'pattern': re.compile(r"^OPENCV_EXTRA_C_FLAGS:INTERNAL=(.*)$")},
    {'name': "cxx_flags_android",        'default': None,       'pattern': re.compile(r"^ANDROID_CXX_FLAGS:INTERNAL=(.*)$")},
    {'name': "android_abi",              'default': None,       'pattern': re.compile(r"^ANDROID_ABI:STRING=(.*)$")},
    {'name': "android_executable",       'default': None,       'pattern': re.compile(r"^ANDROID_EXECUTABLE:FILEPATH=(.*android.*)$")},
    {'name': "ant_executable",           'default': None,       'pattern': re.compile(r"^ANT_EXECUTABLE:FILEPATH=(.*ant.*)$")},
    {'name': "java_test_binary_dir",     'default': None,       'pattern': re.compile(r"^opencv_test_java_BINARY_DIR:STATIC=(.*)$")},
    {'name': "is_x64",                   'default': "OFF",      'pattern': re.compile(r"^CUDA_64_BIT_DEVICE_CODE:BOOL=(ON)$")},#ugly(
    {'name': "cmake_generator",          'default': None,       'pattern': re.compile(r"^CMAKE_GENERATOR:INTERNAL=(.+)$")},
    {'name': "cxx_compiler",             'default': None,       'pattern': re.compile(r"^CMAKE_CXX_COMPILER:\w*PATH=(.+)$")},
    {'name': "cxx_compiler_arg1",        'default': None,       'pattern': re.compile(r"^CMAKE_CXX_COMPILER_ARG1:[A-Z]+=(.+)$")},
    {'name': "with_cuda",                'default': "OFF",      'pattern': re.compile(r"^WITH_CUDA:BOOL=(ON)$")},
    {'name': "cuda_library",             'default': None,       'pattern': re.compile(r"^CUDA_CUDA_LIBRARY:FILEPATH=(.+)$")},
    {'name': "cuda_version",             'default': None,       'pattern': re.compile(r"^CUDA_VERSION:STRING=(.+)$")},
    {'name': "core_dependencies",        'default': None,       'pattern': re.compile(r"^opencv_core_LIB_DEPENDS:STATIC=(.+)$")},
    {'name': "python2",                  'default': None,       'pattern': re.compile(r"^BUILD_opencv_python2:BOOL=(.*)$")},
    {'name': "python3",                  'default': None,       'pattern': re.compile(r"^BUILD_opencv_python3:BOOL=(.*)$")},
)

class CMakeCache:
    def __init__(self, cfg = None):
        self.setDefaultAttrs()
        self.cmake_home_vcver = None
        self.opencv_home_vcver = None
        self.featuresSIMD = None
        self.main_modules = []
        if cfg:
            self.build_type = cfg

    def setDummy(self, path):
        self.tests_dir = os.path.normpath(path)

    def read(self, path, fname):
        rx = re.compile(r'^OPENCV_MODULE_opencv_(\w+)_LOCATION:INTERNAL=(.*)$')
        module_paths = {} # name -> path
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

        self.cmake_home_vcver = readGitVersion(self.git_executable, self.cmake_home)
        if self.opencv_home == self.cmake_home:
            self.opencv_home_vcver = self.cmake_home_vcver
        else:
            self.opencv_home_vcver = readGitVersion(self.git_executable, self.opencv_home)

        for module,path in module_paths.items():
            rel = os.path.relpath(path, self.opencv_home)
            if not ".." in rel:
                self.main_modules.append(module)

        self.flags = [
            self.cxx_flags_android,
            self.cxx_flags,
            self.cxx_flags_release,
            self.opencv_cxx_flags,
            self.cxx_flags_release]
        self.flags = [f for f in self.flags if f]
        self.featuresSIMD = testSIMD(self.cxx_compiler, self.flags, self.cxx_compiler_arg1)

    def setDefaultAttrs(self):
        for p in parse_patterns:
            setattr(self, p["name"], p["default"])

    def gatherTests(self, mask, isGood = None):
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

    def withCuda(self):
        return self.cuda_version and self.with_cuda == "ON" and self.cuda_library and not self.cuda_library.endswith("-NOTFOUND")

    def withJava(self):
        return self.ant_executable and self.java_test_binary_dir

    def withPython2(self):
        return self.python2 == 'ON'

    def withPython3(self):
        return self.python3 == 'ON'

    def getGitVersion(self):
        if self.cmake_home_vcver:
            if self.cmake_home_vcver == self.opencv_home_vcver:
                rev = self.cmake_home_vcver
            elif self.opencv_home_vcver:
                rev = self.cmake_home_vcver + "-" + self.opencv_home_vcver
            else:
                rev = self.cmake_home_vcver
        else:
            rev = None
        if rev:
            rev = rev.replace(":","to")
        else:
            rev = ""
        return rev

    def getTestFullName(self, shortname):
        return os.path.join(self.tests_dir, shortname)

    def getSIMDFeatures(self):
        return self.featuresSIMD

    def getOS(self):
        if self.android_executable:
            return "android"
        else:
            return hostos

    def getArch(self):
        arch = "unknown"
        if self.getOS() == "android":
            if "armeabi-v7a" in self.android_abi:
                arch = "armv7a"
            elif "armeabi-v6" in self.android_abi:
                arch = "armv6"
            elif "armeabi" in self.android_abi:
                arch = "armv5te"
            elif "x86" in self.android_abi:
                arch = "x86"
            elif "mips" in self.android_abi:
                arch = "mips"
            else:
                arch = "ARM"
        elif self.is_x64 and hostmachine in ["AMD64", "x86_64"]:
            arch = "x64"
        elif hostmachine in ["x86", "AMD64", "x86_64"]:
            arch = "x86"
        return arch

    def getDependencies(self):
        if self.core_dependencies:
            candidates = ["tbb", "ippicv", "ipp", "pthreads"]
            return [a for a in self.core_dependencies.split(";") if a and a in candidates]
        return []


#==============================================================================

def getRunningProcessExePathByName_win32(name):
    from ctypes import windll, POINTER, pointer, Structure, sizeof
    from ctypes import c_long , c_int , c_uint , c_char , c_ubyte , c_char_p , c_void_p

    class PROCESSENTRY32(Structure):
        _fields_ = [ ( 'dwSize' , c_uint ) ,
                    ( 'cntUsage' , c_uint) ,
                    ( 'th32ProcessID' , c_uint) ,
                    ( 'th32DefaultHeapID' , c_uint) ,
                    ( 'th32ModuleID' , c_uint) ,
                    ( 'cntThreads' , c_uint) ,
                    ( 'th32ParentProcessID' , c_uint) ,
                    ( 'pcPriClassBase' , c_long) ,
                    ( 'dwFlags' , c_uint) ,
                    ( 'szExeFile' , c_char * 260 ) ,
                    ( 'th32MemoryBase' , c_long) ,
                    ( 'th32AccessKey' , c_long ) ]

    class MODULEENTRY32(Structure):
        _fields_ = [ ( 'dwSize' , c_long ) ,
                    ( 'th32ModuleID' , c_long ),
                    ( 'th32ProcessID' , c_long ),
                    ( 'GlblcntUsage' , c_long ),
                    ( 'ProccntUsage' , c_long ) ,
                    ( 'modBaseAddr' , c_long ) ,
                    ( 'modBaseSize' , c_long ) ,
                    ( 'hModule' , c_void_p ) ,
                    ( 'szModule' , c_char * 256 ),
                    ( 'szExePath' , c_char * 260 ) ]

    TH32CS_SNAPPROCESS = 2
    TH32CS_SNAPMODULE = 0x00000008

    ## CreateToolhelp32Snapshot
    CreateToolhelp32Snapshot= windll.kernel32.CreateToolhelp32Snapshot
    CreateToolhelp32Snapshot.reltype = c_long
    CreateToolhelp32Snapshot.argtypes = [ c_int , c_int ]
    ## Process32First
    Process32First = windll.kernel32.Process32First
    Process32First.argtypes = [ c_void_p , POINTER( PROCESSENTRY32 ) ]
    Process32First.rettype = c_int
    ## Process32Next
    Process32Next = windll.kernel32.Process32Next
    Process32Next.argtypes = [ c_void_p , POINTER(PROCESSENTRY32) ]
    Process32Next.rettype = c_int
    ## CloseHandle
    CloseHandle = windll.kernel32.CloseHandle
    CloseHandle.argtypes = [ c_void_p ]
    CloseHandle.rettype = c_int
    ## Module32First
    Module32First = windll.kernel32.Module32First
    Module32First.argtypes = [ c_void_p , POINTER(MODULEENTRY32) ]
    Module32First.rettype = c_int

    hProcessSnap = c_void_p(0)
    hProcessSnap = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS , 0 )

    pe32 = PROCESSENTRY32()
    pe32.dwSize = sizeof( PROCESSENTRY32 )
    ret = Process32First( hProcessSnap , pointer( pe32 ) )
    path = None

    while ret :
        if name + ".exe" == pe32.szExeFile:
            hModuleSnap = c_void_p(0)
            me32 = MODULEENTRY32()
            me32.dwSize = sizeof( MODULEENTRY32 )
            hModuleSnap = CreateToolhelp32Snapshot( TH32CS_SNAPMODULE, pe32.th32ProcessID )

            ret = Module32First( hModuleSnap, pointer(me32) )
            path = me32.szExePath
            CloseHandle( hModuleSnap )
            if path:
                break
        ret = Process32Next( hProcessSnap, pointer(pe32) )
    CloseHandle( hProcessSnap )
    return path


def getRunningProcessExePathByName_posix(name):
    pids= [pid for pid in os.listdir('/proc') if pid.isdigit()]
    for pid in pids:
        try:
            path = os.readlink(os.path.join('/proc', pid, 'exe'))
            if path and path.endswith(name):
                return path
        except:
            pass

def getRunningProcessExePathByName(name):
    try:
        if hostos == "nt":
            return getRunningProcessExePathByName_win32(name)
        elif hostos == "posix":
            return getRunningProcessExePathByName_posix(name)
        else:
            return None
    except:
        return None


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

#===================================================================================================

if __name__ == "__main__":
    log.error("This is utility file, please execute run.py script")
