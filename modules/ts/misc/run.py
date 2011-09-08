import sys, os, platform, xml, re, tempfile, glob, datetime, getpass
from optparse import OptionParser
from subprocess import Popen, PIPE

hostos = os.name # 'nt', 'posix'
hostmachine = platform.machine() # 'x86', 'AMD64', 'x86_64'
nameprefix = "opencv_perf_"

parse_patterns = (
  {'name': "has_perf_tests",     'default': "OFF",      'pattern': re.compile("^BUILD_PERF_TESTS:BOOL=(ON)$")},
  {'name': "cmake_home",         'default': None,       'pattern': re.compile("^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$")},
  {'name': "opencv_home",        'default': None,       'pattern': re.compile("^OpenCV_SOURCE_DIR:STATIC=(.+)$")},
  {'name': "tests_dir",          'default': None,       'pattern': re.compile("^EXECUTABLE_OUTPUT_PATH:PATH=(.+)$")},
  {'name': "build_type",         'default': "Release",  'pattern': re.compile("^CMAKE_BUILD_TYPE:STRING=(.*)$")},
  {'name': "svnversion_path",    'default': None,       'pattern': re.compile("^SVNVERSION_PATH:FILEPATH=(.*)$")},
  {'name': "cxx_flags",          'default': None,       'pattern': re.compile("^CMAKE_CXX_FLAGS:STRING=(.*)$")},
  {'name': "cxx_flags_debug",    'default': None,       'pattern': re.compile("^CMAKE_CXX_FLAGS_DEBUG:STRING=(.*)$")},
  {'name': "cxx_flags_release",  'default': None,       'pattern': re.compile("^CMAKE_CXX_FLAGS_RELEASE:STRING=(.*)$")},
  {'name': "ndk_path",           'default': None,       'pattern': re.compile("^ANDROID_NDK(?:_TOOLCHAIN_ROOT)?:PATH=(.*)$")},
  {'name': "arm_target",         'default': None,       'pattern': re.compile("^ARM_TARGET:INTERNAL=(.*)$")},
  {'name': "android_executable", 'default': None,       'pattern': re.compile("^ANDROID_EXECUTABLE:FILEPATH=(.*android.*)$")},
  {'name': "is_x64",             'default': "OFF",      'pattern': re.compile("^CUDA_64_BIT_DEVICE_CODE:BOOL=(ON)$")},#ugly(
  {'name': "cmake_generator",    'default': None,       'pattern': re.compile("^CMAKE_GENERATOR:INTERNAL=(.+)$")},
  {'name': "cxx_compiler",       'default': None,       'pattern': re.compile("^CMAKE_CXX_COMPILER:FILEPATH=(.+)$")},
)

def query_yes_no(stdout, question, default="yes"):
    valid = {"yes":True, "y":True, "ye":True, "no":False, "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        stdout.write(os.linesep + question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
            
class RunInfo(object):
    def __init__(self, path, configuration = None):
        self.path = path
        self.error = None
        for p in parse_patterns:
            setattr(self, p["name"], p["default"])
        cachefile = open(os.path.join(path, "CMakeCache.txt"), "rt")
        try:
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
        except:
            pass
        cachefile.close()
        # fix empty tests dir
        if not self.tests_dir:
            self.tests_dir = self.path
        # add path to adb
        if self.android_executable:
            self.adb = os.path.join(os.path.dirname(os.path.dirname(self.android_executable)), ("platform-tools/adb","platform-tools/adb.exe")[hostos == 'nt'])
        else:
            self.adb = None

        # detect target platform    
        if self.android_executable or self.arm_target or self.ndk_path:
            self.targetos = "android"
            if not self.adb:
                try:
                    output = Popen(["adb", "devices"], stdout=PIPE, stderr=PIPE).communicate()
                    self.adb = "adb"
                except OSError:
                    pass
        else:
            self.targetos = hostos

        # fix has_perf_tests param
        self.has_perf_tests = self.has_perf_tests == "ON"
        # fix is_x64 flag
        self.is_x64 = self.is_x64 == "ON"
        if not self.is_x64 and ("X64" in "%s %s %s" % (self.cxx_flags, self.cxx_flags_release, self.cxx_flags_debug) or "Win64" in self.cmake_generator):
            self.is_x64 = True

        # fix test path
        if "Visual Studio" in self.cmake_generator:
            if configuration:
                self.tests_dir = os.path.join(self.tests_dir, configuration)
            else:
                self.tests_dir = os.path.join(self.tests_dir, self.build_type)
        elif not self.is_x64 and self.cxx_compiler:
            #one more attempt to detect x64 compiler
            try:
                output = Popen([self.cxx_compiler, "-v"], stdout=PIPE, stderr=PIPE).communicate()
                if not output[0] and "x86_64" in output[1]:
                    self.is_x64 = True
            except OSError:
                pass

        # detect target arch
        if self.targetos == "android":
            self.targetarch = "arm"
        elif self.is_x64 and hostmachine in ["AMD64", "x86_64"]:
            self.targetarch = "x64"
        elif hostmachine in ["x86", "AMD64", "x86_64"]:
            self.targetarch = "x86"
        else:
            self.targetarch = "unknown"
            
        self.hardware = None
        
        self.getSvnVersion(self.cmake_home, "cmake_home_svn")
        if self.opencv_home == self.cmake_home:
            self.opencv_home_svn = self.cmake_home_svn
        else:
            self.getSvnVersion(self.opencv_home, "opencv_home_svn")
            
        self.tests = self.getAvailableTestApps()
        
    def getSvnVersion(self, path, name):
        if not path:
            setattr(self, name, None)
            return
        if not self.svnversion_path and hostos == 'nt':
            self.tryGetSvnVersionWithTortoise(path, name)
        else:
            svnversion = self.svnversion_path
            if not svnversion:
                svnversion = "svnversion"
            try:
                output = Popen([svnversion, "-n", path], stdout=PIPE, stderr=PIPE).communicate()
                if not output[1]:
                    setattr(self, name, output[0])
                else:
                    setattr(self, name, None)
            except OSError:
                setattr(self, name, None)
        
    def tryGetSvnVersionWithTortoise(self, path, name):
        try:
            wcrev = "SubWCRev.exe"
            dir = tempfile.mkdtemp()
            #print dir
            tmpfilename = os.path.join(dir, "svn.tmp")
            tmpfilename2 = os.path.join(dir, "svn_out.tmp")
            tmpfile = open(tmpfilename, "w")
            tmpfile.write("$WCRANGE$$WCMODS?M:$")
            tmpfile.close();
            output = Popen([wcrev, path, tmpfilename, tmpfilename2, "-f"], stdout=PIPE, stderr=PIPE).communicate()
            if "is not a working copy" in output[0]:
                version = "exported"
            else:
                tmpfile = open(tmpfilename2, "r")
                version = tmpfile.read()
                tmpfile.close()
            setattr(self, name, version)
        except:
            setattr(self, name, None)
        finally:
            if dir:
                import shutil
                shutil.rmtree(dir)
                
    def isTest(self, fullpath):
        if not os.path.isfile(fullpath):
            return False
        if hostos == self.targetos:
            return os.access(fullpath, os.X_OK)
        return True
                
    def getAvailableTestApps(self):
        if self.tests_dir and os.path.isdir(self.tests_dir):
            files = glob.glob(os.path.join(self.tests_dir, nameprefix + "*"))
            if self.targetos == hostos:
                files = [f for f in files if self.isTest(f)]
            return files
        return []
    
    def getLogName(self, app, timestamp):
        app = os.path.basename(app)
        if app.endswith(".exe"):
            app = app[:-4]
        if app.startswith(nameprefix):
            app = app[len(nameprefix):]
        if self.cmake_home_svn:
            if self.cmake_home_svn == self.opencv_home_svn:
                rev = self.cmake_home_svn
            elif self.opencv_home_svn:
                rev = self.cmake_home_svn + "-" + self.opencv_home_svn
            else:
                rev = self.cmake_home_svn
        else:
            rev = None
        if rev:
            rev = rev.replace(":","to") + "_" 
        else:
            rev = ""
        if self.hardware:
            hw = str(self.hardware).replace(" ", "_") + "_"
        else:
            hw = ""
        #stamp = timestamp.strftime("%Y%m%dT%H%M%S")
        stamp = timestamp.strftime("%Y-%m-%d--%H-%M-%S")
        return "%s_%s_%s_%s%s%s.xml" %(app, self.targetos, self.targetarch, hw, rev, stamp)
        
    def getTest(self, name):
        # full path
        if self.isTest(name):
            return name
        
        # name only
        fullname = os.path.join(self.tests_dir, name)
        if self.isTest(fullname):
            return fullname
        
        # name without extension
        fullname += ".exe"
        if self.isTest(fullname):
            return fullname
        
        # short name for OpenCV tests
        for t in self.tests:
            if t == name:
                return t
            fname = os.path.basename(t)
            if fname == name:
                return t
            if fname.endswith(".exe"):
                fname = fname[:-4]
            if fname == name:
                return t
            if fname.startswith(nameprefix):
                fname = fname[len(nameprefix):]
            if fname == name:
                return t
        return None
    
    def runAdb(self, *args):
        cmd = [self.adb]
        cmd.extend(args)
        try:
            output = Popen(cmd, stdout=PIPE, stderr=PIPE).communicate()
            if not output[1]:
                return output[0]
            self.error = output[1]
            print self.error
        except OSError:
            pass
        return None
    
    def isRunnable(self):
        #if not self.has_perf_tests or not self.tests:
            #self.error = "Performance tests are not built (at %s)" % self.path
            #return False
        if self.targetarch == "x64" and hostmachine == "x86":
            self.error = "Target architecture is incompatible with current platform (at %s)" % self.path
            return False
        if self.targetos == "android":
            if not self.adb or not os.path.isfile(self.adb) or not os.access(self.adb, os.X_OK):
                self.error = "Could not find adb executable (at %s)" % self.path
                return False
            adb_res = self.runAdb("devices")
            if not adb_res:
                self.error = "Could not run adb command: %s (at %s)" % (self.error, self.path)
                return False
            connected_devices = len(re.findall(r"^[^ \t]+[ \t]+device$", adb_res, re.MULTILINE))
            if connected_devices == 0:
                self.error = "No Android device connected (at %s)" % self.path
                return False
            if connected_devices > 1:
                self.error = "Too many (%s) devices are connected. Single device is required. (at %s)" % (connected_devices, self.path)
                return False
            if "armeabi-v7a" in self.arm_target:
                adb_res = self.runAdb("shell", "cat /proc/cpuinfo")
                if not adb_res:
                    self.error = "Could not get info about Android platform: %s (at %s)" % (self.error, self.path)
                    return False
                if "ARMv7" not in adb_res:
                    self.error = "Android device does not support ARMv7 commands, but tests are built for armeabi-v7a (at %s)" % self.path
                    return False
                if "NEON" in self.arm_target and "neon" not in adb_res:
                    self.error = "Android device has no NEON, but tests are built for %s (at %s)" % (self.arm_target, self.path)
                    return False
                hw = re.search(r"^Hardware[ \t]*:[ \t]*(.*?)$", adb_res, re.MULTILINE)
                if hw:
                    self.hardware = hw.groups()[0].strip()
        return True
    
    def runTest(self, path, workingDir, _stdout, _stderr, args = []):
        if self.error:
            return
        args = args[:]
        timestamp = datetime.datetime.now()
        logfile = self.getLogName(path, timestamp)
        exe = os.path.abspath(path)
        
        userlog = [a for a in args if a.startswith("--gtest_output=")]
        if len(userlog) == 0:
            args.append("--gtest_output=xml:" + logfile)
        else:
            logfile = userlog[userlog[0].find(":")+1:]
        
        if self.targetos == "android":
            try:
                andoidcwd = "/data/bin/" + getpass.getuser().replace(" ","") + "_perf/"
                exename = os.path.basename(exe)
                androidexe = andoidcwd + exename
                #upload
                print >> _stderr, "Uploading", exename, "to device..."
                output = Popen([self.adb, "push", exe, androidexe], stdout=_stdout, stderr=_stderr).wait()
                if output != 0:
                    print >> _stderr, "adb finishes unexpectedly with error code", output
                    return
                #chmod
                print >> _stderr, "Changing mode of ", androidexe
                output = Popen([self.adb, "shell", "chmod 777 " + androidexe], stdout=_stdout, stderr=_stderr).wait()
                if output != 0:
                    print >> _stderr, "adb finishes unexpectedly with error code", output
                    return
                #run
                command = exename + " " + " ".join(args)
                print >> _stderr, "Running:", command
                Popen([self.adb, "shell", "export OPENCV_TEST_DATA_PATH=" + self.test_data_path + "&& cd " + andoidcwd + "&& ./" + command], stdout=_stdout, stderr=_stderr).wait()
                # try get log
                print >> _stderr, "Pulling", logfile, "from device..."
                hostlogpath = os.path.join(workingDir, logfile)
                output = Popen([self.adb, "pull", andoidcwd + logfile, hostlogpath], stdout=_stdout, stderr=_stderr).wait()
                if output != 0:
                    print >> _stderr, "adb finishes unexpectedly with error code", output
                    return
                #rm log
                Popen([self.adb, "shell", "rm " + andoidcwd + logfile], stdout=_stdout, stderr=_stderr).wait()
            except OSError:
                pass
            if os.path.isfile(hostlogpath):
                return hostlogpath
            return None
        else:
            cmd = [exe]
            cmd.extend(args)
            print >> _stderr, "Running:", " ".join(cmd)
            try: 
                Popen(cmd, stdout=_stdout, stderr=_stderr, cwd = workingDir).wait()
            except OSError:
                pass
            
            logpath = os.path.join(workingDir, logfile)
            if os.path.isfile(logpath):
                return logpath
            return None
            
    def runTests(self, tests, _stdout, _stderr, workingDir, args = []):
        if self.error:
            return []
        if not tests:
            tests = self.tests
        logs = []
        for test in tests:
            t = self.getTest(test)
            if t:
                logfile = self.runTest(t, workingDir, _stdout, _stderr, args)
                if logfile:
                    logs.append(os.path.relpath(logfile, "."))
            else:
                print >> _stderr, "Error: Test \"%s\" is not found in %s" % (test, self.tests_dir)
        return logs

if __name__ == "__main__":
    test_args = [a for a in sys.argv if a.startswith("--perf_") or a.startswith("--gtest_")]
    argv =      [a for a in sys.argv if not(a.startswith("--perf_") or a.startswith("--gtest_"))]
    
    parser = OptionParser()
    parser.add_option("-t", "--tests", dest="tests", help="comma-separated list of modules to test", metavar="SUITS", default="")
    parser.add_option("-w", "--cwd", dest="cwd", help="working directory for tests", metavar="PATH", default=".")
    parser.add_option("", "--android_test_data_path", dest="test_data_path", help="OPENCV_TEST_DATA_PATH for Android run", metavar="PATH", default="/sdcard/opencv_testdata/")
    parser.add_option("", "--configuration", dest="configuration", help="force Debug or Release donfiguration", metavar="CFG", default="")
    
    (options, args) = parser.parse_args(argv)
    
    run_args = []
    
    for path in args:
        path = os.path.abspath(path)
        while (True):
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "CMakeCache.txt")):
                run_args.append(path)
                break
            npath = os.path.dirname(path)
            if npath == path:
                break
            path = npath
    
    if len(run_args) == 0:
        print >> sys.stderr, "Usage:\n", os.path.basename(sys.argv[0]), "<build_path>"
        exit(1)
        
    tests = [s.strip() for s in options.tests.split(",") if s]
            
    if len(tests) != 1 or len(run_args) != 1:
        #remove --gtest_output from params
        test_args = [a for a in test_args if not a.startswith("--gtest_output=")]
    
    logs = []
    for path in run_args:
        info = RunInfo(path, options.configuration)
        #print vars(info),"\n"
        if not info.isRunnable():
            print >> sys.stderr, "Error:", info.error
        else:
            info.test_data_path = options.test_data_path
            logs.extend(info.runTests(tests, sys.stdout, sys.stderr, options.cwd, test_args))

    if logs:            
        print >> sys.stderr, "Collected:", " ".join(logs)
