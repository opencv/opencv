#!/usr/bin/env python
import os
import re
from run_utils import Err, log, execute, getPlatformVersion, isColorEnabled, TempEnvDir
from run_long import LONG_TESTS_DEBUG_VALGRIND, longTestFilter


class TestSuite(object):
    def __init__(self, options, cache, id):
        self.options = options
        self.cache = cache
        self.nameprefix = "opencv_" + self.options.mode + "_"
        self.tests = self.cache.gatherTests(self.nameprefix + "*", self.isTest)
        self.id = id

    def getOS(self):
        return getPlatformVersion() or self.cache.getOS()

    def getLogName(self, app):
        return self.getAlias(app) + '_' + str(self.id) + '.xml'

    def listTests(self, short=False, main=False):
        if len(self.tests) == 0:
            raise Err("No tests found")
        for t in self.tests:
            if short:
                t = self.getAlias(t)
            if not main or self.cache.isMainModule(t):
                log.info("%s", t)

    def getAlias(self, fname):
        return sorted(self.getAliases(fname), key=len)[0]

    def getAliases(self, fname):
        def getCuts(fname, prefix):
            # filename w/o extension (opencv_test_core)
            noext = re.sub(r"\.(exe|apk)$", '', fname)
            # filename w/o prefix (core.exe)
            nopref = fname
            if fname.startswith(prefix):
                nopref = fname[len(prefix):]
            # filename w/o prefix and extension (core)
            noprefext = noext
            if noext.startswith(prefix):
                noprefext = noext[len(prefix):]
            return noext, nopref, noprefext
        # input is full path ('/home/.../bin/opencv_test_core') or 'java'
        res = [fname]
        fname = os.path.basename(fname)
        res.append(fname)  # filename (opencv_test_core.exe)
        for s in getCuts(fname, self.nameprefix):
            res.append(s)
            if self.cache.build_type == "Debug" and "Visual Studio" in self.cache.cmake_generator:
                res.append(re.sub(r"d$", '', s))  # MSVC debug config, remove 'd' suffix
        log.debug("Aliases: %s", set(res))
        return set(res)

    def getTest(self, name):
        # return stored test name by provided alias
        for t in self.tests:
            if name in self.getAliases(t):
                return t
        raise Err("Can not find test: %s", name)

    def getTestList(self, white, black):
        res = [t for t in white or self.tests if self.getAlias(t) not in black]
        if len(res) == 0:
            raise Err("No tests found")
        return set(res)

    def isTest(self, fullpath):
        if fullpath in ['java', 'python2', 'python3']:
            return self.options.mode == 'test'
        if not os.path.isfile(fullpath):
            return False
        if self.cache.getOS() == "nt" and not fullpath.endswith(".exe"):
            return False
        return os.access(fullpath, os.X_OK)

    def wrapInValgrind(self, cmd=[]):
        if self.options.valgrind:
            res = ['valgrind']
            supp = self.options.valgrind_supp or []
            for f in supp:
                if os.path.isfile(f):
                    res.append("--suppressions=%s" % f)
                else:
                    print("WARNING: Valgrind suppression file is missing, SKIP: %s" % f)
            res.extend(self.options.valgrind_opt)
            has_gtest_filter = next((True for x in cmd if x.startswith('--gtest_filter=')), False)
            return res + cmd + ([longTestFilter(LONG_TESTS_DEBUG_VALGRIND)] if not has_gtest_filter else [])
        return cmd

    def tryCommand(self, cmd, workingDir):
        try:
            if 0 == execute(cmd, cwd=workingDir):
                return True
        except:
            pass
        return False

    def runTest(self, path, logfile, workingDir, args=[]):
        args = args[:]
        exe = os.path.abspath(path)
        if path == "java":
            cmd = [self.cache.ant_executable, "-Dopencv.build.type=%s" % self.cache.build_type, "buildAndTest"]
            ret = execute(cmd, cwd=self.cache.java_test_dir)
            return None, ret
        elif path in ['python2', 'python3']:
            executable = os.getenv('OPENCV_PYTHON_BINARY', None)
            if executable is None:
                executable = path
                if not self.tryCommand([executable, '--version'], workingDir):
                    executable = 'python'
            cmd = [executable, self.cache.opencv_home + '/modules/python/test/test.py', '--repo', self.cache.opencv_home, '-v'] + args
            module_suffix = '' if 'Visual Studio' not in self.cache.cmake_generator else '/' + self.cache.build_type
            env = {}
            env['PYTHONPATH'] = self.cache.opencv_build + '/lib' + module_suffix + os.pathsep + os.getenv('PYTHONPATH', '')
            if self.cache.getOS() == 'nt':
                env['PATH'] = self.cache.opencv_build + '/bin' + module_suffix + os.pathsep + os.getenv('PATH', '')
            else:
                env['LD_LIBRARY_PATH'] = self.cache.opencv_build + '/bin' + os.pathsep + os.getenv('LD_LIBRARY_PATH', '')
            ret = execute(cmd, cwd=workingDir, env=env)
            return None, ret
        else:
            if isColorEnabled(args):
                args.append("--gtest_color=yes")
            cmd = self.wrapInValgrind([exe] + args)
            env = {}
            if not self.options.valgrind and self.options.trace:
                env['OPENCV_TRACE'] = '1'
                env['OPENCV_TRACE_LOCATION'] = 'OpenCVTrace-{}'.format(self.getLogBaseName(exe))
                env['OPENCV_TRACE_SYNC_OPENCL'] = '1'
            tempDir = TempEnvDir('OPENCV_TEMP_PATH', "__opencv_temp.")
            tempDir.init()
            log.warning("Run: %s" % " ".join(cmd))
            ret = execute(cmd, cwd=workingDir, env=env)
            try:
                if not self.options.valgrind and self.options.trace and int(self.options.trace_dump) >= 0:
                    import trace_profiler
                    trace = trace_profiler.Trace(env['OPENCV_TRACE_LOCATION']+'.txt')
                    trace.process()
                    trace.dump(max_entries=int(self.options.trace_dump))
            except:
                import traceback
                traceback.print_exc()
                pass
            tempDir.clean()
            hostlogpath = os.path.join(workingDir, logfile)
            if os.path.isfile(hostlogpath):
                return hostlogpath, ret
            return None, ret

    def runTests(self, tests, black, workingDir, args=[]):
        args = args[:]
        logs = []
        test_list = self.getTestList(tests, black)
        if len(test_list) != 1:
            args = [a for a in args if not a.startswith("--gtest_output=")]
        ret = 0
        for test in test_list:
            more_args = []
            exe = self.getTest(test)

            if exe in ["java", "python2", "python3"]:
                logname = None
            else:
                userlog = [a for a in args if a.startswith("--gtest_output=")]
                if len(userlog) == 0:
                    logname = self.getLogName(exe)
                    more_args.append("--gtest_output=xml:" + logname)
                else:
                    logname = userlog[0][userlog[0].find(":")+1:]

            log.debug("Running the test: %s (%s) ==> %s in %s", exe, args + more_args, logname, workingDir)
            if self.options.dry_run:
                logfile, r = None, 0
            else:
                logfile, r = self.runTest(exe, logname, workingDir, args + more_args)
            log.debug("Test returned: %s ==> %s", r, logfile)

            if r != 0:
                ret = r
            if logfile:
                logs.append(os.path.relpath(logfile, workingDir))
        return logs, ret


if __name__ == "__main__":
    log.error("This is utility file, please execute run.py script")
