#!/usr/bin/env python

import os
import argparse
import logging
import datetime
from run_utils import Err, CMakeCache, log, execute
from run_suite import TestSuite
from run_android import AndroidTestSuite

epilog = '''
NOTE:
Additional options starting with "--gtest_" and "--perf_" will be passed directly to the test executables.
'''

if __name__ == "__main__":

    # log.basicConfig(format='[%(levelname)s] %(message)s', level = log.DEBUG)
    # log.basicConfig(format='[%(levelname)s] %(message)s', level = log.INFO)

    parser = argparse.ArgumentParser(
        description='OpenCV test runner script',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build_path", nargs='?', default=".", help="Path to build directory (should contain CMakeCache.txt, default is current) or to directory with tests (all platform checks will be disabled in this case)")
    parser.add_argument("-t", "--tests", metavar="MODULES", default="", help="Comma-separated list of modules to test (example: -t core,imgproc,java)")
    parser.add_argument("-b", "--blacklist", metavar="MODULES", default="", help="Comma-separated list of modules to exclude from test (example: -b java)")
    parser.add_argument("-a", "--accuracy", action="store_true", default=False, help="Look for accuracy tests instead of performance tests")
    parser.add_argument("--check", action="store_true", default=False, help="Shortcut for '--perf_min_samples=1 --perf_force_samples=1'")
    parser.add_argument("-w", "--cwd", metavar="PATH", default=".", help="Working directory for tests (default is current)")
    parser.add_argument("--list", action="store_true", default=False, help="List available tests (executables)")
    parser.add_argument("--list_short", action="store_true", default=False, help="List available tests (aliases)")
    parser.add_argument("--list_short_main", action="store_true", default=False, help="List available tests (main repository, aliases)")
    parser.add_argument("--configuration", metavar="CFG", default=None, help="Force Debug or Release configuration (for Visual Studio and Java tests build)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Do not run the tests")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print more debug information")

    # Valgrind
    parser.add_argument("--valgrind", action="store_true", default=False, help="Run C++ tests in valgrind")
    parser.add_argument("--valgrind_supp", metavar="FILE", action='append', help="Path to valgrind suppression file (example: --valgrind_supp opencv/platforms/scripts/valgrind.supp)")
    parser.add_argument("--valgrind_opt", metavar="OPT", action="append", default=[], help="Add command line option to valgrind (example: --valgrind_opt=--leak-check=full)")

    # QEMU
    parser.add_argument("--qemu", default="", help="Specify qemu binary and base parameters")

    # Android
    parser.add_argument("--android", action="store_true", default=False, help="Android: force all tests to run on device")
    parser.add_argument("--android_sdk", metavar="PATH", help="Android: path to SDK to use adb and aapt tools")
    parser.add_argument("--android_test_data_path", metavar="PATH", default="/sdcard/opencv_testdata/", help="Android: path to testdata on device")
    parser.add_argument("--android_env", action='append', help="Android: add environment variable (NAME=VALUE)")
    parser.add_argument("--android_propagate_opencv_env", action="store_true", default=False, help="Android: propagate OPENCV* environment variables")
    parser.add_argument("--serial", metavar="serial number", default="", help="Android: directs command to the USB device or emulator with the given serial number")
    parser.add_argument("--package", metavar="package", default="", help="Android: run jUnit tests for specified package")

    parser.add_argument("--trace", action="store_true", default=False, help="Trace: enable OpenCV tracing")
    parser.add_argument("--trace_dump", metavar="trace_dump", default=-1, help="Trace: dump highlight calls (specify max entries count, 0 - dump all)")

    args, other_args = parser.parse_known_args()

    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    test_args = [a for a in other_args if a.startswith("--perf_") or a.startswith("--test_") or a.startswith("--gtest_")]
    bad_args = [a for a in other_args if a not in test_args]
    if len(bad_args) > 0:
        log.error("Error: Bad arguments: %s", bad_args)
        exit(1)

    args.mode = "test" if args.accuracy else "perf"

    android_env = []
    if args.android_env:
        android_env.extend([entry.split("=", 1) for entry in args.android_env])
    if args.android_propagate_opencv_env:
        android_env.extend([entry for entry in os.environ.items() if entry[0].startswith('OPENCV')])
    android_env = dict(android_env)
    if args.android_test_data_path:
        android_env['OPENCV_TEST_DATA_PATH'] = args.android_test_data_path

    if args.valgrind:
        try:
            ver = execute(["valgrind", "--version"], silent=True)
            log.debug("Using %s", ver)
        except OSError as e:
            log.error("Failed to run valgrind: %s", e)
            exit(1)

    if len(args.build_path) != 1:
        test_args = [a for a in test_args if not a.startswith("--gtest_output=")]

    if args.check:
        if not [a for a in test_args if a.startswith("--perf_min_samples=")]:
            test_args.extend(["--perf_min_samples=1"])
        if not [a for a in test_args if a.startswith("--perf_force_samples=")]:
            test_args.extend(["--perf_force_samples=1"])
        if not [a for a in test_args if a.startswith("--perf_verify_sanity")]:
            test_args.extend(["--perf_verify_sanity"])

    if bool(os.environ.get('BUILD_PRECOMMIT', None)):
        test_args.extend(["--skip_unstable=1"])

    ret = 0
    logs = []
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.build_path
    try:
        if not os.path.isdir(path):
            raise Err("Not a directory (should contain CMakeCache.txt ot test executables)")
        cache = CMakeCache(args.configuration)
        fname = os.path.join(path, "CMakeCache.txt")

        if os.path.isfile(fname):
            log.debug("Reading cmake cache file: %s", fname)
            cache.read(path, fname)
        else:
            log.debug("Assuming folder contains tests: %s", path)
            cache.setDummy(path)

        if args.android or cache.getOS() == "android":
            log.debug("Creating Android test runner")
            suite = AndroidTestSuite(args, cache, stamp, android_env)
        else:
            log.debug("Creating native test runner")
            suite = TestSuite(args, cache, stamp)

        if args.list or args.list_short or args.list_short_main:
            suite.listTests(args.list_short or args.list_short_main, args.list_short_main)
        else:
            log.debug("Running tests in '%s', working dir: '%s'", path, args.cwd)

            def parseTests(s):
                return [o.strip() for o in s.split(",") if o]
            logs, ret = suite.runTests(parseTests(args.tests), parseTests(args.blacklist), args.cwd, test_args)
    except Err as e:
        log.error("ERROR: test path '%s' ==> %s", path, e.msg)
        ret = -1

    if logs:
        log.warning("Collected: %s", logs)

    if ret != 0:
        log.error("ERROR: some tests have failed")
    exit(ret)
