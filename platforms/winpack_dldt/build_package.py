#!/usr/bin/env python

import os, sys
import argparse
import glob
import re
import shutil
import subprocess
import time

import logging as log

if sys.version_info[0] == 2:
    sys.exit("FATAL: Python 2.x is not supported")

from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Fail(Exception):
    def __init__(self, text=None):
        self.t = text
    def __str__(self):
        return "ERROR" if self.t is None else self.t

def execute(cmd, cwd=None, shell=False):
    try:
        log.debug("Executing: %s" % cmd)
        log.info('Executing: ' + ' '.join(cmd))
        if cwd:
            log.info("    in: %s" % cwd)
        retcode = subprocess.call(cmd, shell=shell, cwd=str(cwd) if cwd else None)
        if retcode < 0:
            raise Fail("Child was terminated by signal: %s" % -retcode)
        elif retcode > 0:
            raise Fail("Child returned: %s" % retcode)
    except OSError as e:
        raise Fail("Execution failed: %d / %s" % (e.errno, e.strerror))

def check_executable(cmd):
    try:
        log.debug("Executing: %s" % cmd)
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if not isinstance(result, str):
            result = result.decode("utf-8")
        log.debug("Result: %s" % (result + '\n').split('\n')[0])
        return True
    except OSError as e:
        log.debug('Failed: %s' % e)
        return False


def rm_one(d):
    d = str(d)  # Python 3.5 may not handle Path
    d = os.path.abspath(d)
    if os.path.exists(d):
        if os.path.isdir(d):
            log.info("Removing dir: %s", d)
            shutil.rmtree(d)
        elif os.path.isfile(d):
            log.info("Removing file: %s", d)
            os.remove(d)


def prepare_dir(d, clean=False):
    d = str(d)  # Python 3.5 may not handle Path
    d = os.path.abspath(d)
    log.info("Preparing directory: '%s' (clean: %r)", d, clean)
    if os.path.exists(d):
        if not os.path.isdir(d):
            raise Fail("Not a directory: %s" % d)
        if clean:
            for item in os.listdir(d):
                rm_one(os.path.join(d, item))
    else:
        os.makedirs(d)
    return Path(d)


def check_dir(d):
    d = str(d)  # Python 3.5 may not handle Path
    d = os.path.abspath(d)
    log.info("Check directory: '%s'", d)
    if os.path.exists(d):
        if not os.path.isdir(d):
            raise Fail("Not a directory: %s" % d)
    else:
        raise Fail("The directory is missing: %s" % d)
    return Path(d)


# shutil.copytree fails if dst exists
def copytree(src, dst, exclude=None):
    log.debug('copytree(%s, %s)', src, dst)
    src = str(src)  # Python 3.5 may not handle Path
    dst = str(dst)  # Python 3.5 may not handle Path
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        return
    def copy_recurse(subdir):
        if exclude and subdir in exclude:
            log.debug('  skip: %s', subdir)
            return
        s = os.path.join(src, subdir)
        d = os.path.join(dst, subdir)
        if os.path.exists(d) or exclude:
            if os.path.isfile(s):
                shutil.copy2(s, d)
            elif os.path.isdir(s):
                if not os.path.isdir(d):
                    os.makedirs(d)
                for item in os.listdir(s):
                    copy_recurse(os.path.join(subdir, item))
            else:
                assert False, s + " => " + d
        else:
            if os.path.isfile(s):
                shutil.copy2(s, d)
            elif os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                assert False, s + " => " + d
    copy_recurse('')


def git_checkout(dst, url, branch, revision, clone_extra_args, noFetch=False):
    assert isinstance(dst, Path)
    log.info("Git checkout: '%s' (%s @ %s)", dst, url, revision)
    if noFetch:
        pass
    elif not os.path.exists(str(dst / '.git')):
        execute(cmd=['git', 'clone'] +
                (['-b', branch] if branch else []) +
                clone_extra_args + [url, '.'], cwd=dst)
    else:
        execute(cmd=['git', 'fetch', 'origin'] + ([branch + ':' + branch] if branch else []), cwd=dst)
    execute(cmd=['git', 'reset', '--hard'], cwd=dst)
    execute(cmd=['git', 'clean', '-f', '-d'], cwd=dst)
    execute(cmd=['git', 'checkout', '--force', '-B', 'winpack_dldt', revision], cwd=dst)
    execute(cmd=['git', 'clean', '-f', '-d'], cwd=dst)
    execute(cmd=['git', 'submodule', 'init'], cwd=dst)
    execute(cmd=['git', 'submodule', 'update', '--force', '--depth=1000'], cwd=dst)
    log.info("Git checkout: DONE")
    execute(cmd=['git', 'status'], cwd=dst)
    execute(cmd=['git', 'log', '--max-count=1', 'HEAD'], cwd=dst)


def git_apply_patch(src_dir, patch_file):
    src_dir = str(src_dir)  # Python 3.5 may not handle Path
    patch_file = str(patch_file)  # Python 3.5 may not handle Path
    assert os.path.exists(patch_file), patch_file
    execute(cmd=['git', 'apply', '--3way', '-v', '--ignore-space-change', str(patch_file)], cwd=src_dir)
    execute(cmd=['git', '--no-pager', 'diff', 'HEAD'], cwd=src_dir)
    os.environ['GIT_AUTHOR_NAME'] = os.environ['GIT_COMMITTER_NAME']='build'
    os.environ['GIT_AUTHOR_EMAIL'] = os.environ['GIT_COMMITTER_EMAIL']='build@opencv.org'
    execute(cmd=['git', 'commit', '-am', 'apply opencv patch'], cwd=src_dir)


#===================================================================================================

class BuilderDLDT:
    def __init__(self, config):
        self.config = config

        cpath = self.config.dldt_config
        log.info('DLDT build configration: %s', cpath)
        if not os.path.exists(cpath):
            cpath = os.path.join(SCRIPT_DIR, cpath)
            if not os.path.exists(cpath):
                raise Fail('Config "%s" is missing' % cpath)
        self.cpath = Path(cpath)

        clean_src_dir = self.config.clean_dldt
        if self.config.dldt_src_dir:
            assert os.path.exists(self.config.dldt_src_dir), self.config.dldt_src_dir
            dldt_dir_name = 'dldt-custom'
            self.srcdir = self.config.dldt_src_dir
            clean_src_dir = False
        else:
            assert not self.config.dldt_src_dir
            self.init_patchset()
            dldt_dir_name = 'dldt-' + self.config.dldt_src_commit + \
                    ('/patch-' + self.patch_hashsum if self.patch_hashsum else '')
            if self.config.build_debug:
                dldt_dir_name += '-debug'
            self.srcdir = None  # updated below
        log.info('DLDT directory: %s', dldt_dir_name)
        self.outdir = prepare_dir(os.path.join(self.config.build_cache_dir, dldt_dir_name))
        if self.srcdir is None:
            self.srcdir = prepare_dir(self.outdir / 'sources', clean=clean_src_dir)
        self.build_dir = prepare_dir(self.outdir / 'build', clean=self.config.clean_dldt)
        self.sysrootdir = prepare_dir(self.outdir / 'sysroot', clean=self.config.clean_dldt)

        if self.config.build_subst_drive:
            if os.path.exists(self.config.build_subst_drive + ':\\'):
                execute(['subst', self.config.build_subst_drive + ':', '/D'])
            execute(['subst', self.config.build_subst_drive + ':', str(self.outdir)])
            def fix_path(p):
                return str(p).replace(str(self.outdir), self.config.build_subst_drive + ':')
            self.srcdir = Path(fix_path(self.srcdir))
            self.build_dir = Path(fix_path(self.build_dir))
            self.sysrootdir = Path(fix_path(self.sysrootdir))


    def init_patchset(self):
        cpath = self.cpath
        self.patch_file = str(cpath / 'patch.config.py')  # Python 3.5 may not handle Path
        with open(self.patch_file, 'r') as f:
            self.patch_file_contents = f.read()

        patch_hashsum = None
        try:
            import hashlib
            patch_hashsum = hashlib.md5(self.patch_file_contents.encode('utf-8')).hexdigest()
        except:
            log.warn("Can't compute hashsum of patches: %s", self.patch_file)
        self.patch_hashsum = patch_hashsum


    def prepare_sources(self):
        if self.config.dldt_src_dir:
            log.info('Using DLDT custom repository: %s', self.srcdir)
            return

        def do_clone(srcdir, noFetch):
            git_checkout(srcdir, self.config.dldt_src_url, self.config.dldt_src_branch, self.config.dldt_src_commit,
                    ['-n', '--depth=100', '--no-single-branch', '--recurse-submodules'] +
                    (self.config.dldt_src_git_clone_extra or []),
                    noFetch=noFetch
            )

        if not os.path.exists(str(self.srcdir / '.git')):
            log.info('DLDT git checkout through "reference" copy.')
            reference_dir = self.config.dldt_reference_dir
            if reference_dir is None:
                reference_dir = prepare_dir(os.path.join(self.config.build_cache_dir, 'dldt-git-reference-repository'))
                do_clone(reference_dir, False)
                log.info('DLDT reference git checkout completed. Copying...')
            else:
                log.info('Using DLDT reference repository. Copying...')
            copytree(reference_dir, self.srcdir)
            do_clone(self.srcdir, True)
        else:
            do_clone(self.srcdir, False)

        log.info('DLDT git checkout completed. Patching...')

        def applyPatch(patch_file, subdir = None):
            if subdir:
                log.info('Patching "%s": %s' % (subdir, patch_file))
            else:
                log.info('Patching: %s' % (patch_file))
            git_apply_patch(self.srcdir / subdir if subdir else self.srcdir, self.cpath / patch_file)

        exec(compile(self.patch_file_contents, self.patch_file, 'exec'))

        log.info('DLDT patches applied')


    def build(self):
        self.cmake_path = 'cmake'
        build_config = 'Release' if not self.config.build_debug else 'Debug'

        cmd = [self.cmake_path, '-G', 'Visual Studio 16 2019', '-A', 'x64']

        cmake_vars = dict(
            CMAKE_BUILD_TYPE=build_config,
            TREAT_WARNING_AS_ERROR='OFF',
            ENABLE_SAMPLES='OFF',
            ENABLE_TESTS='OFF',
            BUILD_TESTS='OFF',
            ENABLE_OPENCV='OFF',
            ENABLE_GNA='OFF',
            ENABLE_SPEECH_DEMO='OFF',  # 2020.4+
            NGRAPH_DOC_BUILD_ENABLE='OFF',
            NGRAPH_UNIT_TEST_ENABLE='OFF',
            NGRAPH_UNIT_TEST_OPENVINO_ENABLE='OFF',
            NGRAPH_TEST_UTIL_ENABLE='OFF',
            NGRAPH_ONNX_IMPORT_ENABLE='OFF',
            CMAKE_INSTALL_PREFIX=str(self.build_dir / 'install'),
            OUTPUT_ROOT=str(self.build_dir),  # 2020.4+
        )

        self.build_config_file = str(self.cpath / 'build.config.py')  # Python 3.5 may not handle Path
        if os.path.exists(str(self.build_config_file)):
            with open(self.build_config_file, 'r') as f:
                cfg = f.read()
            exec(compile(cfg, str(self.build_config_file), 'exec'))
            log.info('DLDT processed build configuration script')

        cmd += [ '-D%s=%s' % (k, v) for (k, v) in cmake_vars.items() if v is not None]
        if self.config.cmake_option_dldt:
            cmd += self.config.cmake_option_dldt

        cmd.append(str(self.srcdir))

        build_dir = self.build_dir
        try:
            execute(cmd, cwd=build_dir)

            # build
            cmd = [self.cmake_path, '--build', '.', '--config', build_config, # '--target', 'install',
                    '--',
                    # '/m:2' is removed, not properly supported by 2021.3
                    '/v:n', '/consoleloggerparameters:NoSummary',
            ]
            execute(cmd, cwd=build_dir)

            # install ngraph only
            cmd = [self.cmake_path, '-DBUILD_TYPE=' + build_config, '-P', 'cmake_install.cmake']
            execute(cmd, cwd=build_dir / 'ngraph')
        except:
            raise

        log.info('DLDT build completed')


    def make_sysroot(self):
        cfg_file = str(self.cpath / 'sysroot.config.py')  # Python 3.5 may not handle Path
        with open(cfg_file, 'r') as f:
            cfg = f.read()
        exec(compile(cfg, cfg_file, 'exec'))

        log.info('DLDT sysroot preparation completed')


    def cleanup(self):
        if self.config.build_subst_drive:
            execute(['subst', self.config.build_subst_drive + ':', '/D'])


#===================================================================================================

class Builder:
    def __init__(self, config):
        self.config = config
        build_dir_name = 'opencv_build' if not self.config.build_debug else 'opencv_build_debug'
        self.build_dir = prepare_dir(Path(self.config.output_dir) / build_dir_name, clean=self.config.clean_opencv)
        self.package_dir = prepare_dir(Path(self.config.output_dir) / 'package/opencv', clean=True)
        self.install_dir = prepare_dir(self.package_dir / 'build')
        self.src_dir = check_dir(self.config.opencv_dir)


    def build(self, builderDLDT):
        self.cmake_path = 'cmake'
        build_config = 'Release' if not self.config.build_debug else 'Debug'

        cmd = [self.cmake_path, '-G', 'Visual Studio 16 2019', '-A', 'x64']

        cmake_vars = dict(
            CMAKE_BUILD_TYPE=build_config,
            INSTALL_CREATE_DISTRIB='ON',
            BUILD_opencv_world='OFF',
            BUILD_TESTS='OFF',
            BUILD_PERF_TESTS='OFF',
            ENABLE_CXX11='ON',
            WITH_INF_ENGINE='ON',
            INF_ENGINE_RELEASE=str(self.config.dldt_release),
            WITH_TBB='ON',
            CPU_BASELINE='AVX2',
            CMAKE_INSTALL_PREFIX=str(self.install_dir),
            INSTALL_PDB='ON',
            INSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL='OFF',

            VIDEOIO_PLUGIN_LIST='all',

            OPENCV_SKIP_CMAKE_ROOT_CONFIG='ON',
            OPENCV_BIN_INSTALL_PATH='bin',
            OPENCV_INCLUDE_INSTALL_PATH='include',
            OPENCV_LIB_INSTALL_PATH='lib',
            OPENCV_CONFIG_INSTALL_PATH='cmake',
            OPENCV_3P_LIB_INSTALL_PATH='3rdparty',
            OPENCV_SAMPLES_SRC_INSTALL_PATH='samples',
            OPENCV_DOC_INSTALL_PATH='doc',
            OPENCV_OTHER_INSTALL_PATH='etc',
            OPENCV_LICENSES_INSTALL_PATH='etc/licenses',

            OPENCV_INSTALL_DATA_DIR_RELATIVE='../../src/opencv',

            BUILD_opencv_python2='OFF',
            BUILD_opencv_python3='ON',
            PYTHON3_LIMITED_API='ON',
            OPENCV_PYTHON_INSTALL_PATH='python',
        )

        cmake_vars['INF_ENGINE_LIB_DIRS:PATH'] = str(builderDLDT.sysrootdir / 'deployment_tools/inference_engine/lib/intel64')
        assert os.path.exists(cmake_vars['INF_ENGINE_LIB_DIRS:PATH']), cmake_vars['INF_ENGINE_LIB_DIRS:PATH']
        cmake_vars['INF_ENGINE_INCLUDE_DIRS:PATH'] = str(builderDLDT.sysrootdir / 'deployment_tools/inference_engine/include')
        assert os.path.exists(cmake_vars['INF_ENGINE_INCLUDE_DIRS:PATH']), cmake_vars['INF_ENGINE_INCLUDE_DIRS:PATH']

        ngraph_DIR = str(builderDLDT.sysrootdir / 'ngraph/cmake')
        if not os.path.exists(ngraph_DIR):
            ngraph_DIR = str(builderDLDT.sysrootdir / 'ngraph/deployment_tools/ngraph/cmake')
        assert os.path.exists(ngraph_DIR), ngraph_DIR
        cmake_vars['ngraph_DIR:PATH'] = ngraph_DIR

        cmake_vars['TBB_DIR:PATH'] = str(builderDLDT.sysrootdir / 'tbb/cmake')
        assert os.path.exists(cmake_vars['TBB_DIR:PATH']), cmake_vars['TBB_DIR:PATH']

        if self.config.build_debug:
            cmake_vars['CMAKE_BUILD_TYPE'] = 'Debug'
            cmake_vars['BUILD_opencv_python3'] ='OFF'  # python3x_d.lib is missing
            cmake_vars['OPENCV_INSTALL_APPS_LIST'] = 'all'

        if self.config.build_tests:
            cmake_vars['BUILD_TESTS'] = 'ON'
            cmake_vars['BUILD_PERF_TESTS'] = 'ON'
            cmake_vars['BUILD_opencv_ts'] = 'ON'
            cmake_vars['INSTALL_TESTS']='ON'

        if self.config.build_tests_dnn:
            cmake_vars['BUILD_TESTS'] = 'ON'
            cmake_vars['BUILD_PERF_TESTS'] = 'ON'
            cmake_vars['BUILD_opencv_ts'] = 'ON'
            cmake_vars['OPENCV_BUILD_TEST_MODULES_LIST'] = 'dnn'
            cmake_vars['OPENCV_BUILD_PERF_TEST_MODULES_LIST'] = 'dnn'
            cmake_vars['INSTALL_TESTS']='ON'

        cmd += [ "-D%s=%s" % (k, v) for (k, v) in cmake_vars.items() if v is not None]
        if self.config.cmake_option:
            cmd += self.config.cmake_option

        cmd.append(str(self.src_dir))

        log.info('Configuring OpenCV...')

        execute(cmd, cwd=self.build_dir)

        log.info('Building OpenCV...')

        # build
        cmd = [self.cmake_path, '--build', '.', '--config', build_config, '--target', 'install',
                '--', '/v:n', '/m:2', '/consoleloggerparameters:NoSummary'
        ]
        execute(cmd, cwd=self.build_dir)

        log.info('OpenCV build/install completed')


    def copy_sysroot(self, builderDLDT):
        log.info('Copy sysroot files')

        copytree(builderDLDT.sysrootdir / 'bin', self.install_dir / 'bin')
        copytree(builderDLDT.sysrootdir / 'etc', self.install_dir / 'etc')

        log.info('Copy sysroot files - DONE')


    def package_sources(self):
        package_opencv = prepare_dir(self.package_dir / 'src/opencv', clean=True)
        package_opencv = str(package_opencv)  # Python 3.5 may not handle Path
        execute(cmd=['git', 'clone', '-s', str(self.src_dir), '.'], cwd=str(package_opencv))
        for item in os.listdir(package_opencv):
            if str(item).startswith('.git'):
                rm_one(os.path.join(package_opencv, item))

        with open(str(self.package_dir / 'README.md'), 'w') as f:
            f.write('See licensing/copying statements in "build/etc/licenses"\n')
            f.write('Wiki page: https://github.com/opencv/opencv/wiki/Intel%27s-Deep-Learning-Inference-Engine-backend\n')

        log.info('Package OpenCV sources - DONE')


#===================================================================================================

def main():

    dldt_src_url = 'https://github.com/openvinotoolkit/openvino'
    dldt_src_commit = '2021.3'
    dldt_release = '2021030000'

    build_cache_dir_default = os.environ.get('BUILD_CACHE_DIR', '.build_cache')
    build_subst_drive = os.environ.get('BUILD_SUBST_DRIVE', None)

    parser = argparse.ArgumentParser(
            description='Build OpenCV Windows package with Inference Engine (DLDT)',
    )
    parser.add_argument('output_dir', nargs='?', default='.', help='Output directory')
    parser.add_argument('opencv_dir', nargs='?', default=os.path.join(SCRIPT_DIR, '../..'), help='Path to OpenCV source dir')
    parser.add_argument('--build_cache_dir', default=build_cache_dir_default, help='Build cache directory (sources and binaries cache of build dependencies, default = "%s")' % build_cache_dir_default)
    parser.add_argument('--build_subst_drive', default=build_subst_drive, help='Drive letter to workaround Windows limit for 260 symbols in path (error MSB3491)')

    parser.add_argument('--cmake_option', action='append', help='Append OpenCV CMake option')
    parser.add_argument('--cmake_option_dldt', action='append', help='Append CMake option for DLDT project')

    parser.add_argument('--clean_dldt', action='store_true', help='Clear DLDT build and sysroot directories')
    parser.add_argument('--clean_opencv', action='store_true', help='Clear OpenCV build directory')

    parser.add_argument('--build_debug', action='store_true', help='Build debug binaries')
    parser.add_argument('--build_tests', action='store_true', help='Build OpenCV tests')
    parser.add_argument('--build_tests_dnn', action='store_true', help='Build OpenCV DNN accuracy and performance tests only')

    parser.add_argument('--dldt_src_url', default=dldt_src_url, help='DLDT source URL (tag / commit, default: %s)' % dldt_src_url)
    parser.add_argument('--dldt_src_branch', help='DLDT checkout branch')
    parser.add_argument('--dldt_src_commit', default=dldt_src_commit, help='DLDT source commit / tag (default: %s)' % dldt_src_commit)
    parser.add_argument('--dldt_src_git_clone_extra', action='append', help='DLDT git clone extra args')
    parser.add_argument('--dldt_release', default=dldt_release, help='DLDT release code for INF_ENGINE_RELEASE (default: %s)' % dldt_release)

    parser.add_argument('--dldt_reference_dir', help='DLDT reference git repository (optional)')
    parser.add_argument('--dldt_src_dir', help='DLDT custom source repository (skip git checkout and patching, use for TESTING only)')

    parser.add_argument('--dldt_config', help='Specify DLDT build configuration (defaults to evaluate from DLDT commit/branch)')

    args = parser.parse_args()

    log.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=os.environ.get('LOGLEVEL', 'INFO'),
            datefmt='%Y-%m-%d %H:%M:%S'
    )
    log.debug('Args: %s', args)

    if not check_executable(['git', '--version']):
        sys.exit("FATAL: 'git' is not available")
    if not check_executable(['cmake', '--version']):
        sys.exit("FATAL: 'cmake' is not available")

    if os.path.realpath(args.output_dir) == os.path.realpath(SCRIPT_DIR):
        raise Fail("Specify output_dir (building from script directory is not supported)")
    if os.path.realpath(args.output_dir) == os.path.realpath(args.opencv_dir):
        raise Fail("Specify output_dir (building from OpenCV source directory is not supported)")

    # Relative paths become invalid in sub-directories
    if args.opencv_dir is not None and not os.path.isabs(args.opencv_dir):
        args.opencv_dir = os.path.abspath(args.opencv_dir)

    if not args.dldt_config:
        if str(args.dldt_src_commit).startswith('releases/20'):  # releases/2020/4
            args.dldt_config = str(args.dldt_src_commit)[len('releases/'):].replace('/', '.')
            if not args.dldt_src_branch:
                args.dldt_src_branch = args.dldt_src_commit
        elif str(args.dldt_src_branch).startswith('releases/20'):  # releases/2020/4
            args.dldt_config = str(args.dldt_src_branch)[len('releases/'):].replace('/', '.')
        else:
            args.dldt_config = args.dldt_src_commit

    _opencv_dir = check_dir(args.opencv_dir)
    _outdir = prepare_dir(args.output_dir)
    _cachedir = prepare_dir(args.build_cache_dir)

    ocv_hooks_dir = os.environ.get('OPENCV_CMAKE_HOOKS_DIR', None)
    hooks_dir = os.path.join(SCRIPT_DIR, 'cmake-opencv-checks')
    os.environ['OPENCV_CMAKE_HOOKS_DIR'] = hooks_dir if ocv_hooks_dir is None else (hooks_dir + ';' + ocv_hooks_dir)

    builder_dldt = BuilderDLDT(args)

    try:
        builder_dldt.prepare_sources()
        builder_dldt.build()
        builder_dldt.make_sysroot()

        builder_opencv = Builder(args)
        builder_opencv.build(builder_dldt)
        builder_opencv.copy_sysroot(builder_dldt)
        builder_opencv.package_sources()
    except:
        builder_dldt.cleanup()
        raise

    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")


if __name__ == "__main__":
    try:
        main()
    except:
        log.info('FATAL: Error occured. To investigate problem try to change logging level using LOGLEVEL=DEBUG environment variable.')
        raise
