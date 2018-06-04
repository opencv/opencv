#!/usr/local/bin/python3
from subprocess import check_output
import os
import re
import shutil
from dodo_helpers import BuildInfo, MarkerFile

build_info = BuildInfo()
global_config = build_info.config['global']


def get_xcode_major():
    ret = check_output(["xcodebuild", "-version"])
    m = re.match(r'XCode\s+(\d)\..*', ret.decode('utf-8'), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


class IOSBuilder:
    def __init__(self, arch, platform):
        self._arch = arch
        self._platform = platform
        self._marker = MarkerFile(build_info.output_dir / '.ios-{}-{}-built'.format(self._arch, self._platform))

    @property
    def marker(self):
        return self._marker

    @property
    def build_directory(self):
        return build_info.ios_output_dir / 'build-{}-{}'.format(self._arch.lower(), self._platform.lower())

    @property
    def install_directory(self):
        return build_info.ios_output_dir / 'install' / '{}-{}'.format(self._arch.lower(), self._platform.lower())

    @property
    def ios_toolchain(self):
        return build_info.root_dir / 'platforms' / 'ios' / 'cmake' / 'Toolchains' / 'Toolchain-{}_Xcode.cmake' \
            .format(self._platform)

    @property
    def arch(self):
        return self._arch

    @property
    def platform(self):
        return self._platform

    def cmake_action(self):
        cmake_command = [
            'cmake',
            '-GXcode',
            '-DAPPLE_FRAMEWORK=ON',
            '-DCMAKE_INSTALL_PREFIX=install',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_TOOLCHAIN_FILE={}'.format(str(self.ios_toolchain.resolve())),
            '-DIOS_ARCH={}'.format(self._arch)
        ]
        if self._platform.lower().startswith('iphoneos'):
            cmake_command += ['-DENABLE_NEON=ON']
        xcode_ver = get_xcode_major()

        if global_config['opencv_contrib_path']:
            modules_path = build_info.script_dir / global_config['opencv_contrib_path'] / 'modules'
            cmake_command += ['-DOPENCV_EXTRA_MODULES_PATH={}'.format(str(modules_path.resolve()))]

        bitcode_enabled = False
        if xcode_ver >= 7 and self._platform.lower().startswith('iphoneos'):
            bitcode_enabled = True
            cmake_command += ['-DCMAKE_C_FLAGS=-fembed-bitcode']
            cmake_command += ['-DCMAKE_CXX_FLAGS=-fembed-bitcode']

        cmake_command += [str(build_info.root_dir.resolve())]
        return ' && '.join([
            'cd {}'.format(self.build_directory),
            'echo bitcode_enabled={}'.format(bitcode_enabled),
            ' '.join(cmake_command),
        ])

    def make_action(self):
        build_command = [
            'xcodebuild',
            'IPHONEOS_DEPLOYMENT_TARGET=6.0',
            'ARCHS={}'.format(self._arch),
            '-sdk {}'.format(self._platform.lower()),
            '-configuration Release',
            '-parallelizeTargets',
        ]
        if build_info.config['ios'] and build_info.config['ios']['num_threads']:
            build_command += ['-jobs {}'.format(build_info.config['ios']['num_threads'])]
        build_command += ['-target', 'ALL_BUILD', 'build']
        build_command += ['|', 'xcpretty']
        return ' && '.join([
            'cd {}'.format(self.build_directory),
            ' '.join(build_command),
        ])

    def task(self):
        actions = [
            'mkdir -p {}'.format(self.build_directory),
            self.cmake_action(),
            self.make_action(),
            self.marker.action(),
        ]
        return {
            'name': 'ios-{}-{}-library'.format(self._arch, self._platform),
            'doc': 'Build ios library for the {} with architecture {}'.format(self._platform, self._arch),
            'actions': actions,
            'file_dep': build_info.standard_source_file_dep,
            'targets': [self.marker.path],
            'verbosity': 2,
            'clean': True,
        }


libraries = [
    IOSBuilder("armv7", "iPhoneOS"),
    IOSBuilder("armv7s", "iPhoneOS"),
    IOSBuilder("arm64", "iPhoneOS"),
    IOSBuilder("i386", "iPhoneSimulator"),
    IOSBuilder("x86_64", "iPhoneSimulator"),
]


def install_library_task(library):
    marker = MarkerFile(build_info.output_dir / '.ios-{}-{}-installed'.format(library.arch, library.platform))

    cmake_command = [
        'cmake',
        '-DCMAKE_INSTALL_PREFIX={}'.format(str(library.install_directory.resolve())),
        '-P',
        'cmake_install.cmake',
    ]

    action = ' && '.join([
        'cd {}'.format(library.build_directory),
        ' '.join(cmake_command),
    ])
    return {
        'name': 'install_{}_{}_ios_task'.format(library.arch, library.platform),
        'doc': 'Installing library for ios platform {} with architecture {}'.format(library.platform, library.arch),
        'actions': [action, marker.action()],
        'file_dep': [library.marker.path],
        'targets': [marker.path],
        'verbosity': 1,
        'clean': True,
    }

def merge_library_task(library):
    marker = MarkerFile(build_info.output_dir / '.ios-{}-{}-libraries-merged'.format(library.arch, library.platform))
    merged_lib = library.install_directory / 'libopencv_merged.a'

    library_lib_directory = library.install_directory / 'lib'
    library_3rdparty_directory = library.install_directory / 'share' / 'OpenCV' / '3rdparty' / 'lib'

    all_libs = [str(lib.resolve()) for lib in library_lib_directory.glob('*.a')] + \
        [str(lib.resolve()) for lib in library_3rdparty_directory.glob('*.a')]
    #print("Merging libraries:\n\t%s" % "\n\t".join([str(lib.resolve()) for lib in all_libs]), file=sys.stderr)
    merge_command = [
        'libtool',
        '-static',
        '-o',
        str(merged_lib.resolve()),
    ] + [lib_path for lib_path in all_libs]

    actions = [
        'mkdir -p {}'.format(str(library.install_directory.resolve())),
        ' && '.join([
            'cd {}'.format(str(library.install_directory.resolve())),
            ' '.join(merge_command),
        ]),
        marker.action(),
    ]
    return {
        'name': 'merge_{}_{}_ios_task'.format(library.arch, library.platform),
        'doc': 'Perform ios install task',
        'actions': actions,
        'file_dep': [library.marker.path for library in libraries],
        'targets': [marker.path, merged_lib],
        'verbosity': 1,
        'clean': True,
    }


def gen_ios_tasks():
    for library in libraries:
        yield library.task()
        yield install_library_task(library)
        yield merge_library_task(library)


def task_build_for_ios():
    yield {
        'basename': 'build_for_ios',
        'name': None,
        'doc': 'Build all ios targets and framework',
        'watch': ['.'],
    }
    yield gen_ios_tasks()



def task_build_ios_framework():
    libraries_merged_marker = MarkerFile(build_info.output_dir / '.ios-libraries-merged')
    marker = MarkerFile(build_info.output_dir / '.ios-framework-installed')
    framework_name = 'opencv2'
    install_dir = build_info.ios_output_dir / 'install'
    merged_lib = install_dir / 'libopencv_merged.a'
    framework_path = install_dir / 'opencv2.framework'

    dest_dir = framework_path / 'Versions' / 'A'

    def clean_framework_dir():
        if os.path.isdir(str(framework_path.resolve())):
            shutil.rmtree(str(framework_path.resolve()))

    def copy_headers():
        shutil.copytree(libraries[0].install_directory / 'include' / 'opencv2', dest_dir / 'Headers')

    def do_lipo():
        lib_paths = [str((lib.install_directory / 'libopencv_merged.a').resolve()) for lib in libraries]
        lipo_command = ["lipo", "-create"]
        lipo_command += lib_paths
        lipo_command += ['-o', str((dest_dir / 'opencv2').resolve())]
        return ' '.join(lipo_command)
        #print("Creating universal library from:\n\t%s" % "\n\t".join(libs), file=sys.stderr)
      
    def copy_resources():
        resources_dir = framework_path / 'Resources'
        os.makedirs(resources_dir)
        info_plist = resources_dir / 'Info.plist'
        shutil.copyfile(libraries[0].build_directory / 'ios' / 'Info.plist', resources_dir / "Info.plist")

    def make_symbolic_links():
        commands = [
            'cd {}'.format(str(framework_path.resolve())),
            'ln -s "A" "Versions/Current"', 
            'ln -s "Versions/Current/Headers" "Headers"', 
            'ln -s "Versions/Current/Resources" "Resources"', 
            'ln -s "Versions/Current/opencv2" "opencv2"',
        ]
        return ' && '.join(commands)

    actions = [
        'mkdir -p {}'.format(str(framework_path.resolve())),
        clean_framework_dir,
        copy_headers,
        do_lipo(),
        copy_resources,
        make_symbolic_links(),
        marker.action(),
    ]

    return {
        'doc': 'Build ios framework task',
        'actions': actions,
        'file_dep': [libraries_merged_marker.path],
        'verbosity': 1,
        'clean': True,
    }

