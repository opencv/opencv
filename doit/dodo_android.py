#!/usr/local/bin/python3
import os
from pathlib import Path
from dodo_helpers import BuildInfo, MarkerFile

build_info = BuildInfo()
global_config = build_info.config['global']

def get_android_toolchain_file():
    return Path(os.environ['ANDROID_NDK']) / 'build' / 'cmake' / 'android.toolchain.cmake'

def task_check_android_environment():
    checks = [
        check_env_var(
            'ANDROID_NDK',
            'The ANDROID_NDK variable is not set. Please ensure you have installed the android NDK'
        ),
        check_env_var(
            'ANDROID_SDK',
            'The ANDROID_SDK variable is not set. Please ensure you have installed the android SDK'
        ),
    ]
    return {
      'doc': 'Checks whether the android SDK and NDK are installed',
      'actions': [run_checks_actions(checks)],
      'verbosity': 2,
    }


class AndroidLibrary:
    def __init__(self, abi_name, toolchain_name, platform_id):
        self._abi_name = abi_name
        self._arch_output = build_info.output_dir / 'mobile' / 'android' / abi_name
        self._toolchain_name = toolchain_name
        self._platform_id = platform_id
        self._marker = MarkerFile(build_info.output_dir / '.android-{}-built'.format(self._abi_name))

    @property
    def marker(self):
        return self._marker

    @property
    def arch_output(self):
        return self._arch_output

    @property
    def abi_name(self):
        return self._abi_name

    def cmake_action(self):
        cmake_vars = dict(
            CMAKE_TOOLCHAIN_FILE=get_android_toolchain_file(),
            WITH_OPENCL="OFF",
            WITH_IPP=("OFF"),
            WITH_TBB="ON",
            BUILD_EXAMPLES="OFF",
            BUILD_TESTS="OFF",
            BUILD_PERF_TESTS="OFF",
            BUILD_DOCS="OFF",
            BUILD_ANDROID_EXAMPLES="ON",
            BUILD_ANDROID_PROJECTS="OFF",
            INSTALL_ANDROID_EXAMPLES="ON",
            NDK_CCACHE='ccache',
            ANDROID_STL="c++_static",
            ANDROID_ABI=self._abi_name,
            ANDROID_TOOLCHAIN_NAME=self._toolchain_name,
            ANDROID_PLATFORM_ID=self._platform_id,
        )
        if global_config['opencv_contrib_path']:
            modules_path = build_info.script_dir / global_config['opencv_contrib_path'] / 'modules'
            cmake_vars['OPENCV_EXTRA_MODULES_PATH'] = str(modules_path.resolve())

        cmake_command = ['cmake', '-GNinja']
        cmake_command += ['-D%s="%s"' % (k, v) for (k, v) in cmake_vars.items() if v is not None]
        cmake_command += [str(build_info.root_dir.resolve())]

        return ' && '.join([
            'cd {}'.format(self._arch_output),
            ' '.join(cmake_command)
        ])

    def make_action(self):
        make_command = ['ninja']
        if build_info.config['android'] and build_info.config['android']['num_threads']:
            make_command += ['-j{}'.format(build_info.config['android']['num_threads'])]

        return ' && '.join([
            'cd {}'.format(self._arch_output),
            ' '.join(make_command),
        ])

    def task(self):
        build_info = BuildInfo()
        actions = [
            'mkdir -p {}'.format(str(self._arch_output.resolve())),
            self.cmake_action(),
            self.make_action(),
            self.marker.action(),
        ]
        return {
            'name': 'android-{}-library'.format(self._abi_name),
            'doc': 'Build android library for the {} architecture'.format(self._abi_name),
            'actions': actions,
            'file_dep': build_info.standard_source_file_dep,
            'targets': [self.marker.path],
            'verbosity': 2,
            'clean': True,
        }

libraries = [
    AndroidLibrary('armeabi-v7a', 'arm-linux-androideabi-clang3.5', '2'),
    AndroidLibrary('arm64-v8a', 'aarch64-linux-android-clang3.5', '3'),
    AndroidLibrary('x86', 'x86-clang3.5', '4'),
    AndroidLibrary('x86_64', 'x86_64-clang3.5', '5'),
]

def gen_android_tasks():
    for library in libraries:
        yield library.task()
        yield install_task(library)


def install_task(library):
    marker = MarkerFile(build_info.output_dir / '.android-{}-installed'.format(library.abi_name))
    install_dir = build_info.android_output_dir / 'install'
    actions = [
        'mkdir -p {}'.format(str(install_dir.resolve())),
        'cd {} && cmake -DCOMPONENT=libs -DCMAKE_INSTALL_PREFIX={} -P cmake_install.cmake' \
                .format(library.arch_output, str(install_dir.resolve())),
        'cd {} && cmake -DCOMPONENT=dev -DCMAKE_INSTALL_PREFIX={} -P cmake_install.cmake' \
                .format(library.arch_output, str(install_dir.resolve())),
        'cd {} && cmake -DCOMPONENT=java -DCMAKE_INSTALL_PREFIX={} -P cmake_install.cmake' \
                .format(library.arch_output, str(install_dir.resolve())),
        'cd {} && cmake -DCOMPONENT=samples -DCMAKE_INSTALL_PREFIX={} -P cmake_install.cmake' \
                .format(library.arch_output, str(install_dir.resolve())),
        marker.action(),
    ]
    return {
        'name': 'install_{}_android_task'.format(library.abi_name),
        'doc': 'Perform android install task for abi={}'.format(library.abi_name),
        'actions': actions,
        'file_dep': [library.marker.path],
        'targets': [marker.path],
        'verbosity': 2,
        'clean': True,
    }

def task_build_for_android():
    yield {
        'basename': 'build_for_android',
        'name': None,
        'doc': 'Build all android targets',
        'watch': ['.'],
        }
    yield gen_android_tasks()
