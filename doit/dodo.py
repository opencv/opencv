#!/usr/local/bin/python3
import os
import re
from pathlib import Path
from termcolor import colored
from dodo_helpers import MarkerFile, BuildInfo, task_setup

from dodo_ios import task_build_for_ios, task_build_ios_framework
from dodo_android import task_build_for_android

build_info = BuildInfo()
global_config = build_info.config['global']

DESKTOP_CMAKE_MARKER = MarkerFile(build_info.output_dir / '.desktop-cmake-built')

def task_desktop_cmake():
    file_deps = build_info.standard_source_file_dep
    actions = []
    cmake_command = [
        'cmake',
    ]
    if global_config['opencv_contrib_path']:
        modules_path = build_info.script_dir / global_config['opencv_contrib_path'] / 'modules'
        message = colored('using opencv_contrib path: ' + str(modules_path.resolve()), 'yellow')
        actions += ['echo "{}"'.format(message)]
        cmake_command += ['-DOPENCV_EXTRA_MODULES_PATH={}'.format(str(modules_path.resolve()))]

    cmake_command += [
        str(build_info.root_dir.resolve()),
    ]

    build_command = ' && '.join([
        'cd {}'.format(str(build_info.desktop_output_dir.resolve())),
        ' '.join(cmake_command)
    ])

    actions += [
        'mkdir -p {}'.format(str(build_info.desktop_output_dir.resolve())),
        build_command,
        DESKTOP_CMAKE_MARKER.action(),
    ]
    return {
        'doc': 'Run cmake for the desktop...',
        'actions': actions,
        'file_dep': file_deps,
        'verbosity': 2,
        'targets': [DESKTOP_CMAKE_MARKER.path],
        'clean': True,
    }


def task_build_for_desktop():
    file_deps = build_info.standard_source_file_dep
    file_deps += [DESKTOP_CMAKE_MARKER.path]
    make_command = [
        'make',
    ]

    if build_info.config['desktop'] and build_info.config['desktop']['num_threads']:
        make_command += ['-j{}'.format(build_info.config['desktop']['num_threads'])]

    build_actions = [
        'cd {}'.format(str(build_info.desktop_output_dir.resolve())),
        ' '.join(make_command),
    ]
    actions = [
        ' && '.join(build_actions)
    ]
    return {
        'doc': 'Build the desktop libraries...',
        'actions': actions,
        'file_dep': file_deps,
        'targets': [],
        'setup': ['setup'],
        'clean': True,
        'verbosity': 2,
    }


def task_build_all():
    return {
      'doc': 'Build opencv for all platforms. This may take a while',
      'actions': ['echo "building all"'],
      'verbosity': 2,
    }
