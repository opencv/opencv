#!/usr/bin/env python
# encoding: utf-8

import multiprocessing
import os
import sys

from distutils import sysconfig
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.command.install_lib import install_lib

version = "2.4.9"


OWN_PATH = os.path.dirname(os.path.realpath(__file__))


class BuildOpenCV(build_ext):

    def build_extensions(self):
        os.chdir(OWN_PATH)
        os.system("cmake "
                  "-D BUILD_EXAMPLES=OFF "
                  "-D BUILD_SHARED_LIBS=OFF "
                  "-D PYTHON_PACKAGES_PATH=%(site_packages)s "
                  "-D PYTHON_EXECUTABLE=%(executable)s "
                  "-D PYTHON_INCLUDES=%(includes)s "
                  "-D PYTHON_LIBRARIES=%(site_packages)s "
                  "CMakeLists.txt" % {
                      'site_packages': sysconfig.get_python_lib(),
                      'includes': sysconfig.get_python_inc(),
                      'executable': sys.executable,
                  })
        os.system("make -j %d" % multiprocessing.cpu_count())


class InstallOpenCV(install_lib):

    def install(self):
        outfile = self.copy_file('lib/cv2.so', self.install_dir)
        return [outfile]


package_path = sysconfig.get_python_lib()

setup(
    name="opencv",
    url="https://github.com/stylight/opencv",
    version=version,
    cmdclass={
        'build_ext': BuildOpenCV,
        'install_lib': InstallOpenCV,
    },
    # Provide a dummy Extension to trigger the execution of the build_ext
    # command class.
    ext_modules=[Extension("foobaz", sources=["foobaz.c"])],
    install_requires=["numpy"],
    include_package_data=True,
    data_files=[
        (package_path, ['modules/python/src2/cv.py']),
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
