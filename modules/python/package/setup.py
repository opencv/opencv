import os
import sys
import platform
import setuptools

SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))

def main():
    os.chdir(SCRIPT_DIR)

    package_name = 'opencv'
    package_version = os.environ.get('OPENCV_VERSION', '4.5.4')  # TODO

    long_description = 'Open Source Computer Vision Library Python bindings'  # TODO

    setuptools.setup(
        name=package_name,
        version=package_version,
        url='https://github.com/opencv/opencv',
        license='Apache 2.0',
        description='OpenCV python bindings',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        maintainer="OpenCV Team",
        install_requires="numpy",
        classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: Apache 2.0 License',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: C++',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
        ],
    )

if __name__ == '__main__':
    main()
