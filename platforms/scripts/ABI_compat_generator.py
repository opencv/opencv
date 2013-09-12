#!/usr/bin/python

from optparse import OptionParser
from shutil import rmtree
import os


architecture = 'armeabi'
excludedHeaders = set(['hdf5.h', 'cap_ios.h', 'ios.h', 'eigen.hpp', 'cxeigen.hpp']) #TOREMOVE
systemIncludes = ['sources/cxx-stl/gnu-libstdc++/4.6/include', \
    '/opt/android-ndk-r8c/platforms/android-8/arch-arm', # TODO: check if this one could be passed as command line arg
    'sources/cxx-stl/gnu-libstdc++/4.6/libs/armeabi-v7a/include']
targetLibs = ['libopencv_java.so']
preamble = ['Eigen/Core']
# TODO: get gcc_options automatically
gcc_options = ['-fexceptions', '-frtti', '-Wno-psabi', '--sysroot=/opt/android-ndk-r8c/platforms/android-8/arch-arm', '-fpic', '-D__ARM_ARCH_5__', '-D__ARM_ARCH_5T__', '-D__ARM_ARCH_5E__', '-D__ARM_ARCH_5TE__', '-fsigned-char', '-march=armv5te', '-mtune=xscale', '-msoft-float', '-fdata-sections', '-ffunction-sections', '-Wa,--noexecstack   ', '-W', '-Wall', '-Werror=return-type', '-Werror=address', '-Werror=sequence-point', '-Wformat', '-Werror=format-security', '-Wmissing-declarations', '-Wundef', '-Winit-self', '-Wpointer-arith', '-Wshadow', '-Wsign-promo', '-Wno-narrowing', '-fdiagnostics-show-option', '-fomit-frame-pointer', '-mthumb', '-fomit-frame-pointer', '-O3', '-DNDEBUG ', '-DNDEBUG']
excludedOptionsPrefix = '-W'



def GetHeaderFiles(root):
    headers = []
    for path in os.listdir(root):
        if not os.path.isdir(os.path.join(root, path)) \
            and os.path.splitext(path)[1] in ['.h', '.hpp'] \
            and not path in excludedHeaders:
            headers.append(os.path.join(root, path))
    return sorted(headers)



def GetClasses(root, prefix):
    classes = []
    if ('' != prefix):
        prefix = prefix + '.'
    for path in os.listdir(root):
        currentPath = os.path.join(root, path)
        if (os.path.isdir(currentPath)):
            classes += GetClasses(currentPath, prefix + path)
        else:
            name = str.split(path, '.')[0]
            ext = str.split(path, '.')[1]
            if (ext == 'class'):
                classes.append(prefix + name)
    return classes



def GetJavaHHeaders():
    print('Generating JNI headers for Java API ...')

    javahHeaders = os.path.join(managerDir, 'javah_generated_headers')
    if os.path.exists(javahHeaders):
        rmtree(javahHeaders)
    os.makedirs(os.path.join(os.getcwd(), javahHeaders))

    AndroidJavaDeps = os.path.join(SDK_path, 'platforms/android-11/android.jar')

    classPath = os.path.join(managerDir, 'sdk/java/bin/classes')
    if not os.path.exists(classPath):
        print('Error: no Java classes found in \'%s\'' % classPath)
        quit()

    allJavaClasses = GetClasses(classPath, '')
    if not allJavaClasses:
        print('Error: no Java classes found')
        quit()

    for currentClass in allJavaClasses:
        os.system('javah -d %s -classpath %s:%s %s' % (javahHeaders, classPath, \
            AndroidJavaDeps, currentClass))

    print('Building JNI headers list ...')
    jniHeaders = GetHeaderFiles(javahHeaders)

    return jniHeaders



def GetImmediateSubdirs(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]



def GetOpenCVModules():
    makefile = open(os.path.join(managerDir, 'sdk/native/jni/OpenCV.mk'), 'r')
    makefileStr = makefile.read()
    left = makefileStr.find('OPENCV_MODULES:=') + len('OPENCV_MODULES:=')
    right = makefileStr[left:].find('\n')
    modules = makefileStr[left:left+right].split()
    modules = filter(lambda x: x != 'ts' and x != 'androidcamera', modules)
    return modules



def FindHeaders(includeJni):
    headers = []

    print('Building Native OpenCV header list ...')

    cppHeadersFolder = os.path.join(managerDir, 'sdk/native/jni/include/opencv2')

    modulesFolders = GetImmediateSubdirs(cppHeadersFolder)
    modules = GetOpenCVModules()

    cppHeaders = []
    for m in modules:
        for f in modulesFolders:
            moduleHeaders = []
            if f == m:
                moduleHeaders += GetHeaderFiles(os.path.join(cppHeadersFolder, f))
                if m == 'flann':
                    flann = os.path.join(cppHeadersFolder, f, 'flann.hpp')
                    moduleHeaders.remove(flann)
                    moduleHeaders.insert(0, flann)
                cppHeaders += moduleHeaders


    cppHeaders += GetHeaderFiles(cppHeadersFolder)
    headers += cppHeaders

    cHeaders = GetHeaderFiles(os.path.join(managerDir, \
        'sdk/native/jni/include/opencv'))
    headers += cHeaders

    if (includeJni):
        headers += GetJavaHHeaders()

    return headers



def FindLibraries():
    libraries = []
    for lib in targetLibs:
        libraries.append(os.path.join(managerDir, 'sdk/native/libs', architecture, lib))
    return libraries



def FindIncludes():
    includes = [os.path.join(managerDir, 'sdk', 'native', 'jni', 'include'),
        os.path.join(managerDir, 'sdk', 'native', 'jni', 'include', 'opencv'),
        os.path.join(managerDir, 'sdk', 'native', 'jni', 'include', 'opencv2')]

    for inc in systemIncludes:
        includes.append(os.path.join(NDK_path, inc))

    return includes



def FilterGCCOptions():
    gcc = filter(lambda x: not x.startswith(excludedOptionsPrefix), gcc_options)
    return sorted(gcc)



def WriteXml(version, headers, includes, libraries):
    xmlName = version + '.xml'

    print '\noutput file: ' + xmlName
    try:
        xml = open(xmlName, 'w')
    except:
        print 'Error: Cannot open output file "%s" for writing' % xmlName
        quit()

    xml.write('<descriptor>')

    xml.write('\n\n<version>')
    xml.write('\n\t%s' % version)
    xml.write('\n</version>')

    xml.write('\n\n<headers>')
    xml.write('\n\t%s' % '\n\t'.join(headers))
    xml.write('\n</headers>')

    xml.write('\n\n<include_paths>')
    xml.write('\n\t%s' % '\n\t'.join(includes))
    xml.write('\n</include_paths>')

    # TODO: uncomment when Eigen problem is solved
    # xml.write('\n\n<include_preamble>')
    # xml.write('\n\t%s' % '\n\t'.join(preamble))
    # xml.write('\n</include_preamble>')

    xml.write('\n\n<libs>')
    xml.write('\n\t%s' % '\n\t'.join(libraries))
    xml.write('\n</libs>')

    xml.write('\n\n<gcc_options>')
    xml.write('\n\t%s' % '\n\t'.join(gcc_options))
    xml.write('\n</gcc_options>')

    xml.write('\n\n</descriptor>')



if __name__ == '__main__':
    usage = '%prog [options] <OpenCV_Manager install directory> <OpenCV_Manager version>'
    parser = OptionParser(usage = usage)
    parser.add_option('--exclude-jni', dest='excludeJni', action="store_true", default=False, metavar="EXCLUDE_JNI", help='Exclude headers for all JNI functions')
    parser.add_option('--sdk', dest='sdk', default='~/NVPACK/android-sdk-linux', metavar="PATH", help='Android SDK path')
    parser.add_option('--ndk', dest='ndk', default='/opt/android-ndk-r8c', metavar="PATH", help='Android NDK path')
    parser.add_option('--java-api-level', dest='java_api_level', default='14', metavar="JAVA_API_LEVEL", help='Java API level for generating JNI headers')

    (options, args) = parser.parse_args()

    if 2 != len(args):
        parser.print_help()
        quit()

    managerDir = args[0]
    version = args[1]

    include_jni = not options.excludeJni
    print 'Include Jni headers: %s' % (include_jni)

    NDK_path = options.ndk
    print 'Using Android NDK from "%s"' % NDK_path

    SDK_path = options.sdk
    print 'Using Android SDK from "%s"' % SDK_path

    headers = FindHeaders(include_jni)

    includes = FindIncludes()

    libraries = FindLibraries()

    gcc_options = FilterGCCOptions()

    WriteXml(version, headers, includes, libraries)
