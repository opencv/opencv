FaceTracker/REAME.txt
2007-05-24, Mark Asbach <asbach@ient.rwth-aachen.de>

Objective:
This document is intended to get you up and running with an OpenCV Framework on Mac OS X

Building the OpenCV.framework:
In the main directory of the opencv distribution, you will find a shell script called
'make_frameworks.sh' that does all of the typical unixy './configure && make' stuff required
to build a universal binary framework. Invoke this script from Terminal.app, wait some minutes
and you are done.

OpenCV is a Private Framework:
On Mac OS X the concept of Framework bundles is meant to simplify distribution of shared libraries,
accompanying headers and documentation. There are however to subtly different 'flavours' of
Frameworks: public and private ones. The public frameworks get installed into the Frameworks
diretories in /Library, /System/Library or ~/Library and are meant to be shared amongst
applications. The private frameworks are only distributed as parts of an Application Bundle.
This makes it easier to deploy applications because they bring their own framework invisibly to
the user. No installation of the framework is necessary and different applications can bring
different versions of the same framework without any conflict.
Since OpenCV is still a moving target, it seems best to avoid any installation and versioning issues
for an end user. The OpenCV framework that currently comes with this demo application therefore
is a Private Framework.

Use it for targets that result in an Application Bundle:
Since it is a Private Framework, it must be copied to the Frameworks/ directory of an Application
Bundle, which means, it is useless for plain unix console applications. You should create a Carbon
or a Cocoa application target in XCode for your projects. Then add the OpenCV.framework just like
in this demo and add a Copy Files build phase to your target. Let that phase copy to the Framework
directory and drop the OpenCV.framework on the build phase (again just like in this demo code).

The resulting application bundle will be self contained and if you set compiler option correctly
(in the "Build" tab of the "Project Info" window you should find 'i386 ppc' for the architectures),
your application can just be copied to any OS 10.4 Mac and used without further installation.
