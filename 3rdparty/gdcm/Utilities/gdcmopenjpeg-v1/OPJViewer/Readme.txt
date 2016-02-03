===============================================================================
	JPEG2000 Visualization Software - OPJViewer

		Version 0.4 beta
===============================================================================


1. Scope
=============

This document describes the installation and use of the OPJViewer in the framework of OpenJPEG library.

This implementation has been developed using the OpenJPEG library as decoding engine and wxWidgets 2.8 as GUI engine.

If you find some bugs or if you have problems using the viewer, please send an e-mail to jpwl@diei.unipg.it

2. Installing the viewer
==========================

There are two options available, at the moment:

a) compile from source code
b) download a precompiled binary.

In order to use option a), it is mandatory to have compiled and built the LibOpenJPEG_JPWL library and the wxWidgets 2.8 framework (you have to download it from http://www.wxwidgets.org/ and compile the wx* libraries).

2.1. Compiling the source code in Windows
-------------------------------------------

The steps required to compile the viewer under windows are:

a) Download at least the libopenjpeg, jpwl, and opjviewer folders from the SVN trunk.
b) Open the OPJViewer.dsw workspace with Visual C++ 6 and activate the "OPJViewer - Win32 Release" configuration.
c) In the configuration settings, go to the C++ tab and modify the wxWidgets paths in order to reflect your wx* install configuration (Preprocessor -> Additional include directories): simply update each instance of the two wx paths, do not remove or add them.
d) In the configuration settings, go to the Link tab and modify the wxWidgets path in order to reflect your wx* install configuration (Input -> Additional library path): simply update the wx path.
e) In the configuration settings, go to the Resources tab and modify the wxWidgets path in order to reflect your wx* install configuration (Additional resource include directories): simply update the wx path.
f) Build!
g) Run!
h) (OPTIONAL) Prepare an installer by compiling the InnoSetup script OPJViewer.iss (you need to download InnoSetup from http://www.jrsoftware.org/isinfo.php).

2.1.1 Additional libraries
----------------------------

Since we are also working on the Digital Cinema JPEG 2000, we are integrating the viewer with the MXF library, which is used to prepare the DCPs for digital movies. You can enable its linking in the code by specifying the USE_MXF preprocessor directive but, remember, the integration is at a very early stage.

2.2. Compiling the source code in Unix-like systems
-----------------------------------------------------

The porting is possible and under way.


3. General information on the viewer
====================================

This viewer is conceived to open and display information and image content of J2K, JP2, and MJ2 files.
The viewer application interface is divided into three main panels:
- a browsing pane;
- a viewing pane;
- a log/peek pane.

The browsing pane will present the markers or boxes hierarchy, with position (byte number where marker/box starts and stops) and length information (i.e., inner length as signalled by marker/box and total length, with marker/box sign included), in the following form:

filename
|
|_ #000: Marker/Box short name (Hex code)
|  |
|  |_ *** Marker/Box long name ***
|  |_ startbyte > stopbyte, inner_length + marker/box sign length (total length)
|  |_ Additional info, depending on the marker/box type
|  |_ ...
|
|_ #001: Marker/Box short name (Hex code)
|  |
|  |_ ...
|
...


The viewing pane will display the decoded image contained in the JPEG 2000 file.
It should display correctly images as large as 4000x2000, provided that a couple of GB of RAM are available. Nothing is known about the display of larger sizes: let us know if you manage to get it working.


The log/peek pane is shared among two different subpanels:

- the log panel will report a lot of debugging info coming out from the wx GUI as well as from the openjpeg library
- the peek pane tries to give a peek on the codestream/file portion which is currently selected in the browsing pane. It shows both hex and ascii values corresponding to the marker/box section. 


4. Known bugs and limitations
===============================

4.1. Bugs
-----------

* 

4.2. Limitations
------------------

* For mj2 files, rendering is only in B/W
