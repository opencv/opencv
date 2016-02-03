This directory contains a subset of the CharLS project (http://charls.codeplex.com/)

It was retrieved on Wed Sep 23 18:30:14 CEST 2009
URL:
http://charls.codeplex.com/Release/ProjectReleases.aspx?ReleaseId=32643
This is the 1.0 Beta Release

Project Description
An optimized implementation of the JPEG-LS standard for lossless and near-lossless image compression. JPEG-LS is a low-complexity standard that matches JPEG 2000 compression ratios. In terms of speed, CharLS outperforms open source and commercial JPEG LS implementations.

About JPEG-LS
JPEG-LS (ISO-14495-1/ITU-T.87) is a standard derived from the Hewlett Packard LOCO algorithm. JPEG LS has low complexity (meaning fast compression) and high compression ratios, similar to JPEG 2000. JPEG-LS is more similar to the old Lossless JPEG than to JPEG 2000, but interestingly the two different techniques result in vastly different performance characteristics.
Wikipedia on lossless JPEG and JPEG-LS: http://en.wikipedia.org/wiki/Lossless_JPEG

Legal
The code in this project is available through a BSD style license, allowing use of the code in commercial closed source applications if you wish. All the code in this project is written from scratch, and not based on other JPEG-LS implementations.


We only include enough of distribution to build the charls library.


Modifications
-------------

- remove tests/* subdirs
- remove *.vcproj/*.sln M$ Visual Studio specific files (use cmake in all cases)
- apply dos2unix to all files
- remove trailing comma (,) in enum {}
