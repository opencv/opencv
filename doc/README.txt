This file describes how to create the opencv.pdf manual in the
.../opencv/doc directory. It further describes how to produce
the html files on disk that populate the online OpenCV documenation
wiki.

-------------------------------------------------------------------------
In Ubuntu or Debian, required to build the opencv.pdf manual:

  sudo apt-get install     texlive texlive-latex-extra latex-xcolor texlive-fonts-extra
 
To build the HTML documentation, these are also required:

  sudo apt-get install python-setuptools             ## See [1] below for another install method
  sudo easy_install -U Sphinx                        ## This is NOT the speech recognition program. 
  sudo apt-get install     dvipng
  sudo easy_install plasTeX            

-------------------------------------------------------------------------
In other Linux distros you will also need to install LiveTeX and,
optionally, if you want to produce the hmtl files, the Sphinx tool (http://sphinx.pocoo.org/)

In MacOSX you can use MacTex (https://www.tug.org/mactex/).

In Windows you can use MiKTeX

--------------------------------------------------------------------------
(1) To build the latex files to create the opencv.pdf manual, in the 
.../opencv/doc directory, issue the command:

sh go

(2) If you want to build the html files that OpenCV uses to populate the 
online documentation, assuming you downloaded easy_install, 
Sphinx and plasTex as above, then from the .../opencv/doc/plastex directory, 
issue the "go" commend there:

sh go

The resulting html files will be be created in:  
.../opencv/doc/plastex/_build/html

--------------------------------------------------------------------------
[1] To install easy install on Ubuntu, try either (as stated above):
 sudo apt-get install python-setuptools
 or try using:
First:
  wget -q http://peak.telecommunity.com/dist/ez_setup.py
Then
  sudo python ez_setup.py
