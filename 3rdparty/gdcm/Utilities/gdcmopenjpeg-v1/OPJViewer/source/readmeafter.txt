This viewer is conceived to open and display information and image content of J2K, JP2,
and MJ2 files.
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

