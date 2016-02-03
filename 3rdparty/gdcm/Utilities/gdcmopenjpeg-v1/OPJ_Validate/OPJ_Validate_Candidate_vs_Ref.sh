#!/bin/bash
cd temp
erase *.ppm
erase *.j2k
erase *.bmp
erase *.tif
erase *.jp2
cd ..

echo
echo "Type the name of the directory (inside OPJ_Binaries) "
echo "containing your executables to compared with reference, followed by [ENTER] (example: rev101):"
read compdir

./OPJ_Validate linux_OPJ_Param_File_v0_1.txt $compdir
echo
