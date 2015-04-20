#!/bin/bash
#Create Reference images and hash

echo
echo "Type the name of the directory (inside OPJ_Binaries) "
echo "containing your reference executables, followed by [ENTER] (example: rev100):"
read refdir
cd temp
rm *.md5
cd ..
./OPJ_Validate linux_OPJ_Param_File_v0_1.txt $refdir
echo

