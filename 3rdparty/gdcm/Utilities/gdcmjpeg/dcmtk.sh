# Script to patch dcmtk in order to compare gdcm version against it
sed -i -e 's/IJG_INT32/INT32/g' *.c *.h
sed -i -e's/jinclude[8|12|16]/jinclude/g' *.c
sed -i -e's/jpeglib[8|12|16]/jpeglib/g' *.c
sed -i -e's/jlossy[8|12|16]/jlossy/g' *.c
sed -i -e's/jmemsys[8|12|16]/jmemsys/g' *.c
sed -i -e's/jdct[8|12|16]/jdct/g' *.c
sed -i -e's/jversion[8|12|16]/jversion/g' *.c
sed -i -e's/jerror[8|12|16]/jerror/g' *.c
sed -i -e's/jdhuff[8|12|16]/jdhuff/g' *.c
sed -i -e's/jlossls[8|12|16]/jlossls/g' *.c
sed -i -e's/jchuff[8|12|16]/jchuff/g' *.c
sed -i -e's/jconfig[8|12|16]/jconfig/g' *.h
sed -i -e's/jmorecfg[8|12|16]/jmorecfg/g' *.h
sed -i -e's/jpegint[8|12|16]/jpegint/g' *.h
sed -i -e's/jerror[8|12|16]/jerror/g' *.h
sed -i -e's/mymain/main/g' *.c
rename 's/8\.h$/\.h/' *.h
rename 's/12\.h$/\.h/' *.h
rename 's/16\.h$/\.h/' *.h
