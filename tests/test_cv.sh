#! /bin/sh
if [ -n "${TESTDATA_DIR}" ] ; then
  ./opencv_test -d $TESTDATA_DIR/cv
else
  ./opencv_test -d $srcdir/../../opencv_extra/testdata/cv
fi
