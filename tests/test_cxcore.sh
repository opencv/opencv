#! /bin/sh
if [ -n "${TESTDATA_DIR}" ] ; then
  ./opencv_test_core -d $TESTDATA_DIR/cxcore
else
  ./opencv_test_core -d $srcdir/../../opencv_extra/testdata/cxcore
fi
