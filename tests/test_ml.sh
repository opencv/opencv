#! /bin/sh
if [ -n "${TESTDATA_DIR}" ] ; then
  ./opencv_test_ml -d $TESTDATA_DIR/ml
else
  ./opencv_test_ml -d $srcdir/../../opencv_extra/testdata/ml
fi
