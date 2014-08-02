#!/bin/sh

BASE_DIR=`dirname $0`
OPENCV_TEST_PATH=$BASE_DIR/@TEST_PATH@
OPENCV_TEST_DATA_PATH=$BASE_DIR/sdk/etc/testdata/

if [ $# -ne 1 ]; then
  echo "Device architecture is not preset in command line"
  echo "Tests are available for architectures: `ls -m ${OPENCV_TEST_PATH}`"
  echo "Usage: $0 <target_device_arch>"
  return 1
else
  TARGET_ARCH=$1
fi

if [ -z `which adb` ]; then
  echo "adb command was not found in PATH"
  return 1
fi

adb push $OPENCV_TEST_DATA_PATH /sdcard/opencv_testdata

adb shell "mkdir -p /data/local/tmp/opencv_test"
SUMMARY_STATUS=0
for t in "$OPENCV_TEST_PATH/$TARGET_ARCH/"opencv_test_* "$OPENCV_TEST_PATH/$TARGET_ARCH/"opencv_perf_*;
do
  test_name=`basename "$t"`
  report="$test_name-`date --rfc-3339=date`.xml"
  adb push $t /data/local/tmp/opencv_test/
  adb shell "export OPENCV_TEST_DATA_PATH=/sdcard/opencv_testdata && /data/local/tmp/opencv_test/$test_name --perf_min_samples=1 --perf_force_samples=1 --gtest_output=xml:/data/local/tmp/opencv_test/$report"
  adb pull "/data/local/tmp/opencv_test/$report" $report
  TEST_STATUS=0
  if [ -e $report ]; then
    if [ `grep -c "<fail" $report` -ne 0 ]; then
      TEST_STATUS=2
    fi
  else
    TEST_STATUS=3
  fi
  if [ $TEST_STATUS -ne 0 ]; then
    SUMMARY_STATUS=$TEST_STATUS
  fi
done

if [ $SUMMARY_STATUS -eq 0 ]; then
  echo "All OpenCV tests finished successfully"
else
  echo "OpenCV tests finished with status $SUMMARY_STATUS"
fi

return $SUMMARY_STATUS