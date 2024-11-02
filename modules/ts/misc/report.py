#!/usr/bin/env python
""" Print performance test run statistics.

Performance data is stored in the GTest log file created by performance tests. Default name is
`test_details.xml`. It can be changed with the `--gtest_output=xml:<location>/<filename>.xml` test
option. See https://github.com/opencv/opencv/wiki/HowToUsePerfTests for more details.

This script produces configurable performance report tables in text and HTML formats. It allows to
filter test cases by name and parameter string and select specific performance metrics columns. One
or multiple test results can be used for input.

### Example

./report.py  -c min,mean,median -f '(LUT|Match).*640' opencv_perf_core.xml  opencv_perf_features.xml

opencv_perf_features.xml, opencv_perf_core.xml

                       Name of Test                            Min        Mean      Median
KnnMatch::OCL_BruteForceMatcherFixture::(640x480, 32FC1)    1365.04 ms 1368.18 ms 1368.52 ms
LUT::OCL_LUTFixture::(640x480, 32FC1)                        2.57 ms    2.62 ms    2.64 ms
LUT::OCL_LUTFixture::(640x480, 32FC4)                        21.15 ms   21.25 ms   21.24 ms
LUT::OCL_LUTFixture::(640x480, 8UC1)                         2.22 ms    2.28 ms    2.29 ms
LUT::OCL_LUTFixture::(640x480, 8UC4)                         19.12 ms   19.24 ms   19.19 ms
LUT::SizePrm::640x480                                        2.22 ms    2.27 ms    2.29 ms
Match::OCL_BruteForceMatcherFixture::(640x480, 32FC1)       1364.15 ms 1367.73 ms 1365.45 ms
RadiusMatch::OCL_BruteForceMatcherFixture::(640x480, 32FC1) 1372.68 ms 1375.52 ms 1375.42 ms

### Options

-o FMT, --output=FMT        - output results in text format (can be 'txt', 'html' or 'auto' - default)
-u UNITS, --units=UNITS     - units for output values (s, ms (default), us, ns or ticks)
-c COLS, --columns=COLS     - comma-separated list of columns to show
-f REGEX, --filter=REGEX    - regex to filter tests
--show-all                  - also include empty and "notrun" lines
"""

import testlog_parser, sys, os, xml, re, glob
from table_formatter import *
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), us, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-c", "--columns", dest="columns", help="comma-separated list of columns to show", metavar="COLS", default="")
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    parser.add_option("", "--show-all", action="store_true", dest="showall", default=False, help="also include empty and \"notrun\" lines")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print("Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml", file=sys.stderr)
        exit(0)

    options.generateHtml = detectHtmlOutputType(options.format)

    # expand wildcards and filter duplicates
    files = []
    files1 = []
    for arg in args:
        if ("*" in arg) or ("?" in arg):
            files1.extend([os.path.abspath(f) for f in glob.glob(arg)])
        else:
            files.append(os.path.abspath(arg))
    seen = set()
    files = [ x for x in files if x not in seen and not seen.add(x)]
    files.extend((set(files1) - set(files)))
    args = files

    # load test data
    tests = []
    files = []
    for arg in set(args):
        try:
            cases = testlog_parser.parseLogFile(arg)
            if cases:
                files.append(os.path.basename(arg))
                tests.extend(cases)
        except:
            pass

    if options.filter:
        expr = re.compile(options.filter)
        tests = [t for t in tests if expr.search(str(t))]

    tbl = table(", ".join(files))
    if options.columns:
        metrics = [s.strip() for s in options.columns.split(",")]
        metrics = [m for m in metrics if m and not m.endswith("%") and m in metrix_table]
    else:
        metrics = None
    if not metrics:
        metrics = ["name", "samples", "outliers", "min", "median", "gmean", "mean", "stddev"]
    if "name" not in metrics:
        metrics.insert(0, "name")

    for m in metrics:
        if m == "name":
            tbl.newColumn(m, metrix_table[m][0])
        else:
            tbl.newColumn(m, metrix_table[m][0], align = "center")

    needNewRow = True
    for case in sorted(tests, key=lambda x: str(x)):
        if needNewRow:
            tbl.newRow()
            if not options.showall:
                needNewRow = False
        status = case.get("status")
        if status != "run":
            if status != "notrun":
                needNewRow = True
            for m in metrics:
                if m == "name":
                    tbl.newCell(m, str(case))
                else:
                    tbl.newCell(m, status, color = "red")
        else:
            needNewRow = True
            for m in metrics:
                val = metrix_table[m][1](case, None, options.units)
                if isinstance(val, float):
                    tbl.newCell(m, "%.2f %s" % (val, options.units), val)
                else:
                    tbl.newCell(m, val, val)
    if not needNewRow:
        tbl.trimLastRow()

    # output table
    if options.generateHtml:
        if options.format == "moinwiki":
            tbl.htmlPrintTable(sys.stdout, True)
        else:
            htmlPrintHeader(sys.stdout, "Report %s tests from %s" % (len(tests), ", ".join(files)))
            tbl.htmlPrintTable(sys.stdout)
            htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)
