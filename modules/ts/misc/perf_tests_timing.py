#!/usr/bin/env python

import testlog_parser, sys, os, xml, glob, re
from table_formatter import *
from optparse import OptionParser
from operator import itemgetter, attrgetter
from summary import getSetName, alphanum_keyselector

if __name__ == "__main__":
    usage = "%prog <log_name>.xml"
    parser = OptionParser(usage = usage)

    for arg in sys.argv:
        print arg

    parser.add_option("-o", "--output", dest="format",
        help="output results in text format (can be 'txt', 'html' or 'auto' - default)",
        metavar="FMT", default="auto")

    (options, args) = parser.parse_args()

    options.generateHtml = detectHtmlOutputType(options.format)

    if 1 != len(args):
        parser.print_help()
        exit(0)

    # expand wildcards and filter duplicates
    file = os.path.abspath(args[0])
    if not os.path.isfile(file):
        print 'Incorrect file name!'
        parser.print_help()
        exit(0)

    # read all passed files
    test_sets = []
    try:
        tests = testlog_parser.parseLogFile(file)
        if tests:
            test_sets.append((os.path.basename(file), tests))
    except IOError as err:
        sys.stderr.write("IOError reading \"" + file + "\" - " + str(err) + os.linesep)
    except xml.parsers.expat.ExpatError as err:
        sys.stderr.write("ExpatError reading \"" + file + "\" - " + str(err) + os.linesep)

    if not test_sets:
        sys.stderr.write("Error: no test data found" + os.linesep)
        quit()

    # find matches
    setsCount = len(test_sets)
    test_cases = {}

    name_extractor = lambda name: str(name)

    for i in range(setsCount):
        for case in test_sets[i][1]:
            name = name_extractor(case)
            if name not in test_cases:
                test_cases[name] = [None] * setsCount
            test_cases[name][i] = case

    testsuits = [] # testsuit name, time, flag for failed tests

    prevGroupName = None
    suit_time = 0
    has_failed = False
    for name in sorted(test_cases.iterkeys(), key=alphanum_keyselector):
        cases = test_cases[name]

        groupName = next(c for c in cases if c).shortName()
        if groupName != prevGroupName:
            if prevGroupName != None:
                testsuits.append((prevGroupName, suit_time, has_failed))
                has_failed = False
                suit_time = 0
            prevGroupName = groupName

        for i in range(setsCount):
            case = cases[i]
            if not case is None:
                if case.get("status") == "run":
                    suit_time += case.get("time")
                if case.get("status") == "failed":
                    has_failed = True

    tbl = table()

    # header
    tbl.newColumn("name", "Name", align = "left", cssclass = "col_name")
    tbl.newColumn("time", "Time (ms)", align = "left", cssclass = "col_name")
    tbl.newColumn("failed", "Failed tests", align = "center", cssclass = "col_name")

    # rows
    for suit in sorted(testsuits, key=lambda suit: suit[1], reverse=True):
        tbl.newRow()
        tbl.newCell("name", suit[0])
        tbl.newCell("time", formatValue(suit[1], "", ""), suit[1])
        if (suit[2]):
            tbl.newCell("failed", "Yes")

    # output table
    if options.generateHtml:
        htmlPrintHeader(sys.stdout, "Timings of %s tests from %s test logs" % (len(test_cases), setsCount))
        tbl.htmlPrintTable(sys.stdout)
        htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)