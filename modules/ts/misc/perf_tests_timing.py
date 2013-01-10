#!/usr/bin/env python

import testlog_parser, sys, os, xml, glob, re
from table_formatter import *
from optparse import OptionParser
from operator import itemgetter, attrgetter
from summary import getSetName, alphanum_keyselector
import re

if __name__ == "__main__":
    usage = "%prog <log_name>.xml [...]"
    parser = OptionParser(usage = usage)

    parser.add_option("-o", "--output", dest = "format",
        help = "output results in text format (can be 'txt', 'html' or 'auto' - default)",
        metavar = 'FMT', default = 'auto')

    (options, args) = parser.parse_args()
    if 1 != len(args):
        parser.print_help()
        exit(0)

    options.generateHtml = detectHtmlOutputType(options.format)

    input_file = args[0]
    file = os.path.abspath(input_file)
    if not os.path.isfile(file):
        sys.stderr.write("IOError reading \"" + file + "\" - " + str(err) + os.linesep)
        parser.print_help()
        exit(0)

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
        exit(0)

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
                testsuits.append({'name': prevGroupName, 'time': suit_time, \
                    'failed': has_failed})
                has_failed = False
                suit_time = 0
            prevGroupName = groupName

        for i in range(setsCount):
            case = cases[i]
            if not case is None:
                if case.get('status') == 'run':
                    suit_time += case.get('time')
                if case.get('status') == 'failed':
                    has_failed = True

    testsuits.append({'name': prevGroupName, 'time': suit_time, \
        'failed': has_failed})

    if len(testsuits)==0:
        print 'No testsuits found'
        exit(0)

    tbl = table()

    # header
    tbl.newColumn('name', 'Name of testsuit', align = 'left', cssclass = 'col_name')
    tbl.newColumn('time', 'Time (ms)', align = 'left', cssclass = 'col_name')
    tbl.newColumn('failed', 'Failed tests', align = 'center', cssclass = 'col_name')

    # rows
    for suit in sorted(testsuits, key = lambda suit: suit['time'], reverse = True):
        tbl.newRow()
        tbl.newCell('name', suit['name'])
        tbl.newCell('time', formatValue(suit['time'], '', ''), suit['time'])
        if (suit['failed']):
            tbl.newCell('failed', 'Yes')
        else:
            tbl.newCell('failed', ' ')

    # output table
    if options.generateHtml:
        tbl.htmlPrintTable(sys.stdout)
        htmlPrintFooter(sys.stdout)
    else:
        input_file = re.sub(r'^[\.\/]*', '', input_file)
        find_module_name = re.search(r'([^_]*)', input_file)
        module_name = find_module_name.group(0)

        splitter = 15 * '*'
        print '\n%s\n  %s\n%s\n' % (splitter, module_name, splitter)
        tbl.consolePrintTable(sys.stdout)
        print 4 * '\n'