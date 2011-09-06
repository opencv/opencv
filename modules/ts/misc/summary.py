import testlog_parser, sys, os, xml, glob
from table_formatter import *
from optparse import OptionParser

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml [<log_name2>.xml ...]"
        exit(0)
        
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-m", "--metric", dest="metric", help="output metric", metavar="NAME", default="gmean")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), mks, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    parser.add_option("", "--no-relatives", action="store_false", dest="calc_relatives", default=True, help="do not output relative values")
    parser.add_option("", "--show-all", action="store_true", dest="showall", default=False, help="also include empty and \"notrun\" lines")
    (options, args) = parser.parse_args()
    
    options.generateHtml = detectHtmlOutputType(options.format)
    if options.metric not in metrix_table:
        options.metric = "gmean"
    if options.metric.endswith("%"):
        options.calc_relatives = False
    
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
    
    # read all passed files
    test_sets = []
    for arg in files:
        try:
            tests = testlog_parser.parseLogFile(arg)
            if options.filter:
                expr = re.compile(options.filter)
                tests = [t for t in tests if expr.search(str(t))] 
            if tests:
                test_sets.append((os.path.basename(arg), tests))
        except IOError as err:
            sys.stderr.write("IOError reading \"" + arg + "\" - " + str(err) + os.linesep)
        except xml.parsers.expat.ExpatError as err:
            sys.stderr.write("ExpatError reading \"" + arg + "\" - " + str(err) + os.linesep)
            
    if not test_sets:
        sys.stderr.write("Error: no test data found" + os.linesep)
        quit()
            
    # find matches
    setsCount = len(test_sets)
    test_cases = {}
    
    for i in range(setsCount):
        for case in test_sets[i][1]:
            name = str(case)
            if name not in test_cases:
                test_cases[name] = [None] * setsCount
            test_cases[name][i] = case
            
    # build table
    getter = metrix_table[options.metric][1]
    if options.calc_relatives:
        getter_p = metrix_table[options.metric + "%"][1]
    tbl = table(metrix_table[options.metric][0])
    
    # header
    tbl.newColumn("name", "Name of Test", align = "left")
    i = 0
    for set in test_sets:
        tbl.newColumn(str(i), set[0].replace(".xml","").replace("_", "\n"), align = "center")
        i += 1
    if options.calc_relatives:
        i = 1
        for set in test_sets[1:]:
            tbl.newColumn(str(i) + "%", set[0].replace(".xml","").replace("_", "\n") + "\nvs\n" + test_sets[0][0].replace(".xml","").replace("_", "\n"), align = "center")
            i += 1
        
    # rows
    needNewRow = True
    for name in sorted(test_cases.iterkeys()):
        cases = test_cases[name]
        if needNewRow:
            tbl.newRow()
            if not options.showall:
                needNewRow = False
        tbl.newCell("name", name)
        for i in range(setsCount):
            case = cases[i]
            if case is None:
                tbl.newCell(str(i), "-")
                if options.calc_relatives and i > 0:
                    tbl.newCell(str(i) + "%", "-")
            else:
                status = case.get("status")
                if status != "run":
                    tbl.newCell(str(i), status, color = "red")
                    if status != "notrun":
                        needNewRow = True
                    if options.calc_relatives and i > 0:
                        tbl.newCell(str(i) + "%", "-", color = "red")
                else:
                    val = getter(case, cases[0], options.units)
                    if options.calc_relatives and i > 0 and val:
                        valp = getter_p(case, cases[0], options.units)
                    else:
                        valp = None
                    if not valp or i == 0:
                        color = None
                    elif valp > 1.05:
                        color = "red"
                    elif valp < 0.95:
                        color = "green"
                    else:
                        color = None
                    if val:
                        needNewRow = True
                        if options.metric.endswith("%"):
                            tbl.newCell(str(i), "%.2f" % val, val, color = color)
                        else:
                            tbl.newCell(str(i), "%.3f %s" % (val, options.units), val, color = color)
                    else:
                        tbl.newCell(str(i), "-")
                    if options.calc_relatives and i > 0:
                        if valp:
                            tbl.newCell(str(i) + "%", "%.2f" % valp, valp, color = color, bold = color)
                        else:
                            tbl.newCell(str(i) + "%", "-")
    if not needNewRow:
        tbl.trimLastRow()

    # output table
    if options.generateHtml:
        htmlPrintHeader(sys.stdout, "Summary report for %s tests from %s test logs" % (len(test_cases), setsCount))
        tbl.htmlPrintTable(sys.stdout)
        htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)
