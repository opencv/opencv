import testlog_parser, sys, os, xml, glob, re
from table_formatter import *
from optparse import OptionParser

numeric_re = re.compile("(\d+)")
cvtype_re = re.compile("(8U|8S|16U|16S|32S|32F|64F)C(\d{1,3})")
cvtypes = { '8U': 0, '8S': 1, '16U': 2, '16S': 3, '32S': 4, '32F': 5, '64F': 6 }

convert = lambda text: int(text) if text.isdigit() else text
keyselector = lambda a: cvtype_re.sub(lambda match: " " + str(cvtypes.get(match.group(1), 7) + (int(match.group(2))-1) * 8) + " ", a)
alphanum_keyselector = lambda key: [ convert(c) for c in numeric_re.split(keyselector(key)) ]

def getSetName(tset, idx, columns, short = True):
    if columns and len(columns) > idx:
        prefix = columns[idx]
    else:
        prefix = None
    if short and prefix:
        return prefix
    name = tset[0].replace(".xml","").replace("_", "\n")
    if prefix:
        return prefix + "\n" + ("-"*int(len(max(prefix.split("\n"), key=len))*1.5)) + "\n" + name
    return name

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml [<log_name2>.xml ...]"
        exit(0)
        
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-m", "--metric", dest="metric", help="output metric", metavar="NAME", default="gmean")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), mks, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    parser.add_option("", "--module", dest="module", default=None, metavar="NAME", help="module prefix for test names")
    parser.add_option("", "--columns", dest="columns", default=None, metavar="NAMES", help="comma-separated list of column aliases")
    parser.add_option("", "--no-relatives", action="store_false", dest="calc_relatives", default=True, help="do not output relative values")
    parser.add_option("", "--with-cycles-reduction", action="store_true", dest="calc_cr", default=False, help="alos output cycle reduction percentages")
    parser.add_option("", "--show-all", action="store_true", dest="showall", default=False, help="also include empty and \"notrun\" lines")
    parser.add_option("", "--match", dest="match", default=None)
    parser.add_option("", "--match-replace", dest="match_replace", default="")
    (options, args) = parser.parse_args()
    
    options.generateHtml = detectHtmlOutputType(options.format)
    if options.metric not in metrix_table:
        options.metric = "gmean"
    if options.metric.endswith("%") or options.metric.endswith("$"):
        options.calc_relatives = False
        options.calc_cr = False
    if options.columns:
        options.columns = [s.strip().replace("\\n", "\n") for s in options.columns.split(",")]
    
    # expand wildcards and filter duplicates
    files = []
    seen = set()
    for arg in args:
        if ("*" in arg) or ("?" in arg):
            flist = [os.path.abspath(f) for f in glob.glob(arg)]
            flist = sorted(flist, key= lambda text: str(text).replace("M", "_"))
            files.extend([ x for x in flist if x not in seen and not seen.add(x)])
        else:
            fname = os.path.abspath(arg)
            if fname not in seen and not seen.add(fname):
                files.append(fname)
    
    # read all passed files
    test_sets = []
    for arg in files:
        try:
            tests = testlog_parser.parseLogFile(arg)
            if options.filter:
                expr = re.compile(options.filter)
                tests = [t for t in tests if expr.search(str(t))]
            if options.match:
                tests = [t for t in tests if t.get("status") != "notrun"]
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
    
    name_extractor = lambda name: str(name)
    if options.match:
        reg = re.compile(options.match)
        name_extractor = lambda name: reg.sub(options.match_replace, str(name))
    
    for i in range(setsCount):
        for case in test_sets[i][1]:
            name = name_extractor(case)
            if options.module:
                name = options.module + "::" + name
            if name not in test_cases:
                test_cases[name] = [None] * setsCount
            test_cases[name][i] = case
            
    # build table
    getter = metrix_table[options.metric][1]
    if options.calc_relatives:
        getter_p = metrix_table[options.metric + "%"][1]
    if options.calc_cr:
        getter_cr = metrix_table[options.metric + "$"][1]
    tbl = table(metrix_table[options.metric][0])
    
    # header
    tbl.newColumn("name", "Name of Test", align = "left", cssclass = "col_name")
    i = 0
    for set in test_sets:
        tbl.newColumn(str(i), getSetName(set, i, options.columns, False), align = "center")
        i += 1
    metric_sets = test_sets[1:]
    if options.calc_cr:
        i = 1
        for set in metric_sets:
            tbl.newColumn(str(i) + "$", getSetName(set, i, options.columns) + "\nvs\n" + getSetName(test_sets[0], 0, options.columns) + "\n(cycles reduction)", align = "center", cssclass = "col_cr")
            i += 1
    if options.calc_relatives:
        i = 1
        for set in metric_sets:
            tbl.newColumn(str(i) + "%", getSetName(set, i, options.columns) + "\nvs\n" + getSetName(test_sets[0], 0, options.columns) + "\n(x-factor)", align = "center", cssclass = "col_rel")
            i += 1
        
    # rows
    prevGroupName = None
    needNewRow = True
    lastRow = None
    for name in sorted(test_cases.iterkeys(), key=alphanum_keyselector):
        cases = test_cases[name]
        if needNewRow:
            lastRow = tbl.newRow()
            if not options.showall:
                needNewRow = False
        tbl.newCell("name", name)
        
        groupName = next(c for c in cases if c).shortName()
        if groupName != prevGroupName:
            prop = lastRow.props.get("cssclass", "")
            if "firstingroup" not in prop:
                lastRow.props["cssclass"] = prop + " firstingroup"
            prevGroupName = groupName

        for i in range(setsCount):
            case = cases[i]
            if case is None:
                tbl.newCell(str(i), "-")
                if options.calc_relatives and i > 0:
                    tbl.newCell(str(i) + "%", "-")
                if options.calc_cr and i > 0:
                    tbl.newCell(str(i) + "$", "-")
            else:
                status = case.get("status")
                if status != "run":
                    tbl.newCell(str(i), status, color = "red")
                    if status != "notrun":
                        needNewRow = True
                    if options.calc_relatives and i > 0:
                        tbl.newCell(str(i) + "%", "-", color = "red")
                    if options.calc_cr and i > 0:
                        tbl.newCell(str(i) + "$", "-", color = "red")
                else:
                    val = getter(case, cases[0], options.units)
                    if options.calc_relatives and i > 0 and val:
                        valp = getter_p(case, cases[0], options.units)
                    else:
                        valp = None
                    if options.calc_cr and i > 0 and val:
                        valcr = getter_cr(case, cases[0], options.units)
                    else:
                        valcr = None
                    if not valp or i == 0:
                        color = None
                    elif valp > 1.05:
                        color = "green"
                    elif valp < 0.95:
                        color = "red"
                    else:
                        color = None
                    if val:
                        needNewRow = True
                    tbl.newCell(str(i), formatValue(val, options.metric, options.units), val, color = color)
                    if options.calc_relatives and i > 0:
                        tbl.newCell(str(i) + "%", formatValue(valp, "%"), valp, color = color, bold = color)
                    if options.calc_cr and i > 0:
                        tbl.newCell(str(i) + "$", formatValue(valcr, "$"), valcr, color = color, bold = color)
    if not needNewRow:
        tbl.trimLastRow()

    # output table
    if options.generateHtml:
        if options.format == "moinwiki":
            tbl.htmlPrintTable(sys.stdout, True)
        else:
            htmlPrintHeader(sys.stdout, "Summary report for %s tests from %s test logs" % (len(test_cases), setsCount))
            tbl.htmlPrintTable(sys.stdout)
            htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)
