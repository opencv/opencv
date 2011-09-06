import testlog_parser, sys, os, xml, re
from table_formatter import *
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), mks, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-c", "--columns", dest="columns", help="comma-separated list of columns to show", metavar="COLS", default="")
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    parser.add_option("", "--show-all", action="store_true", dest="showall", default=False, help="also include empty and \"notrun\" lines")
    (options, args) = parser.parse_args()
    
    if len(args) < 1:
        print >> sys.stderr, "Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml"
        exit(0)

    options.generateHtml = detectHtmlOutputType(options.format)
    args[0] = os.path.basename(args[0])
        
    tests = []
    files = []
    for arg in set(args):
        files.append(os.path.basename(arg))
        tests.extend(testlog_parser.parseLogFile(arg))

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
    for case in sorted(tests):
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
        htmlPrintHeader(sys.stdout, "Report %s tests from %s" % (len(tests), ", ".join(files)))
        tbl.htmlPrintTable(sys.stdout)
        htmlPrintFooter(sys.stdout)
    else:
        tbl.consolePrintTable(sys.stdout)