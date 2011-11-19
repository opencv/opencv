import testlog_parser, sys, os, xml, re
from table_formatter import *
from optparse import OptionParser

cvsize_re = re.compile("^\d+x\d+$")
cvtype_re = re.compile("^(CV_)(8U|8S|16U|16S|32S|32F|64F)(C\d{1,3})?$")

def keyselector(a):
    if cvsize_re.match(a):
        size = [int(d) for d in a.split('x')]
        return size[0] * size[1]
    elif cvtype_re.match(a):
        if a.startswith("CV_"):
            a = a[3:]
        depth = 7
        if a[0] == '8':
            depth = (0, 1) [a[1] == 'S']
        elif a[0] == '1':
            depth = (2, 3) [a[2] == 'S']
        elif a[2] == 'S':
            depth = 4
        elif a[0] == '3':
            depth = 5
        elif a[0] == '6':
            depth = 6
        cidx = a.find('C')
        if cidx < 0:
            channels = 1
        else:
            channels = int(a[a.index('C') + 1:])
        #return (depth & 7) + ((channels - 1) << 3)
        return ((channels-1) & 511) + (depth << 9)
    return a

def getValueParams(test):
    param = test.get("value_param")
    if not param:
        return []
    if param.startswith("("):
        param = param[1:]
    if param.endswith(")"):
        param = param[:-1]
    args = []
    prev_pos = 0
    start = 0
    balance = 0
    while True:
        idx = param.find(",", prev_pos)
        if idx < 0:
            break
        idxlb = param.find("(", prev_pos, idx)
        while idxlb >= 0:
            balance += 1
            idxlb = param.find("(", idxlb+1, idx)
        idxrb = param.find(")", prev_pos, idx)
        while idxrb >= 0:
            balance -= 1
            idxrb = param.find(")", idxrb+1, idx)
        assert(balance >= 0)
        if balance == 0:
            args.append(param[start:idx].strip())
            start = idx + 1
        prev_pos = idx + 1
    args.append(param[start:].strip())
    return args
    #return [p.strip() for p in param.split(",")]

def nextPermutation(indexes, lists, x, y):
    idx = len(indexes)-1
    while idx >= 0:
        while idx == x or idx == y:
            idx -= 1
        if idx < 0:
            return False
        v = indexes[idx] + 1
        if v < len(lists[idx]):
            indexes[idx] = v;
            return True;
        else:
            indexes[idx] = 0;
            idx -= 1
    return False

def getTestWideName(sname, indexes, lists, x, y):
    name = sname + "::("
    for i in range(len(indexes)):
        if i > 0:
            name += ", "
        if i == x:
            name += "X"
        elif i == y:
            name += "Y"
        else:
            name += lists[i][indexes[i]]
    return str(name + ")")    
     
def getTest(stests, x, y, row, col):
    for pair in stests:
        if pair[1][x] == row and pair[1][y] == col:
            return pair[0]
    return None       

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), mks, ns or ticks)", metavar="UNITS", default="ms")
    parser.add_option("-m", "--metric", dest="metric", help="output metric", metavar="NAME", default="gmean")
    parser.add_option("-x", "", dest="x", help="argument number for rows", metavar="ROW", default=1)
    parser.add_option("-y", "", dest="y", help="argument number for columns", metavar="COL", default=0)
    parser.add_option("-f", "--filter", dest="filter", help="regex to filter tests", metavar="REGEX", default=None)
    (options, args) = parser.parse_args()
    
    if len(args) != 1:
        print >> sys.stderr, "Usage:\n", os.path.basename(sys.argv[0]), "<log_name1>.xml"
        exit(1)

    options.generateHtml = detectHtmlOutputType(options.format)
    if options.metric not in metrix_table:
        options.metric = "gmean"
    if options.metric.endswith("%"):
        options.metric = options.metric[:-1]
    getter = metrix_table[options.metric][1]
        
    tests = testlog_parser.parseLogFile(args[0])
    if options.filter:
        expr = re.compile(options.filter)
        tests = [(t,getValueParams(t)) for t in tests if expr.search(str(t))]
    else:
        tests = [(t,getValueParams(t)) for t in tests]
        
    args[0] = os.path.basename(args[0])
        
    if not tests:
        print >> sys.stderr, "Error - no tests matched"
        exit(1)
    
    argsnum = len(tests[0][1])
    sname = tests[0][0].shortName()
    
    arglists = []
    for i in range(argsnum):
        arglists.append({})
        
    names = set()
    names1 = set()
    for pair in tests:
        sn = pair[0].shortName()
        if len(pair[1]) > 1:
            names.add(sn)
        else:
            names1.add(sn)
        if sn == sname:
            if len(pair[1]) != argsnum:
                print >> sys.stderr, "Error - unable to create chart tables for functions having different argument numbers"
                sys.exit(1)
            for i in range(argsnum):
                arglists[i][pair[1][i]] = 1
    
    if names1 or len(names) != 1:
        print >> sys.stderr, "Error - unable to create tables for functions from different test suits:"
        i = 1
        for name in sorted(names):
            print >> sys.stderr, "%4s:   %s" % (i, name)
            i += 1
        if names1:
            print >> sys.stderr, "Other suits in this log (can not be choosen):"
            for name in sorted(names1):
                print >> sys.stderr, "%4s:   %s" % (i, name)
                i += 1
        sys.exit(1)
    
    if argsnum < 2:
        print >> sys.stderr, "Error - tests from %s have less than 2 parameters" % sname
        exit(1)
            
    for i in range(argsnum):
        arglists[i] = sorted([str(key) for key in arglists[i].iterkeys()], key=keyselector)
                
    if options.generateHtml and options.format != "moinwiki":
        htmlPrintHeader(sys.stdout, "Report %s for %s" % (args[0], sname))
            
    indexes = [0] * argsnum
    x = int(options.x)
    y = int(options.y)
    if x == y or x < 0 or y < 0 or x >= argsnum or y >= argsnum:
        x = 1
        y = 0
            
    while True:
        stests = []
        for pair in tests:
            t = pair[0]
            v = pair[1]
            for i in range(argsnum):
                if i != x and i != y:
                    if v[i] != arglists[i][indexes[i]]:
                        t = None
                        break
            if t:
                stests.append(pair)

        tbl = table(metrix_table[options.metric][0] + " for\n" + getTestWideName(sname, indexes, arglists, x, y))
        tbl.newColumn("x", "X\Y")
        for col in arglists[y]:
            tbl.newColumn(col, col, align="center")
        for row in arglists[x]:
            tbl.newRow()
            tbl.newCell("x", row)
            for col in arglists[y]:
                case = getTest(stests, x, y, row, col)
                if case:
                    status = case.get("status")
                    if status != "run":
                        tbl.newCell(col, status, color = "red")
                    else:
                        val = getter(case, None, options.units)
                        if isinstance(val, float):
                            tbl.newCell(col, "%.2f %s" % (val, options.units), val)
                        else:
                            tbl.newCell(col, val, val)
                else:
                    tbl.newCell(col, "-")
        
        if options.generateHtml:
            tbl.htmlPrintTable(sys.stdout, options.format == "moinwiki")
        else:
            tbl.consolePrintTable(sys.stdout)
        if not nextPermutation(indexes, arglists, x, y):
            break
    
    if options.generateHtml and options.format != "moinwiki":
        htmlPrintFooter(sys.stdout)