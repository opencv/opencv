import sys, re, os.path
from xml.dom.minidom import parse

def parseLogFile(filename):
    tests = []
    log = parse(filename)
    fstorage = log.firstChild
    #print help(log)
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        #print case.nodeName
        tests.append(case.nodeName)
    return tests

def process(filename, expr, save_results):
    log = parse(filename)
    fstorage = log.firstChild
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        if expr.match(case.nodeName):
            print case.nodeName
            fstorage.removeChild(case)

    if save_results:
        xmlstr = log.toxml()
        xmlstr = re.sub(r"(\s*\n)+", "\n", xmlstr)
        xmlstr = re.sub(r"(\s*\r\n)+", "\r\n", xmlstr)
        xmlstr = re.sub(r"<(\w*)/>", "<\\1></\\1>", xmlstr)
        xmlstr = xmlstr.replace("&quot;", "\"")
        f = open(filename, 'w')
        f.write(xmlstr)
        f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "This script is used to remove entries from sanity xml"
        print "  Usage:\n", os.path.basename(sys.argv[0]), "<name>.xml <regex>"
        exit(0)

    process(sys.argv[1], re.compile(sys.argv[2]), len(sys.argv) == 4)

