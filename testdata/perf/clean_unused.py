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

def processLogFile(outname, inname, tests):
    log = parse(inname)
    fstorage = log.firstChild
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        if not case.nodeName in tests:
            fstorage.removeChild(case)

    xmlstr = log.toxml()
    xmlstr = re.sub(r"(\s*\n)+", "\n", xmlstr)
    xmlstr = re.sub(r"(\s*\r\n)+", "\r\n", xmlstr)
    f = open(outname, 'w')
    f.write(xmlstr)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:\n", os.path.basename(sys.argv[0]), "<log_name>.xml <log_name>.backup.xml"
        exit(0)

    tests = parseLogFile(sys.argv[1])
    processLogFile(sys.argv[1], sys.argv[2], tests)

