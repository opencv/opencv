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

def processLogFile(filename, tests):
    log = parse(filename)
    fstorage = log.firstChild
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        if not case.nodeName in tests:
            fstorage.removeChild(case)
    
    xmlstr = log.toxml()
    xmlstr = re.sub(r"\n+", "\n", xmlstr)
    xmlstr = re.sub(r"(\r\n)+", "\r\n", xmlstr)
    f = open(filename, 'w')
    f.write(xmlstr)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:\n", os.path.basename(sys.argv[0]), "<old_log_name>.xml <new_log_name>.xml"
        exit(0)

    tests = parseLogFile(sys.argv[2])
    processLogFile(sys.argv[1], tests)

