import sys, re, os.path
from xml.dom.minidom import parse

def parseLogFile(filename):
    tests = {}
    log = parse(open(filename, 'rb'))
    fstorage = log.firstChild
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        tests[case.nodeName] = case
    return tests

def processLogFile(outname, inname):
    tests = parseLogFile(inname)

    log = parse(open(outname, 'rb'))
    fstorage = log.firstChild
    for case in fstorage.childNodes:
        if case.nodeName == "#text":
            continue
        if case.nodeName in tests:
            del tests[case.nodeName]

    for case in tests.items():
        fstorage.appendChild(case[1])

    if tests:
        fstorage.appendChild(log.createTextNode('\n'))

    xmlstr = log.toxml()
    xmlstr = re.sub(r"(\s*\n)+", "\n", xmlstr)
    xmlstr = re.sub(r"(\s*\r\n)+", "\r\n", xmlstr)
    xmlstr = re.sub(r"<(\w*)/>", "<\\1></\\1>", xmlstr)
    xmlstr = xmlstr.replace("&quot;", "\"")
    f = open(outname, 'wb')
    f.write(xmlstr)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:\n", os.path.basename(sys.argv[0]), "<log_name>.xml <new_log_name>.xml"
        exit(0)

    processLogFile(sys.argv[1], sys.argv[2])

