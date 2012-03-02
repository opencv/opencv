import sys, re, os.path
from xml.dom.minidom import parse

class TestInfo(object):

    def __init__(self, xmlnode):
        self.fixture = xmlnode.getAttribute("classname")
        self.name = xmlnode.getAttribute("name")
        self.value_param = xmlnode.getAttribute("value_param")
        self.type_param = xmlnode.getAttribute("type_param")
        if xmlnode.getElementsByTagName("failure"):
            self.status = "failed"
        else:
            self.status = xmlnode.getAttribute("status")
        if self.name.startswith("DISABLED_"):
            self.status = "disabled"
            self.fixture = self.fixture.replace("DISABLED_", "")
            self.name = self.name.replace("DISABLED_", "")
        self.metrix = {}
        self.parseLongMetric(xmlnode, "bytesIn");
        self.parseLongMetric(xmlnode, "bytesOut");
        self.parseIntMetric(xmlnode, "samples");
        self.parseIntMetric(xmlnode, "outliers");
        self.parseFloatMetric(xmlnode, "frequency", 1);
        self.parseLongMetric(xmlnode, "min");
        self.parseLongMetric(xmlnode, "median");
        self.parseLongMetric(xmlnode, "gmean");
        self.parseLongMetric(xmlnode, "mean");
        self.parseLongMetric(xmlnode, "stddev");
        self.parseFloatMetric(xmlnode, "gstddev");
    
    def parseLongMetric(self, xmlnode, name, default = 0):
        if xmlnode.hasAttribute(name):
            tmp = xmlnode.getAttribute(name)
            val = long(tmp)
            self.metrix[name] = val
        else:
            self.metrix[name] = default

    def parseIntMetric(self, xmlnode, name, default = 0):
        if xmlnode.hasAttribute(name):
            tmp = xmlnode.getAttribute(name)
            val = int(tmp)
            self.metrix[name] = val
        else:
            self.metrix[name] = default

    def parseFloatMetric(self, xmlnode, name, default = 0):
        if xmlnode.hasAttribute(name):
            tmp = xmlnode.getAttribute(name)
            val = float(tmp)
            self.metrix[name] = val
        else:
            self.metrix[name] = default

    def parseStringMetric(self, xmlnode, name, default = None):
        if xmlnode.hasAttribute(name):
            tmp = xmlnode.getAttribute(name)
            self.metrix[name] = tmp.strip()
        else:
            self.metrix[name] = default

    def get(self, name, units="ms"):
        if name == "classname":
            return self.fixture
        if name == "name":
            return self.name
        if name == "fullname":
            return self.__str__()
        if name == "value_param":
            return self.value_param
        if name == "type_param":
            return self.type_param
        if name == "status":
            return self.status
        val = self.metrix.get(name, None)
        if not val:
            return val
        if name in ["gmean", "min", "mean", "median", "stddev"]:
            scale = 1.0
            frequency = self.metrix.get("frequency", 1.0) or 1.0
            if units == "ms":
                scale = 1000.0
            if units == "mks":
                scale = 1000000.0
            if units == "ns":
                scale = 1000000000.0
            if units == "ticks":
                frequency = long(1)
                scale = long(1)
            return val * scale / frequency
        return val


    def dump(self, units="ms"):
        print "%s ->\t\033[1;31m%s\033[0m = \t%.2f%s" % (str(self), self.status, self.get("gmean", units), units)
        
    def shortName(self):
        pos = self.name.find("/")
        if pos > 0:
            name = self.name[:pos]
        else:
            name = self.name
        if self.fixture.endswith(name):
            fixture = self.fixture[:-len(name)]
        else:
            fixture = self.fixture
        if fixture.endswith("_"):
            fixture = fixture[:-1]
        return '::'.join(filter(None, [name, fixture]))

    def __str__(self):
        pos = self.name.find("/")
        if pos > 0:
            name = self.name[:pos]
        else:
            name = self.name
        if self.fixture.endswith(name):
            fixture = self.fixture[:-len(name)]
        else:
            fixture = self.fixture
        if fixture.endswith("_"):
            fixture = fixture[:-1]
        return '::'.join(filter(None, [name, fixture, self.type_param, self.value_param]))

    def __cmp__(self, other):
        r = cmp(self.fixture, other.fixture);
        if r != 0:
            return r
        if self.type_param:
            if other.type_param:
                r = cmp(self.type_param, other.type_param);
                if r != 0:
                     return r
            else:
                return -1
        else:
            if other.type_param:
                return 1
        if self.value_param:
            if other.value_param:
                r = cmp(self.value_param, other.value_param);
                if r != 0:
                     return r
            else:
                return -1
        else:
            if other.value_param:
                return 1
        return 0

def parseLogFile(filename):
    tests = []
    log = parse(filename)
    for case in log.getElementsByTagName("testcase"):
        tests.append(TestInfo(case))
    return tests


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:\n", os.path.basename(sys.argv[0]), "<log_name>.xml"
        exit(0)

    for arg in sys.argv[1:]:
        print "Tests found in", arg
        tests = parseLogFile(arg)
        for t in sorted(tests):
            t.dump()
        print
