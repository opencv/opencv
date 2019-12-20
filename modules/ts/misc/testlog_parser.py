#!/usr/bin/env python

from __future__ import print_function
import collections
import re
import os.path
import sys
from xml.dom.minidom import parse

if sys.version_info > (3,):
    long = int
    def cmp(a, b): return (a>b)-(a<b)

class TestInfo(object):

    def __init__(self, xmlnode):
        self.fixture = xmlnode.getAttribute("classname")
        self.name = xmlnode.getAttribute("name")
        self.value_param = xmlnode.getAttribute("value_param")
        self.type_param = xmlnode.getAttribute("type_param")

        custom_status = xmlnode.getAttribute("custom_status")
        failures = xmlnode.getElementsByTagName("failure")

        if len(custom_status) > 0:
            self.status = custom_status
        elif len(failures) > 0:
            self.status = "failed"
        else:
            self.status = xmlnode.getAttribute("status")

        if self.name.startswith("DISABLED_"):
            if self.status == 'notrun':
                self.status = "disabled"
            self.fixture = self.fixture.replace("DISABLED_", "")
            self.name = self.name.replace("DISABLED_", "")
        self.properties = {
            prop.getAttribute("name") : prop.getAttribute("value")
            for prop in xmlnode.getElementsByTagName("property")
            if prop.hasAttribute("name") and prop.hasAttribute("value")
        }
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
        self.parseFloatMetric(xmlnode, "time");
        self.parseLongMetric(xmlnode, "total_memory_usage");

    def parseLongMetric(self, xmlnode, name, default = 0):
        if name in self.properties:
            self.metrix[name] = long(self.properties[name])
        elif xmlnode.hasAttribute(name):
            self.metrix[name] = long(xmlnode.getAttribute(name))
        else:
            self.metrix[name] = default

    def parseIntMetric(self, xmlnode, name, default = 0):
        if name in self.properties:
            self.metrix[name] = int(self.properties[name])
        elif xmlnode.hasAttribute(name):
            self.metrix[name] = int(xmlnode.getAttribute(name))
        else:
            self.metrix[name] = default

    def parseFloatMetric(self, xmlnode, name, default = 0):
        if name in self.properties:
            self.metrix[name] = float(self.properties[name])
        elif xmlnode.hasAttribute(name):
            self.metrix[name] = float(xmlnode.getAttribute(name))
        else:
            self.metrix[name] = default

    def parseStringMetric(self, xmlnode, name, default = None):
        if name in self.properties:
            self.metrix[name] = self.properties[name].strip()
        elif xmlnode.hasAttribute(name):
            self.metrix[name] = xmlnode.getAttribute(name).strip()
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
        if name == "time":
            return self.metrix.get("time")
        if name in ["gmean", "min", "mean", "median", "stddev"]:
            scale = 1.0
            frequency = self.metrix.get("frequency", 1.0) or 1.0
            if units == "ms":
                scale = 1000.0
            if units == "us" or units == "mks":  # mks is typo error for microsecond (<= OpenCV 3.4)
                scale = 1000000.0
            if units == "ns":
                scale = 1000000000.0
            if units == "ticks":
                frequency = long(1)
                scale = long(1)
            return val * scale / frequency
        return val


    def dump(self, units="ms"):
        print("%s ->\t\033[1;31m%s\033[0m = \t%.2f%s" % (str(self), self.status, self.get("gmean", units), units))


    def getName(self):
        pos = self.name.find("/")
        if pos > 0:
            return self.name[:pos]
        return self.name


    def getFixture(self):
        if self.fixture.endswith(self.getName()):
            fixture = self.fixture[:-len(self.getName())]
        else:
            fixture = self.fixture
        if fixture.endswith("_"):
            fixture = fixture[:-1]
        return fixture


    def param(self):
        return '::'.join(filter(None, [self.type_param, self.value_param]))

    def shortName(self):
        name = self.getName()
        fixture = self.getFixture()
        return '::'.join(filter(None, [name, fixture]))


    def __str__(self):
        name = self.getName()
        fixture = self.getFixture()
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

# This is a Sequence for compatibility with old scripts,
# which treat parseLogFile's return value as a list.
class TestRunInfo(collections.Sequence):
    def __init__(self, properties, tests):
        self.properties = properties
        self.tests = tests

    def __len__(self):
        return len(self.tests)

    def __getitem__(self, key):
        return self.tests[key]

def parseLogFile(filename):
    log = parse(filename)

    properties = {
        attr_name[3:]: attr_value
        for (attr_name, attr_value) in log.documentElement.attributes.items()
        if attr_name.startswith('cv_')
    }

    tests = list(map(TestInfo, log.getElementsByTagName("testcase")))

    return TestRunInfo(properties, tests)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n", os.path.basename(sys.argv[0]), "<log_name>.xml")
        exit(0)

    for arg in sys.argv[1:]:
        print("Processing {}...".format(arg))

        run = parseLogFile(arg)

        print("Properties:")

        for (prop_name, prop_value) in run.properties.items():
          print("\t{} = {}".format(prop_name, prop_value))

        print("Tests:")

        for t in sorted(run.tests):
            t.dump()

        print()
