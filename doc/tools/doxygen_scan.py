import traceback

class Symbol(object):
    def __init__(self, anchor, type, cppname):
        self.anchor = anchor
        self.type = type
        self.cppname = cppname
        #if anchor == 'ga586ebfb0a7fb604b35a23d85391329be':
        #    print(repr(self))
        #    traceback.print_stack()

    def __repr__(self):
        return '%s:%s@%s' % (self.type, self.cppname, self.anchor)

def add_to_file(files_dict, file, anchor):
    anchors = files_dict.setdefault(file, [])
    anchors.append(anchor)


def scan_namespace_constants(ns, ns_name, files_dict):
    constants = ns.findall("./member[@kind='enumvalue']")
    for c in constants:
        c_name = c.find("./name").text
        name = ns_name + '::' + c_name
        file = c.find("./anchorfile").text
        anchor = c.find("./anchor").text
        #print('    CONST: {} => {}#{}'.format(name, file, anchor))
        add_to_file(files_dict, file, Symbol(anchor, "const", name))

def scan_namespace_functions(ns, ns_name, files_dict):
    functions = ns.findall("./member[@kind='function']")
    for f in functions:
        f_name = f.find("./name").text
        name = ns_name + '::' + f_name
        file = f.find("./anchorfile").text
        anchor = f.find("./anchor").text
        #print('    FN: {} => {}#{}'.format(name, file, anchor))
        add_to_file(files_dict, file, Symbol(anchor, "fn", name))

def scan_class_methods(c, c_name, files_dict):
    methods = c.findall("./member[@kind='function']")
    for m in methods:
        m_name = m.find("./name").text
        name = c_name + '::' + m_name
        file = m.find("./anchorfile").text
        anchor = m.find("./anchor").text
        #print('    Method: {} => {}#{}'.format(name, file, anchor))
        add_to_file(files_dict, file, Symbol(anchor, "method", name))
