class Anchor(object):
    anchor = ""
    type = ""
    cppname = ""

    def __init__(self, anchor, type, cppname):
        self.anchor = anchor
        self.type = type
        self.cppname = cppname

def add_to_file(files_dict, file, anchor):
    if file in files_dict:
        # if that file already exists as a key in the dictionary
        files_dict[file].append(anchor)
    else:
        files_dict[file] = [anchor]
    return files_dict


def scan_namespace_constants(ns, ns_name, files_dict):
    constants = ns.findall("./member[@kind='enumvalue']")
    for c in constants:
        c_name = c.find("./name").text
        name = ns_name + '::' + c_name
        file = c.find("./anchorfile").text
        anchor = c.find("./anchor").text
        #print('    CONST: {} => {}#{}'.format(name, file, anchor))
        files_dict = add_to_file(files_dict, file, Anchor(anchor, "const", name))
    return files_dict

def scan_namespace_functions(ns, ns_name, files_dict):
    functions = ns.findall("./member[@kind='function']")
    for f in functions:
        f_name = f.find("./name").text
        name = ns_name + '::' + f_name
        file = f.find("./anchorfile").text
        anchor = f.find("./anchor").text
        #print('    FN: {} => {}#{}'.format(name, file, anchor))
        files_dict = add_to_file(files_dict, file, Anchor(anchor, "fn", name))
    return files_dict

def scan_class_methods(c, c_name, files_dict):
    methods = c.findall("./member[@kind='function']")
    for m in methods:
        m_name = m.find("./name").text
        name = c_name + '::' + m_name
        file = m.find("./anchorfile").text
        anchor = m.find("./anchor").text
        #print('    Method: {} => {}#{}'.format(name, file, anchor))
        files_dict = add_to_file(files_dict, file, Anchor(anchor, "method", name))
    return files_dict
