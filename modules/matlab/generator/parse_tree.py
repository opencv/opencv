from string import join
from textwrap import fill
from filters import *

class ParseTree(object):
    def __init__(self, namespaces=None):
        self.namespaces = namespaces if namespaces else []

    def __str__(self):
        return join((ns.__str__() for ns in self.namespaces), '\n\n\n')

    def build(self, namespaces):
        babel = Translator()
        for name, definitions in namespaces.items():
            class_tree = {}
            methods = []
            constants = []
            for defn in definitions:
                obj = babel.translate(defn) 
                if obj is None:
                    continue
                if type(obj) is Class or obj.clss:
                    self.insertIntoClassTree(obj, class_tree)
                elif type(obj) is Method:
                    methods.append(obj)
                elif type(obj) is Constant:
                    constants.append(obj)
                else:
                    raise TypeError('Unexpected object type: '+str(type(obj)))
            self.namespaces.append(Namespace(name, constants, class_tree.values(), methods))

    def insertIntoClassTree(self, obj, class_tree):
        cname = obj.name if type(obj) is Class else obj.clss
        if not cname:
            return
        if not cname in class_tree:
          # add a new class to the tree
            class_tree[cname] = Class(cname)
        # insert the definition into the class
        val = class_tree[cname]
        if type(obj) is Method:
            val.methods.append(obj)
        elif type(obj) is Constant:
            val.constants.append(obj)
        else:
            raise TypeError('Unexpected object type: '+str(type(obj)))



class Translator(object):
    def translate(self, defn):
        # --- class ---
        # classes have 'class' prefixed on their name 
        if 'class' in defn[0].split(' ') or 'struct' in defn[0].split(' '):
            return self.translateClass(defn)
        # --- operators! ---
        #TODO: implement operators: http://www.mathworks.com.au/help/matlab/matlab_oop/implementing-operators-for-your-class.html
        if 'operator' in defn[0]:
            return
        # --- constant ---
        elif convertibleToInt(defn[1]):
            return self.translateConstant(defn)
        # --- function ---
        # functions either need to have input arguments, or not uppercase names
        elif defn[3] or not self.translateName(defn[0]).split('_')[0].isupper():
            return self.translateMethod(defn)
        # --- constant ---
        else:
            return self.translateConstant(defn)

    def translateClass(self, defn):
        return Class()

    def translateMethod(self, defn, class_tree=None):
        name = self.translateName(defn[0])
        clss = self.translateClassName(defn[0])
        rtp  = defn[1]
        static = True if 'S' in ''.join(defn[2]) else False 
        args = defn[3]
        req  = [] 
        opt = []
        for arg in args:
            if arg:
                a = self.translateArgument(arg)
                opt.append(a) if a.default else req.append(a)
        return Method(name, clss, static, '', rtp, False, req, opt)
            
    def translateConstant(self, defn):
        const = True if 'const' in defn[0] else False
        name  = self.translateName(defn[0])
        clss  = self.translateClassName(defn[0])
        tp    = 'int'
        val   = defn[1]
        return Constant(name, clss, tp, const, '', val)

    def translateArgument(self, defn):
        ref   = '*' if '*' in defn[0] else ''
        ref   = '&' if '&' in defn[0] else ref
        const = ' const ' in ' '+defn[0]+' '
        tp    = " ".join([word for word in defn[0].replace(ref, '').split() if not ' const ' in ' '+word+' '])
        name = defn[1]
        default = defn[2] if defn[2] else ''
        modifiers = ''.join(defn[3])
        I = True if not modifiers or 'I' in modifiers else False
        O = True if 'O' in modifiers else False
        return Argument(name, tp, const, I, O, ref, default)

    def translateName(self, name):
        return name.split(' ')[-1].split('.')[-1]

    def translateClassName(self, name):
        name  = name.split(' ')[-1]
        parts = name.split('.')
        return parts[-2] if len(parts) > 1 and not parts[-2] == 'cv' else ''



class Namespace(object):
    def __init__(self, name='', constants=None, classes=None, methods=None):
        self.name = name
        self.constants = constants if constants else []
        self.classes   = classes   if classes   else []
        self.methods = methods if methods else []

    def __str__(self):
        return 'namespace '+self.name+' {\n\n'+\
          (join((c.__str__() for c in self.constants), '\n')+'\n\n' if self.constants else '')+\
          (join((f.__str__() for f in self.methods), '\n')+'\n\n' if self.methods else '')+\
          (join((o.__str__() for o in self.classes), '\n\n')        if self.classes   else '')+'\n};'

class Class(object):
    def __init__(self, name='', namespace='', constants=None, methods=None):
        self.name = name
        self.namespace = namespace
        self.constants = constants if constants else []
        self.methods = methods if methods else []

    def __str__(self):
        return 'class '+self.name+' {\n\t'+\
          (join((c.__str__() for c in self.constants), '\n\t')+'\n\n\t' if self.constants else '')+\
          (join((f.__str__() for f in self.methods), '\n\t')          if self.methods else '')+'\n};'

class Method(object):
    def __init__(self, name='', clss='', static=False, namespace='', rtp='', const=False, req=None, opt=None):
        self.name  = name
        self.clss  = clss
        self.constructor = True if name == clss else False
        self.static = static
        self.const = const
        self.namespace = namespace
        self.rtp = rtp 
        self.req = req if req else []
        self.opt = opt if opt else []

    def __str__(self):
        return (self.rtp+' ' if self.rtp else '')+self.name+'('+\
          join((arg.__str__() for arg in self.req+self.opt), ', ')+\
          ')'+(' const' if self.const else '')+';'

class Argument(object):
    def __init__(self, name='', tp='', const=False, I=True, O=False, ref='', default=''):
        self.name = name
        self.tp   = tp
        self.ref  = ref
        self.I    = I
        self.O    = O
        self.const = const
        self.default = default

    def __str__(self):
        return ('const ' if self.const else '')+self.tp+self.ref+\
                ' '+self.name+('='+self.default if self.default else '')

class Constant(object):
    def __init__(self, name='', clss='', tp='', const=False, ref='', default=''):
        self.name = name
        self.clss = clss
        self.tp   = tp
        self.ref  = ref
        self.const = const
        self.default = default

    def __str__(self):
        return ('const ' if self.const else '')+self.tp+self.ref+\
                ' '+self.name+('='+self.default if self.default else '')+';'

def constants(tree):
    if isinstance(tree, dict) and 'constants' in tree and isinstance(tree['constants'], list):
        for node in tree['constants']:
            yield (node['name'], node['default'])
    if isinstance(tree, dict):
        for key, val in tree.items():
            for gen in constants(val):
                yield gen
    if isinstance(tree, list):
        for val in tree:
            for gen in constants(val):
                yield gen

def todict(obj, classkey=None):
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = todict(obj[k], classkey)
        return obj
    elif hasattr(obj, "__iter__"):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
            for key, value in obj.__dict__.iteritems()
            if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj
