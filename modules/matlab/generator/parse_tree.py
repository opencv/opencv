from string import join
from textwrap import fill

class ParseTree(object):
    def __init__(self, namespaces=[]):
        self.namespaces = namespaces

    def __str__(self):
        return join((ns.__str__() for ns in self.namespaces), '\n\n\n')

    def build(self, namespaces):
        babel = Translator()
        for name, definitions in namespaces.items():
            class_tree = {}
            functions = []
            constants = []
            for defn in definitions:
                obj = babel.translate(defn) 
                if type(obj) is Class or obj.clss:
                    self.insertIntoClassTree(obj, class_tree)
                elif type(obj) is Function:
                    functions.append(obj)
                elif type(obj) is Constant:
                    constants.append(obj)
                else:
                    raise TypeError('Unexpected object type: '+str(type(obj)))
            self.namespaces.append(Namespace(name, class_tree.values(), functions, constants))

    def insertIntoClassTree(self, obj, class_tree):
        cname = obj.name if type(obj) is Class else obj.clss
        if not cname:
            return
        if not cname in class_tree:
          # add a new class to the tree
            class_tree[cname] = Class(cname)
        # insert the definition into the class
        val = class_tree[cname]
        if type(obj) is Function:
            val.functions.append(obj)
        elif type(obj) is Constant:
            val.constants.append(obj)
        else:
            raise TypeError('Unexpected object type: '+str(type(obj)))



class Translator(object):
    def translate(self, defn):
        # --- class ---
        # classes have 'class' prefixed on their name 
        if 'class' in defn[0]:
            return self.translateClass(defn)
        # --- function ---
        # functions either need to have input arguments, or not uppercase names
        elif defn[3] or not self.translateName(defn[0]).isupper():
            return self.translateFunction(defn)
        # --- constant ---
        else:
            return self.translateConstant(defn)

    def translateClass(self, defn):
        return Class()

    def translateFunction(self, defn, class_tree=None):
        name = self.translateName(defn[0])
        clss = self.translateClassName(defn[0])
        rtp  = defn[1]
        args = defn[3]
        req  = [] 
        opt = []
        for arg in args:
            if arg:
                a = self.translateArgument(arg)
                opt.append(a) if a.default else req.append(a)
        return Function(name, clss, '', rtp, False, req, opt)
            
    def translateConstant(self, defn):
        const = True if 'const' in defn[0] else False
        name  = self.translateName(defn[0])
        clss  = self.translateClassName(defn[0])
        tp    = 'int'
        val   = defn[1]
        return Constant(name, clss, tp, const, '', val)

    def translateArgument(self, defn):
        tp   = defn[0]
        name = defn[1]
        default = tp+'()' if defn[2] else ''
        return Argument(name, tp, False, '', default)

    def translateName(self, name):
        return name.split(' ')[-1].split('.')[-1]

    def translateClassName(self, name):
        parts = name.split('.')
        return parts[1] if len(parts) == 3 else ''



class Namespace(object):
    def __init__(self, name='', constants=None, classes=None, functions=None):
        self.name = name
        self.constants = constants if constants else []
        self.classes   = classes   if classes   else []
        self.functions = functions if functions else []

    def __str__(self):
        return 'namespace '+self.name+' {\n\n'+\
          (join((f.__str__() for f in self.functions), '\n')+'\n\n' if self.functions else '')+\
          (join((c.__str__() for c in self.constants), '\n')+'\n\n' if self.constants else '')+\
          (join((o.__str__() for o in self.classes), '\n\n')        if self.classes   else '')+'\n};'

class Class(object):
    def __init__(self, name='', namespace='', constants=None, functions=None):
        self.name = name
        self.namespace = namespace
        self.constants = constants if constants else []
        self.functions = functions if functions else []

    def __str__(self):
        return 'class '+self.name+' {\n\t'+\
          (join((c.__str__() for c in self.constants), '\n\t')+'\n\n\t' if self.constants else '')+\
          (join((f.__str__() for f in self.functions), '\n\t')          if self.functions else '')+'\n};'

class Function(object):
    def __init__(self, name='', clss='', namespace='', rtp='', const=False, req=None, opt=None):
        self.name  = name
        self.clss  = clss
        self.const = const
        self.namespace = namespace
        self.rtp = rtp 
        self.req = req if req else []
        self.opt = opt if opt else []

    def __str__(self):
        return fill((self.rtp+' ' if self.rtp else '')+self.name+'('+\
          join((arg.__str__() for arg in self.req+self.opt), ', ')+\
          ')'+(' const' if self.const else '')+';', 80, subsequent_indent=('\t\t' if self.clss else '\t'))

class Argument(object):
    def __init__(self, name='', tp='', const=False, ref='', default=''):
        self.name = name
        self.tp   = tp
        self.ref  = ref
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
