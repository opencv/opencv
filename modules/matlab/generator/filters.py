from textwrap import TextWrapper
from string import split, join
import re, os

def inputs(args):
    '''Keeps only the input arguments in a list of elements.
    In OpenCV input arguments are all arguments with names
    not beginning with 'dst'
    '''
    try:
      return [arg for arg in args['only'] if arg.I and not arg.O]
    except:
      return [arg for arg in args if arg.I]

def ninputs(fun):
    '''Counts the number of input arguments in the input list'''
    return len(inputs(fun.req)) + len(inputs(fun.opt))

def outputs(args):
    '''Determines whether any of the given arguments is an output
    reference, and returns a list of only those elements.
    In OpenCV, output references are preceeded by 'dst'
    '''
    try:
      return [arg for arg in args['only'] if arg.O and not arg.I]
    except:
      return [arg for arg in args if arg.O]

def only(args):
    '''Returns exclusively the arguments which are only inputs
    or only outputs'''
    d = {};
    d['only'] = args
    return d

def void(arg):
    return arg == 'void'

def flip(arg):
    return not arg

def noutputs(fun):
    '''Counts the number of output arguments in the input list'''
    return int(not void(fun.rtp)) + len(outputs(fun.req)) + len(outputs(fun.opt))

def convertibleToInt(string):
    salt = '1+'
    try:
        exec(salt+string)
        return True
    except:
        return False

def binaryToDecimal(string):
    try:
        return str(eval(string))
    except:
        return string

def formatMatlabConstant(string, table):
    import re
    # split the string into expressions
    words = re.split('(\W+)', string)
    # add a 'cv' prefix if an expression is also a key in the lookup table
    words = ''.join([('cv.'+word if word in table else word) for word in words]) 
    # attempt to convert arithmetic expressions and binary/hex to decimal
    words = binaryToDecimal(words)
    # convert any remaining bitshifts to Matlab 'bitshift' methods
    shift = re.sub('[\(\) ]', '', words).split('<<')
    words = 'bitshift('+shift[0]+', '+shift[1]+')' if len(shift) == 2 else words
    return words

def capitalizeFirst(text):
    return text[0].upper() + text[1:]

def toUpperCamelCase(text):
    return ''.join([capitalizeFirst(word) for word in text.split('_')])

def toLowerCamelCase(text):
    upper_camel = toUpperCamelCase(test)
    return upper_camel[0].lower() + upper_camel[1:]

def toUnderCase(text):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def stripTags(text):
    upper = lambda pattern: pattern.group(1).upper()
    text = re.sub('<code>(.*?)</code>', upper, text)
    text = re.sub('<(.*?)>', '', text)
    return text

def qualify(text, name):
    return re.sub(name.upper(), 'CV.'+name.upper(), text)

def slugify(text):
    return text.lower().replace('_', '-')

def filename(fullpath):
    return os.path.splitext(os.path.basename(fullpath))[0]
    
def comment(text, wrap=80, escape='% ', escape_first='', escape_last=''):
    '''comment filter
    Takes a string in text, and wraps it to wrap characters in length with
    preceding comment escape sequence on each line. escape_first and 
    escape_last can be used for languages which define block comments.
    Examples:
        C++ inline comment    comment(80, '// ')
        C block comment:      comment(80, ' * ', '/*', ' */')
        Matlab comment:       comment(80, '% ')
        Matlab block comment: comment(80, '', '%{', '%}')
        Python docstrings:    comment(80, '', '\'\'\'', '\'\'\'')
    '''

    tw = TextWrapper(width=wrap-len(escape))
    if escape_first:
        escape_first = escape_first+'\n'
    if escape_last:
        escape_last = '\n'+escape_last
    escapn = '\n'+escape
    lines  = text.split('\n')
    wlines = (tw.wrap(line) for line in lines)
    return escape_first+escape+join((join(line, escapn) for line in wlines), escapn)+escape_last
