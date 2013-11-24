from textwrap import TextWrapper
from string import split, join
import re, os
# precompile a URL matching regular expression
urlexpr = re.compile(r"((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)", re.MULTILINE|re.UNICODE)

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
    '''Is the input 'void' '''
    return arg == 'void'

def flip(arg):
    '''flip the sign of the input'''
    return not arg

def noutputs(fun):
    '''Counts the number of output arguments in the input list'''
    return int(not void(fun.rtp)) + len(outputs(fun.req)) + len(outputs(fun.opt))

def convertibleToInt(string):
    '''Can the input string be evaluated to an integer?'''
    salt = '1+'
    try:
        exec(salt+string)
        return True
    except:
        return False

def binaryToDecimal(string):
    '''Attempt to convert the input string to floating point representation'''
    try:
        return str(eval(string))
    except:
        return string

def formatMatlabConstant(string, table):
    '''
    Given a string representing a Constant, and a table of all Constants,
    attempt to resolve the Constant into a valid Matlab expression
    For example, the input
      DEPENDENT_VALUE = 1 << FIXED_VALUE
    needs to be converted to
      DEPENDENT_VALUE = bitshift(1, cv.FIXED_VALUE);
    '''
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

def matlabURL(string):
    """This filter is used to construct a Matlab specific URL that calls the
    system browser instead of the (insanely bad) builtin Matlab browser"""
    return re.sub(urlexpr, '<a href="matlab: web(\'\\1\', \'-browser\')">\\1</a>', string)

def capitalizeFirst(text):
    '''Capitalize only the first character of the text string'''
    return text[0].upper() + text[1:]

def toUpperCamelCase(text):
    '''variable_name --> VariableName'''
    return ''.join([capitalizeFirst(word) for word in text.split('_')])

def toLowerCamelCase(text):
    '''variable_name --> variableName'''
    upper_camel = toUpperCamelCase(test)
    return upper_camel[0].lower() + upper_camel[1:]

def toUnderCase(text):
    '''VariableName --> variable_name'''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def stripTags(text):
    '''
    strip or convert html tags from a text string
    <code>content</code> --> content
    <anything>           --> ''
    &lt                  --> <
    &gt                  --> >
    &le                  --> <=
    &ge                  --> >=
    '''
    upper = lambda pattern: pattern.group(1).upper()
    text = re.sub('<code>(.*?)</code>', upper, text)
    text = re.sub('<([^=\s].*?)>', '', text)
    text = re.sub('&lt', '<', text)
    text = re.sub('&gt', '>', text)
    text = re.sub('&le', '<=', text)
    text = re.sub('&ge', '>=', text)
    return text

def qualify(text, name):
    '''Adds uppercase 'CV.' qualification to any occurrences of name in text'''
    return re.sub(name.upper(), 'CV.'+name.upper(), text)

def slugify(text):
    '''A_Function_name --> a-function-name'''
    return text.lower().replace('_', '-')

def filename(fullpath):
    '''Returns only the filename without an extension from a file path
    eg. /path/to/file.txt --> file
    '''
    return os.path.splitext(os.path.basename(fullpath))[0]

def split(text, delimiter=' '):
    '''Split a text string into a list using the specified delimiter'''
    return text.split(delimiter)

def csv(items, sep=', '):
    '''format a list with a separator (comma if not specified)'''
    return sep.join(item for item in items)

def cellarray(items, escape='\''):
    '''format a list of items as a matlab cell array'''
    return '{' + ', '.join(escape+item+escape for item in items) + '}'

def stripExtraSpaces(text):
    '''Removes superfluous whitespace from a string, including the removal
    of all leading and trailing whitespace'''
    return ' '.join(text.split())

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
