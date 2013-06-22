from textwrap import TextWrapper
from string import split, join
import re

def inputs(args):
    '''Keeps only the input arguments in a list of elements.
    In OpenCV input arguments are all arguments with names
    not beginning with 'dst'
    '''
    out = []
    for arg in args:
        if not arg.name.startswith('dst'):
            out.append(arg)
    return out

def ninputs(args):
    '''Counts the number of input arguments in the input list'''
    return len(inputs(args))

def outputs(args):
    '''Determines whether any of the given arguments is an output
    reference, and returns a list of only those elements.
    In OpenCV, output references are preceeded by 'dst'
    '''
    out = []
    for arg in args:
        if arg.name.startswith('dst'):
            out.append(arg)
    return out

def output(arg):
    return True if arg.name.startswith('dst') else False
        
def noutputs(args):
    '''Counts the number of output arguments in the input list'''
    return len(outputs(args))


def toUpperCamelCase(text):
    return text[0].upper() + text[1:]

def toLowerCamelCase(text):
    return text[0].lower() + text[1:]

def toUnderCase(text):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
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
