from plasTeX import Base
from plasTeX.Base.LaTeX.Verbatim import verbatim
from plasTeX.Base.LaTeX import Sectioning
import sys

class includegraphics(Base.Command):
  args = '[size] file'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvclass(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvfunc(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvCPyFunc(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvCppFunc(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvFunc(Sectioning.subsection):
  args = 'title alt'
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvstruct(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cvmacro(Sectioning.subsection):
  def invoke(self, tex):
    Sectioning.subsection.invoke(self, tex)

class cross(Base.Command):
  args = 'name'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class label(Base.Command):
  args = 'name'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class url(Base.Command):
  args = 'loc'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvarg(Base.Command):
  args = 'item def'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvCross(Base.Command):
  args = 'name altname'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvCPyCross(Base.Command):
  args = 'name'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)
    
class cvCppCross(Base.Command):
  args = 'name'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvdefC(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvcode(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvdefPy(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvdefCpp(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvC(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)
    
class cvCpp(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvCPy(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class cvPy(Base.Command):
  args = 'a'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class xxindex(Base.Command):
  args = 'entry'
  def invoke(self, tex):
    Base.Command.invoke(self, tex)

class lstlisting(verbatim):
  def parse(self, tex):
    verbatim.parse(self, tex)
    return self.attributes

def section_filename(title):
    """Image Processing ==> image_processing.rst"""
    lower_list = [word.lower() for word in title.split()]
    return "_".join(lower_list) + ".rst"

class chapter(Sectioning.chapter):
    @property
    def filenameoverride(self):
        if self.attributes['title'] is not None:
            filename = section_filename(str(self.attributes['title']))
            #assert filename in ['cxcore.rst', 'cvreference.rst']
            return filename
        raise AttributeError, 'This chapter does not generate a new file'
        

class section(Sectioning.section):
    @property
    def filenameoverride(self):
        if self.attributes['title'] is not None:
            filename = section_filename(str(self.attributes['title']))
            print 'section:', filename
            return filename
        raise AttributeError, 'This section does not generate a new file'

class xifthenelse(Base.Command):
    args = 'test then else' 

    class _not(Base.Command):
        macroName = 'not'

    class _and(Base.Command):
        macroName = 'and'

    class _or(Base.Command):
        macroName = 'or'

    class NOT(Base.Command):
        pass

    class AND(Base.Command):
        pass

    class OR(Base.Command):
        pass

    class openParen(Base.Command):
        macroName = '('

    class closeParen(Base.Command):
        macroName = ')'

    class isodd(Base.Command):
        args = 'number:int'

    class isundefined(Base.Command):
        args = 'command:str'

    class equal(Base.Command):
        args = 'first second'

    class lengthtest(Base.Command):
        args = 'test'

    class boolean(Base.Command):
        args = 'name:str'
