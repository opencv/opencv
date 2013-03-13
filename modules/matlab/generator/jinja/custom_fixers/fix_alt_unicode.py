from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, BlankLine


class FixAltUnicode(fixer_base.BaseFix):
    PATTERN = """
    func=funcdef< 'def' name='__unicode__'
                  parameters< '(' NAME ')' > any+ >
    """

    def transform(self, node, results):
        name = results['name']
        name.replace(Name('__str__', prefix=name.prefix))
