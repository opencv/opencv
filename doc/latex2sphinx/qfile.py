import StringIO
import os

class QOpen(StringIO.StringIO):
    def __init__(self, *args):
        self.__args = args
        StringIO.StringIO.__init__(self)

    def close(self):
        import StringIO, os
        fname = self.__args[0]
        if not os.access(fname, os.R_OK) or self.getvalue() != open(fname).read():
            open(*self.__args).write(self.getvalue())
        StringIO.StringIO.close(self)

    def __del__(self):
        if not self.closed:
            self.close()
