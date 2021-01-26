from abc import ABC, ABCMeta, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def get_prepared_models(self):
        pass


class Framework(object):
    in_blob_name = ''
    out_blob_name = ''

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_output(self, input_blob):
        pass
