from abc import ABC, abstractmethod

class AbstractModel(ABC):
    '''
    This is an abstract base class for models. All derived model classes must implement the
    `get_prepared_models` method.
    '''
    @abstractmethod
    def get_prepared_models(self):
        '''
        Abstract method which needs to be implemented by any class that inherits from `AbstractModel`.
        This method is intended to return the prepared model(s) for use.
        '''
        pass


class Framework(ABC):
    '''
    This is an abstract base class for frameworks. All derived framework classes must implement the
    `get_name` and `get_output` methods.
    '''
    in_blob_name = ''
    out_blob_name = ''

    @abstractmethod
    def get_name(self):
        '''
        Abstract method which needs to be implemented by any class that inherits from `Framework`.
        This method is intended to return the name of the framework.
        '''
        pass

    @abstractmethod
    def get_output(self, input_blob):
        '''
        Abstract method which needs to be implemented by any class that inherits from `Framework`.
        This method is intended to process the input blob and return the output.
        '''
        pass
