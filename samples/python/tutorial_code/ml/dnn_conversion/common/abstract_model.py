from abc import ABC, ABCMeta, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def get_prepared_models(self):
        pass


# https://github.com/opencv/opencv/blob/9ba5581d176baf7c8fceb12b4dcef6c49c7a087d/modules/dnn/test/imagenet_cls_test_alexnet.py
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
