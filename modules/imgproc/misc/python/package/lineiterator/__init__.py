import cv2 as cv

class LineIteratorB(cv.LineIterator):
    def __init__(self, pt1, pt2):
        super().__init__(pt1, pt2)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next():
            return self
        else:
            raise StopIteration

LineIteratorB.__module__ = cv.__name__
cv.LineIterator = LineIteratorB
cv._registerMatType(LineIteratorB)
