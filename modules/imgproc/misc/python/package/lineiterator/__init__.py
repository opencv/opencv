import cv2 as cv

class LineIteratorWrapper(cv.LineIterator):
    def __init__(self, pt1, pt2, connectivity=8, leftToRight=False):
        super().__init__(pt1, pt2, connectivity, leftToRight)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next():
            return self.pNext
        else:
            raise StopIteration

LineIteratorWrapper.__module__ = cv.__name__
cv.LineIterator = LineIteratorWrapper
