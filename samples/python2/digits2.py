import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10

def load_base(fn):
    print 'loading "%s" ...' % fn
    digits_img = cv2.imread(fn, 0)
    h, w = digits_img.shape
    digits = [np.hsplit(row, w/SZ) for row in np.vsplit(digits_img, h/SZ)]
    digits = np.array(digits).reshape(-1, SZ, SZ)
    digits = np.float32(digits).reshape(-1, SZ*SZ) / 255.0
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

def cross_validate(model_class, params, samples, labels, kfold = 4, pool = None):
    n = len(samples)
    folds = np.array_split(np.arange(n), kfold)
    def f(i):
        model = model_class(**params)
        test_idx = folds[i]
        train_idx = list(folds)
        train_idx.pop(i)
        train_idx = np.hstack(train_idx)
        train_samples, train_labels = samples[train_idx], labels[train_idx]
        test_samples, test_labels = samples[test_idx], labels[test_idx]
        model.train(train_samples, train_labels)
        resp = model.predict(test_samples)
        score = (resp != test_labels).mean()
        print ".",
        return score
    if pool is None:
        scores = map(f, xrange(kfold))
    else:
        scores = pool.map(f, xrange(kfold))
    return np.mean(scores)

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k

    @staticmethod
    def adjust(samples, labels):
        print 'adjusting KNearest ...'
        best_err, best_k = np.inf, -1
        for k in xrange(1, 11):
            err = cross_validate(KNearest, dict(k=k), samples, labels)
            if err < best_err:
                best_err, best_k = err, k
            print 'k = %d, error: %.2f %%' % (k, err*100)
        best_params = dict(k=best_k)
        print 'best params:', best_params
        return best_params

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF, 
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )

    @staticmethod
    def adjust(samples, labels):
        Cs = np.logspace(0, 5, 10, base=2)
        gammas = np.logspace(-7, -2, 10, base=2)
        scores = np.zeros((len(Cs), len(gammas)))
        scores[:] = np.nan

        print 'adjusting SVM (may take a long time) ...'
        def f(job):
            i, j = job
            params = dict(C = Cs[i], gamma=gammas[j])
            score = cross_validate(SVM, params, samples, labels)
            scores[i, j] = score
            nready = np.isfinite(scores).sum()
            print '%d / %d (best error: %.2f %%, last: %.2f %%)' % (nready, scores.size, np.nanmin(scores)*100, score*100)

        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        pool.map(f, np.ndindex(*scores.shape))
        print scores

        i, j = np.unravel_index(scores.argmin(), scores.shape)
        best_params = dict(C = Cs[i], gamma=gammas[j])
        print 'best params:', best_params
        print 'best error: %.2f %%' % (scores.min()*100)
        return best_params

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

def main_adjustSVM(samples, labels):
    params = SVM.adjust(samples, labels)
    print 'training SVM on all samples ...'
    model = SVN(**params)
    model.train(samples, labels)
    print 'saving "digits_svm.dat" ...'
    model.save('digits_svm.dat')

def main_adjustKNearest(samples, labels):
    params = KNearest.adjust(samples, labels)

def main_showSVM(samples, labels):
    from common import mosaic

    train_n = int(0.9*len(samples))
    digits_train, digits_test = np.split(samples[shuffle], [train_n])
    labels_train, labels_test = np.split(labels[shuffle], [train_n])

    print 'training SVM ...'
    model = SVM(C=2.16, gamma=0.0536)
    model.train(digits_train, labels_train)

    train_err = (model.predict(digits_train) != labels_train).mean()
    resp_test = model.predict(digits_test)
    test_err = (resp_test != labels_test).mean()
    print 'train errors: %.2f %%' % (train_err*100)
    print 'test errors: %.2f %%' % (test_err*100)

    
    # visualize test results
    vis = []
    for img, flag in zip(digits_test, resp_test == labels_test):
        img = np.uint8(img*255).reshape(SZ, SZ)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    vis = mosaic(25, vis)
    cv2.imshow('test', vis)
    cv2.waitKey()
    


if __name__ == '__main__':
    samples, labels = load_base('digits.png')
    shuffle = np.random.permutation(len(samples))
    samples, labels = samples[shuffle], labels[shuffle]
    
    #main_adjustSVM(samples, labels)
    #main_adjustKNearest(samples, labels)
    main_showSVM(samples, labels)
