'''
Digit recognition adjustment. 
Grid search is used to find the best parameters for SVN and KNearest classifiers.
SVM adjustment follows the guidelines given in 
http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

Threading or cloud computing (with http://www.picloud.com/)) may be used 
to speedup the computation.

Usage:
  digits_adjust.py [--model {svm|knearest}] [--cloud] [--env <PiCloud environment>]
  
  --model {svm|knearest}   - select the classifier (SVM is the default)
  --cloud                  - use PiCloud computing platform (for SVM only)
  --env                    - cloud environment name

'''
# TODO dataset preprocessing in cloud
# TODO cloud env setup tutorial

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

from digits import *

def cross_validate(model_class, params, samples, labels, kfold = 3, pool = None):
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

def adjust_KNearest(samples, labels):
    print 'adjusting KNearest ...'
    best_err, best_k = np.inf, -1
    for k in xrange(1, 9):
        err = cross_validate(KNearest, dict(k=k), samples, labels)
        if err < best_err:
            best_err, best_k = err, k
        print 'k = %d, error: %.2f %%' % (k, err*100)
    best_params = dict(k=best_k)
    print 'best params:', best_params
    return best_params

def adjust_SVM(samples, labels, usecloud=False, cloud_env=''):
    Cs = np.logspace(0, 5, 10, base=2)
    gammas = np.logspace(-7, -2, 10, base=2)
    scores = np.zeros((len(Cs), len(gammas)))
    scores[:] = np.nan

    if usecloud:
        try: 
            import cloud
        except ImportError: 
            print 'cloud module is not installed'
            usecloud = False
    if usecloud:
        print 'uploading dataset to cloud...'
        np.savez('train.npz', samples=samples, labels=labels)
        cloud.files.put('train.npz')

    print 'adjusting SVM (may take a long time) ...'
    def f(job):
        i, j = job
        params = dict(C = Cs[i], gamma=gammas[j])
        score = cross_validate(SVM, params, samples, labels)
        return i, j, score
    def fcloud(job):
        i, j = job
        cloud.files.get('train.npz')
        npz = np.load('train.npz')
        params = dict(C = Cs[i], gamma=gammas[j])
        score = cross_validate(SVM, params, npz['samples'], npz['labels'])
        return i, j, score
    
    if usecloud:
        jids = cloud.map(fcloud, np.ndindex(*scores.shape), _env=cloud_env, _profile=True)
        ires = cloud.iresult(jids)
    else:
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        ires = pool.imap_unordered(f, np.ndindex(*scores.shape))

    for count, (i, j, score) in enumerate(ires):
        scores[i, j] = score
        print '%d / %d (best error: %.2f %%, last: %.2f %%)' % (count+1, scores.size, np.nanmin(scores)*100, score*100)
    print scores

    i, j = np.unravel_index(scores.argmin(), scores.shape)
    best_params = dict(C = Cs[i], gamma=gammas[j])
    print 'best params:', best_params
    print 'best error: %.2f %%' % (scores.min()*100)
    return best_params

if __name__ == '__main__':
    import getopt
    import sys
    
    print __doc__

    args, _ = getopt.getopt(sys.argv[1:], '', ['model=', 'cloud', 'env='])
    args = dict(args)
    args.setdefault('--model', 'svm')
    args.setdefault('--env', '')
    if args['--model'] not in ['svm', 'knearest']:
        print 'unknown model "%s"' % args['--model']
        sys.exit(1)

    digits, labels = load_digits('digits.png')
    shuffle = np.random.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]
    digits2 = map(deskew, digits)
    samples = np.float32(digits2).reshape(-1, SZ*SZ) / 255.0
    
    t = clock()
    if args['--model'] == 'knearest':
        adjust_KNearest(samples, labels)
    else:
        adjust_SVM(samples, labels, usecloud='--cloud' in args, cloud_env = args['--env'])
    print 'work time: %f s' % (clock() - t)
        