import numpy as np
import cv2

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses

# TODO move these to cv2
CV_ROW_SAMPLE = 1
CV_VAR_NUMERICAL   = 0
CV_VAR_ORDERED     = 0
CV_VAR_CATEGORICAL = 1


class LetterStatModel(object):
    train_ratio = 0.5
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class RTrees(LetterStatModel):
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([CV_VAR_NUMERICAL] * var_n + [CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
        

class KNearest(LetterStatModel):
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.Boost()
        self.class_n = 26
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([CV_VAR_NUMERICAL] * var_n + [CV_VAR_CATEGORICAL, CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples
    
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses


class SVM(LetterStatModel):
    train_ratio = 0.1
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )


if __name__ == '__main__':
    import argparse

    models = [RTrees, KNearest, Boost, SVM] # MLP, NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='rtrees', choices=models.keys())
    parser.add_argument('-data', nargs=1, default='letter-recognition.data')
    parser.add_argument('-load', nargs=1)
    parser.add_argument('-save', nargs=1)
    args = parser.parse_args()

    print 'loading data %s ...' % args.data
    samples, responses = load_base(args.data)
    Model = models[args.model]
    model = Model()

    train_n = int(len(samples)*model.train_ratio)
    if args.load is None:
        print 'training %s ...' % Model.__name__
        model.train(samples[:train_n], responses[:train_n])
    else:
        fn = args.load[0]
        print 'loading model from %s ...' % fn
        model.load(fn)

    print 'testing...'
    train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n])
    test_rate  = np.mean(model.predict(samples[train_n:]) == responses[train_n:])

    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)

    if args.save is not None:
        fn = args.save[0]
        print 'saving model to %s ...' % fn
        model.save(fn)
