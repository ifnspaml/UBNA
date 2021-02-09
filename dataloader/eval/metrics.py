import numpy as np
import warnings


class Evaluator(object):
    # CONF MATRIX
    #     0  1  2  (PRED)
    #  0 |TP FN FN|
    #  1 |FP TP FN|
    #  2 |FP FP TP|
    # (GT)
    # -> rows (axis=1) are FN
    # -> columns (axis=0) are FP
    @staticmethod
    def iou(conf):  # TP / (TP + FN + FP)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
        meaniu = np.nanmean(iu)
        result = {'iou': dict(zip(range(len(iu)), iu)), 'meaniou': meaniu}
        return result

    @staticmethod
    def accuracy(conf):  # TP / (TP + FN) aka 'Recall'
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            totalacc = np.diag(conf).sum() / (conf.sum())
            acc = np.diag(conf) / (conf.sum(axis=1))
        meanacc = np.nanmean(acc)
        result = {'totalacc': totalacc, 'meanacc': meanacc, 'acc': acc}
        return result

    @staticmethod
    def precision(conf):  # TP / (TP + FP)
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            prec = np.diag(conf) / (conf.sum(axis=0))
        meanprec = np.nanmean(prec)
        result = {'meanprec': meanprec, 'prec': prec}
        return result

    @staticmethod
    def freqwacc(conf):
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
            freq = conf.sum(axis=1) / (conf.sum())
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        result = {'freqwacc': fwavacc}
        return result


class SegmentationRunningScore(object):
    def __init__(self, n_classes=20):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask_true = (label_true >= 0) & (label_true < n_class)
        mask_pred = (label_pred >= 0) & (label_pred < n_class)
        mask = mask_pred & mask_true
        label_true = label_true[mask].astype(np.int)
        label_pred = label_pred[mask].astype(np.int)
        hist = np.bincount(n_class * label_true + label_pred,
                           minlength=n_class*n_class).reshape(n_class, n_class).astype(np.float)
        return hist

    def update(self, label_trues, label_preds):
        # update confusion matrix
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, listofparams=None):
        """Returns the evaluation params specified in the list"""
        possibleparams = {
            'iou': Evaluator.iou,
            'acc': Evaluator.accuracy,
            'freqwacc': Evaluator.freqwacc,
            'prec': Evaluator.precision
        }
        if listofparams is None:
            listofparams = possibleparams

        result = {}
        for param in listofparams:
            if param in possibleparams.keys():
                result.update(possibleparams[param](self.confusion_matrix))
        return result

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
