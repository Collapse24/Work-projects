from enum import Enum

from sklearn import metrics


class Metric(Enum):
    ACCURACY = 'accuracy'
    AUC = 'auc'
    LOG_LOSS = 'log-loss'


metric_func = {
    Metric.ACCURACY:   metrics.accuracy_score,
    Metric.AUC:        metrics.auc,
    Metric.LOG_LOSS:   metrics.log_loss
}