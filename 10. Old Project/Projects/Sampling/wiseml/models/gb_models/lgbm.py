import lightgbm as lgb

from wiseml.train.metric import Metric
from wiseml.models.gb_model import GBModel
from wiseml.models.types.task_type import TaskType
import pickle


class LGBModel(GBModel):

    specific_metric_mapper = {
        Metric.AUC: 'auc',
        Metric.LOG_LOSS: 'logloss'
    }

    def __init__(self, task_type: TaskType):
        super().__init__(task_type=task_type)
        self.task_type = task_type
        self.train_function = self.train
        self.early_stopping_round_alias = 'early_stopping_rounds'
        self.eval_metric_alias = 'metric'
        self.num_rounds_alias = 'num_boost_round'
        self.valid_set_alias = 'valid_sets'
        self.verbose_alias = 'verbose_eval'

    def train(self, **kwargs):
        kwargs['valid_names'] = ['train', 'valid']
        if self.task_type == TaskType.Classification:
            if self.num_class == 2:
                metric_prefix = 'binary_'
            else:
                metric_prefix = 'multi'
                kwargs['num_class'] = self.num_class
            kwargs['metric'] = metric_prefix + self.specific_metric_mapper[kwargs['metric']]

        model = lgb.train(**kwargs)

        return model

    @classmethod
    def load(cls, path):
        model = lgb.Booster(model_file=path)
        return model

    @classmethod
    def load_pickle(cls, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
