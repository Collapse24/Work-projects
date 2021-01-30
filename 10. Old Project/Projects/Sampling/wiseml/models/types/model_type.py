from enum import Enum


class ModelType(Enum):
    pass


class GBModelType(ModelType):
    XGBoost = 'xgb'
    LGBM = 'lgb'
    CatBoost = 'catboost'

