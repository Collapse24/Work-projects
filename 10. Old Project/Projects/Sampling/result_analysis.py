import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def predictions_to_classes(predictions):
    predictions_as_classes = list(map(lambda x: np.where(x == np.max(x))[0][0], predictions))

    return predictions_as_classes

def get_scores(predictions, y_test: list, min_confidence=.5):
    classes_predictions = list(map(lambda x: np.where(x == np.max(x))[0][0], predictions))

    y_pred_conf = []
    y_conf = []
    idx = []

    for i in range(len(predictions)):
        if np.max(predictions[i]) > min_confidence:
            y_conf.append(y_test[i])
            y_pred_conf.append(classes_predictions[i])
            idx.append(i)

    data_part = f"{len(y_pred_conf)} / {len(y_test)} = {(np.round((len(y_pred_conf) / len(y_test) * 100), 2))}%"
    acc_score = np.round(accuracy_score(y_conf, y_pred_conf), 4)
    pre_score = np.round(precision_score(y_conf, y_pred_conf, average='weighted'), 4)
    rec_score = np.round(recall_score(y_conf, y_pred_conf, average='weighted'), 4)

    return y_pred_conf, y_conf, data_part, acc_score, pre_score, rec_score, idx
