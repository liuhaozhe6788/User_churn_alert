import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import metrics


class BasicModel(ABC):
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """
        :return: 模型需要的特征
        """
        return []

    @property
    @abstractmethod
    def label_name(self) -> str:
        """
        :return: 模型预测的标签
        """
        return ""

    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, data: pd.DataFrame):
        """训练模型"""
        pass

    @abstractmethod
    def test(self, data: pd.DataFrame) -> np.ndarray:
        """预测 data 所有行，用于测试模型效果"""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> int:
        pass

    def evaluate(self, predictions: np.array, labels: pd.Series):
        """评估模型效果"""
        predictions = predictions > 0.5
        if isinstance(labels, pd.DataFrame): labels = labels[self.label_name]

        count = labels.value_counts()
        sample_weight = labels.map({0: 1 / count[0], 1: 1 / count[1]})
        accuracy = metrics.accuracy_score(labels, predictions, sample_weight=sample_weight)
        kappa_score = metrics.cohen_kappa_score(labels, predictions, sample_weight=sample_weight)
        f1_score = metrics.f1_score(labels, predictions, sample_weight=sample_weight)
        recall = metrics.recall_score(labels, predictions, sample_weight=sample_weight)
        precision = metrics.precision_score(labels, predictions, sample_weight=sample_weight)
        auc = metrics.roc_auc_score(labels, predictions, sample_weight=sample_weight)
        aucpr = metrics.average_precision_score(labels, predictions, sample_weight=sample_weight)
        confusion_matrix = metrics.confusion_matrix(labels, predictions, sample_weight=sample_weight)
        confusion_matrix = {
            "True Positive(1->1)": confusion_matrix[1, 1],
            "False Negative(1->0)": confusion_matrix[1, 0],
            "True Negative(0->0)": confusion_matrix[0, 0],
            "False Positive(0->1)": confusion_matrix[0, 1],
        }
        metric_dic = {f"f1_score": "%.3f" % f1_score,
                      f"recall": "%.3f" % recall,
                      f"precision": "%.3f" % precision,
                      f"kappa_score": "%.3f" % kappa_score,
                      f"aucpr": "%.3f" % aucpr,
                      f"auc": "%.3f" % auc,
                      f"accuracy": "%.3f" % accuracy,
                      "confusion_mat": confusion_matrix}

        return metric_dic

    def save(self, pickle_path):
        """保存"""
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pickle_path):
        """读取"""
        with open(pickle_path, "rb") as f:
            return pickle.load(f)


__all__ = ["BasicModel"]
