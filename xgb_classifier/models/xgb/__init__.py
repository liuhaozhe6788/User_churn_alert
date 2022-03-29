import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split

import models
from .preprocess import *
from .xgb_classifier import XGBClassifier


class Model(models.BasicModel):
    @property
    def feature_names(self):
        # 模型需要的特征
        return [
            "life_time",
            "login_day_cnt",
            "register_country",
            "login_time",
            "time_zone",
            "grade",
            "last_season_gap_day",
            "battlepass_lvl",
            "curr_asset",
            "curr_diamond",
            "max_chip_stock",
            "max_blind",
            "bankrupt_cnt_3d",
            "bankrupt_cnt",
            "pay_amt",
            "pay_cnt",
            "recency",
            "ads_watch_cnt_1d",
            "ads_click_cnt_1d",
            *[f"{action}_{count}d" for action in ["login_time_hour_list", "login_time_list"] for count in [7, 30]],
            *[f"{action}_{count}d" for action in ["pay", "pay_cnt", "ads_watch_cnt", "login_cnt"] for count in
              [3, 7, 15, 30]],
        ]

    @property
    def label_name(self):
        return "is_churned"

    def __init__(self):
        # 分类器实际使用的特征
        self.__feature_names = [
            "life_time",
            "login_day_cnt",
            # "register_country",
            "grade",
            "last_season_gap_day",
            "battlepass_lvl",
            "curr_asset",
            "curr_diamond",
            "max_chip_stock",
            "max_blind",
            "bankrupt_cnt_3d",
            "bankrupt_cnt",
            "pay_amt",
            "pay_cnt",
            "recency",
            "ads_watch_cnt_1d",
            "ads_click_cnt_1d",
            *[f"{action}_{count}d" for action in ["pay", "pay_cnt", "ads_watch_cnt", "login_cnt"] for count in
              [3, 7, 15, 30]],
            "time_zone",
            "login_time_h",
            *[f"login_time_{f}_{d}d" for f in ["mean", "std"] for d in [7, 30]],
            *[f"login_time_hour_{f}_{d}d" for f in ["sum", "mean", "std"] for d in [7, 30]],
        ]
        self.model = XGBClassifier()

    def train(self, data, with_eval=False):
        if with_eval:
            x_train, x_val, y_train, y_val = train_test_split(*self.__data_process(data, is_train=True), test_size=.2,
                                                              random_state=0)
            self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)])
            return {"train_set": self.evaluate(self.model.predict(x_train), y_train),
                    "val_set": self.evaluate(self.model.predict(x_val), y_val)}
        else:
            self.model.fit(*self.__data_process(data, is_train=True))

    def test(self, data):
        return self.model.predict(self.__data_process(data))

    def predict(self, data):
        return self.model.predict(self.__data_process(data[:1]))[0]

    def plot(self):
        pd.DataFrame({t: self.model.get_booster().get_score(importance_type=t) for t in
                      ["gain", "weight", "cover"]}).sort_values(by="gain", ascending=False).to_excel("importance.xlsx")
        xgboost.to_graphviz(self.model).render("tree", format="png")

    def __data_process(self, data, is_train=False):
        assert isinstance(data, pd.DataFrame), \
            f"错误的数据类型：{type(data)}"
        assert len(set(self.feature_names) - set(data.columns)) == 0, \
            f"缺少特征：{'、'.join(set(self.feature_names) - set(data.dtypes))}"
        assert not getattr(self, "is_train", True) or self.label_name in data.columns, \
            f"训练样本缺少特征：{self.label_name}"
        x = data[set(self.__feature_names).intersection(self.feature_names)]
        x[[
            "login_time_h",
            *[f"login_time_{f}_{d}d" for d in [7, 30] for f in ["mean", "std"]],
            *[f"login_time_hour_{f}_{d}d" for d in [7, 30] for f in ["sum", "mean", "std"]],
        ]] = data.apply(
            lambda line: (
                time_list_to_array(line["login_time"])[0, 3],
                *measure(time_list_to_array(
                    line["login_time"] if pd.isna(line[f"login_time_list_7d"]) else "{},{}".format(
                        line[f"login_time_list_7d"], line["login_time"])))[1:],
                *measure(time_list_to_array(
                    line["login_time"] if pd.isna(line[f"login_time_list_30d"]) else "{},{}".format(
                        line[f"login_time_list_30d"], line["login_time"])))[1:],
                *measure(0 if pd.isna(line[f"login_time_hour_list_7d"]) else to_list(
                    to_list(line[f"login_time_hour_list_7d"]), map=int)),
                *measure(0 if pd.isna(line[f"login_time_hour_list_30d"]) else to_list(
                    to_list(line[f"login_time_hour_list_30d"]), map=int)),
            ),
            axis=1,
            result_type="expand"
        )
        if is_train:
            return x[self.__feature_names], data[[self.label_name]]
        else:
            return x[self.__feature_names]


__all__ = ["Model"]
