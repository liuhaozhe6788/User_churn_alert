import os
import unittest
import warnings

from configs import Config
from models import xgb, BasicModel
from utils.data_load import load_data
from utils.type_cast import to_list

warnings.filterwarnings("ignore")


class XGBTest(unittest.TestCase):
    def setUp(self) -> None:
        # 不是必须通过 configs.Config 和 utils.data_load.load_data获取文件路径，可以通过其他方式，只要保证数据本身没有问题就行
        config = Config()
        data_path = {option: config.get("data_path", option) for option in config.get("data_path")}

        self.feature_names = xgb.Model().feature_names
        self.label_name = xgb.Model().label_name
        self.train_data = load_data(
            [os.path.join(data_path["root_path"], filename) for filename in to_list(data_path["train_set"])],
            columns=self.feature_names + [self.label_name])
        self.test_data = load_data(
            [os.path.join(data_path["root_path"], filename) for filename in to_list(data_path["test_set"])],
            columns=self.feature_names + [self.label_name])
        self.pickle_path = os.path.join(config.get("pickle_path", "root_path"), "xgb.pkl")

    def test_train_with_eval_and_test(self):
        model = xgb.Model()
        results = model.train(self.train_data, with_eval=True)
        model.plot()

        def dict2str(dictionary, deep=0):
            if isinstance(dictionary, dict):
                return "".join([f"\n{'  ' * deep}{n}:{dict2str(v, deep + 1)}" for n, v in dictionary.items()])
            else:
                return "{}".format(dictionary)

        results["test_set"] = model.evaluate(model.test(self.test_data), self.test_data[self.label_name])
        with open("results.txt", 'w') as f:
            f.write(dict2str(results)[1:])

    def test_train_and_save_and_predict(self):
        import numpy as np
        indices = np.random.choice(len(self.test_data), 100)
        model1 = xgb.Model()
        model1.train(self.train_data)
        pred1 = [model1.predict(self.test_data[i:i + 1]) for i in indices]
        model1.save(self.pickle_path)
        """
        上面是训练与保存，下面是加载并验证
        """
        model2 = BasicModel.load(self.pickle_path)
        pred2 = [model2.predict(self.test_data[i:i + 1]) for i in indices]
        self.assertEqual(pred1, pred2)

    # def test_load_and_test(self):
    #     model = BasicModel.load(self.pickle_path)
    #     # test 函数是用来测试的，如果只要预测一行数据，用 predict 函数就可以
    #     results = {
    #         "train_set": model.evaluate(model.test(self.train_data), self.train_data[self.label_name]),
    #         "test_set": model.evaluate(model.test(self.test_data), self.test_data[self.label_name]),
    #     }
    #
    #     def dict2str(dictionary, deep=0):
    #         if isinstance(dictionary, dict):
    #             return "".join([f"\n{'  ' * deep}{n}:{dict2str(v, deep + 1)}" for n, v in dictionary.items()])
    #         else:
    #             return "{}".format(dictionary)
    #     print(dict2str(results)[1:])


if __name__ == '__main__':
    unittest.main()
