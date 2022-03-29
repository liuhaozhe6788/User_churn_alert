import configparser
import os


class Config:
    def __init__(self, path: str = None):
        self.__path = self.__check_path(path)
        self.config = configparser.ConfigParser()
        self.config.read(self.__path)

    def __check_path(self, path: str) -> str:
        if path is None:
            return os.path.join(os.path.dirname(__file__), "config.ini")
        elif os.path.isdir(path):
            return os.path.join(path, "config.ini")
        else:
            return path

    def remove(self, section=None, option=None):
        if section is None:
            for section in self.config.sections():
                self.remove(section)
        elif option is None:
            self.config.remove_section(section)
        else:
            self.config.remove_option(section, option)

    def set(self, section, option, value):
        if section not in self.config.sections():
            self.config.add_section(section)
        self.config.set(section, option, value)

    def get(self, section=None, option=None):
        if section is None:
            return self.config.sections()
        elif section not in self.config.sections():
            return None
        elif option is None:
            return self.config.options(section)
        elif option not in self.config.options(section):
            return None
        else:
            return self.config.get(section, option)

    def save(self, path: str = None) -> None:
        with open(self.__path if path is None else path, 'w') as f:
            self.config.write(f)


if __name__ == "__main__":
    # 默认参数配置
    import datetime

    config = Config()
    config.config.remove_section("data_path")
    root_path = "D:/HOF老用户流失预警项目/数据 20211215 ~ 20220312"
    total_day = 88
    total_list = [(datetime.datetime.strptime("20211215", "%Y%m%d") + datetime.timedelta(i)).strftime("%Y%m%d") for i in
                  range(total_day)]
    train_list = total_list[:7]
    test_list = total_list[14:17]
    config.set("data_path", "root_path", root_path)
    config.set("data_path", "train_set", ",".join([f"fdate={fdate}/train_s4_{fdate}_v5.csv" for fdate in train_list]))
    config.set("data_path", "test_set", ",".join([f"fdate={fdate}/train_s4_{fdate}_v5.csv" for fdate in test_list]))

    config.config.remove_section("pickle_path")
    root_path = "D:/MyCodes/Python/CustomerChurnAlert/src/pickles"

    config.set("pickle_path", "root_path", root_path)
    config.save()
    print(config.config.options("data_path"))
