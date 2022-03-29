from datetime import datetime

import numpy as np

from utils.type_cast import to_list


def time_list_to_array(value: str, sep=",", _format="%Y-%m-%d %H:%M:%S") -> np.ndarray:
    """
    :param value: 一系列时间构成的字符串
    :param sep: 时间之间的分隔符
    :param _format: 时间的格式
    :return: 年月日时分秒矩阵
    """
    arr = np.array(to_list(to_list(value, sep=sep),
                           map=lambda s: to_list(
                               to_list(datetime.strptime(s, _format).strftime("%Y,%m,%d,%H,%M,%S")),
                               map=int)))
    return arr


def measure(values: np.ndarray, _range: tuple = None) -> tuple[int]:
    """
    :param values: 要统计的数据
    :param _range: 罗盘数据（时间等循环的数据）范围
    :return: 总和、均值、方差
    """
    sum = np.sum(values)
    if _range is None:
        mean = np.mean(values)
        std = np.std(values)
    else:
        mean = np.round(np.arctan(np.sin((values - _range[0]) * 2 * np.pi / (_range[1] - _range[0])).sum() /
                                  np.cos((values - _range[0]) * 2 * np.pi / (_range[1] - _range[0])).sum()
                                  ) / (2 * np.pi), 10) % 1 * (_range[1] - _range[0]) + _range[0]
        std = np.std((values + mean - (_range[1] - _range[0]) / 2) % (_range[1] - _range[0]))
    return sum, mean, std
