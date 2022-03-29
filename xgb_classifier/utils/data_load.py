import pandas as pd


def load_data(paths: list[str], columns: list[str], sep='|') -> pd.DataFrame:
    """读取数据
    :param paths: csv 文件地址
    :param columns: 要提取的列名
    :param sep: csv 文件内的分隔符
    :return: 提取并合并后的数据
    """
    if isinstance(paths, str):
        paths = [paths]

    df_list = []
    for path in paths:
        print(path, end="...")
        try:
            df_list.append(pd.read_csv(path, sep=sep, usecols=columns))
        except:
            print("读取失败")
        else:
            print("读取成功")
    return None if len(df_list) == 0 else pd.concat(df_list, axis=0, ignore_index=True)


if __name__ == "__main__":
    import os
    import configs
    from utils.type_cast import *

    config = configs.Config()
    root = config.get("data_path", "root_path")
    paths = to_list(to_list(config.get("data_path", "train_set")), map=lambda filename: os.path.join(root, filename))
    columns = [
        "is_churned",
    ]  # 流失标签
    columns += [
        "uid",
        "life_time",
        "login_day_cnt",
        "register_country",
    ]  # 基本信息
    columns += [
        "login_time",
        "login_time_hour_list_7d",
        "login_time_hour_list_30d",
        "login_time_list_7d",
        "login_time_list_30d",
        "time_zone",
    ]  # 登录习惯
    columns += [
        "grade",
        "last_season_gap_day",
        "battlepass_lvl",
    ]  # 游戏属性
    columns += [
        "curr_asset",
        "curr_diamond",
        "max_chip_stock",
        "max_blind",
        "bankrupt_cnt_3d",
        "bankrupt_cnt",
    ]  # 资产信息
    columns += [
        "pay_amt",
        "pay_cnt",
        "recency",
        *[f"{action}_{count}d" for action in ["pay", "pay_cnt"] for count in [3, 7, 15, 30]],
    ]  # 付费行为
    columns += [
        "ads_watch_cnt_1d",
        "ads_click_cnt_1d",
        *[f"{action}_{count}d" for action in ["ads_watch_cnt"] for count in [3, 7, 15, 30]],
    ]  # 广告行为
    columns += [
        "login_cnt_3d",
        "login_cnt_7d",
        "login_cnt_15d",
        "login_cnt_30d",
    ]  # 登陆行为
    data = load_data(paths[0], columns)
