import pandas as pd


def under_sample(df: pd.DataFrame, label_name) -> pd.DataFrame:
    """根据 label_name 列对 df 欠采样\n"""
    grouped = df.groupby(label_name)
    return grouped.sample(min(len(v) for k, v in grouped))
