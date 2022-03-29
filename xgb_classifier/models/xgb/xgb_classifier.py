import xgboost


class XGBClassifier(xgboost.XGBClassifier):
    def __init__(self, pos_wight=1.1):
        super().__init__(
            learning_rate=0.3,  # 如同学习率
            min_child_weight=1,
            max_depth=6,  # 构建树的深度，越大越容易过拟合
            gamma=0.2,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
            subsample=0.8,  # 随机采样训练样本 训练实例的子采样比
            max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
            colsample_bytree=1,  # 生成树时进行的列采样
            reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            # reg_alpha=0, # L1 正则项参数
            # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
            # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
            # num_class=10, # 类别数，多分类与 multisoftmax 并用
            n_estimators=100,  # 树的个数
            seed=0,  # 随机种子
            use_label_encoder=False,
            objective='binary:logistic',
        )
        super(XGBClassifier, self).__init__()
        self.pos_wight = pos_wight

    def fit(self, X, y, **kwargs):
        count = y.value_counts()
        weight = self.pos_wight * count[0] / count[1]
        self.set_params(scale_pos_weight=weight)
        super().fit(X, y, eval_metric='auc', **kwargs)
