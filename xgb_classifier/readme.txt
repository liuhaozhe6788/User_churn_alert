configs/config.ini 记录了数据路径，运行模型时需要修改为实际地址
models 目录下是模型 class 文件，在设计新类时应该继承 BasicModel 类以便于存取
pickles 下保存训练好的模型
unittests 中是测试单元，用于模型效果和存取功能测试
utils 中是一些通用辅助函数（模型独有的辅助函数放在模型同一目录下）