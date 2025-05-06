import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# 对选择的特征进行错误率计算
def error_rate(xtrain, ytrain, x, opts):
    # 超参数
    k = opts['k']  # KNN的超参数
    fold = opts['fold']  # 数据集参数
    xt = fold['xt']  # 训练数据
    yt = fold['yt']  # 训练标签
    xv = fold['xv']  # 验证数据
    yv = fold['yv']  # 验证标签

    num_train = np.size(xt, 0)  # 训练数据的数量
    num_valid = np.size(xv, 0)  # 验证数据的数量

    xtrain = xt[:, x == 1]  # 选择特征的训练数据
    ytrain = yt.reshape(num_train)
    xvalid = xv[:, x == 1]  # 选择特征的验证数据
    yvalid = yv.reshape(num_valid)

    # 调用KNN模型进行训练
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(xtrain, ytrain)

    # 使用训练好的模型进行预测
    ypred = mdl.predict(xvalid)
    acc = np.sum(yvalid == ypred) / num_valid  # 计算准确率
    error = 1 - acc  # 计算错误率

    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    """
    适应度计算函数
    :param xtrain:  训练数据
    :param ytrain:  训练标签
    :param x:   特征选择向量，传进来的都是一个个体
    :param opts:    参数字典
    :return:    个体的适应度值
    """
    # 超参数
    alpha = 0.99
    beta = 1 - alpha
    # 数据集的特征数量
    max_feat = len(x)
    # 选择的特征个数
    num_feat = np.sum(x == 1)
    # 如果没有特征被选择
    if num_feat == 0:
        cost = 1
    else:
        # 调用错误率计算函数
        error = error_rate(xtrain, ytrain, x, opts)
        # 适应度计算公式，可自行定义
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost
