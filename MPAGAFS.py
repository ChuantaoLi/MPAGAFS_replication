import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def error_rate(xtrain, ytrain, x, opts):
    k = opts['k']
    fold = opts['fold']

    # 获取训练集和验证集
    xt = fold['xt']  # 训练特征
    yt = fold['yt']  # 训练标签
    xv = fold['xv']  # 验证特征
    yv = fold['yv']  # 验证标签

    # 获取样本数量
    num_train = np.size(xt, 0)  # 训练样本数
    num_valid = np.size(xv, 0)  # 验证样本数

    # 根据特征选择向量x选择特征
    xtrain = xt[:, x == 1]  # 选择特征值为1的特征
    ytrain = yt.reshape(num_train)  # 确保标签是一维数组
    xvalid = xv[:, x == 1]  # 对验证集做同样的特征选择
    yvalid = yv.reshape(num_valid)  # 验证标签

    # 创建KNN分类器并训练
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(xtrain, ytrain)

    # 预测并计算准确率
    ypred = mdl.predict(xvalid)
    acc = np.sum(yvalid == ypred) / num_valid  # 计算准确率
    error = 1 - acc  # 错误率=1-准确率
    return error


def Fun(xtrain, ytrain, x, opts):
    # 设置权重参数
    alpha = 0.99  # 分类错误率权重
    beta = 1 - alpha  # 特征数量权重
    max_feat = len(x)  # 最大特征数
    num_feat = np.sum(x == 1)  # 当前选择的特征数

    # 如果没有选择任何特征，则返回最差适应度
    if num_feat == 0:
        cost = 1
    else:
        # 计算分类错误率
        error = error_rate(xtrain, ytrain, x, opts)
        # 适应度是错误率和特征数量的加权和
        cost = alpha * error + beta * (num_feat / max_feat)
    return cost


def binary_competition(Li, Lj):
    dim = len(Li)  # 特征维度
    L_new = np.copy(Li)  # 复制第一个个体作为基础

    # 遍历每个特征位
    for j in range(dim):
        # 如果两个个体在该位不同，则随机选择一个
        if Li[j] != Lj[j]:
            L_new[j] = Lj[j] if np.random.rand() < 0.5 else Li[j]
    return L_new


def jfs(xtrain, ytrain, opts):
    # 从参数中获取算法参数
    dim = opts['dim']  # 特征维度
    N = opts['N']  # 每个子种群大小
    T = opts['T']  # 最大迭代次数
    M = opts['M']  # 子种群数量

    # 初始化种群：M个子种群，每个子种群N个个体，每个个体是dim维二进制向量
    X = np.random.randint(0, 2, (M, N, dim))

    # 初始化适应度矩阵
    fit = np.zeros((M, N))

    # 计算初始适应度
    for i in range(M):  # 遍历每个子种群
        for j in range(N):  # 遍历子种群中的每个个体
            fit[i, j] = Fun(xtrain, ytrain, X[i, j], opts)

    # 跟踪全局最优解
    best_fit = np.min(fit)  # 全局最优适应度
    best_sol = X[np.unravel_index(np.argmin(fit), fit.shape)]  # 全局最优解
    curve = np.zeros(T)  # 记录每次迭代的最优适应度

    # 开始迭代
    for t in range(T):
        # 创建当前适应度的副本用于比较
        prev_fit = np.copy(fit)

        # 第一阶段：动态邻域竞争
        for i in range(M):  # 遍历每个子种群
            for j in range(N):  # 遍历子种群中的每个个体
                current = X[i, j]  # 当前个体
                f_current = fit[i, j]  # 当前适应度

                # 获取邻域个体(环形拓扑)
                idx1 = (j - 1) % N  # 左邻居索引
                idx2 = (j + 1) % N  # 右邻居索引
                nb1 = X[i, idx1]  # 左邻居
                nb2 = X[i, idx2]  # 右邻居
                f1 = fit[i, idx1]  # 左邻居适应度
                f2 = fit[i, idx2]  # 右邻居适应度

                # 如果邻居中有更好的解，则进行竞争
                if f1 < f_current or f2 < f_current:
                    # 选择适应度更好的邻居作为竞争者
                    competitor = nb1 if f1 < f2 else nb2
                    # 执行二进制竞争操作
                    new_sol = binary_competition(current, competitor)
                    # 计算新解的适应度
                    new_fit = Fun(xtrain, ytrain, new_sol, opts)

                    # 只有在新解更好时才接受
                    if new_fit < f_current:
                        X[i, j] = new_sol
                        fit[i, j] = new_fit

        # 共享代理
        elite_pool = []
        elite_fitness = []

        for i in range(M):  # 遍历每个子种群
            # 找到当前子种群的最优个体
            best_idx = np.argmin(fit[i])
            elite_pool.append(X[i, best_idx].copy())
            elite_fitness.append(fit[i, best_idx])

        # 打乱精英池中的个体顺序（实现随机分配）
        indices = np.arange(M)
        np.random.shuffle(indices)
        shuffled_elites = [elite_pool[i] for i in indices]
        shuffled_fitness = [elite_fitness[i] for i in indices]

        # 用打乱后的精英个体替换各子种群的最差个体
        for i in range(M):
            # 找到当前子种群的最差个体
            worst_idx = np.argmax(fit[i])

            # 确保不替换当前子种群自己的精英
            if fit[i, worst_idx] > shuffled_fitness[i]:
                # 替换最差个体
                X[i, worst_idx] = shuffled_elites[i]
                fit[i, worst_idx] = shuffled_fitness[i]

        # 第二阶段：自适应交叉和变异
        for i in range(M):  # 遍历每个子种群
            f_max = np.max(fit[i])  # 子种群最大适应度
            f_ave = np.mean(fit[i])  # 子种群平均适应度
            f_min = np.min(fit[i])  # 子种群最小适应度

            # 找到子种群中的最优个体
            best_idx = np.argmin(fit[i])
            best_in_subpop = X[i, best_idx]

            for j in range(N):  # 遍历子种群中的每个个体
                # 跳过最优个体(保持精英)
                if j == best_idx:
                    continue

                ind = X[i, j]  # 当前个体
                distance = np.sum(ind != best_in_subpop)  # 与最优个体的距离

                # 计算自适应交叉概率
                if f_max > f_ave:
                    # 适应度越差、距离越近，交叉概率越高
                    Pc = ((fit[i, j] - f_min) / (f_max - f_min + 1e-10)) * np.exp(-distance / dim)
                else:
                    Pc = 1  # 如果种群收敛，则提高交叉概率

                # 执行交叉操作
                if np.random.rand() < Pc:
                    # 随机选择交叉点
                    cross_point = np.random.randint(1, dim - 1)
                    # 生成试验个体：前半部分来自当前个体，后半部分来自最优个体
                    trial = np.concatenate([ind[:cross_point], best_in_subpop[cross_point:]])
                    # 计算试验个体的适应度
                    trial_fit = Fun(xtrain, ytrain, trial, opts)

                    # 贪婪选择：只有试验个体更好时才接受
                    if trial_fit < fit[i, j]:
                        X[i, j] = trial
                        fit[i, j] = trial_fit

                # 自适应变异
                pm = 1 / dim  # 基本变异概率
                # 随机选择要变异的位
                mut_pos = np.random.rand(dim) < pm
                if np.any(mut_pos):  # 如果有位被选中变异
                    mutant = X[i, j].copy()  # 复制当前个体
                    mutant[mut_pos] = 1 - mutant[mut_pos]  # 翻转选中的位
                    # 计算变异后的适应度
                    mutant_fit = Fun(xtrain, ytrain, mutant, opts)

                    # 贪婪选择：只有变异后更好时才接受
                    if mutant_fit < fit[i, j]:
                        X[i, j] = mutant
                        fit[i, j] = mutant_fit

        # 更新全局最优解
        current_min = np.min(fit)
        if current_min < best_fit:
            best_fit = current_min
            best_sol = X[np.unravel_index(np.argmin(fit), fit.shape)]

        # 记录当前迭代的最优适应度
        curve[t] = best_fit
        print(f"Iteration {t + 1}/{T}, Best Fitness: {best_fit:.4f}")

    # 获取最终选择的特征索引
    sf = np.where(best_sol == 1)[0]

    return {
        'sf': sf,  # 选择的特征索引
        'nf': len(sf),  # 选择的特征数量
        'c': curve  # 适应度曲线
    }
