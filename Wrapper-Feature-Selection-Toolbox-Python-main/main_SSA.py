import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from ssa import jfs  # 修改导入模块
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 创建结果保存目录
result_dir = f"SSA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(result_dir, exist_ok=True)

# 初始化总结果记录
summary = []

# 运行10次实验
for run in range(10):
    print(f"\n{'=' * 40}")
    print(f"Running Experiment {run + 1}/10")
    print(f"{'=' * 40}")

    # 初始化单次实验记录
    run_record = {
        'run': run + 1,
        'start_time': time.perf_counter(),
        'accuracy': None,
        'recall': None,
        'f1_score': None,
        'precision': None,
        'selected_features': None,
        'num_selected_features': None,
        'convergence': [],
        'duration': None
    }

    # 实验流程
    try:
        # 加载数据并转换标签
        data = pd.read_csv(r'D:\MPCDCGA\dataset\Period Changer.csv').values
        feat = data[:, 0:-1]
        label = LabelEncoder().fit_transform(data[:, -1])

        # 分割数据集
        xtrain, xtest, ytrain, ytest = train_test_split(
            feat, label, test_size=0.2, stratify=label, random_state=run
        )
        fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

        # SSA算法参数设置 ★关键修改点★
        opts = {
            'N': 60,  # 樽海鞘群数量
            'T': 50,  # 最大迭代次数
            'k': 5,  # KNN参数
            'fold': fold  # 数据分割
        }

        # 特征选择
        fmdl = jfs(xtrain, ytrain, opts)
        sf = fmdl['sf']  # 获取选择的特征索引
        run_record['selected_features'] = sf.tolist()
        run_record['num_selected_features'] = fmdl['nf']

        # 保存收敛数据
        convergence_data = fmdl['c'].ravel()
        np.savetxt(f"{result_dir}/convergence_run_{run + 1}.csv",
                   convergence_data, delimiter=",")
        run_record['convergence'] = convergence_data.tolist()

        # 模型训练与评估
        mdl = KNeighborsClassifier(n_neighbors=opts['k'])
        mdl.fit(xtrain[:, sf], ytrain)
        y_pred = mdl.predict(xtest[:, sf])

        # 计算评估指标
        run_record['accuracy'] = accuracy_score(ytest, y_pred)
        run_record['recall'] = recall_score(ytest, y_pred, average='macro')  # 修改为 'macro'
        run_record['f1_score'] = f1_score(ytest, y_pred, average='macro')  # 修改为 'macro'
        run_record['precision'] = precision_score(ytest, y_pred, average='macro')  # 修改为 'macro'

        # 记录运行时间
        run_record['duration'] = time.perf_counter() - run_record['start_time']

        # 保存收敛曲线
        plt.figure(figsize=(14, 8.6))
        plt.plot(convergence_data, 'm-', linewidth=2)  # 洋红色曲线
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title(f'SSA Convergence - Run {run + 1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{result_dir}/convergence_run_{run + 1}.png",
                    dpi=600, bbox_inches='tight')
        plt.close()

        # 打印结果
        print(f"\nRun {run + 1} Results:")
        print(f"Accuracy: {run_record['accuracy'] * 100:.2f}%")
        print(f"Recall: {run_record['recall'] * 100:.2f}%")
        print(f"F1 Score: {run_record['f1_score'] * 100:.2f}%")
        print(f"Precision: {run_record['precision'] * 100:.2f}%")
        print(f"Selected Features: {run_record['selected_features']}")
        print(f"Number of Selected Features: {run_record['num_selected_features']}")
        print(f"Duration: {run_record['duration']:.2f} seconds")

    except Exception as e:
        print(f"Run {run + 1} Failed: {str(e)}")
        continue

    # 保存实验记录
    summary.append(run_record)

# 生成综合报告
df_summary = pd.DataFrame(summary)
df_summary.to_csv(f"{result_dir}/experiment_summary_SSA.csv", index=False)

print(f"\nAll experiments completed! Results saved to: {result_dir}")
