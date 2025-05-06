import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from MPAGAFS import jfs
import warnings
import os
from glob import glob
from tqdm import tqdm
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=RuntimeWarning)

def evaluate_performance(ytrue, ypred):
    acc = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred, average='macro')
    return acc, f1


dataset_paths = glob('dataset/*.csv')
print(f"找到 {len(dataset_paths)} 个数据集:")
for path in dataset_paths:
    print(f"- {os.path.basename(path)}")

results = {
    'dataset': [],
    'run_id': [],
    'accuracy': [],
    'f1_score': [],
    'num_features': [],
    'feature_ratio': [],
    'run_time': []
}

for data_path in tqdm(dataset_paths, desc="处理数据集"):
    try:
        data = pd.read_csv(data_path)
        data = data.values
        feat = np.asarray(data[:, 0:-1])
        label = np.asarray(data[:, -1])

        le = LabelEncoder()
        label = le.fit_transform(label)

        for i in range(1, 6):
            start_time = time.time()

            # 划分训练测试集
            xtrain, xtest, ytrain, ytest = train_test_split(
                feat, label,
                test_size=0.2,
                random_state=i,
                stratify=label
            )

            # 准备参数
            fold = {
                'xt': xtrain,
                'yt': ytrain,
                'xv': xtest,
                'yv': ytest
            }

            opts = {
                'N': 30,  # 种群数量
                'T': 50,  # 迭代次数
                'M': 6,  # 子种群数量
                'dim': xtrain.shape[1],  # 特征维度
                'k': 5,  # kNN中的k
                'fold': fold  # 验证集结构
            }

            result = jfs(xtrain, ytrain, opts)

            sf = result['sf']
            num_feats = len(sf)
            feat_ratio = num_feats / xtrain.shape[1]

            x_train_selected = xtrain[:, sf]
            x_test_selected = xtest[:, sf]

            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(x_train_selected, ytrain)
            ypred = model.predict(x_test_selected)

            acc, f1 = evaluate_performance(ytest, ypred)

            run_time = time.time() - start_time
            results['dataset'].append(os.path.basename(data_path).split('.')[0])
            results['run_id'].append(i)
            results['accuracy'].append(acc)
            results['f1_score'].append(f1)
            results['num_features'].append(num_feats)
            results['feature_ratio'].append(feat_ratio)
            results['run_time'].append(run_time)

    except Exception as e:
        print(f"处理数据集 {os.path.basename(data_path)} 时出错: {str(e)}")
        continue

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 计算汇总统计量
summary_df = results_df.groupby('dataset').agg({
    'accuracy': ['mean', 'std', 'max', 'min'],
    'f1_score': ['mean', 'std', 'max', 'min'],
    'num_features': 'mean',
    'feature_ratio': 'mean',
    'run_time': 'mean'
}).reset_index()

# 重命名列名
summary_df.columns = [
    'dataset',
    'avg_acc', 'std_acc', 'best_acc', 'worst_acc',
    'avg_f1', 'std_f1', 'best_f1', 'worst_f1',
    'avg_num_feats', 'avg_feat_ratio', 'avg_run_time'
]

# 保存详细结果和汇总结果
detailed_output = "detailed_results.csv"
summary_output = "summary_results.csv"

results_df.to_csv(detailed_output, index=False)
summary_df.to_csv(summary_output, index=False)

print(f"\n实验完成! 详细结果已保存到 {detailed_output}")
print(f"汇总结果已保存到 {summary_output}")