import scipy.io
from sklearn.preprocessing import MinMaxScaler
import os
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def load_multiview_dataset(mat_path, dataset, test_size=0.2, random_state=1):

    # 加载MAT文件
    mat_file = os.path.join(mat_path, dataset + '.mat')
    data = scipy.io.loadmat(mat_file)

    # 获取标签 (1×n 矩阵)
    gt = data['gt']
    if gt.shape[0] == 1:
        y = gt.flatten()  # 转换为 (n,) 数组
    else:
        y = gt.squeeze()

    # 获取多视图特征 (元胞数组)
    fea = data['fea']

    if hasattr(fea, 'shape'):  # 如果是numpy数组
        fea = fea.ravel().tolist()  # ravel()会自动展平
    elif isinstance(fea, list) and len(fea) > 0:
        # 处理嵌套列表情况
        if isinstance(fea[0], list):
            fea = [item for sublist in fea for item in sublist]
    else:
        fea = list(fea.flat) if hasattr(fea, 'flat') else list(fea)

    num_views = len(fea)

    # 获取样本数量
    n_samples = fea[0].shape[0]

    # 生成随机索引用于划分训练集和测试集
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_list = []
    train_X_list = []
    test_X_list = []

    for v in range(num_views):
        # 获取第v个视图的特征矩阵
        X_v = MinMaxScaler((0, 1)).fit_transform(fea[v])

        # 分割为训练集和测试集
        X_v_train = X_v[train_idx]
        X_v_test = X_v[test_idx]

        # 转换为PyTorch张量
        X_list.append(torch.FloatTensor(X_v))
        train_X_list.append(torch.FloatTensor(X_v_train))
        test_X_list.append(torch.FloatTensor(X_v_test))

    # 分割标签
    y_train = y[train_idx]
    y_test = y[test_idx]

    # 转换为标签张量

    train_y = torch.LongTensor(y_train)
    test_y = torch.LongTensor(y_test)

    print(f"Data split: {len(train_idx)} train samples, {len(test_idx)} test samples")

    return X_list, y, train_X_list, train_y, test_X_list, test_y, num_views, n_samples