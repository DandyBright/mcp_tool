import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone


def reliefF_multiclass_feature_selection(data_path: str, config_path: str, label_path: str) -> str:
    """
    ReliefF 多类别特征选择算法模块。
    实现 ReliefF 算法，支持多类别问题，返回特征权重和分类性能。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 ReliefF 参数和分类器设置。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含特征权重、选择特征、分类性能等。
    """
    
    # 初始化结果字典
    results = {
        "status": "failed",  # 执行状态
        "message": "",  # 执行消息
        "reliefF_params": {},  # ReliefF 算法参数
        "feature_weights": {},  # 特征权重（原始值）
        "feature_weights_normalized": {},  # 归一化后的特征权重
        "selected_features": [],  # 选择的特征列表
        "classifier_results": {},  # 分类器性能结果
        "reduction_summary": {}  # 特征选择摘要
    }

    try:
        # 1. 加载特征数据
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式，请提供 .csv 或 .xlsx/.xls 文件")
            
        X = df.values  # 特征矩阵
        feature_names = df.columns.tolist()  # 原始特征名称
        
        # 2. 加载和编码标签数据
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='gbk') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        
        le = LabelEncoder()
        y = le.fit_transform(labels)  # 数值标签
        class_names = le.classes_.tolist()  # 类别名称
        n_classes = len(class_names)  # 类别数量
        
        # 3. 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        
        # 提取配置参数
        reliefF_params = config.get('reliefF_params', {})  # ReliefF 参数
        classifier_config = config.get('classifier_params', {})  # 分类器配置
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})  # 训练参数
        
        # ReliefF 参数
        k_nearest = reliefF_params.get('k_nearest', 10)  # 最近邻数量
        n_iterations = reliefF_params.get('n_iterations', 100)  # 迭代次数
        selection_method = reliefF_params.get('selection_method', 'threshold')  # 特征选择方法
        threshold = reliefF_params.get('threshold', 0.0)  # 权重阈值
        n_features_to_select = reliefF_params.get('n_features_to_select', 10)  # 选择特征数量
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2),
            random_state=train_params.get('random_state', 42),
            stratify=y  # 保持类别分布
        )
        
        n_samples, n_features = X_train.shape  # 训练样本数和特征数
        
        # 5. 实现 ReliefF 算法
        def reliefF_algorithm(X, y, k, n_iter):
            """
            ReliefF 算法实现，支持多类别
            
            Args:
                X: 特征矩阵 (n_samples, n_features)
                y: 标签向量 (n_samples,)
                k: 最近邻数量
                n_iter: 迭代次数
                
            Returns:
                特征权重向量 (n_features,)
            """
            n_samples, n_features = X.shape
            weights = np.zeros(n_features)  # 初始化权重向量
            
            # 获取每个类别的样本索引
            class_indices = {}
            for class_label in np.unique(y):
                class_indices[class_label] = np.where(y == class_label)[0]
            
            # 随机选择迭代样本
            sample_indices = np.random.choice(n_samples, min(n_iter, n_samples), replace=False)
            
            for idx in sample_indices:
                current_sample = X[idx]
                current_label = y[idx]
                
                # 计算当前样本与所有样本的距离
                distances = np.sqrt(np.sum((X - current_sample) ** 2, axis=1))
                distances[idx] = np.inf  # 排除自身
                
                # 寻找最近邻：同类最近邻和异类最近邻
                same_class_mask = (y == current_label)
                diff_class_mask = ~same_class_mask
                
                # 同类最近邻
                same_class_distances = distances.copy()
                same_class_distances[~same_class_mask] = np.inf
                same_class_neighbors = np.argsort(same_class_distances)[:k]
                
                # 异类最近邻（每个类别k个）
                diff_class_neighbors = []
                for class_label in np.unique(y):
                    if class_label != current_label:
                        class_mask = (y == class_label)
                        class_distances = distances.copy()
                        class_distances[~class_mask] = np.inf
                        class_neighbors = np.argsort(class_distances)[:k]
                        diff_class_neighbors.extend(class_neighbors)
                
                # 更新权重
                for feature_idx in range(n_features):
                    feature_value = current_sample[feature_idx]
                    
                    # 计算与同类最近邻的平均差异
                    if len(same_class_neighbors) > 0:
                        same_class_diff = np.abs(feature_value - X[same_class_neighbors, feature_idx])
                        avg_same_diff = np.mean(same_class_diff)
                    else:
                        avg_same_diff = 0
                    
                    # 计算与异类最近邻的平均差异
                    if len(diff_class_neighbors) > 0:
                        diff_class_diff = np.abs(feature_value - X[diff_class_neighbors, feature_idx])
                        avg_diff_diff = np.mean(diff_class_diff)
                    else:
                        avg_diff_diff = 0
                    
                    # ReliefF 权重更新公式
                    p_class = len(class_indices[current_label]) / n_samples  # 当前类别概率
                    
                    # 计算其他类别的权重贡献
                    other_class_contrib = 0
                    for class_label in np.unique(y):
                        if class_label != current_label:
                            class_mask = (y == class_label)
                            p_other = len(class_indices[class_label]) / n_samples
                            class_distances = distances.copy()
                            class_distances[~class_mask] = np.inf
                            class_neighbors = np.argsort(class_distances)[:k]
                            
                            if len(class_neighbors) > 0:
                                other_diff = np.abs(feature_value - X[class_neighbors, feature_idx])
                                avg_other_diff = np.mean(other_diff)
                                other_class_contrib += (p_other / (1 - p_class)) * avg_other_diff
                    
                    # 更新权重
                    weights[feature_idx] -= avg_same_diff / (n_iter * k)
                    weights[feature_idx] += other_class_contrib / (n_iter * k)
            
            return weights
        
        # 6. 运行 ReliefF 算法
        feature_weights = reliefF_algorithm(X_train, y_train, k_nearest, n_iterations)
        
        # 归一化权重到 [0, 1] 范围
        if np.max(feature_weights) > np.min(feature_weights):
            weights_normalized = (feature_weights - np.min(feature_weights)) / (np.max(feature_weights) - np.min(feature_weights))
        else:
            weights_normalized = np.ones_like(feature_weights)
        
        # 7. 特征选择
        if selection_method == 'threshold':
            # 基于阈值选择特征
            selected_mask = weights_normalized >= threshold
            selected_indices = np.where(selected_mask)[0]
        elif selection_method == 'top_k':
            # 选择权重最高的k个特征
            n_select = min(n_features_to_select, n_features)
            selected_indices = np.argsort(weights_normalized)[-n_select:][::-1]
        else:
            # 默认使用阈值方法
            selected_mask = weights_normalized >= threshold
            selected_indices = np.where(selected_mask)[0]
        
        # 获取选择的特征名称
        selected_feature_names = [feature_names[i] for i in selected_indices]
        selected_weights = weights_normalized[selected_indices].tolist()
        
        # 8. 使用选择的特征进行分类评估（可选）
        if classifier_config.get('enabled', True):
            # 提取选择的特征
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
            
            # 配置分类器
            classifier_type = classifier_config.get('classifier', 'knn')
            
            if classifier_type.lower() == 'knn':
                base_clf = KNeighborsClassifier()
                param_grid = classifier_config.get('param_grid', {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                })
                
                # 网格搜索
                grid_search = GridSearchCV(
                    estimator=base_clf,
                    param_grid=param_grid,
                    cv=classifier_config.get('cv', 5),
                    scoring=classifier_config.get('scoring', 'accuracy'),
                    n_jobs=1,
                    verbose=0
                )
                
                grid_search.fit(X_train_selected, y_train)
                best_clf = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_cv_score = grid_search.best_score_
                
                # 预测和评估
                y_pred = best_clf.predict(X_test_selected)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # 保存分类器结果
                results["classifier_results"] = {
                    "classifier_type": classifier_type,
                    "best_params": best_params,
                    "best_cv_score": round(float(best_cv_score), 4),
                    "test_accuracy": round(float(accuracy), 4),
                    "classification_report": report
                }
        
        # 9. 构建特征权重字典（按权重排序）
        sorted_indices = np.argsort(weights_normalized)[::-1]
        feature_weights_dict = {
            feature_names[i]: {
                "raw_weight": round(float(feature_weights[i]), 6),
                "normalized_weight": round(float(weights_normalized[i]), 4)
            } for i in sorted_indices
        }
        
        # 10. 构建结果
        results["status"] = "success"
        results["message"] = "ReliefF 特征选择完成"
        results["reliefF_params"] = {
            "k_nearest": k_nearest,
            "n_iterations": n_iterations,
            "selection_method": selection_method,
            "threshold": threshold,
            "n_features_to_select": n_features_to_select
        }
        results["feature_weights"] = feature_weights_dict
        results["selected_features"] = [
            {
                "name": name,
                "weight": weight
            } for name, weight in zip(selected_feature_names, selected_weights)
        ]
        results["reduction_summary"] = {
            "original_features": n_features,
            "selected_features": len(selected_feature_names),
            "reduction_ratio": round(1 - len(selected_feature_names) / n_features, 4),
            "average_weight": round(float(np.mean(weights_normalized[selected_indices])), 4),
            "max_weight": round(float(np.max(weights_normalized)), 4),
            "min_weight": round(float(np.min(weights_normalized)), 4)
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=True, indent=2)