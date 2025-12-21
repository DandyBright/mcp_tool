import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

def compare_svc_kernels(data_path: str, config_path: str, label_path: str) -> str:
    """
    SVC 核函数对比实验工具。
    1. 自动对比 Linear, RBF, Poly 等核函数。
    2. 找出最佳核函数。
    3. 计算最佳模型的特征重要性。
    4. 返回JSON 结果。

    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 SVC 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性和预测摘要。    
    """
    results = {
        "status": "failed",
        "message": "",
        "comparison_results": {},
        "best_kernel": "",
        "best_accuracy": 0.0,
        "feature_importance_best_model": {}  
    }

    try:
        # 1. 加载数据
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式")
            
        X = df.values
        feature_names = df.columns.tolist()
        
        # 2. 加载标签
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='gbk') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        # 3. 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
            
        kernels_to_test = config.get('kernels_to_test', ['linear', 'rbf'])
        common_params = config.get('common_params', {'C': 1.0, 'random_state': 42})
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})
        
        # 4. 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 5. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=train_params.get('test_size', 0.2), 
            random_state=train_params.get('random_state', 42)
        )
        
        comparison_dict = {}
        best_acc = -1.0
        best_k = ""
        best_model_instance = None # 用于存储最佳模型对象
        
        # 6. 循环对比实验
        for kernel in kernels_to_test:
            try:
                model = SVC(kernel=kernel, **common_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                comparison_dict[kernel] = {
                    "accuracy": round(float(acc), 4),
                    "precision_weighted": round(classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'], 4),
                    "recall_weighted": round(classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'], 4)
                }
                
                # 更新最佳记录
                if acc > best_acc:
                    best_acc = acc
                    best_k = kernel
                    best_model_instance = model # 保存这个模型对象，后面算特征重要性用
                    
            except Exception as e_k:
                comparison_dict[kernel] = {"error": str(e_k)}

        # 7. 计算最佳模型的特征重要性
        sorted_features = {}
        if best_model_instance is not None:
            # 使用 Permutation Importance (排列重要性)，这对 Linear/RBF/Poly 都通用
            # n_repeats=5 既能保证一定准确性，又不会太慢
            perm_importance = permutation_importance(
                best_model_instance, X_test, y_test, 
                n_repeats=5, random_state=42
            )
            importances = perm_importance.importances_mean
            
            # 映射特征名
            feature_imp_dict = dict(zip(feature_names, [float(i) for i in importances]))
            # 过滤负值并取 Top 10
            feature_imp_dict = {k: max(0.0, v) for k, v in feature_imp_dict.items()}
            sorted_features = dict(sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True)[:10])

        # 8. 构造结果
        results["status"] = "success"
        results["message"] = f"对比完成。最佳核函数: {best_k} (ACC: {best_acc:.4f})"
        results["comparison_results"] = comparison_dict
        results["best_kernel"] = best_k
        results["best_accuracy"] = round(float(best_acc), 4)
        results["feature_importance_best_model"] = sorted_features 

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=True, indent=2)
