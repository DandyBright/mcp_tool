import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


def lda_supervised_dimension_reduction(data_path: str, config_path: str, label_path: str) -> str:
    """
    LDA 监督降维（多类支持）算法模块。
    返回 JSON 格式的降维结果、模型性能和分类指标。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 LDA 模型参数和训练设置。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含降维信息、模型指标和降维数据统计。
    """
    
    # 初始化结果字典，包含所有可能的返回字段
    results = {
        "status": "failed",  # 执行状态
        "message": "",  # 执行消息
        "lda_params": {},  # LDA 模型参数
        "reduction_info": {},  # 降维信息
        "metrics": {},  # 模型性能指标
        "data_statistics": {},  # 降维后数据统计
        "class_distribution": {}  # 类别分布信息
    }

    try:
        # 1. 加载特征数据
        # 根据文件后缀选择对应的读取方式
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')  # 尝试 UTF-8 编码
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')  # 如果失败，尝试 GBK 编码
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)  # 读取 Excel 文件
        else:
            raise ValueError("不支持的数据格式，请提供 .csv 或 .xlsx/.xls 文件")
            
        # 提取特征矩阵和特征名称
        X = df.values  # 将 DataFrame 转换为 numpy 数组
        feature_names = df.columns.tolist()  # 获取原始特征名称
        
        # 2. 加载和编码标签数据
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='gbk') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        
        # 使用 LabelEncoder 将字符串标签转换为数值
        le = LabelEncoder()
        y = le.fit_transform(labels)  # 转换后的数值标签
        class_names = le.classes_.tolist()  # 获取原始类别名称
        
        # 3. 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)  # 安全加载 YAML 配置
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        
        # 提取配置参数
        lda_params = config.get('lda_params', {})  # LDA 模型参数
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})  # 训练参数
        n_components = lda_params.get('n_components', None)  # 降维维度，None 表示自动选择
        
        # 4. 数据划分
        # 将数据划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2),  # 测试集比例
            random_state=train_params.get('random_state', 42),  # 随机种子
            stratify=y  # 保持类别分布
        )
        
        # 5. 计算最大降维维度
        # LDA 最多可降至 min(特征数, 类别数-1) 维
        n_classes = len(np.unique(y_train))  # 类别数量
        n_features = X_train.shape[1]  # 原始特征数量
        max_components = min(n_features, n_classes - 1)  # 理论最大降维维度
        
        # 调整 n_components，避免超过最大维度
        if n_components is None:
            n_components = max_components  # 使用最大维度
        elif n_components > max_components:
            n_components = max_components  # 限制为最大维度
        
        # 6. 创建和训练 LDA 模型
        lda = LinearDiscriminantAnalysis(
            n_components=n_components,  # 降维维度
            solver=lda_params.get('solver', 'svd'),  # 求解器，默认 SVD
            shrinkage=lda_params.get('shrinkage', None),  # 收缩参数
            priors=lda_params.get('priors', None),  # 先验概率
            store_covariance=lda_params.get('store_covariance', False)  # 是否存储协方差矩阵
        )
        
        # 在训练集上拟合 LDA 模型
        X_train_reduced = lda.fit_transform(X_train, y_train)  # 训练并降维训练集
        X_test_reduced = lda.transform(X_test)  # 降维测试集
        
        # 7. 使用降维后的数据进行分类（使用 LDA 内置的分类功能）
        y_pred = lda.predict(X_test)  # 在原始特征空间预测
        y_pred_reduced = lda.predict(X_test_reduced)  # 在降维空间预测
        
        # 8. 计算模型性能指标
        accuracy = accuracy_score(y_test, y_pred)  # 分类准确率
        accuracy_reduced = accuracy_score(y_test, y_pred_reduced)  # 降维空间准确率
        report = classification_report(y_test, y_pred, output_dict=True)  # 详细分类报告
        
        # 9. 计算类别分布统计
        unique_classes, class_counts = np.unique(y_train, return_counts=True)  # 统计训练集类别分布
        class_distribution = {
            str(class_names[int(cls)]): int(count) for cls, count in zip(unique_classes, class_counts)
        }
        
        # 10. 计算降维数据统计
        # 计算每个降维特征的统计信息
        reduced_stats = {
            f"LD{i+1}": {
                "mean": float(np.mean(X_train_reduced[:, i])),  # 均值
                "std": float(np.std(X_train_reduced[:, i])),  # 标准差
                "min": float(np.min(X_train_reduced[:, i])),  # 最小值
                "max": float(np.max(X_train_reduced[:, i]))  # 最大值
            } for i in range(n_components)
        }
        
        # 计算解释方差比（如果求解器支持）
        explained_variance_ratio = []
        if hasattr(lda, 'explained_variance_ratio_'):
            explained_variance_ratio = lda.explained_variance_ratio_.tolist()
        
        # 11. 构造完整结果
        results["status"] = "success"
        results["message"] = "LDA 监督降维完成"
        results["lda_params"] = {
            "n_components_used": int(n_components),  # 实际使用的降维维度
            "n_components_max": int(max_components),  # 最大理论维度
            "solver": lda.solver,  # 使用的求解器
            "n_classes": int(n_classes)  # 类别数量
        }
        results["reduction_info"] = {
            "original_dimensions": int(n_features),  # 原始维度
            "reduced_dimensions": int(n_components),  # 降维后维度
            "reduction_ratio": round(1 - n_components / n_features, 4),  # 降维比例
            "explained_variance_ratio": explained_variance_ratio  # 解释方差比
        }
        results["metrics"] = {
            "test_accuracy_original": round(float(accuracy), 4),  # 原始空间准确率
            "test_accuracy_reduced": round(float(accuracy_reduced), 4),  # 降维空间准确率
            "accuracy_difference": round(float(accuracy - accuracy_reduced), 4),  # 准确率差异
            "classification_report": report  # 详细分类报告
        }
        results["data_statistics"] = reduced_stats  # 降维数据统计
        results["class_distribution"] = class_distribution  # 类别分布
        results["class_names"] = class_names  # 类别名称列表

    except Exception as e:
        # 异常处理
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=True, indent=2)  # 返回格式化的 JSON 字符串