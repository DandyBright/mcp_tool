import json
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def train_svdd_json_only(data_path: str, config_path: str = None) -> str:
    """
    SVDD (One-Class SVM) 单类分类/异常检测。
    适用于只有大量“正常”样本，只有少量或没有“故障”样本的场景。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx)，包含用于构建正常边界的特征数据。
        config_path (str): [可选] JSON 配置文件路径，包含模型超参数 (nu, gamma, kernel)。
        
    Returns:
        str: JSON 格式的执行结果，包含异常检出率、异常点索引及模型边界信息。
    """
    results = {
        "status": "failed",
        "message": "",
        "model_params": {},
        "metrics": {},
        "anomaly_details": {}
    }

    try:
        # -------------------------------------------------------
        # 1. 加载数据
        # -------------------------------------------------------
        # SVDD 对数据分布敏感，通常假设输入数据大部分是正常的
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式")
            
        X_raw = df.values
        
        # 关键步骤：SVM 基于距离度量，必须进行标准化 (Standardization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw) 
        
        # -------------------------------------------------------
        # 2. 加载配置 (使用 JSON 或 默认值)
        # -------------------------------------------------------
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass # 加载失败则使用默认值

        # SVDD 关键参数
        # nu: 训练误差分数的上界，也是支持向量分数的下界。
        #     通俗理解：允许训练集中有多少比例的数据被判定为“异常/噪音”。
        #     例如 0.05 表示我们认为训练数据中可能有 5% 的异常点。
        svdd_params = config.get('svdd_params', {})
        nu = svdd_params.get('nu', 0.05)         
        kernel = svdd_params.get('kernel', 'rbf') # rbf (高斯核) 是 SVDD 的标准配置
        gamma = svdd_params.get('gamma', 'scale') # 核系数，影响边界的紧致程度
        
        # -------------------------------------------------------
        # 3. 模型训练
        # -------------------------------------------------------
        model = OneClassSVM(
            nu=nu, 
            kernel=kernel, 
            gamma=gamma
        )
        
        # 拟合数据，学习正常数据的边界 (Hypersphere)
        model.fit(X_scaled)
        
        # -------------------------------------------------------
        # 4. 预测与评估
        # -------------------------------------------------------
        # predict 返回: 1 代表正常 (Inlier), -1 代表异常 (Outlier)
        y_pred = model.predict(X_scaled)
        
        # decision_function 返回: 点到超平面的有符号距离
        # 正值: 边界内 (正常), 负值: 边界外 (异常)
        # 绝对值越大，置信度越高
        dist_scores = model.decision_function(X_scaled)
        
        # 统计结果
        n_total = len(y_pred)
        n_normal = np.sum(y_pred == 1)
        n_anomaly = np.sum(y_pred == -1)
        anomaly_rate = n_anomaly / n_total
        
        # 获取异常点的索引 (Python list)
        anomaly_indices = np.where(y_pred == -1)[0].tolist()
        
        # -------------------------------------------------------
        # 5. 构造结果 (JSON)
        # -------------------------------------------------------
        results["status"] = "success"
        results["message"] = "SVDD 异常检测模型训练完成"
        
        results["model_params"] = {
            "nu": nu,
            "kernel": kernel,
            "gamma": str(gamma) # gamma 可能是字符串 'scale' 或浮点数
        }
        
        results["metrics"] = {
            "total_samples": int(n_total),
            "normal_count": int(n_normal),
            "anomaly_count": int(n_anomaly),
            "anomaly_ratio": round(float(anomaly_rate), 4)
        }
        
        # 详细的异常信息，方便前端展示或日志记录
        # 仅返回前 50 个异常点索引，防止 JSON 过大
        display_limit = 50
        results["anomaly_details"] = {
            "detected_indices": anomaly_indices[:display_limit],
            "note": f"仅显示前 {display_limit} 个异常点索引" if len(anomaly_indices) > display_limit else "显示所有异常点索引",
            # 提供最严重的 5 个异常点的索引（距离边界最远/分值最小的负数）
            "top_5_severe_anomalies_indices": np.argsort(dist_scores)[:5].tolist()
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)