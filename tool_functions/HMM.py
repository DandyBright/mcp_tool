import json
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def train_hmm_bearing_json_only(data_path: str, config_path: str = None) -> str:
    """
    HMM 轴承退化三状态建模 (GaussianHMM)。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx)。
        config_path (str): [可选] JSON 配置文件路径。如果不传或读取失败，将使用默认参数。
        
    Returns:
        str: JSON 格式的执行结果。
    """
    results = {
        "status": "failed",
        "message": "",
        "model_params": {},
        "state_definition": {},
        "transition_matrix": [],
        "metrics": {}
    }

    try:
        # 1. 加载数据 (保持不变)
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
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw) 
        feature_names = df.columns.tolist()

        # 2. 加载配置 (修改点：移除 YAML，改用 JSON 或 默认值)
        # -------------------------------------------------------
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    # 尝试解析 JSON 配置
                    config = json.load(f)
            except Exception:
                # 如果 config_path 无效或不是 JSON，这里不做处理，直接用下面的默认值
                pass

        # 从 config 中取值，如果取不到就用后面的默认值
        n_components = config.get('n_components', 3)       # 默认为 3 (正常/退化/故障)
        covariance_type = config.get('covariance_type', 'full') 
        n_iter = config.get('n_iter', 100)
        random_state = config.get('random_state', 42)
        # -------------------------------------------------------

        # 3. 模型训练
        model = GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=n_iter, 
            random_state=random_state,
            verbose=False
        )
        model.fit(X_scaled)

        # 4. 状态物理意义对齐 (关键逻辑：按特征能量排序)
        means = model.means_ 
        state_severity = np.sum(means ** 2, axis=1) # 计算严重程度
        sorted_indices = np.argsort(state_severity) # 排序索引
        state_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        # 重排矩阵和均值
        sorted_trans_mat = model.transmat_[sorted_indices, :][:, sorted_indices]
        sorted_means = means[sorted_indices]

        # 5. 推断与构造结果
        hidden_states = model.predict(X_scaled)
        mapped_states = np.array([state_map[s] for s in hidden_states])
        
        unique, counts = np.unique(mapped_states, return_counts=True)
        state_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
        log_likelihood = model.score(X_scaled)

        results["status"] = "success"
        results["message"] = "HMM 建模完成"
        results["model_params"] = {
            "n_components": n_components,
            "covariance_type": covariance_type
        }
        
        # 状态定义
        state_definitions = {}
        for i in range(n_components):
            feat_means = {feat: round(float(val), 4) for feat, val in zip(feature_names, sorted_means[i])}
            state_definitions[f"State_{i}"] = {
                "description": "健康" if i==0 else ("故障" if i==n_components-1 else "退化"),
                "feature_means": feat_means
            }
        results["state_definition"] = state_definitions
        results["transition_matrix"] = np.round(sorted_trans_mat, 4).tolist()
        results["metrics"] = {
            "log_likelihood": round(float(log_likelihood), 4),
            "state_distribution": state_counts,
            "last_10_states": mapped_states[-10:].tolist() 
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)