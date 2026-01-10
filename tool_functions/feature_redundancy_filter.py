import json
import pandas as pd
import numpy as np

def filter_redundant_features_json_only(data_path: str, config_path: str = None) -> str:
    """
    基于 Pearson 相关系数的特征冗余剔除。
    计算特征间的相关性矩阵，若两个特征相关性高于阈值，则视为冗余，剔除其中一个。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx)。
        config_path (str): [可选] JSON 配置文件路径，包含阈值设置。
        
    Returns:
        str: JSON 格式的执行结果，包含保留特征、剔除特征及高相关性对的详细信息。
    """
    results = {
        "status": "failed",
        "message": "",
        "params": {},
        "analysis_result": {}
    }

    try:
        # -------------------------------------------------------
        # 1. 加载数据
        # -------------------------------------------------------
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式")
            
        # 仅选择数值型列进行相关性分析 (排除字符串/日期列)
        df_numeric = df.select_dtypes(include=[np.number])
        feature_names = df_numeric.columns.tolist()
        
        if len(feature_names) < 2:
            raise ValueError("数值型特征数量不足 2 个，无法进行相关性分析")

        # -------------------------------------------------------
        # 2. 加载配置 (使用 JSON 或 默认值)
        # -------------------------------------------------------
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass

        # 获取相关性阈值，默认 0.9
        # 如果特征A和特征B的相关系数 > 0.9，则认为它们包含重复信息
        threshold = config.get('correlation_threshold', 0.9)
        
        # -------------------------------------------------------
        # 3. 计算相关性矩阵
        # -------------------------------------------------------
        # 使用 Pearson 相关系数 (-1 到 1)
        corr_matrix = df_numeric.corr().abs()

        # -------------------------------------------------------
        # 4. 识别并标记冗余特征 (核心逻辑)
        # -------------------------------------------------------
        # np.triu: 提取上三角矩阵 (Upper Triangle)，不含对角线 (k=1)
        # 为什么要这样做？因为相关性矩阵是对称的，A与B的相关性 等于 B与A。
        # 我们只需要遍历上三角即可避免重复判断，且不包含自己与自己的相关性(即对角线)。
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 寻找所有相关系数绝对值大于阈值的列
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # 确定保留的特征
        kept_features = [f for f in feature_names if f not in to_drop]

        # -------------------------------------------------------
        # 5. 记录高相关性对 (用于解释为什么剔除)
        # -------------------------------------------------------
        # 为了让结果更透明，我们记录下具体是哪两个特征冲突了
        high_corr_pairs = []
        for col in to_drop:
            # 找到该列中，哪些行(即其他特征)与其相关性超标
            # upper[col] 是一个 Series，包含该特征与其他特征的相关性
            correlated_feats = upper[col][upper[col] > threshold].index.tolist()
            for feat in correlated_feats:
                val = upper.loc[feat, col] # 获取具体相关系数值
                high_corr_pairs.append({
                    "drop_feature": col,      # 被剔除的特征
                    "keep_feature": feat,     # 被保留的特征 (因为它在矩阵的前面)
                    "correlation_value": round(float(val), 4)
                })

        # -------------------------------------------------------
        # 6. 构造结果
        # -------------------------------------------------------
        results["status"] = "success"
        results["message"] = f"分析完成，发现 {len(to_drop)} 个冗余特征"
        
        results["params"] = {
            "correlation_threshold": threshold,
            "total_features_input": len(feature_names)
        }
        
        results["analysis_result"] = {
            "kept_features": kept_features,        # 建议保留的特征列表
            "dropped_features": to_drop,           # 建议剔除的特征列表
            "kept_count": len(kept_features),
            "dropped_count": len(to_drop),
            "redundancy_details": high_corr_pairs  # 详细的冲突对
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)