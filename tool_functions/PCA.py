import json
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def train_pca_auto_cutoff(data_path: str, config_path: str) -> str:
    """
    PCA降维 + 累积方差阈值自动截断。
    返回JSON格式的PCA分析结果。
    
    Args:
        data_path (str): 数据文件路径(.csv或.xlsx/xls)，包含特征矩阵。
        config_path (str): YAML配置文件路径，包含PCA参数。
        
    Returns:
        str: JSON格式的执行结果，包含PCA参数、方差解释率、降维数据等。
    """
    # 初始化结果字典，包含所有可能的输出字段
    results = {
        "status": "failed",          # 执行状态：success/error/failed
        "message": "",              # 执行信息或错误信息
        "pca_params": {},           # PCA算法参数
        "variance_analysis": {},    # 方差解释率分析结果
        "components_analysis": {},  # 主成分特征权重分析
        "transformed_data_stats": {},  # 降维后数据的统计信息
        "summary": {}               # 算法执行摘要
    }

    try:
        # 1. 数据加载模块
        # 根据文件扩展名选择对应的读取方法
        if data_path.endswith('.csv'):
            try:
                # 优先尝试UTF-8编码读取CSV文件
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试GBK编码（中文环境常见）
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            # 读取Excel文件，支持.xlsx和.xls格式
            df = pd.read_excel(data_path)
        else:
            # 不支持的文件格式抛出异常
            raise ValueError("不支持的数据格式")
            
        # 提取特征矩阵和特征名称
        X = df.values                # 获取数值矩阵，形状为(n_samples, n_features)
        feature_names = df.columns.tolist()  # 获取所有特征列名
        
        # 2. 配置加载模块
        try:
            # 尝试UTF-8编码读取YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)  # 安全加载YAML，防止执行恶意代码
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GBK编码
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        
        # 从配置中获取PCA相关参数，设置默认值
        pca_config = config.get('pca_params', {})
        variance_threshold = pca_config.get('variance_threshold', 0.85)  # 累积方差阈值，默认85%
        scale_data = pca_config.get('scale_data', True)                 # 是否标准化数据，默认True
        max_components = pca_config.get('max_components', None)         # 最大主成分数限制，默认无限制
        random_state = pca_config.get('random_state', 42)               # 随机种子，保证结果可重现
        
        # 3. 数据预处理
        X_scaled = X  # 初始化缩放后的数据
        if scale_data:
            # 创建标准化器，去除均值和缩放至单位方差
            scaler = StandardScaler()
            # 拟合缩放器并转换数据：X_scaled = (X - mean) / std
            X_scaled = scaler.fit_transform(X)
        
        # 4. 确定主成分数量
        # 首先拟合一个包含所有可能主成分的PCA，用于查看累积方差
        # 限制最大主成分数为50或特征数的最小值，防止内存溢出
        pca_full = PCA(n_components=min(X.shape[1], 50), random_state=random_state)
        pca_full.fit(X_scaled)  # 拟合PCA模型（不进行转换）
        
        # 计算累积方差解释率
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # 找到达到阈值所需的最小主成分数
        # np.argmax返回第一个满足条件的索引，条件为累积方差 >= 阈值
        n_components_auto = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # 处理边界情况
        if n_components_auto == 1 and cumulative_variance[0] < variance_threshold:
            # 情况1：第一个主成分的方差就达不到阈值
            # 选择所有主成分（因为无法达到阈值）
            n_components_auto = X.shape[1]  # 使用所有特征
        elif n_components_auto == 0:
            # 情况2：没有主成分达到阈值（阈值可能为0）
            n_components_auto = 1  # 至少保留一个主成分
        
        # 应用最大主成分数限制（如果配置了的话）
        if max_components is not None:
            # 取自动选择数量和最大限制的较小值
            n_components = min(n_components_auto, max_components)
        else:
            n_components = n_components_auto
        
        # 确保可视化需求：如果数据维度>=2，至少保留2个主成分用于可视化
        if X.shape[1] >= 2:
            # 至少保留2个，但不超过实际计算出的数量
            n_components = max(min(n_components, X.shape[1]), 2)
        else:
            # 如果数据只有1维，使用计算出的数量
            n_components = min(n_components, X.shape[1])
        
        # 5. 执行PCA降维
        # 使用确定的主成分数量创建PCA模型
        pca = PCA(n_components=n_components, random_state=random_state)
        # 拟合模型并进行降维转换：X_pca形状为(n_samples, n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 6. 计算方差解释率
        # explained_variance_ratio_：每个主成分解释的方差比例
        explained_variance = pca.explained_variance_ratio_
        # 计算累积方差解释率
        cumulative_variance_final = np.cumsum(explained_variance)
        
        # 7. 分析主成分特征权重
        # pca.components_：主成分的特征权重矩阵，形状为(n_components, n_features)
        components = pca.components_
        
        # 为每个主成分找出最重要的特征（权重绝对值最大的前5个）
        top_features_per_pc = {}
        for i in range(n_components):
            # 获取第i个主成分的所有特征权重
            component_weights = components[i]
            
            # 找出权重绝对值最大的前5个特征的索引
            # np.argsort返回按值排序的索引，[-5:][::-1]取最后5个并反转（从大到小）
            top_indices = np.argsort(np.abs(component_weights))[-5:][::-1]
            
            # 构建特征名到权重的映射字典
            top_features = {
                feature_names[idx]: float(component_weights[idx])  # 保留原始权重大小
                for idx in top_indices
            }
            
            # 存储到结果字典中，键名为PC1, PC2等
            top_features_per_pc[f"PC{i+1}"] = top_features
        
        # 8. 计算降维数据的统计信息
        transformed_stats = {}
        for i in range(n_components):
            # 提取第i个主成分的所有样本值
            pc_data = X_pca[:, i]
            
            # 计算基本统计量并保留4位小数
            transformed_stats[f"PC{i+1}"] = {
                "mean": round(float(np.mean(pc_data)), 4),  # 平均值
                "std": round(float(np.std(pc_data)), 4),    # 标准差
                "min": round(float(np.min(pc_data)), 4),    # 最小值
                "max": round(float(np.max(pc_data)), 4)     # 最大值
            }
        
        # 9. 构造完整的结果字典
        results["status"] = "success"
        results["message"] = f"PCA降维完成，基于{round(variance_threshold*100, 1)}%累积方差阈值，自动选择{n_components}个主成分"
        
        # 9.1 PCA参数信息
        results["pca_params"] = {
            "variance_threshold": variance_threshold,  # 配置的方差阈值
            "actual_components": n_components,         # 实际选择的主成分数
            "scale_data": scale_data,                  # 是否进行了数据标准化
            "max_components_limit": max_components,    # 最大主成分数限制
            "random_state": random_state               # 随机种子
        }
        
        # 9.2 方差分析结果
        variance_info = {}
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance_final)):
            variance_info[f"PC{i+1}"] = {
                "explained_variance_ratio": round(float(var), 4),    # 单个主成分方差解释率
                "cumulative_variance_ratio": round(float(cum_var), 4)  # 累积方差解释率
            }
        results["variance_analysis"] = variance_info
        
        # 9.3 主成分特征权重分析
        results["components_analysis"] = top_features_per_pc
        
        # 9.4 降维数据统计信息
        results["transformed_data_stats"] = transformed_stats
        
        # 9.5 降维后数据示例（只返回前10行，避免结果过大）
        pca_data_sample = []
        for i in range(min(10, len(X_pca))):
            # 构建每行的主成分值字典
            row = {f"PC{j+1}": round(float(val), 4) for j, val in enumerate(X_pca[i])}
            pca_data_sample.append(row)
        results["transformed_data_sample"] = pca_data_sample
        
        # 9.6 算法执行摘要
        results["summary"] = {
            "original_features": len(feature_names),                    # 原始特征数量
            "selected_components": n_components,                        # 选择的主成分数量
            "total_variance_explained": round(float(cumulative_variance_final[-1]), 4),  # 总方差解释率
            "threshold_achieved": cumulative_variance_final[-1] >= variance_threshold,  # 是否达到阈值
            "data_shape_original": list(X.shape),                       # 原始数据形状
            "data_shape_transformed": list(X_pca.shape)                 # 降维后数据形状
        }

    except Exception as e:
        # 10. 异常处理
        # 捕获所有异常，更新状态和错误信息
        results["status"] = "error"
        results["message"] = str(e)  # 将异常信息转换为字符串
        
    # 11. 返回JSON格式结果
    # ensure_ascii=False允许输出非ASCII字符（如中文），indent=2美化输出格式
    return json.dumps(results, ensure_ascii=False, indent=2)