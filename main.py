"""
基于 fastmcp 构建的 MCP 服务 demo
提供数字相加、文档检索、代码执行和 SQL 查询的工具
"""

from typing import Any, Dict, List, Optional, Union

import httpx
from fastmcp import FastMCP

#from config import settings
from tool_functions import (
    add_numbers,
    random_forest_model,
    train_knn_grid_json_only,
    compare_svc_kernels,
    train_predict_explain_xgboost,
    )


app = FastMCP("ToolsService")


@app.tool()
def add_numbers_tool(a: int, b: int) -> int:
    """
    将两个整数相加并返回结果。（测试使用）

    Args:
        a (int): 第一个整数。
        b (int): 第二个整数。

    Returns:
        int: 两个整数的和。
    """
    return add_numbers(a, b)

@app.tool()
def random_forest_model_tool(csv_path: str,target_column: str,task_type: str,test_size: float,n_estimators: int) -> str:
    """
    一个随机森林模型
    csv_path ：表示 CSV 文件的路径，其类型为字符串（string）。
    target_column ：表示目标列的名称，其类型也为字符串（string）。
    task_type ：表示任务类型，是一个字符串（string），且只能从以下两个选项中选择：classification regression。
    test_size：表示测试集的比例（取值在0-1之间 float类型）
    n_estimators:：生成的决策树数量（int）值太小容易导致欠拟合,值太大时提升效果有限，但会增加计算成本
    """
    return random_forest_model(csv_path,target_column,task_type,test_size,n_estimators)

@app.tool()
def train_knn_grid_json_only_tool(data_path: str, config_path: str, label_path: str) -> str:
    """
    KNN 分类 + GridSearch 网格搜索。
    返回 JSON 格式的训练结果和特征重要性数据。
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 KNN 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性(基于排列重要性)和预测摘要。    
    """
    return train_knn_grid_json_only(data_path, config_path, label_path)

@app.tool()
def compare_svc_kernels_tool(data_path: str, config_path: str, label_path: str) -> str:
    """
    SVC 分类 + GridSearch 网格搜索。
    返回 JSON 格式的训练结果和特征重要性数据。
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 SVC 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性(基于排列重要性)和预测摘要。    
    """
    return compare_svc_kernels(data_path, config_path, label_path)

@app.tool()
def train_predict_explain_xgboost_tool(data_path: str, config_path: str, label_path: str) -> str:
    """
    XGBoost 分类 + GridSearch 网格搜索。
    返回 JSON 格式的训练结果和特征重要性数据。
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 XGBoost 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性(基于排列重要性)和预测摘要。    
    """
    return train_predict_explain_xgboost(data_path, config_path, label_path)

if __name__ == "__main__":
    app.run(transport="stdio")
