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

if __name__ == "__main__":
    app.run(transport="stdio")
