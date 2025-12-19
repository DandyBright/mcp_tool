"""
Notebook 有状态代码执行器工具模块

提供执行 Python 代码片段的功能，支持状态累积。
"""

import ast
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Set, Union


def execute_notebook_code(
    code: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None,
    allowed_modules: Optional[List[str]] = None,
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    在有状态环境中执行给定的 Python 代码片段

    Args:
        code: 要执行的 Python 代码字符串
        globals_dict: 当前全局命名空间（可选，用于状态累积）
        locals_dict: 当前局部命名空间（可选）
        allowed_modules: 允许的模块列表

    Returns:
        包含 stdout、stderr、returncode 和更新后的状态的字典
    """
    # 默认允许的模块
    if allowed_modules is None:
        allowed_modules = [
            "json",
            "re",
            "pathlib",
            "typing",
            "dataclasses",
            "math",
            "datetime",
            "statistics",
            "collections",
            "itertools",
        ]

    # 初始化命名空间
    if globals_dict is None:
        globals_dict = {
            "__builtins__": {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if not name.startswith("_")
            }
        }
    if locals_dict is None:
        locals_dict = {}

    # 验证代码
    if not _validate_code(code, allowed_modules):
        return {
            "stdout": "",
            "stderr": "代码包含禁止的操作。",
            "returncode": -1,
            "globals_dict": globals_dict,
            "locals_dict": locals_dict,
        }

    # 捕获输出
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    try:
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, globals_dict, locals_dict)
        return {
            "stdout": output_buffer.getvalue(),
            "stderr": error_buffer.getvalue(),
            "returncode": 0,
            "globals_dict": globals_dict,
            "locals_dict": locals_dict,
        }
    except Exception as e:
        return {
            "stdout": output_buffer.getvalue(),
            "stderr": f"执行错误: {str(e)}",
            "returncode": -1,
            "globals_dict": globals_dict,
            "locals_dict": locals_dict,
        }


def _validate_code(code: str, allowed_modules: List[str]) -> bool:
    """
    使用 AST 验证代码是否安全

    Args:
        code: 要验证的代码字符串。
        allowed_modules: 允许的模块列表。

    Returns:
        bool: 如果安全则 True，否则 False。
    """
    try:
        tree = ast.parse(code)
        return _check_ast_node(tree, allowed_modules)
    except SyntaxError:
        return False


def _check_ast_node(node: ast.AST, allowed_modules: List[str]) -> bool:
    """
    递归检查 AST 节点

    Args:
        node: AST 节点。
        allowed_modules: 允许的模块列表。

    Returns:
        bool: 如果安全则 True。
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in allowed_modules:
                return False
    elif isinstance(node, ast.ImportFrom):
        if node.module not in allowed_modules:
            return False

    # 递归检查子节点
    for child in ast.iter_child_nodes(node):
        if not _check_ast_node(child, allowed_modules):
            return False

    return True
