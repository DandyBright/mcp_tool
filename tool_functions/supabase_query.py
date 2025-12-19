"""
Supabase 数据库查询模块

使用 psycopg2 直接连接 PostgreSQL 数据库（Supabase 提供的 PostgreSQL 实例）。
"""

import re
from typing import Any, Dict

import psycopg2

from config import settings


def _is_read_only_sql(sql: str) -> bool:
    """
    检查 SQL 语句是否为只读查询（安全检查）

    仅允许执行 SELECT 查询。禁止执行 INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE 等操作。

    参数:
        sql: SQL 语句

    返回:
        True 如果是安全的只读查询，False 否则
    """
    # 移除注释和多余空白
    sql_clean = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)  # 移除单行注释
    sql_clean = re.sub(
        r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL
    )  # 移除多行注释
    sql_clean = sql_clean.strip()

    if not sql_clean:
        return False

    # 获取 SQL 的第一个关键字
    first_keyword = re.match(r"^\s*(\w+)", sql_clean, re.IGNORECASE)
    if not first_keyword:
        return False

    keyword = first_keyword.group(1).upper()

    # 只允许 SELECT 和 WITH (CTE) 操作
    if keyword in ("SELECT", "WITH"):
        return True

    # 禁止的操作
    forbidden_keywords = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "REVOKE",
        "CALL",
        "PRAGMA",
    }

    if keyword in forbidden_keywords:
        return False

    # 其他任何操作都视为不安全
    return False


def get_db_connection():
    """
    获取数据库连接

    返回:
        PostgreSQL 数据库连接对象
    """
    connection = psycopg2.connect(
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
    )
    return connection


def execute_sql_query(sql: str) -> Dict[str, Any]:
    """
    直接执行 SQL 查询语句（仅支持 SELECT）

    参数:
        sql: SQL 查询语句，支持 SELECT 等
             例如: SELECT * FROM solar LIMIT 10
             例如: SELECT date, temperature FROM solar WHERE date LIKE '2021/1/11%' LIMIT 20

    返回:
        包含查询结果和元数据的字典，包括 success, data, count 和错误信息
    """
    connection = None
    cursor = None
    try:
        # 安全检查：仅允许执行只读查询
        if not _is_read_only_sql(sql):
            return {
                "success": False,
                "error": "出于安全考虑，仅允许执行 SELECT 查询语句，不允许执行其他操作",
                "data": [],
            }

        # 创建数据库连接
        connection = get_db_connection()
        cursor = connection.cursor()

        # 执行 SQL 查询
        cursor.execute(sql)

        # 获取列名
        column_names = [description[0] for description in cursor.description]

        # 获取所有查询结果
        rows = cursor.fetchall()

        # 将结果转换为字典列表
        data = []
        for row in rows:
            data.append(dict(zip(column_names, row)))

        return {
            "success": True,
            "data": data,
            "count": len(data),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"数据库错误: {str(e)}",
            "data": [],
        }

    finally:
        # 关闭游标和连接
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
