from typing import Any, Dict, List

from tavily import TavilyClient

from config import settings


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    使用 Tavily API 进行网络搜索

    参数:
        query: 搜索查询字符串
        max_results: 返回的搜索结果数量

    返回:
        包含搜索结果的列表，每个结果是一个字典，包含标题、链接和摘要
    """
    # 验证输入参数
    if not query or query.strip() == "":
        raise ValueError("搜索查询不能为空")

    if max_results <= 0:
        raise ValueError("搜索结果数量必须大于0")

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = client.search(
            query=query, max_results=max_results, auto_parameters=True
        )

        results = []
        for res in response["results"]:
            results.append(
                {
                    "title": res.get("title", "No Title"),
                    "url": res.get("url", "No URL"),
                    "content": res.get("content", "No Content"),
                }
            )

        return results

    except Exception as e:
        return [{"error": str(e)}]
