"""产业链传导：简化版知识图谱（MVP，替代 Neo4j）。

用「行业 → 上游/下游」的有向映射近似产业链。事件文本命中某行业后，
自动传导到上下游。Phase 2 可替换为真正的图数据库（Neo4j + GNN）。
"""

from __future__ import annotations


# 产业链：行业 -> {关键词, 上游, 下游}
INDUSTRY_CHAIN: dict[str, dict[str, list[str]]] = {
    "半导体": {"keywords": ["半导体", "芯片", "晶圆", "封测", "集成电路"],
               "upstream": ["电子", "化学", "通用设备"],
               "downstream": ["计算机", "通信", "汽车", "消费电子"]},
    "计算机": {"keywords": ["计算机", "软件", "互联网", "算力", "数据中心", "人工智能", "大模型", "AGI", "GPT", "AIGC", "AI芯片"],
               "upstream": ["半导体", "电子"],
               "downstream": ["通信", "消费"]},
    "通信": {"keywords": ["通信", "5G", "6G", "光模块"],
             "upstream": ["半导体", "电子"],
             "downstream": ["计算机", "互联网"]},
    "汽车": {"keywords": ["汽车", "新能源车", "电动车", "整车"],
             "upstream": ["钢铁", "电子", "橡胶", "半导体", "电气"],
             "downstream": ["消费"]},
    "钢铁": {"keywords": ["钢铁", "特钢", "板材"],
             "upstream": ["煤炭", "有色"],
             "downstream": ["建筑", "机械", "汽车"]},
    "煤炭": {"keywords": ["煤炭", "焦煤", "动力煤"],
             "upstream": [],
             "downstream": ["电力", "钢铁", "化学"]},
    "电力": {"keywords": ["电力", "火电", "水电", "核电", "风电", "光伏发电"],
             "upstream": ["煤炭", "电气"],
             "downstream": []},
    "有色": {"keywords": ["有色", "铜", "铝", "锂", "稀土", "黄金", "白银", "锌"],
             "upstream": [],
             "downstream": ["电气", "机械", "汽车", "电子"]},
    "化学": {"keywords": ["化学", "化工", "新材料", "化工品"],
             "upstream": ["石油", "煤炭"],
             "downstream": ["医药", "农业", "纺织", "建材", "半导体"]},
    "石油": {"keywords": ["石油", "原油", "天然气"],
             "upstream": [],
             "downstream": ["化学", "交通"]},
    "房地产": {"keywords": ["房地产", "地产", "楼市", "房企"],
               "upstream": ["建材", "钢铁", "水泥"],
               "downstream": ["家电", "家具"]},
    "建材": {"keywords": ["建材", "水泥", "玻璃"],
             "upstream": ["化学", "钢铁", "煤炭"],
             "downstream": ["房地产", "基建"]},
    "食品": {"keywords": ["食品", "白酒", "饮料", "乳品", "调味品"],
             "upstream": ["农业"],
             "downstream": ["消费"]},
    "农业": {"keywords": ["农业", "粮食", "种业", "生猪", "养殖", "饲料", "化肥"],
             "upstream": ["化学"],
             "downstream": ["食品", "消费"]},
    "医药": {"keywords": ["医药", "医疗", "创新药", "疫苗", "医疗器械", "生物"],
             "upstream": ["化学"],
             "downstream": []},
    "机械": {"keywords": ["机械", "工程机械", "设备", "机器人"],
             "upstream": ["钢铁", "通用设备"],
             "downstream": ["汽车", "建筑", "农业"]},
    "电气": {"keywords": ["电气", "电网", "储能", "光伏", "锂电", "电池", "风电"],
             "upstream": ["有色", "半导体"],
             "downstream": ["电力", "汽车"]},
}


def affected_industries(event_text: str) -> list[str]:
    """返回事件文本命中的行业及其上下游（去重）。"""
    hits: list[str] = []
    for industry, info in INDUSTRY_CHAIN.items():
        if any(keyword in event_text for keyword in info["keywords"]):
            hits.append(industry)
            hits.extend(info["upstream"])
            hits.extend(info["downstream"])
    return list(dict.fromkeys(hits))
