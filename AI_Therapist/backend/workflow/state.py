"""
增强的工作流状态 - 包含LSM&nCLiD评分
"""
from typing import TypedDict, Optional, Dict, Any


class ChatStateWithScoring(TypedDict):
    """带评分的聊天状态"""

    # 基础字段
    user_id: str
    user_input: str
    character_id: str
    emotion: str
    cbt_stage: int
    ai_response: str
    understanding_score: float

    # 🆕 评分字段
    lsm_scores: Dict[str, float]  # {"single": 0.1234, "cumulative": 0.2345}
    nclid_scores: Dict[str, float]  # {"single": 0.3456, "cumulative": 0.4567}
    score_analysis: Dict[str, Any]  # 评分分析结果
    score_trends: Dict[str, Any]  # 评分趋势

    # 元数据
    metadata: Optional[Dict[str, Any]]