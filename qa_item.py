from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel, ValidationError

@dataclass
class QAItem(BaseModel):
    question: str
    answer: str
    equipment_problem: str
    tools_required: List[str]
    steps: List[str]
    safety_info: str
    tips: List[str] = field(default_factory=list)
