from typing import Annotated
from pydantic import BaseModel, Field, ValidationError


def min_list_str(min_len: int) -> type:
    return Annotated[list[str], Field(min_length=min_len)]

def fmin(min_len: int) -> type:
    return Field(min_length=min_len)

class QAItem(BaseModel):
    question: str = fmin(20)
    answer: str = fmin(100)
    equipment_problem: str = fmin(10)
    tools_required: min_list_str(1)
    steps: min_list_str(3)
    safety_info: str = fmin(80)
    tips: min_list_str(1)
