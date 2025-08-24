from typing import List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ProblemState:
    filename: str
    label: Optional[str] = None
    test_code: Optional[str] = None
    instruction: Optional[str] = None
    solution: Optional[str] = None
    golden_code: Optional[str] = None # To hold the full original source file

@dataclass
class ProblemStateROCm:
    instruction: str
    label: str
    filename: str
    target_kernel_name: str
    test_code: str
    solution: Optional[str] = None
    pass_call: bool = False
    pass_exe: bool = False