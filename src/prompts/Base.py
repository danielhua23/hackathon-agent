class BasePrompt:
    def __init__(self):
        pass
    
    def get_prompt(self) -> str:
        raise NotImplementedError
