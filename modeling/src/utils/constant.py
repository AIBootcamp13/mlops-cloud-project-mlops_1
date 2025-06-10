from enum import Enum

from modeling.src.model.lstm import MultiOutputLSTM, MultiOutputStackedLSTM

class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]
    
    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class Models(CustomEnum):
    MULTI_OUTPUT_LSTM = MultiOutputLSTM
    MULTI_OUTPUT_STACKED_LSTM = MultiOutputStackedLSTM