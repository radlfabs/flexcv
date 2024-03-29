from .core import cross_validate
from .interface import CrossValidation
from .model_mapping import ModelMappingDict, ModelConfigDict
from .repeated import RepeatedCV

__version__ = "v24.0-beta"

__all__ = [
    "CrossValidation",
    "ModelMappingDict",
    "ModelConfigDict",
    "RepeatedCV",
    "YAML_TEMPLATE_PATH"
]
