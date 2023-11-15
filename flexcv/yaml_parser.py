import yaml
import importlib
from optuna.distributions import (
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution,
)
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict


def int_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> IntDistribution:
    """Construct an optuna IntDistribution from a yaml node.

    Args:
        loader (yaml.SafeLoader): The yaml loader.
        node (yaml.nodes.ScalarNode): The yaml node.

    Returns:
        (IntDistribution): The constructed IntDistribution.
    """
    return IntDistribution(**loader.construct_mapping(node))


def float_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> FloatDistribution:
    """Construct an optuna FloatDistribution from a yaml node.

    Args:
        loader (yaml.SafeLoader): The yaml loader.
        node (yaml.nodes.ScalarNode): The yaml node.

    Returns:
        (FloatDistribution): The constructed FloatDistribution.
    """
    return FloatDistribution(**loader.construct_mapping(node))


def cat_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
) -> CategoricalDistribution:
    """Construct an optuna CategoricalDistribution from a yaml node.

    Args:
        loader (yaml.SafeLoader): The yaml loader.
        node (yaml.nodes.ScalarNode): The yaml node.

    Returns:
        (CategoricalDistribution): The constructed CategoricalDistribution.
    """
    value = loader.construct_mapping(node, deep=True)
    return CategoricalDistribution(**value)


def get_loader():
    """This function returns a yaml loader with the custom constructors for the optuna distributions.
    Custom and safe constructors are added to the following tags:

    - !Int
    - !Float
    - !Cat
    """
    loader = yaml.SafeLoader
    loader.add_constructor("!Int", int_constructor)
    loader.add_constructor("!Float", float_constructor)
    loader.add_constructor("!Cat", cat_constructor)
    return loader


def split_import(import_string) -> tuple[str, str]:
    """This function splits an import string into the class name and the module path.

    Args:
        import_string (str): The import string.

    Returns:
        (tuple[str, str]): The class name and the module path.
    """
    import_list = import_string.split(".")
    class_name = import_list[-1]
    module_path = ".".join(import_list[:-1])
    return class_name, module_path


def parse_yaml_output_to_mapping_dict(yaml_dict) -> ModelMappingDict:
    """This function parses the output of the yaml parser to a ModelMappingDict object.
    Models and post processors are imported automatically.

    Note:
        Despite of automatically importing the classes, no arbitrary code is executed.
        The yaml parser uses the safe loader and a custom constructors for the optuna distributions.
        The imports are done by the importlib module.

    Args:
        yaml_dict (dict): The output of the yaml parser.

    Returns:
        (ModelMappingDict): A dictionary of ModelConfigDict objects.
    """
    model_mapping_dict = ModelMappingDict()
    for model_name, raw_dict in yaml_dict.items():
        if "model" not in raw_dict:
            raise ValueError(f"model name is missing for model {model_name}")

        imports_dict = {}
        import_list = ["model"]
        if "post_processor" in raw_dict:
            import_list.append("post_processor")

        for key in import_list:
            if isinstance(raw_dict[key], str):
                class_name, path = split_import(raw_dict[key])
                foo = importlib.import_module(path)
                imports_dict[key] = getattr(foo, class_name)
            else:
                imports_dict[key] = raw_dict[key]

        # replace the strings in raw_dict with the imported classes
        raw_dict.update(imports_dict)
        model_config_dict = ModelConfigDict(raw_dict)
        model_mapping_dict[model_name] = model_config_dict
    return model_mapping_dict


def read_mapping_from_yaml_file(yaml_file_path: str) -> ModelMappingDict:
    """This function reads in a yaml file and returns a ModelMappingDict object.
    Use the yaml tags !Int, !Float, !Cat to specify the type of the hyperparameter distributions.
    The parser takes care of importing the classes specified in the yaml file in the fields model and post_processor.

    Note:
        Despite of automatically importing the classes, no arbitrary code is executed.
        The yaml parser uses the safe loader and a custom constructors for the optuna distributions.
        The imports are done by the importlib module.

    Args:
        yaml_code (str): The yaml code.

    Returns:
        (ModelMappingDict): A dictionary of ModelConfigDict objects.

    Example:
        Your file.yaml file could look like this:
        ```yaml
        RandomForest:
            model: sklearn.ensemble.RandomForestRegressor
            post_processor: flexcv.model_postprocessing.MixedEffectsPostProcessor
            requires_inner_cv: True
            params:
                max_depth: !Int
                    low: 1
                    high: 10
        ```
        And you would read it in like this:
        ```python
        model_mapping = read_mapping_from_yaml_file("file.yaml")
        ```
    """
    with open(yaml_file_path, "r") as f:
        yaml_output = yaml.load(f, Loader=get_loader())
    model_mapping = parse_yaml_output_to_mapping_dict(yaml_output)
    return model_mapping


def read_mapping_from_yaml_string(yaml_code: str) -> ModelMappingDict:
    """This function reads a yaml string and returns a ModelMappingDict object.
    Use the yaml tags !Int, !Float, !Cat to specify the type of the hyperparameter distributions.
    The parser takes care of importing the classes specified in the yaml file in the fields model and post_processor.

    Note:
        Despite of automatically importing the classes, no arbitrary code is executed.
        The yaml parser uses the safe loader and a custom constructors for the optuna distributions.
        The imports are done by the importlib module.

    Args:
        yaml_code (str): The yaml code.

    Returns:
        (ModelMappingDict): A dictionary of ModelConfigDict objects.

    Example:

        ```python
        yaml_code = '''
                    RandomForest:
                        model: sklearn.ensemble.RandomForestRegressor
                        post_processor: flexcv.model_postprocessing.MixedEffectsPostProcessor
                        requires_inner_cv: True
                        params:
                            max_depth: !Int
                                low: 1
                                high: 10
                    '''
        model_mapping = read_mapping_from_yaml_string(yaml_code)
        ```

    """
    yaml_output = yaml.load(yaml_code, Loader=get_loader())
    model_mapping = parse_yaml_output_to_mapping_dict(yaml_output)
    return model_mapping


if __name__ == "__main__":
    yaml_code = """
    LinearModel:
        model: flexcv.models.LinearModel
        post_processor: flexcv.model_postprocessing.LinearModelPostProcessor
        params:
            max_depth: !Int
                low: 5
                high: 100
                log: true
            min_impurity_decrease: !Float
                low: 0.00000001
                high: 0.02
            features: !Cat 
                choices: [a, b, c]
    """
    print(read_mapping_from_yaml_string(yaml_code))
