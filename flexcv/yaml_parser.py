import yaml
import importlib
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict


def int_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> IntDistribution:
    """Construct an IntDistribution."""
    return IntDistribution(**loader.construct_mapping(node))


def float_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> FloatDistribution:
    """Construct an FloatDistribution."""
    return FloatDistribution(**loader.construct_mapping(node))


def cat_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> CategoricalDistribution:
    """Construct an FloatDistribution."""
    value = loader.construct_mapping(node, deep=True)
    return CategoricalDistribution(**value)
    
    
def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!Int", int_constructor)
    loader.add_constructor("!Float", float_constructor)
    loader.add_constructor("!Cat", cat_constructor)
    return loader


def split_import(import_string) -> tuple[str, str]:
    import_list = import_string.split(".")
    class_name = import_list[-1]
    module_path = ".".join(import_list[:-1])
    return class_name, module_path


def parse_yaml_output_to_mapping_dict(yaml_dict):
    model_mapping_dict = ModelMappingDict()
    for model_name, raw_dict in yaml_dict.items():
        if "model" not in raw_dict:
            raise ValueError(f"model name is missing for model {model_name}")
        
        imports_dict = {}
        import_list = ["model"]
        if "post_processor" in raw_dict:
            import_list.append("post_processor")
        
        for key in import_list:
            class_name, path = split_import(raw_dict[key])
            foo = importlib.import_module(path)
            imports_dict[key] = getattr(foo, class_name)

        # replace the strings in raw_dict with the imported classes
        raw_dict.update(imports_dict)
        model_config_dict = ModelConfigDict(raw_dict)
        model_mapping_dict[model_name] = model_config_dict
    return model_mapping_dict


def read_mapping_from_yaml_file(yaml_file_path):
    with open(yaml_file_path, "r") as f:
        yaml_output = yaml.load(f, Loader=get_loader())
    model_mapping = parse_yaml_output_to_mapping_dict(yaml_output)
    return model_mapping


def read_mapping_from_yaml_string(yaml_code):
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