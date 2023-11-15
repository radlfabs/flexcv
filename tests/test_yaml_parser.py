import pytest
import unittest.mock as mock
from optuna.distributions import IntDistribution, CategoricalDistribution, FloatDistribution
import yaml

from flexcv.models import LinearModel
from flexcv.model_postprocessing import LinearModelPostProcessor
from flexcv.yaml_parser import read_mapping_from_yaml_string, read_mapping_from_yaml_file
from flexcv.yaml_parser import parse_yaml_output_to_mapping_dict
from flexcv.yaml_parser import get_loader
from flexcv.model_mapping import ModelMappingDict, ModelConfigDict

def test_parse_yaml_output_to_mapping_dict():
    # Mock the importlib.import_module function
    with mock.patch('importlib.import_module') as mock_import_module:
        # Define a mock module with a mock class
        mock_module = mock.Mock()
        mock_class = mock.Mock()
        setattr(mock_module, 'MockClass', mock_class)
        mock_import_module.return_value = mock_module

        # Define a sample yaml_dict
        
        yaml_dict = {
            'MockModel': {
                'model': 'mock_module.MockClass',
                'post_processor': 'mock_module.MockClass',
                'param1': 'value1',
                'param2': 'value2',
            }
        }

        # Call the function
        result = parse_yaml_output_to_mapping_dict(yaml_dict)

        # Check the result
        assert isinstance(result, ModelMappingDict)
        assert 'MockModel' in result
        assert isinstance(result['MockModel'], ModelConfigDict)
        assert result['MockModel']['model'] == mock_class
        assert result['MockModel']['post_processor'] == mock_class
        assert result['MockModel']['param1'] == 'value1'
        assert result['MockModel']['param2'] == 'value2'

def test_parse_yaml_output_to_mapping_dict_missing_model():
    # Define a sample yaml_dict with missing 'model'
    yaml_dict = {
        'MockModel': {
            'post_processor': 'mock_module.MockClass',
            'param1': 'value1',
            'param2': 'value2'
        }
    }

    # Call the function and check for ValueError
    with pytest.raises(ValueError) as excinfo:
        parse_yaml_output_to_mapping_dict(yaml_dict)
    assert str(excinfo.value) == 'model name is missing for model MockModel'

def test_read_mapping_from_yaml_string():
    # Mock the yaml.load and parse_yaml_output_to_mapping_dict functions
    with mock.patch('yaml.load') as mock_load, \
         mock.patch('flexcv.yaml_parser.parse_yaml_output_to_mapping_dict') as mock_parse:
        # Define a mock yaml_output and mock model_mapping
        mock_yaml_output = mock.Mock()
        mock_model_mapping = mock.Mock()
        mock_load.return_value = mock_yaml_output
        mock_parse.return_value = mock_model_mapping

        # Define a sample yaml_code
        yaml_code = 'MockYamlCode'

        # Call the function
        result = read_mapping_from_yaml_string(yaml_code)

        # Check the result
        assert result == mock_model_mapping

        # Check the calls to the mocked functions
        mock_load.assert_called_once_with(yaml_code, Loader=get_loader())
        mock_parse.assert_called_once_with(mock_yaml_output)

def test_read_mapping_from_yaml_string_invalid_yaml():
    # Mock the yaml.load function to raise a YAMLError
    with mock.patch('yaml.load') as mock_load:
        mock_load.side_effect = yaml.YAMLError

        # Define a sample yaml_code
        yaml_code = 'InvalidYamlCode'

        # Call the function and check for YAMLError
        with pytest.raises(yaml.YAMLError):
            read_mapping_from_yaml_string(yaml_code)
            
def test_read_mapping_from_yaml_string_valid_yaml():
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

    python_result = read_mapping_from_yaml_string(yaml_code)
    assert isinstance(python_result, ModelMappingDict)
    assert 'LinearModel' in python_result
    assert isinstance(python_result['LinearModel'], ModelConfigDict)
    assert python_result['LinearModel']['model'] == LinearModel
    assert python_result['LinearModel']['post_processor'] == LinearModelPostProcessor
    assert python_result['LinearModel']['params']['max_depth'] == IntDistribution(low=5, high=100, log=True)
    assert python_result['LinearModel']['params']['min_impurity_decrease'] == FloatDistribution(low=0.00000001, high=0.02)
    assert python_result['LinearModel']['params']['features'] == CategoricalDistribution(choices=['a', 'b', 'c'])

def test_read_mapping_from_yaml_file_valid_yaml():
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
    
    # write it to a file
    with open('test.yaml', 'w') as f:
        f.write(yaml_code)
    
    python_result = read_mapping_from_yaml_file("test.yaml")
    assert isinstance(python_result, ModelMappingDict)
    assert 'LinearModel' in python_result
    assert isinstance(python_result['LinearModel'], ModelConfigDict)
    assert python_result['LinearModel']['model'] == LinearModel
    assert python_result['LinearModel']['post_processor'] == LinearModelPostProcessor
    assert python_result['LinearModel']['params']['max_depth'] == IntDistribution(low=5, high=100, log=True)
    assert python_result['LinearModel']['params']['min_impurity_decrease'] == FloatDistribution(low=0.00000001, high=0.02)
    assert python_result['LinearModel']['params']['features'] == CategoricalDistribution(choices=['a', 'b', 'c'])
    # remove the file
    import pathlib
    pathlib.Path("test.yaml").unlink()
    assert not pathlib.Path("test.yaml").exists()