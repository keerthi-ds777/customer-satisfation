import sys
import pytest
import logging
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_run_pipeline_logic():
    """
    Unit test for run_pipeline logic bypassing ZenML orchestration.
    """
    logger.info("Starting test_run_pipeline_logic")

    # Create a mock for zenml module
    mock_zenml = MagicMock()
    
    # Define behavior for @pipeline decorator
    # Usage: @pipeline(enable_cache=True) -> returns decorator -> returns function
    def pipeline_wrapper(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    mock_zenml.pipeline.side_effect = pipeline_wrapper

    logger.info("Mocking 'zenml' module and '@pipeline' decorator")

    # Patch 'zenml' in sys.modules so the import in _run_pipeline uses our mock
    with patch.dict(sys.modules, {'zenml': mock_zenml}):
        # Ensure imports are fresh
        if 'pipeline._run_pipeline' in sys.modules:
            del sys.modules['pipeline._run_pipeline']
        
        # We might also need to mock steps if they depend on real zenml or heavy libs
        # For now, let's assume we can patch them *after* import or mock them via sys.modules too
        # But since they are imported `from steps...`, if we install deps, they might load fine.
        # Alternatively, we can mock `steps` package to speed up.
        
        # Let's mock the steps modules to avoid importing them (and their deps like tensorflow)
        mock_steps = MagicMock()

        logger.info("Mocking 'steps' and 'src' modules to isolate logic")
        with patch.dict(sys.modules, {
            'steps': mock_steps,
            'steps.ingest_data': mock_steps.ingest_data,
            'steps.preprocessing': mock_steps.preprocessing,
            'steps.training_data': mock_steps.training_data,
            'steps.evaluate': mock_steps.evaluate,
            'src': MagicMock(),
            'src.data_cleaning': MagicMock(),
        }):
            # Import the module under test
            from pipeline import _run_pipeline
            
            # Since we mocked the modules `steps...`, the imports in _run_pipeline
            # `from steps.ingest_data import ingesting` will get attributes from our mocks.
            # So `_run_pipeline.ingesting` is `mock_steps.ingest_data.ingesting`
            
            # Setup the mocked functions
            mock_ingesting = _run_pipeline.ingesting
            mock_cleaning = _run_pipeline.cleaning
            mock_train = _run_pipeline.train_model
            mock_evaluate = _run_pipeline.evaluate_model
            
            # Configure return values
            mock_data = MagicMock(name="ingested_data")
            mock_ingesting.return_value = mock_data
            
            mock_cleaning.return_value = ("xtrain", "xtest", "ytrain", "ytest")
            mock_train.return_value = ("model", "pred")
            mock_evaluate.return_value = (0.95, 0.02, 0.98, 0.05)
            
            # Act
            # Because of our zenml mock, run_pipeline is just the plain function now
            logger.info("Calling _run_pipeline.run_pipeline")
            _run_pipeline.run_pipeline("data/dummy.csv")
            
            # Assert
            logger.info("Verifying mock calls")
            mock_ingesting.assert_called_once_with(data_path="data/dummy.csv")
            mock_cleaning.assert_called_once_with(mock_data)
            mock_train.assert_called_once_with("xtrain", "xtest", "ytrain", "ytest")
            mock_evaluate.assert_called_once_with("ytest", "pred")
            
    logger.info("test_run_pipeline_logic completed successfully")

if __name__ == "__main__":
    test_run_pipeline_logic()