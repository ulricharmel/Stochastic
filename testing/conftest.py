import pytest
from pathlib import Path
import numpy as np

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")

ms_path = Path(test_data_path, "test8chans2.ms")
initmodel_path = Path(test_data_path, "t80-ms-init-point-cc-model.npy")
dummymodel_path = Path(test_data_path, "t80-ms-dummy-cc-model.npy")

@pytest.fixture(scope="session")
def msname():
    """Session level fixture for test data path."""

    return str(ms_path)

@pytest.fixture(scope="session")
def initmodel():
    """Session level fixture for init model"""

    return str(initmodel_path)

@pytest.fixture(scope="session")
def dummymodel():
    """Session level fixture for dummy model path."""

    return str(dummymodel_path)

@pytest.fixture(scope="module")
def freq0():
    return 1450000000
