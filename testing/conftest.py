import pytest
from pathlib import Path
import requests
import tarfile
from shutil import rmtree
import numpy as np

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")

ms_path = Path(test_data_path, "test8chans2.ms")
initmodel_path = Path(test_data_path, "t80-ms-init-point-cc-model.npy")
dummymodel_path = Path(test_data_path, "t80-ms-dummy-cc-model.npy")

data_lnk = "https://www.dropbox.com/s/8e49mfgsh4h6skq/C147_subset.tar.gz"
_data_tar_name = "C147_subset.tar.gz"
data_tar_path = Path(test_data_path, _data_tar_name)

tar_lnk_list = [data_lnk]
tar_pth_list = [data_tar_path]
dat_pth_list = [ms_path]

def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        for lnk, pth in zip(tar_lnk_list, tar_pth_list):
            download = requests.get(lnk, params={"dl": 1})
            with open(pth, 'wb') as f:
                f.write(download.content)
            with tarfile.open(pth, "r:gz") as tar:
                tar.extractall(path=test_data_path)
            pth.unlink()
        print("Test data successfully downloaded.")

# def pytest_sessionfinish(session, exitstatus):
#     """Called after test run finished, before returning exit status."""

#     for pth in dat_pth_list:
#         if pth.exists():
#             print("\nRemoving test data ({}).".format(pth))
#             rmtree(pth)
#             print("Test data successfully removed.")

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
