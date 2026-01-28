from __future__ import annotations
import pytest
import torch
from tests.utils import clear_cuda_cache

def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Command line options.
    """
    parser.addoption("--test-mode",
                     action="store",
                     default="full",
                     choices=["forward", "backward", "full"],
                     help="Test mode: forward only, backward only, or full (default: full)",)
    parser.addoption("--warmup-iterations",
                     action="store",
                     default=3,
                     type=int,
                     help="Number of warmup iterations (default: 3)",)
    parser.addoption("--show-timing",
                     action="store_true",
                     default=False,
                     help="Show detailed timing information",)
    parser.addoption("--dtype",
                     action="store",
                     default=None,
                     choices=["float16", "float32", "float64", "bfloat16"],
                     help="Test only specific dtype (default: all)",)
    parser.addoption("--dtypes",
                     action="store",
                     default=None,
                     help="Comma-separated list of dtypes (e.g., 'float16,float32')",)

@pytest.fixture(scope="session")
def test_mode(request: pytest.FixtureRequest) -> str:
    """
    Get the test mode from command line options.
    """
    return request.config.getoption("--test-mode")

@pytest.fixture(scope="session")
def warmup_iterations(request: pytest.FixtureRequest) -> int:
    """
    Get the number of warmup iterations from command line options.
    """
    return request.config.getoption("--warmup-iterations")

@pytest.fixture(scope="session")
def show_timing(request: pytest.FixtureRequest) -> bool:
    """
    Get whether to show timing information from command line options.
    """
    return request.config.getoption("--show-timing")

@pytest.fixture(scope="session")
def device() -> str:
    """
    Get the CUDA device.
    """
    if not torch.cuda.is_available():
        pytest.fail("CUDA is not available")
    return "cuda"

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """
    Clean up CUDA cache after each test.
    """
    yield
    clear_cuda_cache()

@pytest.fixture
def seed() -> int:
    """
    Default random seed for reproducibility.
    """
    return 42

DTYPE_MAP = {"float16": torch.float16,
             "float32": torch.float32,
             "float64": torch.float64,
             "bfloat16": torch.bfloat16,}

def parse_dtypes(dtype_str: str | None, dtypes_str: str | None) -> set[torch.dtype] | None:
    """
    Parse dtype options into a set of torch dtypes.
    """
    if dtype_str is None and dtypes_str is None:
        return None

    selected = set()

    if dtype_str and dtype_str in DTYPE_MAP:
        selected.add(DTYPE_MAP[dtype_str])

    if dtypes_str:
        for dt in dtypes_str.split(","):
            dt = dt.strip().lower()
            if dt in DTYPE_MAP:
                selected.add(DTYPE_MAP[dt])

    return selected if selected else None

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Filter tests by dtype if specified.
    """
    dtype_opt = config.getoption("--dtype")
    dtypes_opt = config.getoption("--dtypes")
    selected = parse_dtypes(dtype_opt, dtypes_opt)

    if selected is not None:
        selected_items = []
        deselected_items = []
        for item in items:
            if hasattr(item, "callspec") and "dtype" in item.callspec.params:
                test_dtype = item.callspec.params["dtype"]
                if test_dtype not in selected:
                    deselected_items.append(item)
                    continue
            selected_items.append(item)
        
        items[:] = selected_items
        config.hook.pytest_deselected(items=deselected_items)