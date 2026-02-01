import ast
from pathlib import Path

import pytest


def get_function_loggers(file_path):
    """Parses a file and returns a list of function names that contain logging.getLogger calls."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    function_loggers = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        if (
                            child.func.attr == "getLogger"
                            and isinstance(child.func.value, ast.Name)
                            and child.func.value.id == "logging"
                        ):
                            if node.name != "_apply_logging_preferences":
                                function_loggers.append(node.name)
    return function_loggers


@pytest.mark.parametrize(
    "file_path",
    [
        "src/blur_api/serve.py",
        "src/blur_cli/cli.py",
        "src/anonymizer/blurring.py",
        "src/anonymizer/detection/core.py",
        "src/anonymizer/tracking/fused.py",
        "src/anonymizer/tracking/botsort.py",
        "src/anonymizer/core.py",
        "src/anonymizer/sahi_integration.py",
        "src/anonymizer/tracking/base.py",
        "src/anonymizer/tracking/embeddings.py",
        "src/anonymizer/detection/model.py",
        "src/anonymizer/tracking/hybrid.py",
    ],
)
def test_no_logger_in_functions(file_path):
    """Verifies that no logging.getLogger calls exist inside functions."""
    full_path = Path(file_path)
    if not full_path.exists():
        pytest.skip(f"File {file_path} not found")

    loggers_in_functions = get_function_loggers(full_path)
    assert (
        not loggers_in_functions
    ), f"Found logging.getLogger in functions: {loggers_in_functions} in {file_path}"
