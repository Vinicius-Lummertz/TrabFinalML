"""Top-level utils module to satisfy unpickler imports (e.g. utils.to_string)."""

def to_string(value):
    """Simple to-string converter used by FunctionTransformer in the saved pipeline."""
    return str(value)
