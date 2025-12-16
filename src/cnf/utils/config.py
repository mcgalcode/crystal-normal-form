import os

def should_use_rust():
    return os.environ.get("USE_RUST") is not None