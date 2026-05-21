import os

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def resolve_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)
