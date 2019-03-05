import glob
from os.path import basename, dirname, isfile

files = glob.glob(dirname(__file__) + "/*.py")
modules = [
    f"purls.algorithms.{basename(f).strip('.py')}"
    for f in files
    if isfile(f) and not basename(f) == "base.py" and not f.endswith("__init__.py")
]


class AlgorithmError(Exception):
    def __init__(self, msg):
        self.msg = msg
