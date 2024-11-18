import os
import sys
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

pkg = "simm"


def scansubdir(dir):
    """should be recursive if there are submodules inside modules"""
    subdirs = []
    for f in os.listdir(dir):
        p = os.path.join(dir, f)
        if f is not None and os.path.isdir(p) and not f.startswith(".") and not f.startswith("__"):
            subdirs.append(p.replace(os.path.sep, "."))
    return subdirs


subdirs = scansubdir(pkg)
# print('\n'.join(subdirs))
def scandir(dir, files=[]):
    for f in os.listdir(dir):
        if f == "__init__.py":
            continue
        p = os.path.join(dir, f)
        if os.path.isfile(p) and p.endswith(".py"):
            files.append(p.replace(os.path.sep, ".")[:-3])
        elif os.path.isdir(p):
            scandir(p, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep) + ".py"
    return Extension(
        extName,
        [extPath],
        include_dirs=["."],
    )


extNames = scandir(pkg)
extensions = [makeExtension(name) for name in extNames]
print("\n".join([e.sources[0] for e in extensions]))
setup(name=pkg, packages=subdirs, ext_modules=cythonize(extensions))
