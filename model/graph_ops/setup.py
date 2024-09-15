from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "graph_ops",
        [
            "src/common/dsu.cpp",
            "src/common/edge.cpp",
            "src/common/operation.cpp",
            "src/mst.cpp",
            "src/interface.cpp"
        ],
        include_dirs = ["/usr/include/eigen3"],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args = ["-O3"]
    ),
]

setup(
    name="graph_ops",
    version=__version__,
    author="Shiqi Li",
    author_email="lishiqi@stu.xjtu.edu.cn",
    description="",
    long_description="",
    install_requires=["pybind11", "numpy"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7"
)