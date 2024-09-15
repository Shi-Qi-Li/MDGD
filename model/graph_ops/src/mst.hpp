#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<long> kruskal(
    int camera_num,
    py::array_t<double> weight_matrix
);

