#pragma once

#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double find_median(std::vector<double> array);

py::array_t<double> get_pose_from_tree(
    int camera_num,
    int root,
    py::array_t<long> tree_matrix,
    py::array_t<double> w_obs
);

py::array_t<double> get_absolute_weight_from_tree(
    int camera_num,
    int root,
    py::array_t<long> tree_matrix,
    py::array_t<double> weight_matrix
);

int get_tree_centroid(
    int camera_num,
    py::array_t<long> tree_matrix
);