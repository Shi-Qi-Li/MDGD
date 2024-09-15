#include "mst.hpp"
#include "common/operation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(graph_ops, m) {
    m.def("kruskal", &kruskal);
    m.def("get_pose_from_tree", &get_pose_from_tree);
    m.def("get_tree_centroid", &get_tree_centroid);
    m.def("get_absolute_weight_from_tree", &get_absolute_weight_from_tree);
}