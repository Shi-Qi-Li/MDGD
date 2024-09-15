#include <cmath>
#include <string>
#include <cassert>
#include <algorithm>

#include "mst.hpp"
#include "common/dsu.hpp"
#include "common/edge.hpp"

py::array_t<long> kruskal(
        int camera_num,
        py::array_t<double> weight_matrix
    ) {

    py::buffer_info weight_mat = weight_matrix.request();

    assert (weight_mat.ndim == 2);
    assert (weight_mat.shape[0] == weight_mat.shape[1]);
    assert (weight_mat.shape[0] == camera_num);

    auto adjacent = py::array_t<long>(weight_mat.size); 
    py::buffer_info adjacent_mat = adjacent.request();
    
    double* weight_mat_ptr = (double*)weight_mat.ptr;
    long* adjacent_mat_ptr = (long*)adjacent_mat.ptr;

    memset(adjacent_mat_ptr, 0, sizeof(long) * camera_num * camera_num);

    dsu node(camera_num);
    int edge_num = camera_num * (camera_num - 1) / 2;
    std::vector<edge> edges(edge_num);

    int temp = 0;
    for (int i = 0; i < camera_num; i++) {
        for (int j = i + 1; j < camera_num; j++) {
            edges[temp++].set(i, j, weight_mat_ptr[i * camera_num + j]);
        }
    }
    std::sort(edges.begin(), edges.end());

    temp = 0;
    for (auto graph_edge: edges) {
        int pa_x = node.find(graph_edge.x), pa_y = node.find(graph_edge.y); 
        if (pa_x == pa_y) {
            continue;
        }

        adjacent_mat_ptr[graph_edge.x * camera_num + graph_edge.y] = 1;
        adjacent_mat_ptr[graph_edge.y * camera_num + graph_edge.x] = 1;
        
        node.unite(graph_edge.x, graph_edge.y);
        
        temp++;
        if (temp == camera_num - 1) {
            break;
        }
    }

    return adjacent;
}