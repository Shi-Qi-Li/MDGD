#include <queue>
#include <cassert>
#include <algorithm>
#include <Eigen/Dense>

#include "operation.hpp"

double find_median(std::vector<double> array) { 
  
    int n = array.size();
    assert (n > 0);
    if (n % 2 == 0) { 
        nth_element(array.begin(), array.begin() + n / 2, array.end()); 
        nth_element(array.begin(), array.begin() + (n - 1) / 2, array.end()); 
  
        return (array[(n - 1) / 2] + array[n / 2]) / 2.0; 
    } else { 
        nth_element(array.begin(), array.begin() + n / 2, array.end()); 
  
        return array[n / 2]; 
    } 
}

py::array_t<double> get_pose_from_tree(
        int camera_num,
        int root,
        py::array_t<long> tree_matrix,
        py::array_t<double> w_obs
    ) {

    py::buffer_info tree_mat = tree_matrix.request();
    py::buffer_info w = w_obs.request();
    
    assert (0 <= root && root < camera_num);
    
    assert (tree_mat.ndim == 2);
    assert (w.ndim == 2);

    assert (tree_mat.shape[0] == tree_mat.shape[1]);
    assert (w.shape[0] == w.shape[1]);

    assert (tree_mat.shape[0] == camera_num);
    
    int dim = w.shape[0] / camera_num;
    assert (w.shape[0] == camera_num * dim);

    auto result = py::array_t<double>(camera_num * dim * dim);
    py::buffer_info result_mat = result.request();

    long* tree_mat_ptr = (long*)tree_mat.ptr;
    double* w_ptr = (double*)w.ptr;
    double* result_ptr = (double*)result_mat.ptr;
    memset(result_ptr, 0, sizeof(double) * camera_num * dim * dim);

    std::queue<int> q;
    bool* vis = new bool[camera_num];
    memset(vis, 0, sizeof(bool) * camera_num);
    
    q.push(root);
    for (int i = 0; i < dim; i++) {
        result_ptr[root * dim * dim + i * dim + i] = 1;
    }
    vis[root] = true;

    while (!q.empty()) {
        auto temp = q.front();
        q.pop();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pos(dim, dim), pos_i(dim, dim);
        for (int dx = 0; dx < dim; dx++) {
            for (int dy = 0; dy < dim; dy++) {
                pos(dx, dy) = result_ptr[(temp * dim + dx) * dim + dy];
            }
        }

        for (int i = 0; i < camera_num; i++) {
            if (tree_mat_ptr[temp * camera_num + i] == 1 && !vis[i]) {
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> relative(dim, dim);
                for (int dx = 0; dx < dim; dx++) {
                    for (int dy = 0; dy < dim; dy++) {
                        // notice inverse the temp and i order here to avoid inverse the transform
                        relative(dx, dy) = w_ptr[(i * dim + dx) * camera_num * dim + temp * dim + dy];
                    }
                }

                pos_i = relative * pos;

                for (int dx = 0; dx < dim; dx++) {
                    for (int dy = 0; dy < dim; dy++) {
                        result_ptr[(i * dim + dx) * dim + dy] = pos_i(dx, dy);
                    }
                }

                q.push(i);
                vis[i] = true;
            }
        }
    }

    return result;
}

py::array_t<double> get_absolute_weight_from_tree(
        int camera_num,
        int root,
        py::array_t<long> tree_matrix,
        py::array_t<double> weight_matrix
    ) {

    py::buffer_info tree_mat = tree_matrix.request();
    py::buffer_info weight_mat = weight_matrix.request();
    
    assert (0 <= root && root < camera_num);
    
    assert (tree_mat.ndim == 2);
    assert (weight_mat.ndim == 2);

    assert (tree_mat.shape[0] == tree_mat.shape[1]);
    assert (weight_mat.shape[0] == weight_mat.shape[1]);

    assert (tree_mat.shape[0] == camera_num);
    assert (weight_mat.shape[0] == camera_num);
    
    auto result = py::array_t<double>(camera_num * 2);
    py::buffer_info result_mat = result.request();

    long* tree_mat_ptr = (long*)tree_mat.ptr;
    double* weight_ptr = (double*)weight_mat.ptr;
    double* result_ptr = (double*)result_mat.ptr;
    memset(result_ptr, 0, sizeof(double) * camera_num * 2);

    std::queue<int> q;
    bool* vis = new bool[camera_num];
    memset(vis, 0, sizeof(bool) * camera_num);
    
    q.push(root);
    result_ptr[root * 2] = 1;
    result_ptr[root * 2 + 1] = 0;
    vis[root] = true;

    while (!q.empty()) {
        auto temp = q.front();
        q.pop();

        double weight = result_ptr[temp * 2];
        double step = result_ptr[temp * 2 + 1];

        for (int i = 0; i < camera_num; i++) {
            if (tree_mat_ptr[temp * camera_num + i] == 1 && !vis[i]) {
                q.push(i);
                vis[i] = true;
                result_ptr[i * 2] = weight * weight_ptr[temp * camera_num + i];
                result_ptr[i * 2 + 1] = step + 1;
            }
        }
    }

    return result;
}

void dfs_centroid(int u, int fa, int camera_num, std::vector<int> node[], std::vector<int>& size, std::vector<int>& weight) {
    size[u] = 1;
    weight[u] = 0;
    for (auto v: node[u]) {
        if (v == fa) continue;
        dfs_centroid(v, u, camera_num, node, size, weight);
        size[u] += size[v];
        weight[u] = std::max(weight[u], size[v]);
    }
    weight[u] = std::max(weight[u], camera_num - size[u]);
}

int get_tree_centroid(int camera_num, py::array_t<long> tree_matrix) {
    py::buffer_info tree_mat = tree_matrix.request();
    
    assert (tree_mat.ndim == 2);
    assert (tree_mat.shape[0] == tree_mat.shape[1]);
    assert (tree_mat.shape[0] == camera_num);

    long* tree_mat_ptr = (long*)tree_mat.ptr;
    
    std::vector<int> node[camera_num];
    std::vector<int> size(camera_num, 0);
    std::vector<int> weight(camera_num, 0);

    for (int i = 0; i < camera_num; i++) {
        for (int j = i + 1; j < camera_num; j++) {
            if (tree_mat_ptr[i * camera_num + j] == 1) {
                node[i].push_back(j);
                node[j].push_back(i);
            }
        }
    }

    dfs_centroid(0, -1, camera_num, node, size, weight);
    
    int max_num = camera_num, centroid = -1;
    for (int i = 0; i < camera_num; i++) {
        if (weight[i] < max_num) {
            max_num = weight[i];
            centroid = i;
        }
    }

    assert (centroid != -1);

    return centroid;
}