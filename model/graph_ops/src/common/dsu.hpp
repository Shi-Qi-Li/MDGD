#pragma once

#include <vector>

class dsu {
private:
    std::vector<int> pa;
public:
    explicit dsu(int size_);
    void unite(int x, int y);
    int find(int x);
};