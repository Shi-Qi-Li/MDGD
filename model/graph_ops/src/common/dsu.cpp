#include <numeric>
#include <algorithm>

#include "dsu.hpp"

dsu::dsu(int size_) : pa(size_) {
    iota(pa.begin(), pa.end(), 0);
}

void dsu::unite(int x, int y) {
    x = find(x), y = find(y);
    if (x == y) return;
    pa[y] = x;
}

int dsu::find(int x) { 
    return pa[x] == x ? x : pa[x] = find(pa[x]); 
}