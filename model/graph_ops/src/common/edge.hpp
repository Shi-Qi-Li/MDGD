#pragma once

class edge {
public:
    int x, y;
    double weight;
    int support_count;
    explicit edge(int x_ = 0, int y_ = 0, double weight_ = 0.0, int support_count_ = 0);
    void set(int x_, int y_, double weight_, int support_count_ = 0);
    bool operator<(const edge e);
};