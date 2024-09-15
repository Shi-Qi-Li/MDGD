#include "edge.hpp"

edge::edge(int x_, int y_, double weight_, int support_count_) : x(x_), y(y_), weight(weight_), support_count(support_count_) {}

void edge::set(int x_, int y_, double weight_, int support_count_) {
    this->x = x_;
    this->y = y_;
    this->weight = weight_;
    this->support_count = support_count_;
}

bool edge::operator<(const edge e) {
    return weight > e.weight;
}