#include <iostream>
#include <valarray>
#include "KNeighborsClassifier.hpp"

int main() {
    int n;
    std::cin >> n;
    std::valarray<double> v(n);

    for (int i = 0; i < n; ++i) {
        std::cin >> v[i];
    }

    auto distResult = knn::distanceWeights(v);
    auto uniformResult = knn::uniformWeights(v);

    for (int i  = 0; i < n; ++i) {
        std::cout <<  distResult[i] << " ";
    }

    return 0;
}