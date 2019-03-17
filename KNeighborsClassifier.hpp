//
// Created by Soarex16 on 16.03.2019.
//

#ifndef KNEIGHBORSCLASSIFIER_HPP
#define KNEIGHBORSCLASSIFIER_HPP

#include <valarray>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <functional>

namespace knn {
    std::valarray<double> uniformWeights(const std::valarray<double> &distances) {
        std::valarray<double> equalWeights(distances.size(), 1.0 / distances.size());

        return equalWeights;
    }

    std::valarray<double> distanceWeights(const std::valarray<double> &distances) {
        std::valarray<double> inverseDist = distances.apply([](double dist) -> double {return std::pow(dist, -2);});

        return inverseDist;
    }

    double euclideanDistance(const std::valarray<double>& x, const std::valarray<double>& y, const std::valarray<double>& metricParams) {
        return std::sqrt(std::pow(x - y, 2).sum());
    }

    double minkowskiDistance(const std::valarray<double>& x, const std::valarray<double>& y, const std::valarray<double>& metricParams) {
        return std::pow(std::pow(std::abs(x - y), metricParams[0]).sum(), 1 / metricParams[0]);
    }

    void brute(const std::vector<std::valarray<double>>& x, std::vector<std::vector<double>>& distances, int k) {
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> distancesToCurrent(x.size() + 1);
            distancesToCurrent[x.size()] = 0;

        }
    }

    class KNeighborsClassifier {
    private:
        //TODO: ОБУЧАЮЩУЮ ВЫБОРКУ НАДО ВСЕ-ТАКИ ГДЕ-ТО ХРАНИТЬ, ПОЭТОМУ ПРИДЕТСЯ ВСЕ ПЕРЕДЕЛАТЬ (НО ЭТО ХОРОШО,
        // ПОТОМУ ЧТО АЛГОРИТМЫ СТАНУТ ПРОЩЕ) С УЧЕТОМ ТОГО, ЧТО ДАННЫЕ МЫ ХРАНИМ
        // ЕСТЬ ОТЛИЧНОЕ ПРЕДЛОЖЕНИЕ ПЕРЕПИСАТЬ НЕКОТОРЫЕ ФУНКЦИИ С НУЛЯ
        int neighborsNum;
        std::vector<std::vector<double>> distances;
        std::map<std::valarray<double>, int> datasetMapping;
        std::vector<double> labels;
        const std::valarray<double> metricParams;

        std::function<void (const std::vector<std::valarray<double>>& x, std::vector<std::vector<double>>& distances, int k)> nnAlgorithm;
        std::function<std::valarray<double>(const std::valarray<double>&)> weights;
        std::function<double(const std::valarray<double>&, const std::valarray<double>&, const std::valarray<double>&)> metric;

    public:
        KNeighborsClassifier(int neighborsNum = 5,
            std::function<std::valarray<double>(const std::valarray<double>&)> weights = uniformWeights,
            std::function<void (const std::vector<std::valarray<double>>& x, std::vector<std::vector<double>>& distances, int k)> nnAlgorithm = brute,
            std::function<double(const std::valarray<double>&, const std::valarray<double>&, const std::valarray<double>&)> metric = minkowskiDistance,
            std::valarray<double> metricParams = {2.0}) :
            neighborsNum(neighborsNum), weights(std::move(weights)), nnAlgorithm(std::move(nnAlgorithm)),
            metric(std::move(metric)), metricParams(std::move(metricParams)) {}

            // 1 - dists, 2 - idx
            // indexes are global in this->distances
            // sorted by distance
            std::vector<std::pair<std::valarray<double>, std::valarray<int>>> kneighbors(const std::vector<std::valarray<double>>& queryPoints) {
                std::vector<std::pair<std::valarray<double>, std::valarray<int>>> neighbors(queryPoints.size());

                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int idx = datasetMapping[queryPoints[i]];
                    std::vector<std::pair<double, int>> dists(distances[idx].size());

                    for (size_t j = 0; j < distances[idx].size(); ++j) {
                        dists[j] = std::make_pair(distances[idx][j], j);
                    }

                    std::sort(dists.begin(), dists.end());

                    for (size_t j = 0; j < neighborsNum; ++j) {
                        neighbors[i].first[j] = dists[j].first;
                        neighbors[i].second[j] = dists[j].second;
                    }
                }
            }

            void fit(const std::vector<std::valarray<double>>& x, const std::valarray<int>& y) {
                // calculate distances
                nnAlgorithm(x, y, distances, neighborsNum);

                // find K nearest neighbors for each point in train data
                std::vector<std::pair<std::valarray<double>, std::valarray<int>>> neighborsForEveryPoint = kneighbors(x);

                int beginningIdx = labels.size();
                labels.resize(labels.size(), y.size());

                for(size_t i = 0; i < neighborsForEveryPoint.size(); ++i) {
                    auto pointData = neighborsForEveryPoint[i];

                    std::map<int, double> weightedDecision;

                    std::valarray<double> neighborsWeights = weights(pointData.first);

                    for (size_t j = 0; i < neighborsWeights.size(); ++i) {
                        weightedDecision[pointData.second[j]] += neighborsWeights[j];
                    }

                    auto predictedClass = (*std::max_element(weightedDecision.begin(), weightedDecision.end(),
                        [](const std::pair<int, double>& p1, std::pair<int, double>& p2) {
                          return p1.second < p2.second;})).first;

                    datasetMapping[x[i]] = beginningIdx + i;
                    labels[beginningIdx + i] = predictedClass;
                }
            }

            std::vector<int> predict(const std::vector<std::valarray<double>>& x) {
                std::vector<int> prediction;

                std::vector<std::pair<std::valarray<double>, std::valarray<int>>> neighborsForEveryPoint = kneighbors(x);

                for(size_t i = 0; i < neighborsForEveryPoint.size(); ++i) {
                    auto pointData = neighborsForEveryPoint[i];

                    std::map<int, double> weightedDecision;

                    std::valarray<double> neighborsWeights = weights(pointData.first);

                    for (size_t j = 0; i < neighborsWeights.size(); ++i) {
                        weightedDecision[pointData.second[j]] += neighborsWeights[j];
                    }

                    auto predictedClass = (*std::max_element(weightedDecision.begin(), weightedDecision.end(),
                                                             [](const std::pair<int, double>& p1, std::pair<int, double>& p2) {
                                                               return p1.second < p2.second;})).first;

                    prediction[i] = predictedClass;
                }

                return prediction;
            }

            double score(const std::vector<std::valarray<double>>& samples, std::vector<double> labels) {
                std::cout << "WTF?" << std::endl;
            }
    };
}

#endif //KNEIGHBORSCLASSIFIER_HPP
