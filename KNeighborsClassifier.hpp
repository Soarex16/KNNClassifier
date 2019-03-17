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
    /**
     * To improve the readability of the code, a number of types are declared for:
     * feature vector
     * test data labels
     * metric function
     */
    typedef const std::valarray<double>& featureVec;
    typedef double label;
    typedef const std::function<double(featureVec, featureVec, const std::valarray<double>&)>& metricFunc;

    /**
     * Uniform distribution of neighbors weights
     * @param distances vector of distances from one point to their neighbors
     * @return vector of weights
     */
    std::valarray<double> uniformWeights(const std::valarray<double> &distances) {
        std::valarray<double> equalWeights(distances.size(), 1.0 / distances.size());

        return equalWeights;
    }

    /**
     * Inverse square (1/d^2) distribution of neighbors weights
     * @param distances vector of distances from one point to their neighbors
     * @return vector of weights
     */
    std::valarray<double> distanceWeights(const std::valarray<double> &distances) {
        std::valarray<double> inverseDist = distances.apply([](double dist) -> double {return std::pow(dist, -2);});

        return inverseDist;
    }

    /**
     * Standard euclidean metric (square root of the sum of squared differences)
     * @param x fisrt point
     * @param y second point
     * @param metricParams additional metric parameters (i.e. p for minkowsi metric)
     * @return distance between points in current metric
     */
    double euclideanDistance(featureVec x, featureVec y, const std::valarray<double>& metricParams) {
        return std::sqrt(std::pow(x - y, 2).sum());
    }

    /**
     * Minkowski metric sum[(x - y)^p]^(1/p)
     * @param x fisrt point
     * @param y second point
     * @param metricParams vector, the first component of which is the exponent (p)
     * @return distance between points in current metric
     */
    double minkowskiDistance(featureVec x, featureVec y, const std::valarray<double>& metricParams) {
        return std::pow(std::pow(std::abs(x - y), metricParams[0]).sum(), 1 / metricParams[0]);
    }

    /**
     * Base class for all storages of points.
     * Point storage must provide function for calculating distances.
     * The main goal of this class is to move data to a separate storage, which allows adding,
     * deleting points, finding the distance between them, and also searching for K nearest neighbors
     */
    class PointsStorage {
     protected:
      metricFunc metric;
      const std::valarray<double>& metricParams;

     public:
      PointsStorage(metricFunc m, const std::valarray<double>& params) : metric(m), metricParams(params) {}

      /**
       * Note: need to do something about it to increase the readability of the code
       * @param queryPoints vector of points for which neighbors are searched
       * @param k number of neighbors of each point
       * @return A vector of pairs, where the first component of the pair contains
       * the distances to points, and the second component stores their IDs
       */
      virtual std::vector<std::pair< std::valarray<double>, std::valarray<int> >> getNeighbors(const std::vector<featureVec>& queryPoints, int k) = 0;

      // The following method are needed to establish a bijection between training data, their labels and identifiers.
      virtual label getLabel(int idx) = 0;

      /**
       * Operator that allows to get the index of a point by the vector, which represents that point
       * @param feature vector, that represents point coordinates in N dimensional space
       * @return index inside structure
       */
      virtual int operator[](featureVec point) = 0;

      /**
       * Operator that allows to get the feature vector of a point by its internal index
       * @param idx index of a point inside storage
       * @return feature vector of a point with given index
       */
      virtual featureVec operator[] (int idx) = 0;

      /**
       * Methods that implement the addition and removal of a set of points from the storage.
       * These methods are necessary because some storage implementations (for example, a KD tree)
       * can work more efficiently (as I think). Also, these methods allow you to implement additional fitting.
       * @param points point vector to add or remove
       */
      virtual void addPoints(std::vector<featureVec> points) = 0;
      virtual void deletePoints(std::vector<featureVec> points) = 0;

      /**
       * Calculates distance between two points
       * @param p1 first point
       * @param p2 second point
       * @return distance between points
       */
      double distanceBetween(featureVec p1, featureVec p2) {
          return metric(p1, p2, metricParams);
      }

      virtual ~PointsStorage() = default;
    };





    void brute(const std::vector<featureVec>& x, int k) {
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
