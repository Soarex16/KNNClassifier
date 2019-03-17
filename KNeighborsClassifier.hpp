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
     * metric (distance) function
     * weights calculation function
     */
    typedef const std::valarray<double>& featureVec;
    typedef double label;
    typedef const std::function<double (featureVec, featureVec, const std::valarray<double>&)>& metricFunc;
    typedef const std::function<std::valarray<double> (const std::valarray<double> &distances)>& weightsFunc;

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
      virtual std::vector< std::valarray<std::pair<double, int>> > getNeighbors(const std::vector<featureVec>& queryPoints, int k) = 0;

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
       * @param pointsVec point vector to add or remove
       * @param labels labels for each point
       */
      virtual void addPoints(const std::vector<featureVec>& pointsVec, const std::vector<label>& labelsVec) = 0;
      virtual void deletePoints(const std::vector<featureVec>& pointsVec) = 0;

      /**
       * Calculates distance between two points
       * @param p1 first point
       * @param p2 second point
       * @return distance between points
       */
      double distanceBetween(featureVec p1, featureVec p2) {
          return metric(p1, p2, metricParams);
      }

      ~PointsStorage() = default;
    };

    /**
    * Very inefficient storage implementation
    */
    class NaiveDataStorage: public PointsStorage {
     private:
      std::vector<featureVec> points;
      std::vector<featureVec> labels;

      // stupidly store the matrix of distances
      std::vector<std::vector<double>> distances;
     public:
      virtual int operator[](featureVec point) {
          auto it = std::find(points.begin(), points.end(), point);

          return *it;
      }

      virtual featureVec operator[] (int idx) {
          return points[idx];
      }

      virtual std::vector< std::valarray<std::pair<double, int>> > getNeighbors(const std::vector<featureVec>& queryPoints, int k) {
          std::vector< std::valarray<std::pair<double, int>> > result(queryPoints.size());

          for (size_t i = 0; i < queryPoints.size(); ++i) {
              std::valarray< std::pair<double, int> > dists(queryPoints[i].size());

              // get index of a point
              auto pointIdx = this->operator[](queryPoints[i]);

              // copy dists and indexes, sort them and select first k
              std::vector<std::pair<double, int>> v(distances[i].size());
              for (int j = 0; j < distances[i].size(); ++j) {
                  v[j] = std::make_pair(distances[i][j], j);
              }

              std::sort(v.begin(), v.end());

              for (size_t j = 1; j <= k; ++j) {
                  dists[i] = v[j];
              }

              result[i] = dists;
          }

          return result;
      }

      virtual label getLabel(int idx) {
          return labels[idx];
      }

      /**
       * For a good account, it is not necessary to recalculate
       * the entire table of distances, but only to change its rows and columns,
       * but since this implementation is purely training, efficiency issues are
       * not considered here. Perhaps in the future will be optimized.
       *
       * Premature optimization is the root of all evil (or at least most of it) in programming.
       * /~ Donald knuth
       */
      virtual void addPoints(const std::vector<featureVec>& pointsVec, const std::vector<label>& labelsVec) {
          points.insert(points.end(), pointsVec.begin(), pointsVec.end());
          labels.insert(labels.end(), labelsVec.begin(), labelsVec.end());

          std::vector<std::vector<double>> newDists(points.size());

          for (size_t i = 0; i < points.size(); ++i) {
              std::vector<double> v(points.size(), 0);
              newDists[i] = v;
          }

          // fills the elements above the main diagonal
          for (size_t i = 0; i < points.size(); ++i) {
              for (size_t j = i + 1; j < points.size(); ++j) {
                  newDists[i][j] = distanceBetween(points[i], points[j]);
              }
          }

          // because p (x, y) = p (y, x) we just copy the values
          for (size_t i = 0; i < points.size(); ++i) {
              for (size_t j = 0; j < i; ++j) {
                  newDists[i][j] = newDists[j][i];
              }
          }
      }

      virtual void deletePoints(const std::vector<featureVec>& pointsVec) {
          for (size_t i = 0; i < pointsVec.size(); ++i) {
              auto idx = this->operator[](pointsVec[i]);
              points.erase(idx);
              labels.erase(idx);
          }
      }

      ~NaiveDataStorage() = default;
    };

    /**
     * Main class of the library
     * Naive implementation of the K nearest neighbors method
     */
    class KNeighborsClassifier {
    private:
        int neighborsNum;
        PointsStorage& points;
        weightsFunc weights;

    public:
        KNeighborsClassifier(PointsStorage& pointsStorage, int neighborsNum = 5, weightsFunc w = uniformWeights) :
            neighborsNum(neighborsNum), points(pointsStorage), weights(w) {}

            /**
             * Adds points to point storage, that calculates distances between them
             * @param x feature vectors of points
             * @param y true labels
             */
            void fit(const std::vector<featureVec>& x, const std::vector<label>& y) {
                points.addPoints(x, y);
            }

            /**
             * Predicts class for each feature vector
             * @param x data to predict
             * @return vector of predicted labels
             */
            std::vector<label> predict(const std::vector<featureVec>& x) {
                std::vector<label> prediction;

                auto neighborsForEveryPoint = points.getNeighbors(x, neighborsNum);

                for(size_t i = 0; i < neighborsForEveryPoint.size(); ++i) {
                    // valarray that contains distances to points and their indexes
                    auto pointData = neighborsForEveryPoint[i];

                    std::valarray<double> neighborsWeights(pointData.size());
                    for (int j = 0; j < pointData.size(); ++j) {
                        neighborsWeights[j] = pointData[j].first;
                    }

                    // calculate weights for every neighbor
                    neighborsWeights = weights(neighborsWeights);

                    std::map<int, double> weightedDecision;
                    // add to decision weighted values
                    for (size_t j = 0; i < neighborsWeights.size(); ++i) {
                        weightedDecision[pointData[j].second] += neighborsWeights[j];
                    }

                    // find index with maximum value
                    auto predictedClass = (*std::max_element(weightedDecision.begin(), weightedDecision.end(),
                                                             [](const std::pair<int, double>& p1, std::pair<int, double>& p2) {
                                                               return p1.second < p2.second;})).first;

                    prediction[i] = predictedClass;
                }

                return prediction;
            }

            /**
             * Returns a score of KNNClassifier
             * @param samples test data
             * @param labels true labels
             * @return MSE
             */
            double score(const std::vector<featureVec>& samples, std::vector<double> labels) {
                auto predicted = predict(samples);

                double sum = 0;
                for (size_t i = 0; i < predicted.size(); ++i) {
                    sum += std::pow(predicted[i] - labels[i], 2);
                }

                return sum / predicted.size();
            }
    };
}

#endif //KNEIGHBORSCLASSIFIER_HPP
