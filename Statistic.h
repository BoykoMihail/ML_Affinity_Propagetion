/* 
 * File:   Statistic.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:42
 */
#include <vector>
#include <eigen3/Eigen/Core>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef STATISTIC_H
#define	STATISTIC_H

class Statistic {
public:

    static void findeStatistic(const VectorXd &v, double &mean, double &sig) {

        float summOfElements = 0;
        float summOfSquareElements = 0;
        for (int i = 0; i < v.size(); ++i) {
            summOfSquareElements += v(i) * v(i);
        }
        mean = v.mean();
        sig = sqrt(summOfSquareElements / v.size() - mean * mean);
    }

    static void findeStatistic(const vector<double> &v, double &mean, double &sig) {

        float summOfElements = 0;
        float summOfSquareElements = 0;
        for (int i = 0; i < v.size(); ++i) {
            summOfElements += v[i];
            summOfSquareElements += v[i] * v[i];
        }
        mean = summOfElements / v.size();
        sig = sqrt(summOfSquareElements / v.size() - mean * mean);

    }

};

#endif	/* STATISTIC_H */

