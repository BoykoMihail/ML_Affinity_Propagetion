/* 
 * File:   main.cpp
 * Author: boyko_mihail
 *
 * Created on 24 ноября 2019 г., 12:36
 */

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Statistic.h"
#include <random>
#include <set>
#include <eigen3/Eigen/Sparse>

using namespace std;
using namespace Eigen;

/*
 * 
 */

struct Edge {
    int source;
    int destination;
    double similarity;
    double responsibility;
    double availability;

    Edge(int source, int destination, double similarity) {
        this->source = source;
        this->destination = destination;
        this->similarity = similarity;
        this->availability = 0;
        this->responsibility = 0;
    }

    bool operator<(const Edge& rhs) const {
        return similarity < rhs.similarity;
    }
};

typedef vector<Edge*> Edges;

struct Graph {
    int n; // the number of vertices
    Edges* outEdges; // array of out edges of corresponding vertices
    Edges* inEdges; // array of in edges of corresponding vertices
    vector<Edge> edges; // all edges
};


void readLocation(char* data, vector<vector<int>> &vec);
void read_training_text_edges(char* data, Graph* graph);
void destroyGraph(Graph* graph);
inline void update(double& variable, double newValue, double damping);
void updateResponsibilities(Graph* graph, double damping);
void updateAvailabilities(Graph* graph, double damping);
bool updateExamplars(Graph* graph, vector<int>& examplar);
void clusteringGraph(Graph* graph, vector<int>& examplar, int maxit, double damping);

bool pred(const std::pair< int, int > &a, const std::pair< int, int > &b) {
    return a.second > b.second;
}

int main(int argc, char** argv) {

    Graph* graph = new Graph; // граф пользователей
    int maxit = 30;
    double damping = 0.1;
    std::vector<double> Results(0);
    std::vector<double> ResultsAll(0);

    // Считываем граф пользователей
    read_training_text_edges("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/AP/Gowalla_edges.txt", graph);


    //Вектор локаций
    vector<vector<int>> vecOfLocation(graph->n);

    // Считываем Локализации по пользователям
    readLocation("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/AP/Gowalla_totalCheckins.txt", vecOfLocation);

    vector<int> examplar(graph->n, -1); // К какому классу принадлежит i-ый экземпляр

    // Кластеризация
    clusteringGraph(graph, examplar, maxit, damping);


    std::ofstream myfileHist;
    myfileHist.open("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/AP/AfinityPropagetion_Boyko/Results_Hist.csv");
    for (int i = 0; i < examplar.size(); ++i) {
        myfileHist << examplar[i] << ",";
    }
    myfileHist << "\n";
    myfileHist.close();

    // Shuffle
    std::vector<int> indexes(examplar.size());
    for (int i = 0; i < examplar.size(); ++i) {
        indexes[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);


    int crossValCount = examplar.size() / 5;

    for (int cr = 0; cr < 5; cr++) {

        vector<int> examplarTrain;
        vector<int> examplarTest;
        vector<vector<int>> vecTrain(examplar.size() - crossValCount);
        vector<vector<int>> vecTest(crossValCount);

        int indexTrain = 0;
        int indexTest = 0;
        for (int j = 0; j < indexes.size(); j++) {
            if (j < crossValCount * cr || j >= crossValCount * (cr + 1)) {
                examplarTrain.push_back(examplar[indexes[j]]);
                vecTrain[indexTrain] = vecOfLocation[indexes[j]];
                indexTrain++;
            } else {
                examplarTest.push_back(examplar[indexes[j]]);
                vecTest[indexTest] = vecOfLocation[indexes[j]];
                indexTest++;
            }
        }

        std::map<int, int> topByTrainAll;
        for (int i = 0; i < vecTrain.size(); ++i) {
            for (int j = 0; j < vecTrain[i].size(); ++j) {
                topByTrainAll[vecTrain[i][j]]++;
            }
        }

        std::vector < std::pair< int, int > > TOPLocationclusterAllTrain;

        std::vector< std::pair< int, int > > veccc(topByTrainAll.begin(), topByTrainAll.end());
        std::sort(veccc.begin(), veccc.end(), pred);
        TOPLocationclusterAllTrain = std::vector < std::pair< int, int > >(veccc.begin(), veccc.begin() + (veccc.size() < 10 ? veccc.size() : 10));

        // Считаем количества вхождений мест для каждого кластера
        std::map< int, std::map<int, int>> topByTrainInEachCluster;
        for (int i = 0; i < vecTrain.size(); ++i) {
            for (int j = 0; j < vecTrain[i].size(); ++j) {
                topByTrainInEachCluster[examplarTrain[i]][vecTrain[i][j]]++;
            }
        }

        // Одбираем Top-10
        std::map< int, std::vector < std::pair< int, int > > > TOPLocationcluster;

        for (std::pair< int, std::map<int, int>> map : topByTrainInEachCluster) {
            std::vector< std::pair< int, int > > veccc(map.second.begin(), map.second.end());
            std::sort(veccc.begin(), veccc.end(), pred);
            TOPLocationcluster[map.first] = std::vector < std::pair< int, int > >(veccc.begin(), veccc.begin() + (veccc.size() < 10 ? veccc.size() : 10));

        }

        // Считаем качество разбиения
        double AllSumm = 0;
        double AllSummTopGlobal = 0;
        int sizeLocation = 0;
        for (int i = 0; i < vecTest.size(); ++i) {
            sizeLocation += vecTest[i].size();

            for (int o = 0; o < TOPLocationclusterAllTrain.size(); o++) {
                if (std::find(vecTest[i].begin(), vecTest[i].end(), TOPLocationclusterAllTrain[o].first) != vecTest[i].end()) {
                    AllSummTopGlobal++;
                }
            }


            if (TOPLocationcluster[examplarTest[i]].size() > 0) {
                for (int o = 0; o < TOPLocationcluster[examplarTest[i]].size(); o++) {
                    if (std::find(vecTest[i].begin(), vecTest[i].end(), TOPLocationcluster[examplarTest[i]][o].first) != vecTest[i].end()) {
                        AllSumm++;
                    }
                }
            } else {
                for (int o = 0; o < TOPLocationclusterAllTrain.size(); o++) {
                    if (std::find(vecTest[i].begin(), vecTest[i].end(), TOPLocationclusterAllTrain[o].first) != vecTest[i].end()) {
                        AllSumm++;
                    }
                }
            }
        }
        cout << "AllSumm/sizeLocation = " << AllSumm / (double) sizeLocation << endl;
        Results.push_back(AllSumm / (double) sizeLocation);
        cout << "AllSummTopGlobal/sizeLocation = " << AllSummTopGlobal / (double) sizeLocation << endl;
        ResultsAll.push_back(AllSummTopGlobal / (double) sizeLocation);
        cout << "--------------------------" << endl;
    }

    double M = 0;
    double sig = 0;

    Statistic::findeStatistic(Results, M, sig);

    double M2 = 0;
    double sig2 = 0;

    Statistic::findeStatistic(ResultsAll, M2, sig2);

    std::ofstream myfile;
    myfile.open("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/AP/AfinityPropagetion_Boyko/Results_Table.csv");
    myfile << ",1,2,3,4,5,E,SD,\n";
    myfile << "Metric: ," << (ResultsAll[0]) << "," << (ResultsAll[1]) << "," << (ResultsAll[2]) << "," << (ResultsAll[3]) << "," << (ResultsAll[4]) << "," << M2 << "," << sig2 << ",\n";
    myfile << "Metric: ," << (Results[0]) << "," << (Results[1]) << "," << (Results[2]) << "," << (Results[3]) << "," << (Results[4]) << "," << M << "," << sig << ",\n";

    myfile.close();
    // Уничтожаем кластер
    destroyGraph(graph);

    return 0;
}

vector<string>& split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void readLocation(char* data, vector<vector<int>> &vec) {

    std::string buf(data);
    ifstream file(buf);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<string> tokens = split(line, '\t');
            if (tokens.size() == 5) {

                unsigned int userId = stod(tokens[0]);
                unsigned int locationId = stod(tokens[4]);
                vec[userId].push_back(locationId);
            }

        }
        file.close();
    }
}

void read_training_text_edges(char* data, Graph * graph) {

    graph->n = 196590;
    graph->outEdges = new Edges[graph->n];
    graph->inEdges = new Edges[graph->n];
    vector<Edge>& edges = graph->edges;

    long index = 0;
    std::string buf(data);
    ifstream file(buf);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<string> tokens = split(line, '\t');
            if (tokens.size() == 2) {
                unsigned int user1 = stod(tokens[0]);
                unsigned int user2 = stod(tokens[1]);
                index++;
                edges.push_back(Edge(user1, user2, -1));
            }
        }
        file.close();
    }

    for (int i = 0; i < graph->n; ++i) {
        edges.push_back(Edge(i, i, -1e300));
    }

    for (size_t i = 0; i < edges.size(); ++i) {
        Edge* p = &edges[i];
        p->similarity += (1e-16 * p->similarity + 1e-300) * (rand() / (RAND_MAX + 1.0));
        graph->outEdges[p->source].push_back(p);
        graph->inEdges[p->destination].push_back(p);
    }
}

void destroyGraph(Graph * graph) {
    delete [] graph->outEdges;
    delete [] graph->inEdges;
    delete graph;
}

void clusteringGraph(Graph* graph, vector<int>& examplar, int maxit, double damping) {
    // Кластеризуем пользоватлей
    for (int i = 0; i < maxit; ++i) {
        updateResponsibilities(graph, damping);
        updateAvailabilities(graph, damping);
        if (updateExamplars(graph, examplar)) {
            cout << " is update! " << " i = " << i << endl;
        } else {
            cout << " is not update! " << " i = " << i << endl;
        }
    }

}

inline void update(double& variable, double newValue, double damping) {
    variable = damping * variable + (1.0 - damping) * newValue;
}

void updateResponsibilities(Graph* graph, double damping) {
    for (int i = 0; i < graph->n; ++i) {
        Edges& edges = graph->outEdges[i];
        int m = edges.size();
        double max1 = -HUGE_VAL, max2 = -HUGE_VAL;
        double argmax1 = -1;
        for (int k = 0; k < m; ++k) {
            double value = edges[k]->similarity + edges[k]->availability;
            if (value > max1) {
                swap(max1, value);
                argmax1 = k;
            }
            if (value > max2) {
                max2 = value;
            }
        }
        // update responsibilities
        for (int k = 0; k < m; ++k) {
            if (k != argmax1) {
                update(edges[k]->responsibility, edges[k]->similarity - max1, damping);
            } else {
                update(edges[k]->responsibility, edges[k]->similarity - max2, damping);
            }
        }
    }
}

void updateAvailabilities(Graph* graph, double damping) {
    for (int k = 0; k < graph->n / 2; ++k) {
        Edges& edges = graph->inEdges[k];
        int m = edges.size();
        // calculate sum of positive responsibilities
        double sum = 0.0;
        double rkk = 0.0;
        for (int i = 0; i < m; ++i) {
            if (i < m - 1) {
                sum += max(0.0, edges[i]->responsibility);
            } else {
                rkk = edges[i]->responsibility;
            }
        }
        for (int i = 0; i < m; ++i) {
            if (i < m - 1) {
                double t = min(0.0, rkk + sum - max(0.0, edges[i]->responsibility));
                update(edges[i]->availability, t, damping);
            } else {
                update(edges[i]->availability, sum, damping);
            }
        }
    }
}

bool updateExamplars(Graph* graph, vector<int>& examplar) {
    bool changed = false;
    for (int i = 0; i < graph->n; ++i) {
        Edges& edges = graph->outEdges[i];
        int m = edges.size();
        double maxValue = -HUGE_VAL;
        int argmax = i;
        for (int k = 0; k < m; ++k) {
            double value = edges[k]->availability + edges[k]->responsibility;
            if (value > maxValue) {
                maxValue = value;
                argmax = edges[k]->destination;
            }
        }
        if (examplar[i] != argmax) {
            examplar[i] = argmax;
            changed = true;
        }
    }
    return changed;
}

//void drawHist(const vector<int>& data, Mat3b& dst, int binSize = 3, int height = 0) {
//    int max_value = *max_element(data.begin(), data.end());
//    int rows = 0;
//    int cols = 0;
//    if (height == 0) {
//        rows = max_value + 10;
//    } else {
//        rows = max(max_value + 10, height);
//    }
//
//    cols = data.size() * binSize;
//
//    dst = Mat3b(rows, cols, Vec3b(0, 0, 0));
//
//    for (int i = 0; i < data.size(); ++i) {
//        int h = rows - data[i];
//      //  rectangle(dst, Point(i*binSize, h), Point((i + 1) * binSize - 1, rows), (i % 2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255), cv::FILLED);
//    }
//
//}