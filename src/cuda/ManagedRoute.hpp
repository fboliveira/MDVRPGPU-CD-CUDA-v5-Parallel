/*
 * ManagedRoute.h
 *
 *  Created on: Apr 16, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef MANAGEDROUTE_H_
#define MANAGEDROUTE_H_

#include "../classes/MDVRPProblem.hpp"
#include "../classes/AlgorithmConfig.hpp"
#include "../classes/CustomerPosition.hpp"

#include "ManagedDoubleLinkedList.hpp"

class ManagedRoute {

    DoubleLinkedList *tour;

    float cost;
    float penaltyDuration;
    float penaltyDemand;

    int id;
    int depot;
    int demand;

    MDVRPProblem *problem;
    AlgorithmConfig *config;

public:

    ManagedRoute();
    ManagedRoute(MDVRPProblem *problem, AlgorithmConfig *config, int depot, int routeID);

    // Copy constructor
    ManagedRoute(const ManagedRoute& other);

    // Getters and Setters
    float getCost() const;
    void setCost(float cost);

    float getPenaltyDuration() const;
    void setPenaltyDuration(float penalty);

    int getDemand() const;
    void setDemand(int demand);

    float getPenaltyDemand() const;
    void setPenaltyDemand(float penaltyDemand);

    int getId() const;
    void setId(int id);

    int getDepot() const;
    void setDepot(int depot);

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig *config);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem *problem);

    DoubleLinkedList* getTour();
    void setTour(DoubleLinkedList *tour);

    // Methods
    void setCustomersPosition(vector<CustomerPosition>& position);

    void startValues();

    float getTotalCost();
    void updatePenalty();

    Node* addAtFront(int customer);
    Node* addAtBack(int customer);

    Node* addAfterPrevious(Node* previous, int customer);
    Node* addAfterPrevious(int previousCustomer, int customer);

    Node* find(int customer);

    void insertBestPosition(int customer);

    void remove(Node* position);
    void remove(int customer);

    void calculateCost();

    //float calculateCost(Node* start, Node* end, int& demand);

    void changeCustomer(Node* position, int newCustomer);
    void swap(Node* source, Node* dest);
    //void reverse(Node* begin, Node* end);

    bool isPenalized();

    void print();
    void printSolution();

    bool operator==(ManagedRoute* right);

};

#endif /* MANAGEDROUTE_H_ */
