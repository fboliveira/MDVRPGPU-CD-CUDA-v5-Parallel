/* 
 * File:   Route.hpp
 * Author: fernando
 *
 * Created on July 15, 2014, 7:41 PM
 */

#ifndef ROUTE_HPP
#define	ROUTE_HPP

#include "../global.h"

using namespace std;

#include "DoubleLinkedList.hpp"

class Route {
    
    DoubleLinkedList *tour;

    float cost;
    float penaltyDuration;
    float penaltyDemand;

    int id;
    int depot;
    int demand;
    
    tipo_problema *problem; 
    tipo_config *config;

public:

    Route(tipo_problema *problem, tipo_config *config, int depot, int routeID);    
    ~Route();
    
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

    DoubleLinkedList* getTour() const;
    
    tipo_config* getConfig() const;
    void setConfig(tipo_config *config);

    tipo_problema* getProblem() const;
    void setProblem(tipo_problema *problem);
    
    float getTotalCost();
    void updatePenalty();
    
    void startValues();
    
    void add(int customer);    
    void add(Node* node);
    void add(Node* previous, Node* node);
    
    void remove(Node* node);
    
    Node* findNode(int customer);
    
    void calculateCost();
    void print();
    
};

#endif	/* ROUTE_HPP */