/* 
 * File:   Route.cpp
 * Author: fernando
 * 
 * Created on July 15, 2014, 8:44 PM
 */

#include <iostream>
#include <cfloat>
#include "Route.hpp"

/*
 * Constructor and Destructor
 */

Route::Route(tipo_problema *problem, tipo_config *config, int depot, int routeID) {

    this->setProblem(problem);
    this->setConfig(config);
    this->setDepot(depot);
    this->setId(routeID);

    this->tour = new DoubleLinkedList();
    this->startValues();

}

Route::~Route() {
        
    delete this->tour;
    //this->problem = NULL;
    //this->config = NULL;
}

/*
 * Getters and Setters
 */

float Route::getCost() const {
    return this->cost;
}

void Route::setCost(float cost) {
    this->cost = cost;
    this->updatePenalty();
}

float Route::getPenaltyDuration() const {
    return this->penaltyDuration;
}

void Route::setPenaltyDuration(float penalty) {
    this->penaltyDuration = penalty;
}

int Route::getDemand() const {
    return this->demand;
}

void Route::setDemand(int demand) {
    this->demand = demand;
}

float Route::getPenaltyDemand() const {
    return this->penaltyDemand;
}

void Route::setPenaltyDemand(float penaltyDemand) {
    this->penaltyDemand = penaltyDemand;
}

int Route::getId() const {
    return this->id;
}

void Route::setId(int id) {
    this->id = id;
}

int Route::getDepot() const {
    return this->depot;
}

void Route::setDepot(int depot) {
    this->depot = depot;
}

DoubleLinkedList* Route::getTour() const {
    return this->tour;
}

tipo_config* Route::getConfig() const {
    return this->config;
}

void Route::setConfig(tipo_config *config) {
    this->config = config;
}

tipo_problema* Route::getProblem() const {
    return this->problem;
}

void Route::setProblem(tipo_problema *problem) {
    this->problem = problem;
}

/* 
 * Methods
 */

void Route::startValues() {
    this->setCost(0.0);
    this->setPenaltyDuration(0.0);
    this->setPenaltyDuration(0.0);
    this->setDemand(0);
}

float Route::getTotalCost() {
    return this->getCost() + this->getPenaltyDuration() + this->getPenaltyDemand();
}

void Route::updatePenalty() {

    if (this->getProblem()->duracao > 0 && this->getCost() > this->getProblem()->duracao)
        this->setPenaltyDuration(this->getConfig()->routeDurationPenalty * (this->getCost() - this->getProblem()->duracao));
    else
        this->setPenaltyDuration(0.0);

    if (demand > this->getProblem()->capacidade)
        this->setPenaltyDemand(this->getConfig()->capacityPenalty * (this->getDemand() - this->getProblem()->capacidade));
    else
        this->setPenaltyDemand(0.0);

}

// Add customer at the end

void Route::add(int customer) {

    Node *node = new Node(customer);
    this->add(node);

}

void Route::add(Node* node) {

    // Get the last customer
    Node *back = this->getTour()->getBack();

    // Calculate the difference from the last to depot
    float distLastDep = this->getProblem()->distDeposito.at(this->getDepot()).at(back->customer - 1);
    float distLastNew = this->getProblem()->distancia.at(back->customer - 1).at(node->customer - 1);
    float distNewDep = this->getProblem()->distDeposito.at(this->getDepot()).at(node->customer - 1);

    // Update demand
    this->setDemand(this->getDemand() + this->getProblem()->demanda.at(node->customer - 1));
    this->setCost(this->getCost() - distLastDep + distLastNew + distNewDep);

    this->getTour()->appendNodeBack(node);

}

// Add customer node after previous node

void Route::add(Node* previous, Node* node) {

    // First Position
    if (previous == NULL) {

        Node *first = this->getTour()->getFront();
        
        // The route is empty
        if ( first == NULL ) {
            this->getTour()->appendNodeFront(node);
            this->calculateCost();
            return;
        }
        
        float distDepFirst = this->getProblem()->distDeposito.at(this->getDepot()).at(first->customer - 1);
        float distDepNew = this->getProblem()->distDeposito.at(this->getDepot()).at(node->customer - 1);
        float distNewFirst = this->getProblem()->distancia.at(node->customer - 1).at(first->customer - 1);

        // Update demand    
        this->setDemand(this->getDemand() + this->getProblem()->demanda.at(node->customer - 1));
        this->setCost(this->getCost() - distDepFirst + distDepNew + distNewFirst);
        this->getTour()->appendNodeFront(node);

    }// Last position
    else if (previous->Next == NULL)
        this->add(node);
        // In the middle
    else {

        float distPrevAfter = this->getProblem()->distancia.at(previous->customer - 1).at(previous->Next->customer - 1);
        float distPrevNew = this->getProblem()->distancia.at(previous->customer - 1).at(node->customer - 1);
        float distNewAfter = this->getProblem()->distancia.at(node->customer - 1).at(previous->Next->customer - 1);

        // Update demand    
        this->setDemand(this->getDemand() + this->getProblem()->demanda.at(node->customer - 1));
        this->setCost(this->getCost() - distPrevAfter + distPrevNew + distNewAfter);
        this->getTour()->appendNodeAfter(previous, node);

    }

}

void Route::remove(Node* node) {

    float previous, after, newCost;

    // Just one node
    if (node == this->getTour()->getFront() && node == this->getTour()->getBack()) {
        this->startValues();
    } else {

        if (node == this->getTour()->getFront()) {
            previous = this->getProblem()->distDeposito.at( this->getDepot() ).at( node->customer - 1 );
            after = this->getProblem()->distancia.at( node->customer - 1 ).at( node->Next->customer - 1 );
            newCost = this->getProblem()->distDeposito.at( this->getDepot() ).at( node->Next->customer - 1 );
        } else if (node == this->getTour()->getBack()) {

            previous = this->getProblem()->distancia.at( node->Previous->customer - 1 ).at( node->customer - 1 );
            after = this->getProblem()->distDeposito.at( this->getDepot() ).at( node->customer - 1 );
            newCost = this->getProblem()->distDeposito.at( this->getDepot()).at( node->Previous->customer - 1 );

        } else {

            previous = this->getProblem()->distancia.at( node->Previous->customer - 1 ).at( node->customer - 1 );
            after = this->getProblem()->distancia.at( node->customer - 1 ).at( node->Next->customer - 1 );
            newCost = this->getProblem()->distancia.at( node->Previous->customer - 1 ).at( node->Next->customer - 1 );

        }

        this->setDemand(this->getDemand() - this->getProblem()->demanda.at( node->customer - 1 ));
        this->setCost(this->getCost() - previous - after + newCost);

    }

    this->getTour()->removeNode(node);

}

Node* Route::findNode(int customer) {

    Node *it = this->getTour()->getFront();
    Node *node = NULL;

    while (it != NULL) {

        if (it->customer == customer) {
            node = it;
            break;
        }

        it = it->Next;
    }

    return node;

}

void Route::calculateCost() {

    int demand = 0;
    float cost = 0.0;

    Node *node = this->getTour()->getFront();

    // D->C1
    cost += this->getProblem()->distDeposito.at(this->getDepot()).at( node->customer - 1 );
    // Cn->D
    cost += this->getProblem()->distDeposito.at(this->getDepot()).at( this->getTour()->getBack()->customer - 1 );

    while (node != NULL) {
        if (node->Next != NULL)
            cost += this->getProblem()->distancia.at( node->customer - 1 ).at( node->Next->customer - 1 );

        demand += this->getProblem()->demanda.at( node->customer - 1 );
        node = node->Next;
    }

    this->setDemand(demand);
    this->setCost(cost);

}

void Route::print() {

    cout << "[D: " << this->getDepot() << " - R: " << this->getId() << "] => ";
    cout << "Cost: " << this->getCost() << " + P_Dur: " << this->getPenaltyDuration()
            << " + P_Dem: " << this->getPenaltyDemand() << " => TOTAL = " << this->getTotalCost() << endl;

    cout << "Demand: " << this->getDemand() << " => Route: D -> ";

    Node *node = this->getTour()->getFront();
    int num = 0;

    while (node != NULL) {

        cout << node->customer << " (" << this->getProblem()->demanda.at(node->customer - 1) << ") -> ";
        num++;
        node = node->Next;

    }

    cout << "D [ " << num << " ]\n\n";

}
