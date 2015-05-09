/* 
 * File:   RouteCost.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on August 4, 2014, 8:34 PM
 */

#include "RouteCost.hpp"

/*
 * Constructors and Destructor
 */

RouteCost::RouteCost(MDVRPProblem* problem, AlgorithmConfig* config, int depot) :
    config(config), depot(depot), problem(problem) {
    this->startValues();
}

RouteCost::RouteCost(const RouteCost& orig) {
}

RouteCost::~RouteCost() {
}

/*
 * Getters and Setters
 */

AlgorithmConfig* RouteCost::getConfig() const {
    return config;
}

float RouteCost::getCost() const {
    return cost;
}

int RouteCost::getDemand() const {
    return demand;
}

int RouteCost::getDepot() const {
    return depot;
}

float RouteCost::getPenaltyDemand() const {
    return penaltyDemand;
}

float RouteCost::getPenaltyDuration() const {
    return penaltyDuration;
}

MDVRPProblem* RouteCost::getProblem() const {
    return problem;
}

void RouteCost::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

void RouteCost::setCost(float cost) {
    this->cost = cost;
}

void RouteCost::setDemand(int demand) {
    this->demand = demand;
}

void RouteCost::setDepot(int depot) {
    this->depot = depot;
}

void RouteCost::setPenaltyDemand(float penaltyDemand) {
    this->penaltyDemand = penaltyDemand;
}

void RouteCost::setPenaltyDuration(float penaltyDuration) {
    this->penaltyDuration = penaltyDuration;
}

void RouteCost::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}
/*
 * Public Methods
 */

float RouteCost::getTotalCost() {
    return this->getCost() + this->getPenaltyDuration() + this->getPenaltyDemand();
}

void RouteCost::updatePenalty() {

    if (this->getProblem()->getDuration() > 0 && this->getCost() > this->getProblem()->getDuration())
        this->setPenaltyDuration(this->getConfig()->getRouteDurationPenalty() * (this->getCost() - this->getProblem()->getDuration()));
    else
        this->setPenaltyDuration(0.0);

    if (this->getDemand() > this->getProblem()->getCapacity())
        this->setPenaltyDemand(this->getConfig()->getCapacityPenalty() * (this->getDemand() - this->getProblem()->getCapacity()));
    else
        this->setPenaltyDemand(0.0);
    
}

void RouteCost::startValues() {
    this->setCost(0.0);
    this->setPenaltyDuration(0.0);
    this->setPenaltyDuration(0.0);
    this->setDemand(0);
}

template<typename Iter>
void RouteCost::calculateCost(Iter begin, Iter end) {

    
    
}


/*
 * Private Methods
 */
