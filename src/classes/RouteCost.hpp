/* 
 * File:   RouteCost.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on August 4, 2014, 8:34 PM
 */

#ifndef ROUTECOST_HPP
#define	ROUTECOST_HPP

class RouteCost {

    float cost;
    float penaltyDuration;
    float penaltyDemand;

    int depot;
    int demand;

    MDVRPProblem *problem; 
    AlgorithmConfig *config;

public:
    
    RouteCost(MDVRPProblem* problem, AlgorithmConfig* config, int depot);

    RouteCost(const RouteCost& orig);
    virtual ~RouteCost();

    AlgorithmConfig* getConfig() const;
    void setConfig(AlgorithmConfig* config);

    float getCost() const;
    void setCost(float cost);

    int getDemand() const;
    void setDemand(int demand);

    int getDepot() const;
    void setDepot(int depot);

    float getPenaltyDemand() const;
    void setPenaltyDemand(float penaltyDemand);

    float getPenaltyDuration() const;
    void setPenaltyDuration(float penaltyDuration);

    MDVRPProblem* getProblem() const;
    void setProblem(MDVRPProblem* problem);        

    float getTotalCost();
    void updatePenalty();
    
    void startValues();
    
    template<typename Iter>
    void calculateCost(Iter begin, Iter end);
    
private:

};

#endif	/* ROUTECOST_HPP */

