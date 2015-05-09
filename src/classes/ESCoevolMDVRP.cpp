/* 
 * File:   ESCoevolMDVRP.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 24, 2014, 3:50 PM
 */

#include "ESCoevolMDVRP.hpp"
#include "PathRelinking.hpp"

/*
 * Constructors and Destructor
 */

ESCoevolMDVRP::ESCoevolMDVRP(MDVRPProblem *problem, AlgorithmConfig *config) :
problem(problem), config(config) {
}

/*
 * Getters and Setters
 */

AlgorithmConfig* ESCoevolMDVRP::getConfig() const {
    return this->config;
}

void ESCoevolMDVRP::setConfig(AlgorithmConfig* config) {
    this->config = config;
}

MDVRPProblem* ESCoevolMDVRP::getProblem() const {
    return this->problem;
}

void ESCoevolMDVRP::setProblem(MDVRPProblem* problem) {
    this->problem = problem;
}

/*
 * Public Methods
 */

/*
 * ES - Evolution Strategy - Coevolucionary - MDVRP
 */
void ESCoevolMDVRP::run() {

    // # Instanciar variaveis 
    time_t start;
    time(&start);

    //ESCoevolMDVRP::testFunction(this->getProblem(), this->getConfig());
    //return;
    
    this->getProblem()->getMonitor().setStart(start);

    // Elite group
    EliteGroup *eliteGroup = new EliteGroup(this->getProblem(), this->getConfig());

    // Create Monitor Locks
    this->getProblem()->getMonitor().createLocks(this->getProblem()->getDepots(), this->getConfig()->getNumSubIndDepots());
    
    // Create structure and subpopulation for each depot       
    Community *community = new Community(this->getProblem(), this->getConfig(), eliteGroup);
    community->pairingRandomly();

    // Evaluate Randomly
    community->evaluateSubpops(true);

    // Associate all versus best
    community->pairingAllVsBest();

    // Evaluate All Vs Best
    community->evaluateSubpops(true);

    //ESCoevolMDVRP::testFunction(this->getProblem(), this->getConfig(), community);
    //return;
    
    // Print evolution
    community->printEvolution();
    this->getProblem()->getMonitor().updateGeneration();
    //eliteGroup->getBest().printSolution();

    // ##### Start manager ###### ----------------
    try {
        community->manager();
    }    catch (exception& e) {
        cout << e.what();
    }
    // ########################## ----------------

    // Print result    
    if (this->getConfig()->isSaveLogRunFile())
        community->writeLogToFile();
    
    // Print final solution
    community->getEliteGroup()->getBest().printSolution();

    cout << "\nCustomers: " << community->getEliteGroup()->getBest().getNumTotalCustomers() << endl;
    cout << "Route Customers: " << community->getEliteGroup()->getBest().getNumTotalCustomersFromRoutes() << endl << endl;
    
    community->getEliteGroup()->printValues();

    //community->printSubpopList();
    
    // Clear memory
    delete community;
    
    // Destroy Monitor Locks
    this->getProblem()->getMonitor().destroyLocks(this->getProblem()->getDepots());
    
}

/*
 * Private Methods
 */


void ESCoevolMDVRP::testFunction(MDVRPProblem* problem, AlgorithmConfig* config) {
    
    Route r = Route(problem, config, 0, 0);
    Route v = Route(problem, config, 0, 1);

    int vr1[] = {1,2,3,4,5};
    
    for(int i = 0; i < 5; ++i)
        r.addAtBack(vr1[i]);
    
    r.print();

    int vr2[] = {6,7,8,9,10};

    for(int i = 0; i < 5; ++i)
        v.addAtBack(vr2[i]);
    
    v.print();
    cout << endl;
    cout << endl;

    cout << LocalSearch::processMoveDepotRoute(r, v, 4, true) << endl;
    r.printSolution();
    v.printSolution();

}

void ESCoevolMDVRP::testFunction(MDVRPProblem* problem, AlgorithmConfig* config, Community* community) {

    PathRelinking pathRelinking = PathRelinking(problem, config);
    
    community->getEliteGroup()->getEliteGroup().at(2).printSolution();
    
    pathRelinking.operate(community->getEliteGroup()->getEliteGroup().at(2), community->getEliteGroup()->getBest());

    community->getEliteGroup()->getEliteGroup().at(2).printSolution();
    
}
