/* 
 * File:   ESCoevolMDVRP.cpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 * 
 * Created on July 24, 2014, 3:50 PM
 */

#include "ESCoevolMDVRP.hpp"

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
	EliteGroup *eliteGroup = new EliteGroup(this->getProblem(),
			this->getConfig());

	// Create Monitor Locks
	this->getProblem()->getMonitor().createLocks(
			this->getProblem()->getDepots(),
			this->getConfig()->getNumSubIndDepots());

	// Create structure and subpopulation for each depot
	Community *community = new Community(this->getProblem(), this->getConfig(),
			eliteGroup);
	community->pairingRandomly();

	// Evaluate Randomly
	cout << "Evaluate Randomly..." << endl;
	community->evaluateSubpops(true);

	// Associate all versus best
	cout << "Associate all versus best" << endl;
	community->pairingAllVsBest();

	// Evaluate All Vs Best
	cout << "Evaluate All Vs Best" << endl;
	community->evaluateSubpops(true);

	//ESCoevolMDVRP::testFunction(this->getProblem(), this->getConfig(), community);
	//return;

	// Print evolution
	cout << "Print evolution" << endl;
	community->printEvolution();
	this->getProblem()->getMonitor().updateGeneration();
	//eliteGroup->getBest().printSolution();

	// ##### Start manager ###### ----------------
	try {
		community->manager();
	} catch (exception& e) {
		cout << e.what();
	}
	// ########################## ----------------

	// Print result
	if (this->getConfig()->isSaveLogRunFile())
		community->writeLogToFile();

	// Print final solution
	community->getEliteGroup()->getBest().printSolution();

	cout << "\nCustomers: "
			<< community->getEliteGroup()->getBest().getNumTotalCustomers()
			<< endl;
	cout << "Route Customers: "
			<< community->getEliteGroup()->getBest().getNumTotalCustomersFromRoutes()
			<< endl << endl;

	community->getEliteGroup()->printValues();

	//community->printSubpopList();

	// Clear memory
	delete community;

	// Destroy Monitor Locks
	this->getProblem()->getMonitor().destroyLocks(
			this->getProblem()->getDepots());

}

/*
 * Private Methods
 */

void ESCoevolMDVRP::testFunction(MDVRPProblem* problem,
		AlgorithmConfig* config) {

//    Route r = Route(problem, config, 0, 0);
//    Route v = Route(problem, config, 0, 1);
//
//    int vr1[] = {1,2,3,4,5};
//
//    for(int i = 0; i < 5; ++i)
//        r.addAtBack(vr1[i]);
//
//    r.print();
//
//    int vr2[] = {6,7,8,9,10};
//
//    for(int i = 0; i < 5; ++i)
//        v.addAtBack(vr2[i]);
//
//    v.print();
//    cout << endl;
//    cout << endl;
//
//    cout << LocalSearch::processMoveDepotRoute(r, v, 4, true) << endl;
//    r.printSolution();
//    v.printSolution();

	cout << "Declaring..." << endl;
	Individual ind = Individual(problem, config, 0, 0);
	cout << "Creating..." << endl;
	ind.create();
	ind.print(true);
	cout << "Evaluating..." << endl;

	Node* n = new Node(10);
	cout << "Node: " << n->customer << endl;

	ManagedRoute route = ManagedRoute(problem, config, 0, 0);
	for (auto ite = ind.getGene().begin(); ite != ind.getGene().end(); ++ite) {
		cout << (*ite) << endl;
		route.getTour()->appendNodeBack((*ite));
	}

	ind.evaluate(true);
	cout << "Printing..." << endl;
	ind.printSolution(true);

	ManagedRoute r2 = ManagedRoute(problem, config, 0, 1);
	for (auto ite = ind.getGene().begin(); ite != ind.getGene().end(); ++ite) {
		cout << (*ite) << endl;
		r2.getTour()->appendNodeBack((*ite));
	}

	vector<ManagedRoute> vm;

//	route.getTour()->dispNodesForward();
//	route.getTour()->dispNodesReverse();
//
//	r2.getTour()->dispNodesForward();
//	r2.getTour()->dispNodesReverse();

	vm.push_back(route);
	vm.push_back(r2);

	cout << "Clearing routes... ";
	for (auto ite = vm.begin(); ite != vm.end(); ++ite) {
		(*ite).getTour()->destroyList();
	}
	cout << "OK." << endl;

	//ind.getRoutes().at(0).getTour()->dispNodesForward();
	cout << "Copy" << endl;
	Individual j = ind.copy();
	cout << "Mutate" << endl;
	j.mutate();
	cout << "Eval" << endl;
	j.evaluate(true);
	cout << "LS" << endl;
	j.localSearch();
	cout << "RtoG" << endl;
	j.routesToGenes();

	for (int i = 0; i < 10; ++i) {
		ind.split();
		ind.evaluate(true);
		cout << "Printing..." << endl;
		ind.printSolution(true);
	}

}

void ESCoevolMDVRP::testFunction(MDVRPProblem* problem, AlgorithmConfig* config,
		Community* community) {

	/*
	 PathRelinking pathRelinking = PathRelinking(problem, config);

	 community->getEliteGroup()->getEliteGroup().at(2).printSolution();

	 pathRelinking.operate(community->getEliteGroup()->getEliteGroup().at(2), community->getEliteGroup()->getBest());

	 community->getEliteGroup()->getEliteGroup().at(2).printSolution();
	 */

	Individual i =
			community->getSubpops().at(0).getIndividualsGroup().getIndividuals().at(
					0);

//	i.print(true);
//	cudaMutate(i.getGene());
//	i.evaluate(true);
//	i.print(true);
	cudaTeste();

}
