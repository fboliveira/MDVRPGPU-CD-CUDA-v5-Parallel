/*
 * ManagedRoute.cpp
 *
 *  Created on: Apr 16, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#include "ManagedRoute.hpp"

/*
 * Constructors and Destructor
 */

ManagedRoute::ManagedRoute() {
	tour = new DoubleLinkedList();
}

ManagedRoute::ManagedRoute(MDVRPProblem *problem, AlgorithmConfig *config,
		int depot, int routeID) {

	this->setProblem(problem);
	this->setConfig(config);
	this->setDepot(depot);
	this->setId(routeID);
	this->startValues();
	tour = new DoubleLinkedList();

}

ManagedRoute::ManagedRoute(const ManagedRoute& other) :
		tour(other.tour), cost(other.cost), penaltyDuration(
				other.penaltyDuration), penaltyDemand(other.penaltyDemand), id(
				other.id), depot(other.depot), demand(other.demand), problem(
				other.problem), config(other.config) {
}

/*
 * Getters and Setters
 */

float ManagedRoute::getCost() const {
	return this->cost;
}

void ManagedRoute::setCost(float cost) {
	this->cost = cost;
	this->updatePenalty();
}

float ManagedRoute::getPenaltyDuration() const {
	return this->penaltyDuration;
}

void ManagedRoute::setPenaltyDuration(float penalty) {
	this->penaltyDuration = penalty;
}

int ManagedRoute::getDemand() const {
	return this->demand;
}

void ManagedRoute::setDemand(int demand) {
	this->demand = demand;
}

float ManagedRoute::getPenaltyDemand() const {
	return this->penaltyDemand;
}

void ManagedRoute::setPenaltyDemand(float penaltyDemand) {
	this->penaltyDemand = penaltyDemand;
}

int ManagedRoute::getId() const {
	return this->id;
}

void ManagedRoute::setId(int id) {
	this->id = id;
}

int ManagedRoute::getDepot() const {
	return this->depot;
}

void ManagedRoute::setDepot(int depot) {
	this->depot = depot;
}

AlgorithmConfig* ManagedRoute::getConfig() const {
	return this->config;
}

void ManagedRoute::setConfig(AlgorithmConfig *config) {
	this->config = config;
}

MDVRPProblem* ManagedRoute::getProblem() const {
	return this->problem;
}

void ManagedRoute::setProblem(MDVRPProblem *problem) {
	this->problem = problem;
}

DoubleLinkedList* ManagedRoute::getTour() {
	if (tour == NULL)
		cout << "tour == NULL" << endl;
	return this->tour;
}

void ManagedRoute::setTour(DoubleLinkedList *tour) {
	this->tour = tour;
}

/*
 * Methods
 */

void ManagedRoute::setCustomersPosition(vector<CustomerPosition>& position) {

	for (Node* i = this->getTour()->getFront(); i != NULL;
			i = this->getTour()->getNext(i)) {
		int customer = i->customer;

		position.at(customer - 1).setCustomer(customer);
		position.at(customer - 1).setDepot(this->getDepot());
		position.at(customer - 1).setRoute(this->getId());
	}

}

void ManagedRoute::startValues() {
	this->setCost(0.0);
	this->setPenaltyDuration(0.0);
	this->setPenaltyDemand(0.0);
	this->setDemand(0);
}

float ManagedRoute::getTotalCost() {
	return this->getCost() + this->getPenaltyDuration()
			+ this->getPenaltyDemand();
}

void ManagedRoute::updatePenalty() {

	if (this->getProblem()->getDuration() > 0
			&& this->getCost() > this->getProblem()->getDuration())
		this->setPenaltyDuration(
				this->getConfig()->getRouteDurationPenalty()
						* (this->getCost() - this->getProblem()->getDuration()));
	else
		this->setPenaltyDuration(0.0);

	if (this->getDemand() > this->getProblem()->getCapacity())
		this->setPenaltyDemand(
				this->getConfig()->getCapacityPenalty()
						* (this->getDemand() - this->getProblem()->getCapacity()));
	else
		this->setPenaltyDemand(0.0);

}

// Add customer at the front of the list

Node* ManagedRoute::addAtFront(int customer) {

	float distFirstDep = 0.0, distFirstNew = 0.0, distNewDep = 0.0;

	// If the list is empty
	if (this->getTour()->empty()) {
		return this->addAtBack(customer);
	} else {

		// Get the first customer
		int firstCustomer = this->getTour()->getFront()->customer;

		// Calculate the distance from the first to the depot
		distFirstDep = this->getProblem()->getDepotDistances().at(
				this->getDepot()).at(firstCustomer - 1);
		// Calculate the distance from the first to the new customer
		distFirstNew = this->getProblem()->getCustomerDistances().at(
				firstCustomer - 1).at(customer - 1);
		// Calculate the distance from the new to the depot
		distNewDep = this->getProblem()->getDepotDistances().at(
				this->getDepot()).at(customer - 1);

	}

	// Update demand
	this->setDemand(
			this->getDemand()
					+ this->getProblem()->getDemand().at(customer - 1));
	this->setCost(this->getCost() - distFirstDep + distFirstNew + distNewDep);

	this->getTour()->appendNodeFront(customer);

	return this->getTour()->getFront();

}

// Add customer at the end of the list
Node* ManagedRoute::addAtBack(int customer) {

	float distLastDep = 0.0, distLastNew = 0.0, distNewDep = 0.0;

	// If the list is empty
	if (this->getTour()->empty()) {
		// D -> C -> D
		distNewDep = this->getProblem()->getDepotDistances().at(
				this->getDepot()).at(customer - 1) * 2;
	} else {

		// Get the last customer
		int lastCustomer = this->getTour()->getBack()->customer;

		// Calculate the distance from the last to depot
		distLastDep = this->getProblem()->getDepotDistances().at(
				this->getDepot()).at(lastCustomer - 1);
		// Calculate the distance from the last to the new customer
		distLastNew = this->getProblem()->getCustomerDistances().at(
				lastCustomer - 1).at(customer - 1);
		// Calculate the distance from the new to depot
		distNewDep = this->getProblem()->getDepotDistances().at(
				this->getDepot()).at(customer - 1);

	}

	// Update demand
	this->setDemand(
			this->getDemand()
					+ this->getProblem()->getDemand().at(customer - 1));
	this->setCost(this->getCost() - distLastDep + distLastNew + distNewDep);

	this->getTour()->appendNodeBack(customer);
	return this->getTour()->getBack();

}

// Add customer after previous one
Node* ManagedRoute::addAfterPrevious(Node* previous, int customer) {

	try {

		if (this->getTour()->empty())
			return this->addAtBack(customer);
		else if (previous == NULL || previous->next == NULL) {
			return this->addAtBack(customer);
		} else {

			int prevCustomer = previous->customer;
			int nextCustomer = previous->next->customer;

			float distPrevAfter = this->getProblem()->getCustomerDistances().at(
					prevCustomer - 1).at(nextCustomer - 1);
			float distPrevNew = this->getProblem()->getCustomerDistances().at(
					prevCustomer - 1).at(customer - 1);
			float distNewAfter = this->getProblem()->getCustomerDistances().at(
					customer - 1).at(nextCustomer - 1);

			// Update demand
			this->setDemand(
					this->getDemand()
							+ this->getProblem()->getDemand().at(customer - 1));
			this->setCost(
					this->getCost() - distPrevAfter + distPrevNew
							+ distNewAfter);

			this->getTour()->appendNodeAfter(previous, customer);
			return previous->next;

		}

	} catch (exception &e) {
		cout << endl;
		cout << endl;
		this->printSolution();
		cout << "ManagedRoute::addAfterPrevious: " << e.what() << '\n';
	}

	return this->find(customer);

}

Node* ManagedRoute::addAfterPrevious(int previousCustomer, int customer) {

	if (previousCustomer <= 0)
		return this->addAtFront(customer);

	Node* previous = this->find(previousCustomer);
	return this->addAfterPrevious(previous, customer);
}

Node* ManagedRoute::find(int customer) {
	return this->getTour()->find(customer);
}

void ManagedRoute::insertBestPosition(int customer) {

	float bestCost;
	bool front = true;

	Node* bestPos;
	Node* pos = this->addAtFront(customer);

	bestCost = this->getTotalCost();

	this->remove(pos);

	for (Node* i = this->getTour()->getFront(); i != NULL;
			i = this->getTour()->getNext(i)) {

		pos = this->addAfterPrevious(i, customer);

		if (Util::isBetterSolution(this->getTotalCost(), bestCost)) {
			bestCost = this->getTotalCost();
			bestPos = i;
			front = false;
		}

		this->remove(pos);

	}

	if (front)
		this->addAtFront(customer);
	else
		this->addAfterPrevious(bestPos, customer);
}

void ManagedRoute::remove(Node* position) {

	float previous, after, newCost;

	int customer = position->customer;

	// Just one node
	if (customer == this->getTour()->getFront()->customer
			&& customer == this->getTour()->getBack()->customer) {
		this->startValues();
	} else {

		if (customer == this->getTour()->getFront()->customer) {

			int nextCustomer = position->next->customer;

			// From Depot to Customer
			previous = this->getProblem()->getDepotDistances().at(
					this->getDepot()).at(customer - 1);
			// From Customer to Next
			after =
					this->getProblem()->getCustomerDistances().at(customer - 1).at(
							nextCustomer - 1);
			// From Next to Depot
			newCost = this->getProblem()->getDepotDistances().at(
					this->getDepot()).at(nextCustomer - 1);

		} else if (customer == this->getTour()->getBack()->customer) {

			int prevCustomer = position->previous->customer;

			previous = this->getProblem()->getCustomerDistances().at(
					prevCustomer - 1).at(customer - 1);
			after =
					this->getProblem()->getDepotDistances().at(this->getDepot()).at(
							customer - 1);
			newCost = this->getProblem()->getDepotDistances().at(
					this->getDepot()).at(prevCustomer - 1);

		} else {

			int nextCustomer = position->next->customer;
			int prevCustomer = position->previous->customer;

			previous = this->getProblem()->getCustomerDistances().at(
					prevCustomer - 1).at(customer - 1);
			after =
					this->getProblem()->getCustomerDistances().at(customer - 1).at(
							nextCustomer - 1);
			newCost = this->getProblem()->getCustomerDistances().at(
					prevCustomer - 1).at(nextCustomer - 1);

		}

		this->setDemand(
				this->getDemand()
						- this->getProblem()->getDemand().at(customer - 1));
		this->setCost(this->getCost() - previous - after + newCost);

	}

	this->getTour()->removeNode(position);

}

void ManagedRoute::remove(int customer) {
	this->remove(this->find(customer));
}

void ManagedRoute::calculateCost() {

	int demand = 0;
	float cost = 0.0;

	if (!this->getTour()->empty()) {

		int customer = this->getTour()->getFront()->customer;

		if (customer > this->getProblem()->getCustomers())
			cout << "Erro => " << customer << "\n";

		// D->C1
		cost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(
				customer - 1);
		// Cn->D
		customer = this->getTour()->getBack()->customer;
		cost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(
				customer - 1);

		int nextCustomer;

		for (Node* i = this->getTour()->getFront(); i != NULL;
				i = this->getTour()->getNext(i)) {

			customer = i->customer;

			auto nextPosition = next(i);

			if (i->next != NULL) {

				nextCustomer = i->next->customer;
				cost += this->getProblem()->getCustomerDistances().at(
						customer - 1).at(nextCustomer - 1);
			}

			demand += this->getProblem()->getDemand().at(customer - 1);
		}

	}

	this->setDemand(demand);
	this->setCost(cost);
}

void ManagedRoute::changeCustomer(Node* position, int newCustomer) {

	float oldCustomerCost = 0, newCustomerCost = 0;

    int customer = position->customer;

    // Just one node
    if (customer == this->getTour()->getFront()->customer && customer == this->getTour()->getBack()->customer) {
        this->remove(position);
        this->addAtBack(newCustomer);
    } else {

        //auto nextPosition = next(position);
        //auto prevPosition = prev(position);

        // If it is in the front of
        if (position->customer == this->getTour()->getFront()->customer) {

            int nextCustomer = position->next->customer;

            // D->C
            oldCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);
            // C->C+1
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);

            // D->NewC
            newCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(newCustomer - 1);
            // NewC->C+1
            newCustomerCost += this->getProblem()->getCustomerDistances().at(newCustomer - 1).at(nextCustomer - 1);

        } else if (position->customer == this->getTour()->getBack()->customer) { // Last one

            int prevCustomer = position->previous->customer;

            // C-1->C
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            // C->D
            oldCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(customer - 1);

            // C-1->NewC
            newCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(newCustomer - 1);

            // D->NewC
            newCustomerCost += this->getProblem()->getDepotDistances().at(this->getDepot()).at(newCustomer - 1);

        } else { // Anywhere...

            int nextCustomer = position->next->customer;
            int prevCustomer = position->previous->customer;

            // C-1->C
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(customer - 1);
            // C->C+1
            oldCustomerCost += this->getProblem()->getCustomerDistances().at(customer - 1).at(nextCustomer - 1);

            // C-1->NewC
            newCustomerCost += this->getProblem()->getCustomerDistances().at(prevCustomer - 1).at(newCustomer - 1);
            // NewC->C+1
            newCustomerCost += this->getProblem()->getCustomerDistances().at(newCustomer - 1).at(nextCustomer - 1);

        }

        this->setDemand(this->getDemand() - this->getProblem()->getDemand().at(customer - 1)
                + this->getProblem()->getDemand().at(newCustomer - 1));
        this->setCost(this->getCost() - oldCustomerCost + newCustomerCost);

        // Change value of customer
        position->customer = newCustomer;

    }

}

void ManagedRoute::swap(Node* source, Node* dest) {
    int customerSource = source->customer;
    int customerDest = dest->customer;

    this->changeCustomer(source, customerDest);
    this->changeCustomer(dest, customerSource);
}

bool ManagedRoute::isPenalized() {
	return this->getPenaltyDemand() > 0 || this->getPenaltyDuration() > 0;
}

void ManagedRoute::print() {

    if (this->getTour()->empty())
        return;

    cout << "[D: " << this->getDepot() << " - R: " << this->getId() << "] => ";
    cout << "Cost: " << this->getCost() << " + P_Dur: " << this->getPenaltyDuration()
            << " + P_Dem: " << this->getPenaltyDemand() << " => TOTAL = " << this->getTotalCost() << endl;

    cout << "Demand: " << this->getDemand() << " => Route: D -> ";

    int num = 0;
    MDVRPProblem * problem = this->getProblem();

	for (Node* i = this->getTour()->getFront(); i != NULL;
			i = this->getTour()->getNext(i)) {

        cout << i->customer << " (" << problem->getDemand().at(i->customer - 1) << ") -> ";
        num++;
    };

    cout << "D [ " << num << " ]\n\n";
}

void ManagedRoute::printSolution() {

	if (this->getTour()->empty())
		return;

	cout << this->getDepot() + 1 << "\t" << this->getId() + 1 << "\t";
	printf("%.2f\t%d", this->getTotalCost(), this->getDemand());

	cout << "\t0 ";

	int num = 0;
	MDVRPProblem * problem = this->getProblem();

	for (Node* i = this->getTour()->getFront(); i != NULL;
			i = this->getTour()->getNext(i)) {
		cout << i->customer << " ";
		num++;
	};

	cout << "0\n";
}

bool ManagedRoute::operator ==(ManagedRoute* right) {

	Node* j = right->getTour()->getFront();

	for (Node* i = this->getTour()->getFront(); i != NULL;
			i = this->getTour()->getNext(i)) {
		if (j == NULL || i->customer != j->customer)
			return false;

		j = right->getTour()->getNext(j);
	}

	return true;

}
