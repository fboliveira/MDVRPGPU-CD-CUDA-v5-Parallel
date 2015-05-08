/*
 * ManagedLocalSearch.cpp
 *
 *  Created on: Apr 29, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#include "ManagedLocalSearch.hpp"

bool ManagedLocalSearch::processMoveDepotRoute(ManagedRoute& ru,
		ManagedRoute& rv, int move, bool equal) {

	ManagedRoute newRu = ru;
	ManagedRoute newRv = rv;

	bool result = true;

	//---bool result = operateMoveDepotRouteFacade(newRu, newRv, move, equal);

	newRu.calculateCost();
	if (!equal)
		newRv.calculateCost();

	if (Util::isBetterSolution(newRu.getTotalCost() + newRv.getTotalCost(),
			ru.getTotalCost() + rv.getTotalCost())) {
		ru = newRu;
		if (!equal)
			rv = newRv;
		result = true;
	} else
		result = false;

	return result;
}

bool ManagedLocalSearch::operateMoves(MDVRPProblem* problem,
		AlgorithmConfig* config, ManagedRoute& ru, ManagedRoute& rv,
		bool equal) {

	return true;
}
