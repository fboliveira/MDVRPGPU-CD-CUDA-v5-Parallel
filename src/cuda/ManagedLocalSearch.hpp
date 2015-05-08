/*
 * ManagedLocalSearch.hpp
 *
 *  Created on: Apr 29, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef MANAGEDLOCALSEARCH_HPP_
#define MANAGEDLOCALSEARCH_HPP_

#include "../classes/MDVRPProblem.hpp"
#include "../classes/AlgorithmConfig.hpp"
#include "ManagedRoute.hpp"

class ManagedLocalSearch {

public:

    static bool processMoveDepotRoute(ManagedRoute& ru, ManagedRoute& rv, int move, bool equal);
    static bool operateMoves(MDVRPProblem *problem, AlgorithmConfig *config, ManagedRoute& ru, ManagedRoute& rv, bool equal);


private:

    static bool operateMoveDepotRouteFacade(ManagedRoute& ru, ManagedRoute& rv, int move, bool equal);
    static bool operateMoveDepotRouteM1(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM2(ManagedRoute& ru, ManagedRoute& rv, bool equal, bool operateM3 = false);
    static bool operateMoveDepotRouteM3(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM4(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM5(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM6(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM7(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM8(ManagedRoute& ru, ManagedRoute& rv, bool equal);
    static bool operateMoveDepotRouteM9(ManagedRoute& ru, ManagedRoute& rv, bool equal);

};

#endif /* MANAGEDLOCALSEARCH_HPP_ */

