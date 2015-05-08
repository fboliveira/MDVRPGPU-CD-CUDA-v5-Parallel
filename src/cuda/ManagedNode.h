/*
 * ManagedNode.h
 *
 *  Created on: Apr 15, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef MANAGEDNODE_H_
#define MANAGEDNODE_H_

#include "Managed.h"

// Adapted from:
// http://www.cprogramming.com/snippets/source-code/double-linked-list-cplusplus
//An example of a simple double linked list using OOP techniques

struct ManagedNode : public Managed {
    int customer;
    ManagedNode *next = NULL;
    ManagedNode *previous = NULL;

    ManagedNode(int y) {
    	cout << "ManagedNode: " << y << endl;
        customer = y;
        next = previous = NULL;
    }
};

struct Node {
    int customer;
    Node *next = NULL;
    Node *previous = NULL;

    Node(int y) {
    	cout << "Node: " << y << endl;
        customer = y;
        next = previous = NULL;
    }
};

#endif /* MANAGEDNODE_H_ */
