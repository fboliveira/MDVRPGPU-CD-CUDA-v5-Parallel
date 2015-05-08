/*
 * ManagedDoubleLinkedList.h
 *
 *  Created on: Apr 16, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef MANAGEDDOUBLELINKEDLIST_H_
#define MANAGEDDOUBLELINKEDLIST_H_

#include <iostream>
#include <cstdio>
#include <vector>

#include "ManagedNode.h"

using namespace std;

// Adapted from:
// http://www.cprogramming.com/snippets/source-code/double-linked-list-cplusplus
//An example of a simple double linked list using OOP techniques

class DoubleLinkedList
{
private:

	Node *front;
	Node *back;
	int _size;

public:

	DoubleLinkedList() {
		front = NULL;
		back = NULL;
		_size = 0;
	}

	DoubleLinkedList(DoubleLinkedList* other);

	~DoubleLinkedList() {
		cout << "DoubleLinkedList::destroyList()" << endl;
		destroyList();
	}

	Node* getFront();
	Node* getBack();
	Node* getNext(Node *n);
	Node* getPrevious(Node *n);
	Node* find(int customer);

	void appendNodeAfter(Node *pos, int x);
	void appendNodeFront(int x);
	void appendNodeBack(int x);

	void removeNode(Node *n);

	void dispNodesForward();
	void dispNodesReverse();

	void destroyList();

	bool empty();
	int size();

	vector<int> getVector();
	void setVector(vector<int> v);

	// Operators

private:

	void incSize();
	void decSize();

};

#endif /* MANAGEDDOUBLELINKEDLIST_H_ */
