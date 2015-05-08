/*
 * ManagedDoubleLinkedList.cpp
 *
 *  Created on: Apr 16, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#include "ManagedDoubleLinkedList.hpp"

Node* DoubleLinkedList::getFront() {
	return front;
}

Node* DoubleLinkedList::getBack() {
	return back;
}

Node* DoubleLinkedList::getNext(Node *n) {
	return n->next;
}

Node* DoubleLinkedList::getPrevious(Node *n) {
	return n->previous;
}

void DoubleLinkedList::appendNodeAfter(Node *pos, int x) {

	Node *n = new Node(x);

	n->previous = pos;
	n->next = pos->next;
	pos->next->previous = n;

	pos->next = n;

	if (pos == back)
		back = n;

	incSize();

}

void DoubleLinkedList::appendNodeFront(int x) {
	Node *n = new Node(x);
	if (front == NULL) {
		front = n;
		back = n;
	} else {
		front->previous = n;
		n->next = front;
		front = n;
	}

	incSize();

}

void DoubleLinkedList::appendNodeBack(int x) {

	Node *n = new Node(x);
	if (back == NULL) {
		front = n;
		back = n;
	} else {
		back->next = n;
		n->previous = back;
		back = n;
	}

	incSize();

}

void DoubleLinkedList::removeNode(Node *n) {

	//  p-[ ]-n

	if (n == front)
		front = n->next;

	if (n == back)
		back = n->previous;

	if (n->previous != NULL)
		n->previous->next = n->next;

	if (n->next != NULL)
		n->next->previous = n->previous;

	decSize();

}

void DoubleLinkedList::dispNodesForward() {
	Node *temp = front;
	cout << "\n\nNodes in forward order:" << endl;
	while (temp != NULL) {
		cout << temp->customer << "[" << temp->next << "]\t";
		temp = temp->next;
	}
	cout << endl;
}

void DoubleLinkedList::dispNodesReverse() {
	Node *temp = back;
	cout << "\n\nNodes in reverse order :" << endl;
	while (temp != NULL) {
		cout << temp->customer << "   ";
		temp = temp->previous;
	}
	cout << endl;
}

DoubleLinkedList::DoubleLinkedList(DoubleLinkedList* other) {

	for (Node* i = other->getFront(); i != NULL; i = other->getNext(i)) {
		this->appendNodeBack(i->customer);
	}

}

void DoubleLinkedList::destroyList() {
	Node *T = back;
	while (T != NULL) {
		Node *T2 = T;
		T = T->previous;
		delete T2;
	}
	front = NULL;
	back = NULL;
	_size = 0;
}

Node* DoubleLinkedList::find(int customer) {

	for (Node* i = this->getFront(); i != NULL; i = this->getNext(i)) {
		if (i->customer == customer)
			return i;
	}

	return NULL;

}

bool DoubleLinkedList::empty() {
	return _size == 0; // front == NULL && back == NULL;
}

int DoubleLinkedList::size() {
	return _size;
}

/*
 * Operators
 */

/*
 * Private methods
 */

void DoubleLinkedList::incSize() {
	_size++;
}

vector<int> DoubleLinkedList::getVector() {

	vector<int> v;
	Node *temp = front;

	while (temp != NULL) {
		v.push_back(temp->customer);
		temp = temp->next;
	}

}

void DoubleLinkedList::setVector(vector<int> v) {

	destroyList();

	for(auto ite = v.begin(); ite != v.end(); ++ite)
		appendNodeBack((*ite));

}

void DoubleLinkedList::decSize() {
	_size--;
	if (_size < 0)
		destroyList();
}
