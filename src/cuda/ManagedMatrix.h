/*
 * ManagedMatrix.h
 *
 *  Created on: Apr 15, 2015
 *      Author: Fernando B Oliveira - fboliveira25@gmail.com
 *
 *  Description:
 *	
 */

#ifndef MANAGEDMATRIX_H_
#define MANAGEDMATRIX_H_

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

template<class T>
class ManagedMatrix
{
private:

	// Attributes
	size_t lines;
	size_t rows;

	T **data;

	// Private methods

public:

	// Constructor
	ManagedMatrix();
	ManagedMatrix(size_t lines, size_t rows);

	// Destructor
	~ManagedMatrix();

	size_t getLines() const;
	void setLines(size_t lines);

	size_t getRows() const;
	void setRows(size_t rows);

	T **getData();

	T get(int line, int row);
	void set(int line, int row, T value);

	void init();
	void print();


};

// Constructor
template<class T>
ManagedMatrix<T>::ManagedMatrix() {
	this->setLines(0);
	this->setRows(0);
}

template<class T>
ManagedMatrix<T>::ManagedMatrix(size_t lines, size_t rows) {
	this->setLines(lines);
	this->setRows(rows);
	init();
}

// Destructor
template<class T>
ManagedMatrix<T>::~ManagedMatrix() {

	for (int i = 0; i < this->lines; i++)
		cudaFree(data[i]);

	cudaFree(data);

}
// Getters and Setters
template<class T>
size_t ManagedMatrix<T>::getLines() const {
	return lines;
}

template<class T>
void ManagedMatrix<T>::setLines(size_t lines) {
	this->lines = lines;
}

template<class T>
size_t ManagedMatrix<T>::getRows() const {
	return rows;
}

template<class T>
void ManagedMatrix<T>::setRows(size_t rows) {
	this->rows = rows;
}

template<class T>
T **ManagedMatrix<T>::getData() {
	return data;
}

// Methods
template<class T>
T ManagedMatrix<T>::get(int line, int row) {
	return data[line][row];
}

template<class T>
void ManagedMatrix<T>::set(int line, int row, T value) {
	data[line][row] = value;
}

template<class T>
void ManagedMatrix<T>::init() {
	cudaMallocManaged(&data, lines * sizeof(T *));
	for (int i = 0; i < lines; i++)
		cudaMallocManaged(&data[i], rows * sizeof(T));
}

template<class T>
void ManagedMatrix<T>::print() {

	for(int i = 0; i < this->getLines(); ++i) {
		for(int j = 0; j < this->getRows(); ++j)
			cout << this->get(i, j) << "\t";
		cout << endl;
	}
}

#endif /* MANAGEDMATRIX_H_ */
