#ifndef MATRIX_CLASS_H
#define MATRIX_CLASS_H
 
#include <iostream>
#include <iomanip>
#include "mkl_lapacke.h"
 
class Solver;
class Matrix;
 
class Matrix
{
public:
	Matrix(std::size_t rVal = 0)
	{
		_Rows = rVal;
		_Cols = _Rows;
		_Data = new double[_Rows * _Cols];
	}
	Matrix(std::size_t rVal, std::size_t cVal)
	{
		_Rows = rVal;
		_Cols = cVal;
		_Data = new double[_Rows * _Cols];
	}
	Matrix(const Matrix& rhs)
	{
		_Rows = rhs._Rows;
		_Cols = rhs._Cols;
		_Data = new double[_Rows * _Cols];
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(_Data + ix) = *(rhs._Data + ix);
		}
	}
	// the deconstructor function
	~Matrix()
	{
		delete[] _Data;
	}
	void resize(std::size_t rVal, std::size_t cVal)
	{
		delete[] _Data;
		_Rows = rVal;
		_Cols = cVal;
		_Data = new double[_Rows * _Cols];
	}
	// member functions
	std::size_t rows() const
	{
		return _Rows;
	}
	std::size_t cols() const
	{
		return _Cols;
	}
	// inverse matrix
	void inverse()
	{
		int *ipiv = new int[_Rows * _Cols];
		LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'U', _Rows, _Data, _Rows, ipiv);
		LAPACKE_dsytri(LAPACK_ROW_MAJOR, 'U', _Rows, _Data, _Rows, ipiv);
		delete[] ipiv;
		for(std::size_t j = 0; j < _Cols; ++j)
		{
			for(std::size_t i = j + 1; i < _Rows; ++i)
				_Data[i * _Cols + j] = _Data[j * _Cols + i];
		}
	}
	// overloaded operations
	Matrix& operator=(const Matrix& rhs)
	{
		resize(rhs._Rows, rhs._Cols);
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(_Data + ix) = *(rhs._Data + ix);
		}
		return *this;
	}
	Matrix& operator*=(double scala)
	{
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(_Data + ix) *= scala;
		}
		return *this;
	}
	Matrix& operator+=(const Matrix& mat)
	{
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(_Data + ix) += *(mat._Data + ix);
		}
		return *this;
	}
	Matrix& operator-=(const Matrix& mat)
	{
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(_Data + ix) -= *(mat._Data + ix);
		}
		return *this;
	}
	Matrix operator-() const
	{
		Matrix mat(_Rows, _Cols);
		for(std::size_t ix = 0; ix != _Rows * _Cols; ++ix)
		{
			*(mat._Data + ix) = 0 - *(_Data + ix);
		}
		return mat;
	}
	double& operator()(std::size_t rVal, std::size_t cVal)
	{
		return _Data[rVal*_Cols + cVal];
	}
	const double& operator()(std::size_t rVal, std::size_t cVal) const
	{
		return _Data[rVal*_Cols + cVal];
	}
	// friend classes and functions
	friend class Solver;
	friend Matrix operator+(const Matrix&, const Matrix&);
	friend Matrix operator-(const Matrix&, const Matrix&);
	friend Matrix operator*(const Matrix&, double);
	friend Matrix operator*(double, const Matrix&);	
	// for test
	friend std::ostream& operator<<(std::ostream&, const Matrix&);
protected:
	std::size_t _Rows;
	std::size_t _Cols;
	double* _Data; // row major array
};
 
Matrix operator+(const Matrix& lhs, const Matrix& rhs)
{
	Matrix sum(lhs._Rows, lhs._Cols);
	for(std::size_t ix = 0; ix != lhs._Rows * lhs._Cols; ++ix)
	{
		*(sum._Data + ix) = *(lhs._Data + ix) + *(rhs._Data + ix);
	}
	return sum;
}
 
Matrix operator-(const Matrix& lhs, const Matrix& rhs)
{
	Matrix diff(lhs._Rows, lhs._Cols);
	for(std::size_t ix = 0; ix != lhs._Rows * lhs._Cols; ++ix)
	{
		*(diff._Data + ix) = *(lhs._Data + ix) - *(rhs._Data + ix);
	}
	return diff;
}
 
Matrix operator*(const Matrix& mat, double scala)
{
	Matrix prod(mat._Rows, mat._Cols);
	for(std::size_t ix = 0; ix != mat._Rows * mat._Cols; ++ix)
	{
		*(prod._Data + ix) = *(mat._Data + ix) * scala;
	}
	return prod;
}
 
Matrix operator*(double scala, const Matrix& mat)
{
	return mat * scala;
}
 
std::ostream& operator<<(std::ostream& os, const Matrix& mat)
{
	for(std::size_t i = 0; i != mat.rows(); ++i)
	{
		for(std::size_t j = 0; j != mat.cols(); ++j)
		{
			os << mat(i, j) << "\t";
		}
		os << std::endl;
	}
	return os;
}
 
class Solver
{
public:
	Solver(Matrix& mat) : _EigenVecs(mat)
	{
		_Size = mat.rows();
		_EigenVals.resize(_Size, 1);
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', _Size, _EigenVecs._Data, _Size, _EigenVals._Data);
	}
	const Matrix& eigenvalues() const
	{
		return _EigenVals;
	}
	const Matrix& eigenvectors() const
	{
		return _EigenVecs;
	}
	friend std::ostream& operator<<(std::ostream&, const Solver&);
private:
	std::size_t _Size;
	Matrix _EigenVals;
	Matrix& _EigenVecs;
};
 
std::ostream& operator<<(std::ostream& os, const Solver& eigen)
{
	for(std::size_t ix = 0; ix != eigen._Size; ++ix)
	{
		os << "eigen value: " << eigen._EigenVals(ix, 0) << std::endl;
		os << "corresponding eigen vector:" << std::endl;
		for(std::size_t j = 0; j != eigen._Size; ++j)
		{
			os << eigen._EigenVecs(j, ix) << std::endl;
		}
	}
	return os;
}
 
#endif

