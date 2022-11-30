

#ifndef COMPLEX_H
#define COMPLEX_H
#include <string>
class Complex{
	//private:
	//double real;
	//double imag;

	public:
	double real;
	double imag;
	Complex(double real, double imag);
	double getReal() const;
	double getImag() const;
	void setReal(double real);
	void setImag(double imag);
	void conjugate();
	Complex operator + (Complex const &obj);
	Complex operator - (Complex const &obj);
	Complex operator * (const double d);
	Complex operator * (Complex const &obj);
	Complex operator / (const double d);
	Complex & operator +=(Complex const &obj);
		
	std::string toString();
	//functionality for adding, subtracting, equal


};



#endif




