#include "complex.h"
#include <string>


using namespace std;


Complex::Complex(double real = 0.0, double imag = 0.0){this->real = real; this->imag = imag;}

double Complex::getReal()const{
	return real;
}

double Complex::getImag()const{
	return this->imag;
}

void Complex::setReal(double real){
	this->real = real;
}

void Complex::setImag(double imag){
	this->imag = imag;
}
void Complex::conjugate(){
	this->real = this->real;
	this->imag = -this->imag;
}
Complex Complex::operator+(Complex const &c){
	Complex comp;
	comp.setReal(this->getReal() + c.getReal());
	comp.setImag(this->getImag() + c.getImag());
	return comp;
}

Complex Complex::operator-(Complex const &c){
	Complex comp;
	comp.setReal(this->getReal() - c.getReal());
	comp.setImag(this->getImag() - c.getImag());
	return comp;

}
Complex Complex::operator*(const Complex &c){
	Complex comp;
	comp.real = (this->getReal() * c.getReal() - this->getImag() * c.getImag());
	comp.imag = (this->getReal() * c.getImag() + this->getImag() * c.getReal());
	return comp;


}
Complex Complex::operator*(const double d){
	Complex comp;
	comp.setReal(this->getReal() * d);
	comp.setImag(this->getImag() * d);
	return comp;
}
Complex Complex::operator/(const double d){
	Complex comp;
	comp.setReal(this->getReal() / d);
	comp.setImag(this->getImag() / d);
	return comp;
}
Complex & Complex::operator +=(Complex const&obj){
	real += obj.real;
	imag += obj.imag;
	return *this;
	
}
string Complex::toString(){
	string output;

	if(getImag() < 0){
		output.append(to_string(getReal()));
		output.append(to_string(getImag()));
		output.push_back('i');
		//cout << getReal() << getImag() << "i" << endl;
	}
	else{
		output.append(to_string(getReal()));
		output.push_back('+');
		output.append(to_string(getImag()));
		output.push_back('i');
		//cout << getReal() << "+" << getImag() << "i" << endl;
	}
	return output;
}
