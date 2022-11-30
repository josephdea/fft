#include <iostream>
#include <string>
#include <vector>
#include "complex.h"
#include <math.h>
using namespace std;



Complex expC(double theta){
	return Complex(cos(theta),sin(theta));
}

vector<Complex> forwardFT(vector<double>x){


	int n = x.size();
	vector<Complex>res;
	for(int k = 0;k<n;k++){
		Complex c(0.0,0.0);
		for(int j = 0;j<n;j++){
			c += (expC(-2 * M_PI * j * k / n)) * x[j];
		}
		res.push_back(c);
	}
	return res;

}

vector<Complex> backwardFT(vector<Complex>x){
	int n = x.size();
	vector<Complex>res;
	for(int k = 0;k<n;k++){
		Complex c(0.0,0.0);
		for(int j = 0;j<n;j++){
			c += x[j] * expC(2 * M_PI * j * k / n);
		}
		res.push_back(c);
	}
	return res;

}


vector<Complex>forwardFFT(vector<Complex>x){
	if(x.size() == 1){
		return x;
	}
	int N = x.size();
	vector<Complex>xEven;
	vector<Complex>xOdd;
	vector<Complex>factors;
	vector<Complex>res;

	for(int i = 0;i<N;i += 2){
		xEven.push_back(x[i]);
	}
	for(int i = 1;i<N;i+=2){
		xOdd.push_back(x[i]);
	}
	xEven = forwardFFT(xEven);
	xOdd = forwardFFT(xOdd);
	
	for(int i = 0;i<N;i++){
		factors.push_back(expC(-2 * M_PI * i / N));
	}
	for(int i = 0;i<N/2;i++){
		res.push_back(xEven[i] + factors[i] * xOdd[i]);
	}
	for(int i = N/2;i<N;i++){
		res.push_back(xEven[i- N/2] + factors[i] * xOdd[i - N/2]);
	}
	return res;
}

vector<Complex>inverseFFT(vector<Complex>x){
	//cout <<"debug" << endl;
	for(int i = 0;i<x.size();i++){
		x[i].conjugate();
	}
	vector<Complex>res = forwardFFT(x);
	for(int i= 0;i<res.size();i++){
		res[i].conjugate();
		res[i] = res[i] / res.size();
	}
	return res;

}


int main(){
	Complex c1(5.0,5.0);
	Complex c2 = Complex(2.0,3.0);
	c2 += c1 * 5;
	//cout << c2.toString() << endl;
	//cout << expC(3.14).toString() << endl;
	//exit(0);
	vector<double>d;
	vector<Complex>c;
	for(int i = 0;i<16;i++){
		d.push_back((double)(rand()%100));
		c.push_back(Complex(d[i],0.0));
	}


	vector<Complex>res = forwardFT(d);
	vector<Complex>resFFT = forwardFFT(c);
	vector<Complex>orig = inverseFFT(resFFT);
	for(int i = 0;i<res.size();i++){
		//cout << orig[i].toString() << endl;
		cout << (c[i] - orig[i]).toString() << endl;
		//cout << (res[i] - resFFT[i]).toString() << endl;
	}
	
	//vector<Complex>orig = backwardFT(res);
	//for(int i = 0;i<d.size();i++){
	//	cout << d[i] << "  -  " << (orig[i]/orig.size()).toString() << endl;
	//}




}
