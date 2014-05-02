#ifndef INVMVGAUSSIAN_H
#define INVMVGAUSSIAN_H

#include "objcokus.h"

class InverseGaussian
{
public:
	InverseGaussian(void);
	~InverseGaussian(void);
	InverseGaussian(double dMu, double dScale) {
		m_iSet = 0;
		m_dMu = dMu;
		m_dScale = dScale;
	}

	double sample();

	void reset(double dMu, double dScale) {
		m_iSet = 0;
		m_dMu = dMu;
		m_dScale = dScale;
	}

	double nextGaussian();

private:
	double m_dMu;
	double m_dScale;

	// for Gaussian random variable
	double m_dGset;
	int m_iSet;
	
	objcokus cokus;
};

#endif