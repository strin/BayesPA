#include "InverseGaussian.h"
#include <math.h>
#include "defs.h"
#include "stdio.h"

InverseGaussian::InverseGaussian(void)
{
	cokus.reloadMT();
	cokus.seedMT(time(NULL));
}

InverseGaussian::~InverseGaussian(void)
{
}

double InverseGaussian::sample()
{
	  double v = nextGaussian();   // sample from a normal distribution with a mean of 0 and 1 standard deviation

	  double y = v*v;
      double x = m_dMu + (m_dMu * m_dMu * y) / (2* m_dScale) - (m_dMu/(2*m_dScale)) * sqrt(4*m_dMu*m_dScale*y + m_dMu*m_dMu*y*y);

	  double test = cokus.random01();  // sample from a uniform distribution between 0 and 1

	  if (test <= (m_dMu)/(m_dMu + x))
             return x;
      else
             return (m_dMu*m_dMu) / x;
}

double InverseGaussian::nextGaussian()
{
	//double dMu = 0;
	//double dSigma = 1;

	if ( m_iSet == 0 ) {
		double dRsq = 0;
		double v1, v2;
		do {
			v1 = 2.0 * cokus.random01() - 1.0;
			v2 = 2.0 * cokus.random01() - 1.0;
			dRsq = v1 * v1 + v2 * v2;
		} while (dRsq > 1.0 || dRsq < 1e-300);

		double dFac = sqrt(-2.0 * log(dRsq) / dRsq);
		m_dGset = v1 * dFac;
		m_iSet = 1;
		return /*dMu + dSigma * */ v2 * dFac;
	} else {
		m_iSet = 0;
		return /*dMu + dSigma **/ m_dGset;
	}
}

