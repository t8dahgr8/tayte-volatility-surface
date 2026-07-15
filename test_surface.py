import unittest
import numpy as np
from surface import black,implied,svi,density,synthetic,run
from tempfile import TemporaryDirectory
from pathlib import Path

class SurfaceTests(unittest.TestCase):
    def test_black_parity_and_inversion(self):
        for t in [.05,.5,2]:
            for k in [80,100,130]:
                c=black(103,k,t,.27,.98,'C');p=black(103,k,t,.27,.98,'P')
                self.assertAlmostEqual(c-p,.98*(103-k),places=10)
                for kind,price in [('C',c),('P',p)]:self.assertAlmostEqual(implied(price,103,k,t,.98,kind),.27,places=5)
    def test_noninvertible(self):
        for price in [-1,0,100,101]:
            with self.assertRaises(ValueError):implied(price,100,100,1,1,'C')
    def test_reject_mixed_snapshot(self):
        df=synthetic();df.loc[0,'timestamp']='different'
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):run(df,Path(d),'test')
    def test_derivatives(self):
        p=[.02,.03,-.4,.01,.2];k=np.linspace(-.8,.8,100);h=1e-4
        w=svi(k,p);wp=(svi(k+h,p)-svi(k-h,p))/(2*h);wpp=(svi(k+h,p)-2*w+svi(k-h,p))/h**2
        ref=(1-k*wp/(2*w))**2-wp**2/4*(1/w+.25)+wpp/2
        np.testing.assert_allclose(density(k,p),ref,atol=1e-7)

if __name__=='__main__':unittest.main()
