"""Tayte Choudhury: quote cleaning, Black-76 inversion and constrained SVI.
CSV columns: timestamp, spot, expiry_years, strike, type (C/P), bid, ask.
--demo generates explicitly synthetic quotes, never historical SPY evidence.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq,minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def black(f,k,t,v,d,kind):
    z=v*np.sqrt(t); x=np.log(f/k)/z+z/2
    call=d*(f*ndtr(x)-k*ndtr(x-z))
    return call if kind=='C' else call-d*(f-k)

def implied(price,f,k,t,d,kind):
    intrinsic=d*max((f-k) if kind=='C' else (k-f),0)
    upper=d*(f if kind=='C' else k)
    if not intrinsic < price < upper: raise ValueError('Price outside invertible bounds')
    return brentq(lambda v:black(f,k,t,v,d,kind)-price,1e-6,5,xtol=1e-6)

def svi(k,p):
    a,b,r,m,s=p; z=k-m
    return a+b*(r*z+np.sqrt(z*z+s*s))

def density(k,p):
    a,b,r,m,s=p;z=k-m;root=np.sqrt(z*z+s*s);w=svi(k,p)
    wp=b*(r+z/root);wpp=b*s*s/root**3
    return (1-k*wp/(2*w))**2-wp**2/4*(1/w+.25)+wpp/2

def synthetic():
    rows=[]
    for t in np.linspace(.05,1.5,18):
        f=500*np.exp(.025*t);d=np.exp(-.04*t)
        for k in np.linspace(350,650,120):
            v=np.sqrt(svi(np.log(k/f),[.025*t,.05*t,-.35,0,.2])/t)
            for kind in ['C','P']:
                mid=black(f,k,t,v,d,kind);half=max(.0001,mid*.01)
                rows.append(['synthetic',500,t,k,kind,max(0,mid-half),mid+half])
    return pd.DataFrame(rows,columns=['timestamp','spot','expiry_years','strike','type','bid','ask'])

def run(df,out,label):
    out.mkdir(parents=True,exist_ok=True)
    required=['timestamp','spot','expiry_years','strike','type','bid','ask']
    if any(c not in df for c in required):raise ValueError('Required columns: '+','.join(required))
    if df.timestamp.nunique()!=1 or df.spot.nunique()!=1:raise ValueError('Require one synchronized snapshot and spot')
    if df.duplicated(['expiry_years','strike','type']).any():raise ValueError('Duplicate option contracts')
    if not df.type.isin(['C','P']).all():raise ValueError('Option type must be C or P')
    if not np.isfinite(df[['spot','expiry_years','strike','bid','ask']]).all().all():raise ValueError('Nonfinite input')
    if (df[['spot','expiry_years','strike']]<=0).any().any():raise ValueError('Nonpositive input')
    raw=len(df);filters={};df=df.copy()
    for name,mask in [('zero_bid',lambda x:x.bid>0),('crossed_locked',lambda x:x.ask>x.bid),('relative_spread',lambda x:(x.ask-x.bid)/((x.ask+x.bid)/2)<=.3),('short_expiry',lambda x:x.expiry_years>=7/365),('sparse_expiry',lambda x:x.groupby('expiry_years').strike.transform('nunique')>=6)]:
        keep=mask(df);filters[name]=int((~keep).sum());df=df[keep].copy()
    df['mid']=(df.bid+df.ask)/2
    grid=np.linspace(-1,1,2000);params=[];summary=[];observed=[];failed=0;previous=None
    for t,part in df.groupby('expiry_years',sort=True):
        paired=part.pivot(index='strike',columns='type',values='mid').dropna()
        if len(paired)<3 or not {'C','P'}<=set(paired):raise ValueError('Insufficient parity pairs')
        slope,intercept=np.polyfit(paired.index,paired.C-paired.P,1);d=-slope;f=intercept/d
        if not 0<d<1.1 or not f>0:raise ValueError('Invalid parity estimate')
        points=[]
        for row in part.itertuples():
            if (row.type=='C' and row.strike<f) or (row.type=='P' and row.strike>=f):continue
            try:v=implied(row.mid,f,row.strike,t,d,row.type)
            except ValueError:failed+=1;continue
            points.append((np.log(row.strike/f),v))
        if len(points)<6:raise ValueError('Insufficient invertible OTM quotes')
        k,v=np.array(points).T;target=v*v*t
        # Four groups: positive variance, Lee wing slopes, butterfly density,
        # and calendar monotonicity. Verification is bounded to the stated grid.
        def constraints(p):
            a,b,r,m,s=p
            c=[a+b*s*np.sqrt(1-r*r)-1e-8,2-b*(1+r)-1e-6,2-b*(1-r)-1e-6]
            return np.r_[c,density(grid,p),svi(grid,p)-(svi(grid,previous) if previous is not None else 0)]
        initial=np.array([np.min(target)*.6,.03*t,-.3,0,.2])
        if previous is not None:initial=previous.copy();initial[0]+=max(1e-5,np.mean(target)-np.mean(svi(k,initial)))
        fit=minimize(lambda p:np.mean((svi(k,p)-target)**2)*1e5,initial,method='SLSQP',bounds=[(1e-9,2),(1e-8,1),(-.999,.999),(-1,1),(.001,2)],constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':600,'ftol':1e-10})
        if not fit.success or np.min(constraints(fit.x)) < -1e-7:raise RuntimeError('SVI constraints failed: '+fit.message)
        p=fit.x;rmse=float(np.sqrt(np.mean((np.sqrt(svi(k,p)/t)-v)**2))*100)
        summary.append(dict(expiry_years=float(t),forward=float(f),discount=float(d),quotes=len(k),rmse_vol_points=rmse,min_density=float(density(grid,p).min()),min_calendar_gap=None if previous is None else float((svi(grid,p)-svi(grid,previous)).min())))
        params.append(dict(expiry_years=float(t),parameters=p.tolist()));observed.append((t,k,v));previous=p
    if not params:raise ValueError('No calibrated slices')
    report=dict(author='Tayte Choudhury',dataset=label,raw_quotes=raw,filters=filters,noninvertible=failed,grid=dict(min=-1,max=1,points=2000),scope='Finite-grid verification only; no global arbitrage guarantee or market performance claim.',slices=summary,parameters=params)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(summary).to_csv(out/'slices.csv',index=False)
    fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d');ks=np.linspace(-.5,.5,100);ts=np.array([p['expiry_years'] for p in params]);K,T=np.meshgrid(ks,ts);V=np.array([np.sqrt(svi(ks,p['parameters'])/p['expiry_years'])*100 for p in params]);ax.plot_surface(K,T,V,cmap='viridis');ax.set(xlabel='Log forward moneyness',ylabel='Years to expiry',zlabel='Implied volatility (%)',title='Tayte Choudhury | SVI surface\n'+label);fig.tight_layout();fig.savefig(out/'surface.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for t,k,v in observed[::max(1,len(observed)//4)]:
        p=next(x['parameters'] for x in params if x['expiry_years']==t);ax.scatter(k,v*100,s=9);ax.plot(k,np.sqrt(svi(k,p)/t)*100,label=f'{t:.2f} years')
    ax.set(xlabel='Log forward moneyness',ylabel='Implied volatility (%)',title='Observed quotes and fitted SVI | '+label);ax.legend();fig.tight_layout();fig.savefig(out/'smiles.png',dpi=160);plt.close(fig)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--csv',type=Path);ap.add_argument('--demo',action='store_true');ap.add_argument('--out',type=Path,default=Path('results'));a=ap.parse_args()
    if bool(a.csv)==a.demo:ap.error('Choose --csv or --demo')
    r=run(synthetic() if a.demo else pd.read_csv(a.csv),a.out,'Synthetic validation dataset' if a.demo else a.csv.name)
    print(json.dumps({k:r[k] for k in ['dataset','raw_quotes','filters','noninvertible']},indent=2))
