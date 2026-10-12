from pathlib import Path
import sys,json,hashlib,shutil,time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(r'D:\石阳\博士工作\科研\周报')
REF=BASE/'week update-132-shiyang/simulation/响应曲线'
RAW=BASE/'week update-65-shiyang/gamry/20250313/night-1st test/全部条件切分周期-auto-complete response'
OUT=BASE/'week update-140-shiyang/200Hz_10与50光强_连续动态拟合'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
PERIOD_MS=5.;WIDTH_MS=2.5

def periodic_states(t_ms,delays_ms,taus_ms,taus_off_ms=None):
    """Periodic steady state; different positive charging and discharging time constants."""
    if taus_off_ms is None:taus_off_ms=taus_ms
    phi=(np.asarray(t_ms)[:,None]-delays_ms[None,:])%PERIOD_MS
    alpha=np.exp(-WIDTH_MS/taus_ms)
    beta=np.exp(-(PERIOD_MS-WIDTH_MS)/taus_off_ms)
    q0=(1-alpha)*beta/(1-alpha*beta)
    qon=1+(q0-1)*alpha
    return np.where(phi<WIDTH_MS,1+(q0-1)*np.exp(-phi/taus_ms),qon*np.exp(-np.maximum(phi-WIDTH_MS,0)/taus_off_ms))

def predict_periodic(t_ms,params):
    q=periodic_states(np.asarray(t_ms),np.array(params['delays_ms']),np.array(params['taus_ms']),np.array(params['taus_off_ms']))
    return params['offset_uA']+q@np.array(params['amplitudes_uA'])

def simulate_continuous(t_s,illumination,params,initial_state=None):
    """ZOH input integration. Keep each state across events, including after light leaves.

    illumination is the measured/assumed normalized optical drive at the same geometry
    as calibration. For array applications the input and spatial delays must be specified
    consistently; do not count geometric delay twice. This is an empirical linear model.
    """
    t=np.asarray(t_s,float);u=np.asarray(illumination,float)
    if len(t)!=len(u) or np.any(np.diff(t)<=0):raise ValueError('Need equal-length input and increasing times')
    taus=np.asarray(params['taus_ms'])*1e-3;taus_off=np.asarray(params['taus_off_ms'])*1e-3;delays=np.asarray(params['delays_ms'])*1e-3
    q=np.zeros(len(taus)) if initial_state is None else np.array(initial_state,dtype=float)
    states=np.empty((len(t),len(q)));states[0]=q
    # Evaluate delayed input with zero-order hold; prehistory is dark, unless caller extends input.
    for n in range(1,len(t)):
        sample_t=t[n-1]-delays
        ix=np.searchsorted(t,sample_t,side='right')-1
        drive=np.where(ix>=0,u[np.maximum(ix,0)],0)
        effective_tau=np.where(drive>=q,taus,taus_off)
        decay=np.exp(-(t[n]-t[n-1])/effective_tau)
        q=q*decay+drive*(1-decay)
        states[n]=q
    return params['offset_uA']+states@np.array(params['amplitudes_uA']),states

def fit_case(t,y,k,restarts=7):
    scale=np.ptp(y)
    ridge=1e-5
    def coefficients(p):
        tau=np.exp(p[:k]);tau_off=np.exp(p[k:2*k]);delay=p[2*k:]
        H=np.column_stack([np.ones(len(t)),periodic_states(t,delay,tau,tau_off)])
        penalty=np.zeros((k,k+1));penalty[:,1:]=np.eye(k)*np.sqrt(ridge)
        coef=np.linalg.lstsq(np.vstack([H,penalty]),np.r_[y,np.zeros(k)],rcond=None)[0]
        return H,coef
    def residual(p):
        H,c=coefficients(p)
        return np.r_[(H@c-y)/scale,np.sqrt(ridge)*c[1:]/scale]
    rng=np.random.default_rng(200+k)
    best=None
    starts=[(np.array([.015,.12,.035,.45]),np.array([.74,.76,1.35,1.37]))] if k==4 else []
    while len(starts)<restarts:
        starts.append((np.exp(rng.uniform(np.log(.008),np.log(.7),k)),rng.uniform(.65,1.45,k)))
    for index,(tau,delay) in enumerate(starts):
        opt=least_squares(residual,np.r_[np.log(tau),np.log(tau),delay],bounds=(np.r_[np.full(2*k,np.log(.003)),np.full(k,.3)],np.r_[np.full(2*k,np.log(5.)),np.full(k,1.8)]),max_nfev=650,ftol=2e-9,xtol=2e-9,gtol=2e-9)
        H,c=coefficients(opt.x);yp=H@c;rmse=float(np.sqrt(np.mean((yp-y)**2)))
        print('restart',k,index+1,'RMSE_uA',round(rmse,5),flush=True)
        if best is None or np.sum(residual(opt.x)**2)<best[0]:best=(np.sum(residual(opt.x)**2),opt,c,yp,rmse)
    _,opt,c,yp,rmse=best
    order=np.argsort(opt.x[2*k:])
    params={'n_branches':k,'offset_uA':float(c[0]),'amplitudes_uA':c[1:][order].tolist(),'taus_ms':np.exp(opt.x[:k])[order].tolist(),'taus_off_ms':np.exp(opt.x[k:2*k])[order].tolist(),'delays_ms':opt.x[2*k:][order].tolist(),'model':'empirical delayed state model with asymmetric charging/discharging rates','calibration_frequency_hz':200,'assumed_duty_cycle':.5,'rmse_uA':rmse,'nrmse_by_peak_to_peak':rmse/scale,'r_squared':float(1-np.sum((yp-y)**2)/np.sum((y-y.mean())**2)),'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message)}
    return params,yp

if __name__=='__main__':
    all_params={};metrics=[];validation=[]
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,light in enumerate([10,50]):
        source=REF/f'混合响应曲线-{light}%.csv';data=pd.read_csv(source);t=data.Time_s.to_numpy()*1e3;y=data.Current_A.to_numpy()*1e6
        print('FIT',light,flush=True)
        p,yp=fit_case(t,y,4)
        # An additional branch is tried only when four branches cannot reach 1% full-scale RMSE.
        if p['nrmse_by_peak_to_peak']>.01:
            candidate,cy=fit_case(t,y,5,restarts=8)
            if candidate['rmse_uA']<.8*p['rmse_uA']:p,yp=candidate,cy
        p['reference_path']=str(source);p['reference_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();p['raw_source']=str(RAW/f'peaks_auto_{light}%_0_200Hz/cycle_1.csv')
        all_params[str(light)]=p
        pd.DataFrame({'Time_s':t*1e-3,'measured_current_A':y*1e-6,'fitted_current_A':yp*1e-6,'residual_A':(yp-y)*1e-6}).to_csv(OUT/f'fit_{light}percent.csv',index=False,encoding='utf-8-sig')
        metrics.append({'intensity_percent':light,**{key:p[key] for key in ['n_branches','rmse_uA','nrmse_by_peak_to_peak','r_squared']}})
        axs[row,0].plot(t,y,color='#1664b5',lw=2,label='实测参考周期');axs[row,0].plot(t,yp,'--',color='#e77826',lw=1.8,label='连续动态模型');axs[row,0].set(title=f'{light}% 光强｜完整周期',xlabel='时间 (ms)',ylabel='电流 (µA)');axs[row,0].legend();axs[row,0].grid(alpha=.2)
        axs[row,1].plot(t,yp-y,color='#7a3f9b');axs[row,1].axhline(0,color='grey',lw=.8);axs[row,1].set(title=f'残差｜RMSE {p["rmse_uA"]:.3f} µA；R² {p["r_squared"]:.5f}',xlabel='时间 (ms)',ylabel='拟合 − 实测 (µA)');axs[row,1].grid(alpha=.2)
        # Same operating conditions only; these cycles are not used for fitting.
        folder=RAW/f'peaks_auto_{light}%_0_200Hz'
        for number in range(2,22):
            f=folder/f'cycle_{number}.csv'
            if not f.exists():continue
            d=pd.read_csv(f);actual=d.Current_A.to_numpy()*1e6;pred=predict_periodic(d.Time_s.to_numpy()*1e3,p)
            validation.append({'intensity_percent':light,'cycle':number,'file':str(f),'rmse_uA':float(np.sqrt(np.mean((actual-pred)**2))),'nrmse_by_peak_to_peak':float(np.sqrt(np.mean((actual-pred)**2))/np.ptp(actual))})
        print('BEST',light,json.dumps(p,ensure_ascii=False),flush=True)
    fig.suptitle('200 Hz、0°：仅拟合阵列模拟所用的10%与50%光强',fontsize=17);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'01_完整周期拟合与残差.png',dpi=220);fig.savefig(OUT/'01_完整周期拟合与残差.pdf');plt.close(fig)
    (OUT/'fitted_parameters.json').write_text(json.dumps(all_params,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(metrics).to_csv(OUT/'fit_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(validation).to_csv(OUT/'same_condition_other_cycles.csv',index=False,encoding='utf-8-sig')
    # Edge views retain the recovery tail, and use no event reset.
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,light in enumerate([10,50]):
        d=pd.read_csv(OUT/f'fit_{light}percent.csv')
        for col,(name,anchor) in enumerate([('上升沿',.000716),('下降沿',.003219)]):
            rel=(d.Time_s.to_numpy()-anchor)*1e3;sel=(rel>=-.716)&(rel<=1.784);ax=axs[row,col]
            ax.plot(rel[sel],d.measured_current_A.to_numpy()[sel]*1e6,lw=2,color='#1664b5',label='实测参考');ax.plot(rel[sel],d.fitted_current_A.to_numpy()[sel]*1e6,'--',lw=1.8,color='#e77826',label='动态拟合')
            ax.set(title=f'{light}% 光强｜{name}',xlabel='相对边沿锚点的时间 (ms)',ylabel='电流 (µA)');ax.legend();ax.grid(alpha=.2)
            if col==1:ax.axvspan(.737,1.784,color='#2a9b52',alpha=.08)
    fig.suptitle('进入与离开均保留连续动态：下降沿尾部自然恢复',fontsize=17);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'02_上升沿与下降沿拟合.png',dpi=220);fig.savefig(OUT/'02_上升沿与下降沿拟合.pdf');plt.close(fig)
    # Continuous pulse histories illustrate unrelaxed states; they are extrapolations, not validation.
    ts=np.arange(0,.025,1e-5);pulse=np.zeros(len(ts));pulse[(ts>=.002)&(ts<.010)]=1;pulse[(ts>=.012)&(ts<.0126)]=1;pulse[(ts>=.013)&(ts<.015)]=1
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True);axs[0].plot(ts*1e3,pulse,color='black');axs[0].set_ylabel('归一化光照');axs[0].set_title('改变照明持续时间与间隔：仅用于演示连续状态，不代表实测验证')
    for ax,light in zip(axs[1:],[10,50]):
        current,states=simulate_continuous(ts,pulse,all_params[str(light)]);ax.plot(ts*1e3,current);ax.set_ylabel(f'{light}% 电流 (µA)');ax.grid(alpha=.2)
        pd.DataFrame({'time_s':ts,'illumination':pulse,'simulated_current_A':current*1e-6}).to_csv(OUT/f'continuous_demo_{light}percent.csv',index=False,encoding='utf-8-sig')
    axs[-1].set_xlabel('时间 (ms)');fig.tight_layout();fig.savefig(OUT/'03_连续充放电演示_非验证.png',dpi=180);plt.close(fig)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print('DONE',OUT,flush=True)
