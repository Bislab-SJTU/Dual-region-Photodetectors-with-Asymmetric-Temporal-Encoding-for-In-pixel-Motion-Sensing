from pathlib import Path
from collections import OrderedDict
import sys,json,hashlib,shutil,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imageio_ffmpeg

sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(r'D:\石阳\博士工作\科研\周报')
OLD=BASE/'week update-132-shiyang/simulation'
FIT=BASE/'week update-140-shiyang/200Hz_10与50光强_连续动态拟合'
ROOT=BASE/'week update-140-shiyang/simulation_连续动态模型'
ROOT.mkdir(exist_ok=True)
RESULT=ROOT/'result/亮度变化'
CODE=ROOT/'code';CODE.mkdir(exist_ok=True)
CONFIG={'rows':100,'cols':100,'pixel_width_mm':5.4,'pixel_height_mm':3.25,'pixel_spacing_mm':.5,'array_center_x_mm':4.,'start_x_mm':-850.,'start_y_mm':-80.,'move_distance_mm':1150.,'speed_mm_s':7330.4,'motion_angle_deg':0.,'image_scale_factor':15.,'nominal_dt_s':1e-5,'time_scale':.03,'video_fps':15,'brightness_boundary_column':50,'left_reference_percent':50,'right_reference_percent':10,'occlusion_enabled':False,'reference_event_anchor_s':.000716,'current_display_vmax_uA':135.,'illumination_sampling':'same point / opaque-alpha>=1 rule as previous generator','model_role':'empirical combined hybrid response calibrated at 0deg,200Hz; no separate RC/C identification','network_inference_performed':False}
PARAMS=json.loads((FIT/'fitted_parameters.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
ts=np.linspace(0,CONFIG['move_distance_mm']/CONFIG['speed_mm_s'],int((CONFIG['move_distance_mm']/CONFIG['speed_mm_s'])/CONFIG['nominal_dt_s'])+1)
dt=ts[1]-ts[0]
step=max(1,int((.03/15)/1e-5));frame_idx=np.arange(0,len(ts),step);frame_times=ts[frame_idx]
im=Image.open(OLD/'car_bin.png');dpi=im.info['dpi'][0];mm_per_px=25.4/dpi
rgba=np.array(im.convert('RGBA'));alpha=np.flipud(rgba[:,:,3])/255.;height,width=alpha.shape
image_w=width*mm_per_px*15;image_h=height*mm_per_px*15
CONFIG.update({'image_dpi':dpi,'image_width_mm':image_w,'image_height_mm':image_h,'actual_dt_s':float(dt),'time_samples':len(ts),'frame_count':len(frame_idx),'frame_sample_step':int(step),'first_x_mm':4-(100*5.4+99*.5)/2})
xpos=CONFIG['first_x_mm']+np.arange(100)*5.9
ypos=np.arange(100)*3.75
u_index=np.trunc((xpos[:,None]-(-850+7330.4*ts[None,:]))/image_w*width).astype(int)
v_index=np.trunc((ypos+80)/image_h*height).astype(int)
valid_u=(u_index>=0)&(u_index<width)

def events_from_mask(mask):
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    return list(zip(starts.tolist(),ends.tolist()))

def pixel_mask(r,c):
    if not 0<=v_index[r]<height:return np.zeros(len(ts),bool)
    return valid_u[c] & (alpha[v_index[r],np.clip(u_index[c],0,width-1)]>=1)

def dynamic_current(times,event_times,params,reference_anchor_s=.000716):
    """Exact evolution of the fitted state model across piecewise-constant illumination.

    Remove reference-cycle phase once. Delays retained between branches are empirical;
    do not add another sub-unit geometric delay to this combined response model.
    """
    output=np.full(len(times),params['offset_uA'],float)
    for amp,tau_on,tau_off,delay_ms in zip(params['amplitudes_uA'],params['taus_ms'],params['taus_off_ms'],params['delays_ms']):
        delay=delay_ms*.001-reference_anchor_s
        if delay<0:raise ValueError('Calibration phase yields a negative causal delay')
        edge_times=[];drives=[]
        for a,b in event_times:edge_times.extend([a+delay,b+delay]);drives.extend([1.,0.])
        state=0.;previous=times[0];drive=0.
        values=np.empty(len(times),float)
        for boundary,next_drive in zip(edge_times+[times[-1]+(times[1]-times[0])],drives+[0.]):
            if boundary<previous:raise ValueError('Overlapping/unsorted input events')
            lo=np.searchsorted(times,previous,side='left');hi=np.searchsorted(times,boundary,side='left')
            tau=(tau_on if drive else tau_off)*.001
            values[lo:hi]=drive+(state-drive)*np.exp(-(times[lo:hi]-previous)/tau)
            state=drive+(state-drive)*np.exp(-(boundary-previous)/tau)
            previous=boundary;drive=next_drive
            if boundary>times[-1]:break
        output+=amp*values
    return output*1e-6

STEADY={i:pd.read_csv(OLD/'响应曲线'/f'持续响应曲线-{i}%.csv') for i in [10,50]}
def old_steady_response(events,light):
    d=STEADY[light];rt=d.Time_s.to_numpy();ry=d.Current_A.to_numpy();series=np.zeros(len(ts));s2=.000737;s4=.000737
    i2=np.interp(.000716+s2,rt,ry);i4=np.interp(.003219,rt,ry)
    for a,b in events:
        length=7330.4*(ts[b-1]-ts[a]);excess=max(0,length-5.4)/7330.4;total=s2+excess+s4
        end=np.searchsorted(ts,ts[a]+total,side='left');local=ts[a:end]-ts[a]
        vals=np.empty(len(local));rise=local<s2;plateau=(local>=s2)&(local<s2+excess);fall=~(rise|plateau)
        vals[rise]=np.interp(.000716+local[rise],rt,ry,right=ry[-1])
        if excess>0:vals[plateau]=i2+(local[plateau]-s2)/excess*(i4-i2)
        vals[fall]=np.interp(.003219+local[fall]-s2-excess,rt,ry,right=ry[-1])
        series[a:end]=vals
    return series

TIME_PREFIX=[f'{value:.12g},' for value in ts]
CSV_CACHE_PATHS={}
def complete_csv(path):
    if not path.exists() or path.stat().st_size<100:return False
    try:
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-128));line=handle.read().decode('ascii').splitlines()[-1]
        parts=line.split(',')
        return len(parts)==2 and parts[0]==format(ts[-1],'.12g') and np.isfinite(float(parts[1]))
    except (OSError,ValueError,UnicodeError):return False

def save_csv(path,series,cache_key):
    if complete_csv(path):
        CSV_CACHE_PATHS.setdefault(cache_key,path);return
    if cache_key in CSV_CACHE_PATHS:
        shutil.copyfile(CSV_CACHE_PATHS[cache_key],path);return
    temp=path.with_suffix('.tmp')
    temp.write_text('Time_s,Current_A\n'+''.join(prefix+format(value,'.12g')+'\n' for prefix,value in zip(TIME_PREFIX,series)),encoding='ascii')
    temp.replace(path);CSV_CACHE_PATHS[cache_key]=path

if __name__=='__main__':
    # Verify scene timing against existing waveforms before creating the large output set.
    checks=[]
    for r,c in [(20,20),(30,30),(50,30),(50,70),(70,85),(0,0)]:
        events=events_from_mask(pixel_mask(r,c))
        previous=pd.read_csv(OLD/f'result/亮度变化/混合响应/pixel_waveforms/pixel_{r}_{c}.csv')
        assert np.allclose(previous.Time_s,ts,atol=1e-12,rtol=0)
        starts=np.flatnonzero((previous.Current_A.to_numpy()!=0)&~np.r_[False,previous.Current_A.to_numpy()[:-1]!=0])
        assert starts.tolist()==[a for a,b in events],(r,c,starts,events)
        old_s=pd.read_csv(OLD/f'result/亮度变化/稳态响应/pixel_waveforms/pixel_{r}_{c}.csv').Current_A.to_numpy()
        error=float(np.max(np.abs(old_steady_response(events,50 if c<50 else 10)-old_s)))
        assert error<1e-12,(r,c,error)
        checks.append({'row':r,'col':c,'event_onsets_match_previous':True,'steady_max_difference_A':error})
    print('Scene timing and steady-control checks passed',flush=True)
    # Check event integrator against the fitted periodic analytic solution.
    spec=importlib.util.spec_from_file_location('fit',FIT/'fit_array_response_dynamics.py');fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
    test_times=np.arange(0,.05,1e-5);test_events=[(k*.005,k*.005+.0025) for k in range(10)]
    for light in [10,50]:
        y=dynamic_current(test_times,test_events,PARAMS[str(light)])*1e6;tail=test_times>=.045;expected=fit.predict_periodic((test_times[tail]+.000716)*1000,PARAMS[str(light)])
        err=float(np.sqrt(np.mean((y[tail]-expected)**2)));assert err<1e-5
        checks.append({'light_percent':light,'continuous_vs_periodic_rmse_uA':err})
    outputs={};cubes={};cache=OrderedDict();event_rows=[];status_rows=[]
    for kind in ['混合响应','稳态响应']:
        folder=RESULT/kind;folder.mkdir(parents=True,exist_ok=True);(folder/'pixel_waveforms').mkdir(exist_ok=True)
        outputs[kind]=folder
        cubes[kind]=np.lib.format.open_memmap(folder/'array_current_A.npy',mode='w+',dtype=np.float32,shape=(100,100,len(ts)))
    for r in range(100):
        for c in range(100):
            light=50 if c<50 else 10;events=events_from_mask(pixel_mask(r,c));key=(light,tuple(events))
            if key in cache:
                hybrid,steady=cache.pop(key);cache[key]=(hybrid,steady)
            else:
                event_times=[(float(ts[a]),float(ts[b]) if b<len(ts) else float(ts[-1]+dt)) for a,b in events]
                hybrid=dynamic_current(ts,event_times,PARAMS[str(light)])
                steady=old_steady_response(events,light)
                cache[key]=(hybrid,steady)
                if len(cache)>64:cache.popitem(last=False)
            assert np.isfinite(hybrid).all() and np.isfinite(steady).all()
            for kind,series in [('混合响应',hybrid),('稳态响应',steady)]:
                cubes[kind][r,c]=series
                save_csv(outputs[kind]/'pixel_waveforms'/f'pixel_{r}_{c}.csv',series,(kind,key))
            status_rows.append({'row':r,'col':c,'reference_light_percent':light,'event_count':len(events),'status':'active' if events else 'no_optical_event'})
            for number,(a,b) in enumerate(events):
                event_rows.append({'row':r,'col':c,'event_id':number,'reference_light_percent':light,'input_on_index':a,'input_off_index_exclusive':b,'input_on_s':ts[a],'input_off_s':ts[b] if b<len(ts) else ts[-1]+dt})
        if (r+1)%10==0:print('Generated rows',r+1,'/100; CSVs',(r+1)*200,flush=True)
    for cube in cubes.values():cube.flush()
    pd.DataFrame(event_rows).to_csv(RESULT/'illumination_events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(status_rows).to_csv(RESULT/'pixel_status.csv',index=False,encoding='utf-8-sig')
    np.save(RESULT/'time_s.npy',ts);pd.DataFrame({'Time_s':ts}).to_csv(RESULT/'time_axis.csv',index=False,float_format='%.12g')
    pd.DataFrame({'frame':np.arange(len(frame_idx)),'sample_index':frame_idx,'Time_s':frame_times}).to_csv(RESULT/'frame_times.csv',index=False)
    # Regenerate the same plain current frames. Arrow/flow frames belong to later inference.
    cmap=matplotlib.colormaps['Blues'];aspect=5.4/3.25;frame_size=(int(100*aspect)*2,200)
    def current_image(current):
        rgb=(cmap(np.clip(np.flipud(np.abs(current))*1e6/135.,0,1))[:,:,:3]*255).astype('uint8')
        return Image.fromarray(rgb).resize(frame_size,Image.Resampling.NEAREST)
    for kind,folder in outputs.items():
        frames=folder/'frames';frames.mkdir(exist_ok=True);images=[]
        writer=imageio_ffmpeg.write_frames(str(folder/'array_current.mp4'),frame_size,fps=15,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2);writer.send(None)
        for fi,index in enumerate(frame_idx):
            image=current_image(cubes[kind][:,:,index]);image.save(frames/f'frame_{fi:04d}.png');writer.send(np.asarray(image));images.append(image)
        writer.close();images[0].save(folder/'array_current.gif',save_all=True,append_images=images[1:],duration=67,loop=0)
        fig,axs=plt.subplots(1,5,figsize=(15,3.3))
        for ax,fi in zip(axs,[8,18,32,52,68]):ax.imshow(images[fi]);ax.set_title(f'{frame_times[fi]*1000:.1f} ms');ax.axis('off')
        fig.suptitle(f'{kind}｜电流幅值图，色标0–135 µA｜连续动态模型' if kind=='混合响应' else f'{kind}｜沿用原稳态对照模型');fig.tight_layout();fig.savefig(folder/'five_frame_current.png',dpi=200);plt.close(fig)
        print('Saved frames/video',kind,flush=True)
    # Optical scene uses the same spatial brightness boundary as the electrical simulation.
    optical=RESULT/'光照场景';optical.mkdir(exist_ok=True);(optical/'frames').mkdir(exist_ok=True)
    optical_frames=[];optical_size=(660,400)
    ow=imageio_ffmpeg.write_frames(str(optical/'light_motion.mp4'),optical_size,fps=15,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2);ow.send(None)
    for fi,index in enumerate(frame_idx):
        illuminated=np.zeros((100,100),float)
        for r in range(100):
            if 0<=v_index[r]<height:illuminated[r]=valid_u[:,index]*(alpha[v_index[r],np.clip(u_index[:,index],0,width-1)]>=1)
        illuminated[:,50:]*=.2
        rgb=(cmap(np.flipud(illuminated)*.8)[:,:,:3]*255).astype('uint8')
        image=Image.fromarray(rgb).resize(optical_size,Image.Resampling.NEAREST);image.save(optical/'frames'/f'frame_{fi:04d}.png');optical_frames.append(image);ow.send(np.asarray(image))
    ow.close();optical_frames[0].save(optical/'light_motion.gif',save_all=True,append_images=optical_frames[1:],duration=67,loop=0)
    # Representative complete waveforms and departure-tail check.
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for row,c in enumerate([30,70]):
        ev=events_from_mask(pixel_mask(50,c));new=cubes['混合响应'][50,c].astype(float);old=pd.read_csv(OLD/f'result/亮度变化/混合响应/pixel_waveforms/pixel_50_{c}.csv').Current_A.to_numpy()
        axs[row,0].plot(ts*1e3,old*1e6,color='grey',label='原拼接模型');axs[row,0].plot(ts*1e3,new*1e6,color='#1664b5',label='新连续动态模型');axs[row,0].set(title=f'像素(50,{c})｜{50 if c<50 else 10}% 光强',xlabel='时间(ms)',ylabel='电流(µA)');axs[row,0].legend()
        off=ts[ev[-1][1]];sel=(ts>=off-.001)&(ts<=off+.004);axs[row,1].plot((ts[sel]-off)*1e3,old[sel]*1e6,color='grey',label='原拼接模型');axs[row,1].plot((ts[sel]-off)*1e3,new[sel]*1e6,color='#1664b5',label='新连续动态模型');axs[row,1].set(title='离开后的恢复尾部',xlabel='相对光照离开的时间(ms)',ylabel='电流(µA)');axs[row,1].legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'新旧模拟波形与恢复尾部对比.png',dpi=200);plt.close(fig)
    # Read exported CSVs back and compare with in-memory high-precision output.
    for r,c in [(0,0),(30,30),(50,30),(50,70),(70,85),(99,99)]:
        for kind in outputs:
            f=outputs[kind]/'pixel_waveforms'/f'pixel_{r}_{c}.csv';d=pd.read_csv(f)
            assert len(d)==len(ts) and list(d.columns)==['Time_s','Current_A']
            assert np.max(np.abs(d.Time_s.to_numpy()-ts))<1e-12
            assert np.max(np.abs(d.Current_A.to_numpy()-cubes[kind][r,c]))<2e-11
    for kind in outputs:
        assert len(list((outputs[kind]/'pixel_waveforms').glob('pixel_*.csv')))==10000
        assert len(list((outputs[kind]/'frames').glob('*.png')))==79
    manifest={'configuration':CONFIG,'hybrid_parameters':PARAMS,'model_parameter_sha256':hashlib.sha256((FIT/'fitted_parameters.json').read_bytes()).hexdigest(),'car_image_sha256':hashlib.sha256((OLD/'car_bin.png').read_bytes()).hexdigest(),'scene_and_integrator_checks':checks,'total_pixels_per_response':10000,'samples_per_pixel':len(ts),'input_events':len(event_rows),'active_pixels':sum(s['event_count']>0 for s in status_rows),'steady_control':'regenerated using original steady reference and original piecewise algorithm; not refit with hybrid parameters','arrow_frames':'not produced; previous frame_pngs and matched_array.png are template/inference outputs, not raw array simulation','cache_format':'array_current_A.npy float32, axes row,col,time; CSV current float64 rounded to 12 significant digits','source_generator':str(OLD/'阵列模拟-带mask和logo-18.py')}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(__file__,CODE/Path(__file__).name);shutil.copy2(FIT/'fitted_parameters.json',CODE/'fitted_parameters.json');shutil.copy2(FIT/'fit_array_response_dynamics.py',CODE/'fit_array_response_dynamics.py');shutil.copy2(OLD/'car_bin.png',CODE/'car_bin.png')
    print('DONE',ROOT,flush=True)
