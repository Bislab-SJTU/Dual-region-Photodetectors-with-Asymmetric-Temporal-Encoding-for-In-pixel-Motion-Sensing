from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio_ffmpeg

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:\石阳\博士工作\科研\周报\week update-140-shiyang')
SOURCE=ROOT.parent/'week update-132-shiyang/simulation/result/亮度变化'
OUT=ROOT/'Farneback_稳态响应_亮度变化'
META=OUT/'02_输入与参数';META.mkdir(exist_ok=True)
PARAMS=dict(pyr_scale=.2,levels=3,winsize=30,iterations=3,poly_n=7,poly_sigma=1.5,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
STEP=8;THRESHOLD=.5;ANGLE=5
# Reproduce the historical steady-response experiment on its original frames.
# Original renderer advances 200 samples per frame. Use recorded sample times.
source_time=pd.read_csv(SOURCE/'稳态响应/pixel_waveforms/pixel_40_40.csv',usecols=['Time_s']).Time_s.to_numpy()
indices=np.arange(0,len(source_time),200)
times=pd.DataFrame({'frame':np.arange(len(indices)),'sample_index':indices,'Time_s':source_time[indices]})
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def draw(ax,rgb,xy,uv,correct):
    h,w=rgb.shape[:2];ax.imshow(rgb,origin='upper',interpolation='nearest')
    # Open arrowheads; display only. Numeric flow values are never changed.
    if len(xy):
        length=np.linalg.norm(uv,axis=1);unit=uv/length[:,None]
        tip=xy+uv;head=np.minimum(length*.3,2.5)
        base=tip-unit*head[:,None];normal=np.stack([-unit[:,1],unit[:,0]],axis=1)*head[:,None]*.45
        colors=np.where(correct[:,None],np.array([51,204,51])/255,np.array([237,125,49])/255)
        ax.add_collection(LineCollection(np.stack([xy,tip],axis=1),colors=colors,linewidths=.65))
        ax.add_collection(LineCollection(np.concatenate([np.stack([tip,base+normal],axis=1),np.stack([tip,base-normal],axis=1)]),colors=np.concatenate([colors,colors]),linewidths=.65))
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_aspect('equal');ax.axis('off')

inputs=[];summaries=[];verification=[]
for branch in ['稳态响应']:
    dest=OUT/'03_实验结果'/branch
    for d in ['帧图_原尺寸','帧图_600dpi','数值结果','汇总图与动画']:(dest/d).mkdir(parents=True,exist_ok=True)
    files=sorted((SOURCE/branch/'frames').glob('frame_*.png'))
    assert len(files)==len(times)==79
    images=[]
    for i,p in enumerate(files):
        rgb=np.array(Image.open(p).convert('RGB'));images.append(rgb)
        index=int(times.iloc[i].sample_index)
        inputs.append(dict(branch=branch,frame=i,path=str(p),sha256=sha(p),sample_index=index,time_s=float(times.iloc[i].Time_s)))
    verification.append(dict(branch=branch,source_frames=79,dimensions=list(images[0].shape),source_role='original week132 steady-response frames; no regenerated or hybrid array',timing_note='200 samples/frame from historical renderer; exact times from pixel_40_40.csv; flow units pixels/frame'))
    rows=[];vectors=[];flows=[];selected={};native=[]
    prev=cv2.cvtColor(images[0],cv2.COLOR_RGB2GRAY)
    h,w=images[0].shape[:2];yy,xx=np.mgrid[0:h:STEP,0:w:STEP];all_xy=np.column_stack([xx.ravel(),yy.ravel()])
    for i,rgb in enumerate(images[1:],1):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        flow=cv2.calcOpticalFlowFarneback(prev,gray,None,**PARAMS);prev=gray
        assert np.isfinite(flow).all();flows.append(flow)
        uv=flow[yy,xx].reshape(-1,2);mag=np.linalg.norm(uv,axis=1);keep=mag>=THRESHOLD
        xy=all_xy[keep];uv=uv[keep];mag=mag[keep]
        angle=np.degrees(np.arctan2(uv[:,1],uv[:,0]));correct=(uv[:,0]>0)&(np.abs(angle)<=ANGLE)
        dt=float(times.iloc[i].Time_s-times.iloc[i-1].Time_s)
        for pt,v,m,a,c in zip(xy,uv,mag,angle,correct):
            vectors.append(dict(frame=i,time_s=float(times.iloc[i].Time_s),x_image_px=int(pt[0]),y_image_px=int(pt[1]),dx_image_px=float(v[0]),dy_image_px=float(v[1]),dx_array_columns=float(v[0]*100/w),dy_array_rows=float(-v[1]*100/h),magnitude_image_px=float(m),angle_image_deg=float(a),within_rightward_5deg=bool(c)))
        rows.append(dict(frame=i,time_s=float(times.iloc[i].Time_s),dt_s=dt,grid_vectors=len(all_xy),vectors_above_threshold=len(xy),within_rightward_5deg=int(correct.sum()),outside_rightward_5deg=int((~correct).sum()),rightward_fraction=float(correct.mean()) if len(correct) else np.nan))
        frame=rgb.copy()
        for pt,v,c in zip(xy,uv,correct):
            cv2.arrowedLine(frame,tuple(pt.astype(int)),tuple((pt+v).astype(int)),(51,204,51) if c else (237,125,49),1,cv2.LINE_AA,tipLength=.3)
        Image.fromarray(frame).save(dest/'帧图_原尺寸'/f'flow_{i:04d}.png');native.append(Image.fromarray(frame))
        fig=plt.figure(figsize=(6,4));ax=fig.add_axes([.015,.04,.97,.87]);draw(ax,rgb,xy,uv,correct)
        fig.suptitle(f'Farnebäck | {branch} | {times.iloc[i].Time_s*1000:.1f} ms',fontsize=11)
        fig.savefig(dest/'帧图_600dpi'/f'flow_{i:04d}.png',dpi=600);plt.close(fig)
        if i in [8,18,32,52,68]:selected[i]=(rgb,xy,uv,correct)
        if i%20==0:print(branch,i,'/78',flush=True)
    np.save(dest/'数值结果/dense_flow_image_pixels.npy',np.stack(flows))
    pd.DataFrame(rows).to_csv(dest/'数值结果/frame_statistics.csv',index=False)
    pd.DataFrame(vectors).to_csv(dest/'数值结果/displayed_vectors.csv',index=False)
    total=sum(r['vectors_above_threshold'] for r in rows);ok=sum(r['within_rightward_5deg'] for r in rows)
    summaries.append(dict(branch=branch,frame_pairs=78,displayed_vectors=total,within_rightward_5deg=ok,outside_rightward_5deg=total-ok,rightward_fraction=ok/total if total else None))
    fig,axs=plt.subplots(1,5,figsize=(18,3.6))
    for ax,(i,args) in zip(axs,selected.items()):draw(ax,*args);ax.set_title(f'{times.iloc[i].Time_s*1000:.1f} ms')
    fig.suptitle(f'Farnebäck | {branch} | Green: rightward ±5°; orange: other directions',fontsize=12)
    fig.tight_layout();fig.savefig(dest/'汇总图与动画/five_frames_600dpi.png',dpi=600);fig.savefig(dest/'汇总图与动画/five_frames.pdf');plt.close(fig)
    native[0].save(dest/'汇总图与动画/optical_flow.gif',save_all=True,append_images=native[1:],duration=67,loop=0)
    writer=imageio_ffmpeg.write_frames(str(dest/'汇总图与动画/optical_flow.mp4'),(w*4,h*4),fps=15,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2);writer.send(None)
    for im in native:writer.send(np.array(im.resize((w*4,h*4),Image.Resampling.NEAREST)))
    writer.close()
    print('DONE',branch,summaries[-1],flush=True)
pd.DataFrame(inputs).to_csv(META/'input_frames_sha256.csv',index=False)
times.to_csv(META/'frame_times.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'03_实验结果/summary.csv',index=False)
(META/'parameters.json').write_text(json.dumps(dict(source=str(SOURCE),farneback=PARAMS,grid_step_image_px=STEP,min_magnitude_image_px=THRESHOLD,direction_tolerance_image_deg=ANGLE,image_mapping='original week132 steady-response PNGs, read unchanged as RGB then OpenCV grayscale; no current preprocessing during rerun',flow_units='image pixels per successive source frame; raw dy positive downward',arrow_rgb_correct=[51,204,51],arrow_rgb_other=[237,125,49],frame_dpi=600,video_fps=15,summary_interpretation='direction agreement among thresholded sampled vectors; not all-pixel accuracy, not NN event accuracy',python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),ensure_ascii=False,indent=2),encoding='utf-8')
(META/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print('ALL COMPLETE',flush=True)
