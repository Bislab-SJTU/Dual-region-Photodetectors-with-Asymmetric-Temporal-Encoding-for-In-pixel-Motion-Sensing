from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:\石阳\博士工作\科研\周报\week update-140-shiyang')
EXP=ROOT/'Farneback_稳态响应_亮度变化'
NN=ROOT/'局部波形_变长序列神经网络_200Hz/汽车亮度变化_识别结果'
WIDTH,HEIGHT,DPI=3600,2168,600
def figax():
    fig=plt.figure(figsize=(WIDTH/DPI,HEIGHT/DPI),dpi=DPI)
    ax=fig.add_axes([0,0,1,1]);ax.axis('off');return fig,ax
def save(fig,p):
    fig.savefig(p,dpi=DPI,pad_inches=0);plt.close(fig)
    with Image.open(p) as im:assert im.size==(WIDTH,HEIGHT)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
flowpath=EXP/'03_实验结果/稳态响应/数值结果/dense_flow_image_pixels.npy'
predpath=NN/'edge_predictions.csv'
before={str(p):sha(p) for p in [flowpath,predpath]}
inputs=pd.read_csv(EXP/'02_输入与参数/input_frames_sha256.csv')
flow=np.load(flowpath,mmap_mode='r')
dest=EXP/'03_实验结果/稳态响应/帧图_600dpi'
for i in range(1,79):
    rgb=np.array(Image.open(inputs.iloc[i].path).convert('RGB'));h,w=rgb.shape[:2]
    yy,xx=np.mgrid[0:h:8,0:w:8];xy=np.column_stack([xx.ravel(),yy.ravel()]);uv=flow[i-1,yy,xx].reshape(-1,2)
    mag=np.linalg.norm(uv,axis=1);keep=mag>=.5;xy=xy[keep];uv=uv[keep];mag=mag[keep]
    correct=(uv[:,0]>0)&(np.abs(np.degrees(np.arctan2(uv[:,1],uv[:,0])))<=5)
    fig,ax=figax();ax.imshow(rgb,origin='upper',interpolation='nearest',aspect='auto')
    if len(xy):
        unit=uv/mag[:,None];tip=xy+uv;head=np.minimum(mag*.3,2.5);base=tip-unit*head[:,None]
        normal=np.column_stack([-unit[:,1],unit[:,0]])*head[:,None]*.45
        colors=np.where(correct[:,None],np.array([51,204,51])/255,np.array([237,125,49])/255)
        ax.add_collection(LineCollection(np.stack([xy,tip],axis=1),colors=colors,linewidths=.65))
        ax.add_collection(LineCollection(np.concatenate([np.stack([tip,base+normal],axis=1),np.stack([tip,base-normal],axis=1)]),colors=np.concatenate([colors,colors]),linewidths=.65))
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
    save(fig,dest/f'flow_{i:04d}.png')
print('Farneback clean frames: 78',flush=True)
source=ROOT/'simulation_连续动态模型/result/亮度变化'
cube=np.load(source/'混合响应/array_current_A.npy',mmap_mode='r')
df=pd.read_csv(predpath);times=pd.read_csv(source/'frame_times.csv')
available=df.window_end_s.to_numpy();angles=np.deg2rad(df.pred_angle_deg.to_numpy());rr=df.row.to_numpy();cc=df.col.to_numpy();correct=df.direction_correct.to_numpy()
dest=NN/'frames_600dpi_开放箭头_RGB33CC33'
for f in times.itertuples(index=False):
    show=(available<=f.Time_s)&(f.Time_s<available+.002)
    fig,ax=figax();ax.imshow(np.abs(cube[:,:,int(f.sample_index)])*1e6,origin='lower',cmap='Blues',vmin=0,vmax=135,extent=[0,100,0,100],aspect='auto',interpolation='nearest')
    if show.any():
        direction=np.column_stack([np.cos(angles[show]),np.sin(angles[show])]);normal=np.column_stack([-direction[:,1],direction[:,0]])
        tail=np.column_stack([cc[show]+.5,rr[show]+.5]);tip=tail+1.8*direction;back=tip-.70*direction
        segments=np.stack([np.stack([tail,tip],axis=1),np.stack([tip,back+.48*normal],axis=1),np.stack([tip,back-.48*normal],axis=1)],axis=1).reshape(-1,2,2)
        colors=np.repeat(np.where(correct[show],'#33cc33','#ef7c28'),3)
        ax.add_collection(LineCollection(segments,colors=colors,linewidths=np.tile([.8,.45,.45],show.sum()),capstyle='butt',joinstyle='miter'))
    ax.set_xlim(0,100);ax.set_ylim(0,100);save(fig,dest/f'frame_{int(f.frame):04d}.png')
print('NN clean frames: 79',flush=True)
assert all(sha(Path(p))==v for p,v in before.items())
record={'size':[WIDTH,HEIGHT],'dpi':DPI,'no_titles_axes_margins':True,'farneback_frames':78,'nn_frames':79,'numeric_data_unchanged':True,'sha256':before,'nn_arrow_duration_ms':2,'correct_arrow_rgb':[51,204,51]}
(EXP/'02_输入与参数/纯画面导出检查.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(NN/'600dpi_export.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
