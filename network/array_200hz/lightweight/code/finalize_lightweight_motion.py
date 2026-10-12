from pathlib import Path
import json,sys,hashlib,shutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import imageio_ffmpeg
from lightweight_motion import ROOT,BASE,OUT,LightModel,LABELS,tensor_inputs,evaluate
from evaluate_lightweight_motion import loadmodel

sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
dest=OUT/'04_耗时与对比'
raw,length,y,meta=tensor_inputs();valids=torch.tensor(np.flatnonzero(meta.split.eq('val')));records=[]
for kind in ['mlp','cnn']:
    model=loadmodel(kind,'cuda')
    for frac in [1,.65,.85]:
        metrics,_=evaluate(model,raw,length,y,valids,frac,'cuda');records.append(dict(kind=kind,retained_fraction=frac,**metrics))
valdf=pd.DataFrame(records);valdf.to_csv(dest/'validation_selection_metrics.csv',index=False)
quality=valdf.groupby('kind')[['direction_accuracy','intensity_accuracy']].min().min(axis=1)
eligible=quality[quality==quality.max()].index.tolist()
timing=pd.read_csv(dest/'timing_summary.csv')
chosen=timing[(timing.device=='cuda')&timing.kind.isin(eligible)].sort_values('median_seconds').iloc[0]['kind']
(OUT/'selected_model.json').write_text(json.dumps(dict(kind=chosen,initial_loss_preference='cnn',selection='highest minimum validation direction/intensity accuracy across full/65/85 prefixes, then lowest measured GPU latency; checkpoint selection uses validation only; test and array not used',validation_minimum_accuracies=quality.to_dict(),labels=LABELS,frequency_hz=200,simulation_used_in_training_or_selection=False),indent=2),encoding='utf-8')
info=json.loads((dest/'verification_and_scope.json').read_text(encoding='utf-8'));info['selected_kind']=chosen
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in info['sources'].items())
(dest/'verification_and_scope.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
# Compute dense layer/conv MACs for the actual fixed-time storage shape.
stats=[]
for kind in ['mlp','cnn']:
    model=loadmodel(kind,'cpu');counts=[];handles=[]
    def hook(m,x,out):
        if isinstance(m,torch.nn.Linear):counts.append(out.numel()*m.in_features)
        elif isinstance(m,torch.nn.Conv1d):counts.append(out.numel()*(m.in_channels//m.groups)*m.kernel_size[0])
    for m in model.modules():
        if isinstance(m,(torch.nn.Linear,torch.nn.Conv1d)):handles.append(m.register_forward_hook(hook))
    with torch.inference_mode():model(torch.zeros(1,250),torch.tensor([250]))
    for h in handles:h.remove()
    stats.append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),linear_conv_MAC_per_waveform=sum(counts),note='excludes normalization, mask/pooling, activation and softmax; fixed 250-time-position storage'))
pd.DataFrame(stats).to_csv(dest/'model_size_and_MAC.csv',index=False)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,device in zip(axs,['cpu','cuda']):
    g=timing[timing.device==device].set_index('kind').loc[['baseline_gru','mlp','cnn']]
    ax.bar(['原GRU','轻量MLP','轻量CNN'],g.median_seconds,color=['#8293aa','#33cc33','#6ca6d9'])
    ax.set_yscale('log');ax.set_ylabel('处理32400条波形的时间（秒，对数坐标）');ax.set_title('CPU 4线程' if device=='cpu' else 'RTX 4060 Ti')
    for i,v in enumerate(g.median_seconds):ax.text(i,v*1.12,f'{v:.3f}s',ha='center')
fig.tight_layout();fig.savefig(dest/'耗时对比.png',dpi=220);fig.savefig(dest/'耗时对比.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for kind in ['mlp','cnn']:
    h=pd.read_csv(OUT/'01_模型训练'/kind/'training_history.csv');axs[0].plot(h.epoch,h.val_loss,label=kind);axs[1].plot(h.epoch,h.val_direction,label=kind)
axs[0].set(title='验证损失',xlabel='Epoch',yscale='log');axs[1].set(title='验证方向准确率',xlabel='Epoch')
for a in axs:a.legend()
fig.tight_layout();fig.savefig(OUT/'01_模型训练/训练曲线.png',dpi=220);plt.close(fig)
# Render only the final selected model, with the previously requested clean frame style.
results=OUT/'03_汽车阵列识别'/chosen
frames=results/'frames_600dpi_纯画面';frames.mkdir(exist_ok=True)
source=ROOT/'simulation_连续动态模型/result/亮度变化'
cube=np.load(source/'混合响应/array_current_A.npy',mmap_mode='r');times=pd.read_csv(source/'frame_times.csv')
df=pd.read_csv(results/'edge_predictions.csv');available=df.window_end_s.to_numpy();angles=np.deg2rad(df.pred_angle_deg.to_numpy());rr=df.row.to_numpy();cc=df.col.to_numpy();correct=df.direction_correct.to_numpy()
small=[];five=[]
writer=imageio_ffmpeg.write_frames(str(results/'network_predictions_纯画面.mp4'),(1200,722),fps=15,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2);writer.send(None)
for f in times.itertuples(index=False):
    show=(available<=f.Time_s)&(f.Time_s<available+.002)
    fig=plt.figure(figsize=(6,2168/600),dpi=600);ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    ax.imshow(np.abs(cube[:,:,int(f.sample_index)])*1e6,origin='lower',cmap='Blues',vmin=0,vmax=135,extent=[0,100,0,100],aspect='auto',interpolation='nearest')
    if show.any():
        direction=np.column_stack([np.cos(angles[show]),np.sin(angles[show])]);normal=np.column_stack([-direction[:,1],direction[:,0]])
        tail=np.column_stack([cc[show]+.5,rr[show]+.5]);tip=tail+1.8*direction;back=tip-.70*direction
        seg=np.stack([np.stack([tail,tip],1),np.stack([tip,back+.48*normal],1),np.stack([tip,back-.48*normal],1)],1).reshape(-1,2,2)
        colors=np.repeat(np.where(correct[show],'#33cc33','#ef7c28'),3)
        ax.add_collection(LineCollection(seg,colors=colors,linewidths=np.tile([.8,.45,.45],show.sum()),capstyle='butt',joinstyle='miter'))
    ax.set_xlim(0,100);ax.set_ylim(0,100)
    p=frames/f'frame_{int(f.frame):04d}.png';fig.savefig(p,dpi=600,pad_inches=0);plt.close(fig)
    with Image.open(p) as im:
        assert im.size==(3600,2168) and abs(im.info['dpi'][0]-600)<.1
        writer.send(np.array(im.convert('RGB').resize((1200,722),Image.Resampling.NEAREST)))
        small.append(im.convert('RGB').resize((900,542),Image.Resampling.NEAREST))
        if int(f.frame) in [8,18,32,52,68]:five.append(im.convert('RGB').resize((1200,722),Image.Resampling.NEAREST))
    if int(f.frame)%20==0:print('clean frames',int(f.frame),'/79',flush=True)
writer.close()
small[0].save(results/'network_predictions_纯画面.gif',save_all=True,append_images=small[1:],duration=67,loop=0)
montage=Image.new('RGB',(6000,722))
for i,im in enumerate(five):montage.paste(im,(i*1200,0))
montage.save(results/'five_frames_纯画面.png',dpi=(600,600))
(results/'显示设置.json').write_text(json.dumps(dict(frames=79,width=3600,height=2168,dpi=600,arrow_duration_ms=2,arrow_length_array_pixels=1.8,arrow_rgb=[51,204,51],open_arrowheads=True,no_axes_titles_or_margins=True),indent=2),encoding='utf-8')
for name in ['lightweight_motion.py','evaluate_lightweight_motion.py','finalize_lightweight_motion.py']:
    p=Path(__file__).parent/name
    if p.resolve()!=(OUT/'code'/name).resolve():shutil.copy2(p,OUT/'code'/name)
print('FINAL SELECTED',chosen,'PARAMETERS',stats,flush=True)
