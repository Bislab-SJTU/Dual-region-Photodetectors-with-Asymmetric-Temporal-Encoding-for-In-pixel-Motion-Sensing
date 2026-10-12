from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import pandas as pd
import torch
sys.stdout.reconfigure(encoding='utf-8')
from train_partial_200hz import Model,feature,collate,LABELS
BASE=Path(r'D:\石阳\博士工作\科研\周报');ROOT=BASE/'week update-140-shiyang'
TRAIN=ROOT/'局部波形_变长序列神经网络_200Hz'
SOURCE=ROOT/'simulation_连续动态模型/result/亮度变化'
OUT=TRAIN/'汽车亮度变化_识别结果';OUT.mkdir(exist_ok=True)
code=TRAIN/'code';code.mkdir(exist_ok=True)
cube=np.load(SOURCE/'混合响应/array_current_A.npy',mmap_mode='r');ts=np.load(SOURCE/'time_s.npy');dt=np.median(np.diff(ts))
df=pd.read_csv(TRAIN/'event_segmentation_metadata.csv')
keep=[c for c in df.columns if not c.startswith(('pred_','confidence_','p_')) and c!='direction_correct'];df=df[keep].copy()
assert df.other_edges_in_window.eq(0).all()
torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
model=Model().to(device);model.load_state_dict(torch.load(TRAIN/'best_model.pth',map_location=device,weights_only=True));model.eval()
probs=[[],[],[]];raw_store=np.zeros((len(df),250),dtype=np.float32);lengths=[]
with torch.inference_mode():
    for first in range(0,len(df),256):
        batch=[]
        for ix,p in df.iloc[first:first+256].iterrows():
            start=round(p.window_start_s/dt);end=round(p.window_end_s/dt)
            raw=np.asarray(cube[int(p.row),int(p.col),start:end],dtype=np.float32)
            # Apply the same measured dominant-edge alignment used during training.
            smooth=np.convolve(np.pad(raw,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
            measured_edge=int(np.argmax(np.abs(np.diff(smooth))))+1
            offset=max(0,measured_edge-72)
            raw=raw[offset:]
            df.loc[ix,'pre_alignment_window_start_s']=p.window_start_s
            df.loc[ix,'window_start_s']=(start+offset)*dt
            df.loc[ix,'measured_edge_offset_samples']=measured_edge
            df.loc[ix,'alignment_trim_left_samples']=offset
            df.loc[ix,'model_input_samples']=len(raw)
            seq,aux=feature(raw);batch.append((seq,aux,[0,0,0],int(ix)))
            lengths.append(len(raw));raw_store[ix,:len(raw)]=raw
        x,aux,length,y,ids=collate(batch)
        for j,z in enumerate(model(x.to(device),aux.to(device),length)):probs[j].append(z.softmax(-1).cpu().numpy())
        if first%4096==0:print('Inferred',first+len(batch),'/',len(df),flush=True)
probs=[np.concatenate(p) for p in probs]
for j,branch in enumerate(['intensity_percent','angle_deg','frequency_hz']):
    df['pred_'+branch]=np.array(LABELS[j])[probs[j].argmax(1)];df['confidence_'+branch]=probs[j].max(1)
    for k,value in enumerate(LABELS[j]):df[f'p_{branch}_{value}']=probs[j][:,k]
df['direction_correct']=df.pred_angle_deg.eq(0)
df.to_csv(OUT/'edge_predictions.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(OUT/'actual_variable_length_inputs_and_probabilities.npz',raw_current_A=raw_store,valid_lengths=np.array(lengths),intensity_prob=probs[0],angle_prob=probs[1],frequency_prob=probs[2])
summary=[]
for region,sub in [('all',df),('left_50percent',df[df.col<50]),('right_10percent',df[df.col>=50])]:
    for edge in ['both','rise','fall']:
        g=sub if edge=='both' else sub[sub.edge.eq(edge)]
        summary.append(dict(region=region,edge=edge,n=len(g),correct=int(g.direction_correct.sum()),direction_consistency=float(g.direction_correct.mean()),intensity_consistency=float(g.pred_intensity_percent.eq(g.reference_light_percent).mean())))
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
shutil.copy2(SOURCE/'pixel_status.csv',OUT/'pixel_status.csv')
frame_info=pd.read_csv(SOURCE/'frame_times.csv')
manifest=dict(source=str(SOURCE),trained_model=str(TRAIN/'best_model.pth'),model_sha256=hashlib.sha256((TRAIN/'best_model.pth').read_bytes()).hexdigest(),summary=summary,
    arrow_hold_s=.002,arrow_available_at="input window end",no_time_stretching=True,no_endpoint_filling=True,no_direction_override=True,no_confidence_filter=True,actual_sequence_lengths=sorted(set(lengths)),
    batch_padding_ignored='packed GRU recurrence and masked attention; auxiliary statistics only use actual samples',input_sampling_interval_s=1e-5,
    segmentation='previous adaptive event boundaries, followed by identical measured dominant-edge alignment as experimental training; no other geometric edge in valid interval',limitations=['Same acquisition experimental split, not independent device/session validation.','Array generated from references in same acquisition; no simulated data used to train/select checkpoint.','Array ground truth direction only 0 deg.','Partial observations may not uniquely determine direction/frequency.'])
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('ARRAY SUMMARY',json.dumps(summary),flush=True)
# Verify predictions are invariant to arbitrary batch storage padding.
with torch.inference_mode():
    p=df[df.window_shortened].iloc[0];start=round(p.window_start_s/dt);end=round(p.window_end_s/dt)
    seq,aux=feature(np.array(cube[int(p.row),int(p.col),start:end]));n=len(seq)
    x=seq.unsqueeze(0).to(device);ax=aux.unsqueeze(0).to(device);length=torch.tensor([n])
    original=model(x,ax,length)
    extended=torch.cat([x,torch.full((1,77,3),12345.,device=device)],dim=1)
    extra=model(extended,ax,length)
    maxdiff=max((a-b).abs().max().item() for a,b in zip(original,extra))
    assert maxdiff<1e-6
(OUT/'padding_invariance_verification.json').write_text(json.dumps({'extra_ignored_points':77,'extra_point_value':12345,'max_logit_difference':maxdiff,'no_other_edges':bool(df.other_edges_in_window.eq(0).all()),'segments':len(df)},indent=2),encoding='utf-8')
shutil.copy2(__file__,code/Path(__file__).name)
print('INFERENCE DONE',OUT,flush=True)
