from pathlib import Path
import sys,json,time,importlib.util,hashlib,shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from lightweight_motion import ROOT,BASE,OUT,LightModel,features,LABELS,tensor_inputs,evaluate

class Baseline(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,raw,length):
        seq,aux,_=features(raw,length)
        return self.model(seq.transpose(1,2),aux,length.cpu())[:2]

def loadmodel(kind,device):
    if kind=='baseline_gru':
        spec=importlib.util.spec_from_file_location('old_train',BASE/'code/train_partial_200hz.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        model=m.Model();model.load_state_dict(torch.load(BASE/'best_model.pth',map_location='cpu',weights_only=True));model=Baseline(model)
    else:
        model=LightModel(kind);model.load_state_dict(torch.load(OUT/'01_模型训练'/kind/'best_model.pth',map_location='cpu',weights_only=True))
    return model.to(device).eval()

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
    selected=json.loads((OUT/'selected_model.json').read_text())['kind']
    source=BASE/'汽车亮度变化_识别结果'
    z=np.load(source/'actual_variable_length_inputs_and_probabilities.npz');raw=torch.from_numpy(z['raw_current_A']);length=torch.tensor(z['valid_lengths'],dtype=torch.long)
    old=pd.read_csv(source/'edge_predictions.csv');device='cuda' if torch.cuda.is_available() else 'cpu'
    summaries=[];checks=[]
    for kind in ['mlp','cnn']:
        model=loadmodel(kind,device);probs=[[],[]]
        with torch.inference_mode():
            for first in range(0,len(raw),256):
                outputs=model(raw[first:first+256].to(device),length[first:first+256].to(device))
                for j,o in enumerate(outputs):probs[j].append(o.softmax(-1).cpu().numpy())
        probs=[np.concatenate(x) for x in probs]
        df=old[[c for c in old.columns if not c.startswith(('pred_','confidence_','p_')) and c!='direction_correct']].copy()
        for j,branch in enumerate(['intensity_percent','angle_deg']):
            df['pred_'+branch]=np.array(LABELS[j])[probs[j].argmax(1)];df['confidence_'+branch]=probs[j].max(1)
            for k,value in enumerate(LABELS[j]):df[f'p_{branch}_{value}']=probs[j][:,k]
        df['direction_correct']=df.pred_angle_deg.eq(0);df['intensity_correct']=df.pred_intensity_percent.eq(df.reference_light_percent)
        dest=OUT/'03_汽车阵列识别'/kind;dest.mkdir(parents=True,exist_ok=True)
        df.to_csv(dest/'edge_predictions.csv',index=False,encoding='utf-8-sig')
        np.savez_compressed(dest/'probabilities.npz',valid_lengths=length.numpy(),intensity_probability=probs[0],angle_probability=probs[1])
        for region,sub in [('all',df),('left_50percent',df[df.col<50]),('right_10percent',df[df.col>=50])]:
            for edge,g in [('both',sub),('rise',sub[sub.edge.eq('rise')]),('fall',sub[sub.edge.eq('fall')])]:
                summaries.append(dict(kind=kind,region=region,edge=edge,n=len(g),direction_correct=int(g.direction_correct.sum()),direction_accuracy=float(g.direction_correct.mean()),intensity_accuracy=float(g.intensity_correct.mean())))
        # Explicit check: unobserved storage values must have no effect.
        r=raw[:256].clone();l=length[:256];r2=r.clone();mask=torch.arange(250)[None,:]>=l[:,None];r2[mask]=12345
        with torch.inference_mode():
            a=model(r.to(device),l.to(device));b=model(r2.to(device),l.to(device))
        diff=max((u-v).abs().max().item() for u,v in zip(a,b));assert diff==0
        checks.append(dict(kind=kind,arbitrary_padding_max_logit_difference=diff,parameter_count=sum(p.numel() for p in model.parameters())))
        print('ARRAY',kind,[s for s in summaries if s['kind']==kind and s['region']=='all'],flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'03_汽车阵列识别/summary.csv',index=False)
    timings=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        for kind in ['baseline_gru','mlp','cnn']:
            model=loadmodel(kind,device)
            with torch.inference_mode():
                for start in [0,256]:model(raw[start:start+256].to(device),length[start:start+256].to(device))
                if device=='cuda':torch.cuda.synchronize()
                for rep in range(3):
                    t=time.perf_counter()
                    for first in range(0,len(raw),256):
                        outputs=model(raw[first:first+256].to(device),length[first:first+256].to(device))
                        result=[o.softmax(-1).cpu().numpy() for o in outputs]
                    if device=='cuda':torch.cuda.synchronize()
                    elapsed=time.perf_counter()-t
                    timings.append(dict(kind=kind,device=device,repeat=rep+1,seconds=elapsed,n=len(raw)))
                    print('BENCH',kind,device,rep+1,elapsed,flush=True)
    dest=OUT/'04_耗时与对比';dest.mkdir(exist_ok=True)
    pd.DataFrame(timings).to_csv(dest/'timing_repeats.csv',index=False)
    summary=pd.DataFrame(timings).groupby(['kind','device'],as_index=False).agg(median_seconds=('seconds','median'),min_seconds=('seconds','min'),max_seconds=('seconds','max'))
    summary.to_csv(dest/'timing_summary.csv',index=False)
    # For a speed-focused task, equally accurate validation models are ranked
    # by measured GPU latency. Neither test nor array accuracy is used here.
    trainraw,trainlength,trainy,meta=tensor_inputs()
    valids=torch.tensor(np.flatnonzero(meta.split.eq('val')));validation=[]
    for kind in ['mlp','cnn']:
        model=loadmodel(kind,'cuda')
        for frac in [1,.65,.85]:
            metrics,_=evaluate(model,trainraw,trainlength,trainy,valids,frac,'cuda')
            validation.append(dict(kind=kind,retained_fraction=frac,**metrics))
    valdf=pd.DataFrame(validation);valdf.to_csv(dest/'validation_selection_metrics.csv',index=False)
    quality=valdf.groupby('kind')[['direction_accuracy','intensity_accuracy']].min().min(axis=1)
    eligible=quality[quality==quality.max()].index.tolist()
    selected=summary[(summary.device=='cuda')&summary.kind.isin(eligible)].sort_values('median_seconds').iloc[0]['kind']
    (OUT/'selected_model.json').write_text(json.dumps(dict(kind=selected,initial_loss_preference=json.loads((OUT/'selected_model.json').read_text())['kind'],selection='highest minimum validation direction/intensity accuracy across full/65/85 prefixes, then lowest measured GPU latency; checkpoint selection uses validation only; test and array not used',validation_minimum_accuracies=quality.to_dict(),labels=LABELS,frequency_hz=200,simulation_used_in_training_or_selection=False),indent=2),encoding='utf-8')
    info=dict(selected_kind=selected,input_segments=len(raw),input_valid_samples=int(length.sum()),array_used_in_training_or_selection=False,batch_size=256,cpu_threads=4,gpu=torch.cuda.get_device_name(0),scope='preloaded signed raw currents and effective lengths; includes vectorized normalization/features, host-to-device transfer, inference, softmax and host probability return; excludes disk IO, event segmentation/alignment, plotting, training; baseline remeasured with identical vectorized features',padding_checks=checks,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'experimental_inputs.npz',BASE/'experimental_split.csv',source/'actual_variable_length_inputs_and_probabilities.npz',BASE/'best_model.pth']})
    (dest/'verification_and_scope.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(__file__,OUT/'code/evaluate_lightweight_motion.py')
    print('COMPLETE SELECTED',selected,summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
