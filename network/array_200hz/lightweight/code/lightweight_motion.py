from pathlib import Path
import sys,random,time,json,hashlib,importlib.util,shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(r'D:\石阳\博士工作\科研\周报\week update-140-shiyang')
BASE=ROOT/'局部波形_变长序列神经网络_200Hz'
OUT=ROOT/'轻量网络_200Hz_波形识别'
LABELS=[[10,50,100],[0,30,60,90,120,150,180]]
MAXLEN=250
def features(raw,length):
    l=length.to(raw.device);mask=torch.arange(MAXLEN,device=raw.device)[None,:]<l[:,None]
    # Arbitrary batch storage outside observed interval has no influence.
    a=raw.masked_fill(~mask,0);mean=a.sum(1)/l
    centered=(a-mean[:,None]).masked_fill(~mask,0)
    std=(centered.square().sum(1)/l).sqrt().clamp_min(1e-8)
    norm=centered/std[:,None]
    seq=torch.stack([norm,a/1e-4,torch.arange(MAXLEN,device=raw.device)[None,:].expand_as(a)*.01*mask],1)
    aux=torch.stack([mean/1e-4,std/1e-4,l*.01],1)
    return seq,aux,mask

class LightModel(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='mlp':
            self.body=nn.Sequential(nn.Linear(503,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        else:
            self.convs=nn.ModuleList([nn.Conv1d(3,8,7,stride=2),nn.Conv1d(8,12,5,stride=2),nn.Conv1d(12,16,3,stride=2)])
            self.fc=nn.Sequential(nn.Linear(16*8*2+3,32),nn.ReLU())
            # Fixed physical time bins, never stretch shortened observations.
            self.register_buffer('bin_mask',(torch.arange(32)[None,:]//4==torch.arange(8)[:,None]))
        self.heads=nn.ModuleList([nn.Linear(32,k) for k in [3,7]])
    def forward(self,raw,length):
        seq,aux,mask=features(raw,length)
        if self.kind=='mlp':h=self.body(torch.cat([seq[:,:2,:].flatten(1),aux],1))
        else:
            z=seq;l=length.to(raw.device)
            for conv in self.convs:
                z=F.relu(conv(F.pad(z,(conv.kernel_size[0]-1,0))))
                l=(l+1)//2
                m=torch.arange(z.shape[-1],device=z.device)[None,:]<l[:,None]
                z=z*m[:,None,:]
            valid=self.bin_mask[None,:,:]&m[:,None,:]
            count=valid.sum(-1).clamp_min(1)
            mean=torch.einsum('bct,bkt->bck',z,valid.to(z.dtype))/count[:,None,:]
            mx=z[:,:,None,:].masked_fill(~valid[:,None,:,:],-1e30).max(-1).values
            mx=mx.masked_fill(~valid.any(-1)[:,None,:],0)
            h=self.fc(torch.cat([mean.flatten(1),mx.flatten(1),aux],1))
        return [head(h) for head in self.heads]

def tensor_inputs():
    z=np.load(BASE/'experimental_inputs.npz');meta=pd.read_csv(BASE/'experimental_split.csv')
    raw=torch.from_numpy(z['raw'].astype(np.float32));length=torch.tensor(z['length'],dtype=torch.long)
    y=torch.tensor(np.column_stack([[LABELS[j].index(int(v)) for v in meta[c]] for j,c in enumerate(['light','angle'])]),dtype=torch.long)
    return raw,length,y,meta

@torch.inference_mode()
def evaluate(model,raw,length,y,ids,frac,device):
    model.eval();rows=[];loss=0
    for ix in ids.split(512):
        l=length[ix].clone()
        if frac!=1:l=(l*frac).round().long().clamp_min(32)
        out=model(raw[ix].to(device),l.to(device));truth=y[ix].to(device)
        loss+=sum(F.cross_entropy(o,truth[:,j]).item() for j,o in enumerate(out))*len(ix)
        pred=torch.stack([o.argmax(1) for o in out],1).cpu().numpy()
        for k,i in enumerate(ix.tolist()):rows.append([i,*y[i].tolist(),*pred[k].tolist()])
    df=pd.DataFrame(rows,columns=['metadata_index','true_light_index','true_angle_index','pred_light_index','pred_angle_index'])
    return dict(loss=loss/len(ids),direction_accuracy=float(df.true_angle_index.eq(df.pred_angle_index).mean()),intensity_accuracy=float(df.true_light_index.eq(df.pred_light_index).mean())),df

def train(kind,raw,length,y,meta,device):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    dest=OUT/'01_模型训练'/kind;dest.mkdir(parents=True,exist_ok=True)
    trainids=torch.tensor(np.flatnonzero(meta.split.eq('train')));valid=torch.tensor(np.flatnonzero(meta.split.eq('val')))
    model=LightModel(kind).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    best=(-1,-1);hist=[];clock=time.perf_counter();bestepoch=0;wait=0
    for epoch in range(1,151):
        model.train();tot=0
        for ix in trainids[torch.randperm(len(trainids))].split(256):
            l=length[ix].clone();crop=torch.rand(len(l))<.8;frac=.6+.4*torch.rand(len(l))
            l=torch.where(crop,(l*frac).round().long().clamp_min(32),l)
            # Same prefix augmentation policy as baseline; no simulated data.
            opt.zero_grad(set_to_none=True);out=model(raw[ix].to(device),l.to(device));truth=y[ix].to(device)
            loss=sum(F.cross_entropy(o,truth[:,j]) for j,o in enumerate(out));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();tot+=loss.item()*len(ix)
        vals=[evaluate(model,raw,length,y,valid,f,device)[0] for f in [1,.65,.85]]
        va=np.mean([v['direction_accuracy'] for v in vals]);vl=np.mean([v['loss'] for v in vals])
        score=(float(va),-float(vl))
        if score>best:best=score;bestepoch=epoch;wait=0;torch.save(model.state_dict(),dest/'best_model.pth')
        else:wait+=1
        hist.append(dict(epoch=epoch,train_loss=tot/len(trainids),val_loss=vl,val_direction=va,val_intensity=np.mean([v['intensity_accuracy'] for v in vals])))
        pd.DataFrame(hist).to_csv(dest/'training_history.csv',index=False)
        if epoch%10==0 or epoch==1:print(kind,'epoch',epoch,'val direction',va,'val intensity',hist[-1]['val_intensity'],'elapsed',round(time.perf_counter()-clock),flush=True)
        if wait>=25:break
    (dest/'training_metadata.json').write_text(json.dumps(dict(kind=kind,best_epoch=bestepoch,parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-clock,selection='validation mean direction accuracy then loss across full/65%/85%; test and simulation not used',seed=42),indent=2),encoding='utf-8')
    return best

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
    raw,length,y,meta=tensor_inputs();device='cuda' if torch.cuda.is_available() else 'cpu'
    assert meta.groupby(['condition','pair_group']).split.nunique().max()==1
    print('Splits',meta.groupby('split').size().to_dict(),device,flush=True)
    scores={kind:train(kind,raw,length,y,meta,device) for kind in ['mlp','cnn']}
    # Choose architecture on validation only before looking at array or test.
    chosen=max(scores,key=scores.get)
    (OUT/'selected_model.json').write_text(json.dumps(dict(kind=chosen,validation_scores=scores,labels=LABELS,frequency_hz=200,simulation_used_in_training_or_selection=False),indent=2),encoding='utf-8')
    tests=[];testids=torch.tensor(np.flatnonzero(meta.split.eq('test')))
    for kind in ['mlp','cnn']:
        model=LightModel(kind).to(device);model.load_state_dict(torch.load(OUT/'01_模型训练'/kind/'best_model.pth',map_location=device,weights_only=True))
        for fraction in [1,.65,.85]:
            metrics,frame=evaluate(model,raw,length,y,testids,fraction,device)
            frame=frame.merge(meta.reset_index(names='metadata_index')[['metadata_index','light','angle','condition']],on='metadata_index')
            dest=OUT/'02_留出测试'/kind;dest.mkdir(parents=True,exist_ok=True)
            frame.to_csv(dest/f'predictions_{fraction}.csv',index=False)
            tests.append(dict(kind=kind,retained_fraction=fraction,n=len(frame),**metrics))
            for column in ['light','angle']:
                grouped=frame.assign(direction_correct=frame.true_angle_index.eq(frame.pred_angle_index),intensity_correct=frame.true_light_index.eq(frame.pred_light_index)).groupby(column).agg(n=('metadata_index','size'),direction_accuracy=('direction_correct','mean'),intensity_accuracy=('intensity_correct','mean'))
                grouped.to_csv(dest/f'by_{column}_{fraction}.csv')
    pd.DataFrame(tests).to_csv(OUT/'02_留出测试/test_summary.csv',index=False)
    shutil.copy2(__file__,OUT/'code/lightweight_motion.py')
    print('SELECTED',chosen,'TEST',json.dumps(tests),flush=True)

if __name__=='__main__':main()
