from pathlib import Path
import sys,json,re,random,hashlib,shutil,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset,DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(r'D:\石阳\博士工作\科研\周报')
OUT=BASE/'week update-140-shiyang/局部波形_变长序列神经网络_200Hz'
DATA=BASE/'week update-65-shiyang/gamry/20250313/night-1st test/全部条件切分周期-auto-two stages'
LABELS=[[10,50,100],[0,30,60,90,120,150,180],[200]]
DT=1e-5
def feature(raw):
    a=np.asarray(raw,dtype=np.float32);mean=a.mean();std=a.std()
    # All statistics and samples come only from the observed interval.
    norm=(a-mean)/max(float(std),1e-8)
    seq=np.stack([norm,a/1e-4,np.arange(len(a),dtype=np.float32)*.01],axis=1)
    aux=np.array([mean/1e-4,std/1e-4,len(a)*DT*1e3],dtype=np.float32)
    return torch.from_numpy(seq),torch.from_numpy(aux)
def collate(batch):
    seq,aux,y,ids=zip(*batch);length=torch.tensor([len(x) for x in seq])
    return pad_sequence(seq,batch_first=True),torch.stack(aux),length,torch.tensor(np.array(y)),list(ids)
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.gru=nn.GRU(32,64,batch_first=True,bidirectional=True)
        self.attn=nn.Linear(128,1);self.fc=nn.Sequential(nn.Linear(131,128),nn.ReLU(),nn.Dropout(.15))
        self.heads=nn.ModuleList([nn.Linear(128,n) for n in [3,7,1]])
    def forward(self,x,aux,length):
        packed=pack_padded_sequence(self.embed(x),length.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.gru(packed);z,_=pad_packed_sequence(out,batch_first=True)
        mask=torch.arange(z.shape[1],device=z.device)[None,:]<length.to(z.device)[:,None]
        scores=self.attn(z).squeeze(-1).masked_fill(~mask,-torch.inf)
        pool=(z*scores.softmax(-1).unsqueeze(-1)).sum(1)
        h=self.fc(torch.cat([pool,aux],1));return [head(h) for head in self.heads]
class Waves(Dataset):
    def __init__(self,waves,meta,split,mode):
        self.waves=waves;self.meta=meta;self.ids=np.flatnonzero(meta.split.eq(split));self.mode=mode
    def __len__(self):return len(self.ids)
    def __getitem__(self,k):
        ix=self.ids[k];raw=self.waves[ix];n=len(raw)
        if self.mode=='random':
            # Retain the beginning/edge, vary only how much later response is observed.
            frac=random.uniform(.60,1.0) if random.random()<.8 else 1.0
            raw=raw[:max(32,round(n*frac))]
        elif isinstance(self.mode,float):raw=raw[:max(32,round(n*self.mode))]
        x,aux=feature(raw);r=self.meta.iloc[ix];y=[LABELS[j].index(int(r[c])) for j,c in enumerate(['light','angle','frequency'])]
        return x,aux,y,int(ix)
def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
    cache=OUT/'experimental_inputs.npz';mp=OUT/'experimental_split.csv'
    if cache.exists():
        z=np.load(cache);return [z['raw'][i,:n] for i,n in enumerate(z['length'])],pd.read_csv(mp)
    waves=[];rows=[];rng=np.random.default_rng(42)
    for count,folder in enumerate(sorted(DATA.iterdir())):
        m=re.fullmatch(r'peaks_auto_(\d+)%_(\d+)_(\d+)Hz',folder.name)
        if not m or int(m[3])!=200:continue
        files=sorted(folder.glob('cycle_*.csv'),key=lambda f:int(f.stem.split('_')[-1]))
        groups=sorted(set(int(f.stem.split('_')[-1])//2 for f in files));groups=[g for g in groups if g!=0]
        # Blocked split by consecutive paired cycles, with a two-cycle gap at boundaries.
        a=int(len(groups)*.7);b=int(len(groups)*.85)
        split_groups={'train':groups[:max(1,a-1)],'val':groups[a+1:max(a+2,b-1)],'test':groups[b+1:]}
        for split,gg in split_groups.items():
            cap={'train':100,'val':30,'test':30}[split]
            chosen=set(rng.choice(gg,min(cap,len(gg)),replace=False)) if gg else set()
            for f in files:
                idx=int(f.stem.split('_')[-1]);group=idx//2
                if group not in chosen:continue
                d=pd.read_csv(f,usecols=['Time_s','Current_A']);t=d.Time_s.to_numpy();raw=d.Current_A.to_numpy()
                # Find dominant edge using smoothed measured data; cut a physical 2.5ms maximum.
                smooth=np.convolve(np.pad(raw,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
                edge=int(np.argmax(np.abs(np.diff(smooth))))+1
                start=max(0,edge-72);end=min(len(raw),start+250)
                target=np.arange(t[start],t[end-1]+DT*.1,DT)
                signal=np.interp(target,t,raw).astype(np.float32)
                if len(signal)<32:continue
                waves.append(signal);rows.append(dict(file=str(f),condition=folder.name,source_samples=d.shape[0],cycle_index=idx,pair_group=group,split=split,light=int(m[1]),angle=int(m[2]),frequency=int(m[3]),edge_sample=edge,start_sample=start,samples=len(signal)))
        if (count+1)%15==0:print('Prepared conditions',count+1,'waveforms',len(waves),flush=True)
    meta=pd.DataFrame(rows);meta.to_csv(mp,index=False,encoding='utf-8-sig')
    padded=np.zeros((len(waves),250),dtype=np.float32)
    for i,w in enumerate(waves):padded[i,:len(w)]=w
    np.savez_compressed(cache,raw=padded,length=np.array([len(w) for w in waves]))
    assert meta.groupby(['condition','pair_group']).split.nunique().max()==1
    return waves,meta
def evaluate(model,loader,device):
    model.eval();correct=np.zeros(3);n=0;loss=0;records=[]
    with torch.inference_mode():
        for x,aux,length,y,ids in loader:
            x=x.to(device);aux=aux.to(device);y=y.to(device);outs=model(x,aux,length)
            loss+=sum(nn.functional.cross_entropy(z,y[:,j]).item() for j,z in enumerate(outs))*len(y)
            preds=np.stack([z.argmax(-1).cpu().numpy() for z in outs],1);truth=y.cpu().numpy()
            correct+=(preds==truth).sum(0);n+=len(y)
            for i,ix in enumerate(ids):records.append([ix,*truth[i],*preds[i]])
    return loss/n,correct/n,records
def main():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    waves,meta=prepare();device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Data',meta.groupby('split').size().to_dict(),'device',device,flush=True)
    config=dict(labels=LABELS,sampling_interval_s=DT,max_samples=250,crop_retained_fraction=[.6,1.0],seed=42,train_data=str(DATA),model='bidirectional packed GRU + masked attention',padding_policy='batch storage padding ignored by packed recurrence and masked attention',selection='validation direction accuracy over full and 65/85 percent crops, then validation loss',simulation_used_in_training=False,excluded_initial_pair_group=0,split='blocked by paired cycles within acquisition; gap at boundaries; same-acquisition rather than independent device validation',max_epochs=150,patience=25,scope="200Hz reference condition, all 7 angles and all 3 intensities; frequency classification not evaluated")
    (OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    train=DataLoader(Waves(waves,meta,'train','random'),batch_size=192,shuffle=True,collate_fn=collate)
    vals=[DataLoader(Waves(waves,meta,'val',f),batch_size=256,collate_fn=collate) for f in ['full',.65,.85]]
    model=Model().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=(-1,-1);patience=0;history=[];clock=time.time()
    for epoch in range(1,151):
        model.train();total=0;n=0
        for x,aux,length,y,ids in train:
            x=x.to(device);aux=aux.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            outs=model(x,aux,length);loss=sum(nn.functional.cross_entropy(z,y[:,j]) for j,z in enumerate(outs))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()*len(y);n+=len(y)
        ev=[evaluate(model,loader,device) for loader in vals];vl=np.mean([e[0] for e in ev]);va=np.mean([e[1][1] for e in ev])
        score=(float(va),-float(vl));history.append(dict(epoch=epoch,train_loss=total/n,val_loss=vl,val_angle=va,val_full_angle=ev[0][1][1],val_65_angle=ev[1][1][1],val_85_angle=ev[2][1][1]))
        if score>best:best=score;patience=0;torch.save(model.state_dict(),OUT/'best_model.pth');best_epoch=epoch
        else:patience+=1
        pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
        print(f'Epoch {epoch}: train={total/n:.4f} val_angle={va:.4%} val_loss={vl:.4f} best={best_epoch} elapsed={time.time()-clock:.0f}s',flush=True)
        if patience>=25:break
    model.load_state_dict(torch.load(OUT/'best_model.pth',map_location=device,weights_only=True))
    results=[]
    for fraction in ['full',.65,.85]:
        loader=DataLoader(Waves(waves,meta,'test',fraction),batch_size=256,collate_fn=collate)
        loss,acc,records=evaluate(model,loader,device);name=str(fraction)
        frame=pd.DataFrame(records,columns=['metadata_index','true_light_index','true_angle_index','true_frequency_index','pred_light_index','pred_angle_index','pred_frequency_index'])
        frame.to_csv(OUT/f'test_predictions_{name}.csv',index=False)
        results.append(dict(retained_fraction=name,n=len(frame),intensity_accuracy=float(acc[0]),direction_accuracy=float(acc[1]),frequency_accuracy=float(acc[2]),loss=loss))
        for j,branch in enumerate(['light','angle','frequency']):
            cm=pd.crosstab(frame[f'true_{branch}_index'],frame[f'pred_{branch}_index']).reindex(index=range(len(LABELS[j])),columns=range(len(LABELS[j])),fill_value=0)
            cm.index=LABELS[j];cm.columns=LABELS[j];cm.to_csv(OUT/f'test_confusion_{branch}_{name}.csv')
    pd.DataFrame(results).to_csv(OUT/'test_summary.csv',index=False)
    print('TEST',json.dumps(results),flush=True)
    fig,ax=plt.subplots(1,2,figsize=(11,4));h=pd.DataFrame(history)
    ax[0].plot(h.epoch,h.train_loss,label='train');ax[0].plot(h.epoch,h.val_loss,label='validation');ax[0].legend();ax[0].set(title='Loss',xlabel='Epoch')
    for c in ['val_full_angle','val_65_angle','val_85_angle']:ax[1].plot(h.epoch,h[c],label=c)
    ax[1].legend();ax[1].set(title='Validation direction accuracy',xlabel='Epoch');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=170);plt.close(fig)
    (OUT/'training_result.json').write_text(json.dumps(dict(best_epoch=best_epoch,test=results,model_sha256=hashlib.sha256((OUT/'best_model.pth').read_bytes()).hexdigest(),elapsed_s=time.time()-clock),indent=2),encoding='utf-8')
    script=Path(__file__).resolve()
    if script!=(OUT/'code'/script.name).resolve():shutil.copy2(script,OUT/'code'/script.name)
    print('TRAINING DONE',OUT,flush=True)
if __name__=='__main__':main()
