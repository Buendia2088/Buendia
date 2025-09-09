#!/usr/bin/env python3
"""
continual_minimal_proj_reg_replay.py
------------------------------------
Minimal MNIST continual‑learning demo supporting:

• CIL / TIL / DIL 场景
• 三种抗遗忘技术 (可独立组合):
    1) Replay  缓存回放        --replay / --no-replay
    2) EWC     参数正则        --regularization / --no-regularization
    3) KD      蒸馏            --distill / --no-distill
    4) **Proj  梯度投影(A‑GEM)** --proj / --no-proj

额外超参:
    --ewc-lambda   (EWC 强度, 默认 10)
    --kd-lambda    (KD 强度, 默认 1)
    --kd-temp      (KD 温度, 默认 2)
    --m-proj       (梯度投影参考批大小, 默认 128)

示例:
python continual_minimal_proj_reg_replay.py --scenario CIL --tasks 5 --epochs 1 \
       --replay --proj --m-proj 64
"""
from __future__ import annotations
import argparse, copy, random
from typing import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ----------------------------------------------------------------- Scenario
class ScenarioConfig:
    def __init__(self, name:str, shared_head:bool):
        self.name, self.shared_head = name, shared_head
SCENARIOS = {"CIL":ScenarioConfig("CIL",True),
             "TIL":ScenarioConfig("TIL",False),
             "DIL":ScenarioConfig("DIL",True)}

# ----------------------------------------------------------------- Data split
def split_mnist_by_class(k, root="./data"):
    per = 10//k
    tf = transforms.ToTensor()
    tr= datasets.MNIST(root, True , tf, download=True)
    te= datasets.MNIST(root, False, tf, download=True)
    tr_tasks, te_tasks, cmap = [], [], {}
    for t in range(k):
        cls=list(range(t*per,(t+1)*per))
        cmap[t]=cls
        tr_idx=(torch.isin(tr.targets, torch.tensor(cls))).nonzero(as_tuple=True)[0] # get a bool tensor
        te_idx=(torch.isin(te.targets, torch.tensor(cls))).nonzero(as_tuple=True)[0] # get a bool tensor
        tr_tasks.append(Subset(tr,tr_idx)) # Subset() is used to repackage the tr with index tr_idx
        te_tasks.append(Subset(te,te_idx))
    return tr_tasks, te_tasks, cmap

def split_mnist_by_domain(k=5, root="./data"):
    tfs=[transforms.ToTensor(),
         transforms.Compose([transforms.ToTensor(),
                             transforms.Lambda(lambda x:(x+0.2*torch.randn_like(x)).clamp(0,1))]),
         transforms.Compose([transforms.RandomRotation(15), transforms.ToTensor()]),
         transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x:1.-x)]),
         transforms.Compose([transforms.RandomRotation((-15,-15)), transforms.ToTensor()])]
    tr_tasks, te_tasks, cmap = [], [], {}
    for t in range(k):
        cmap[t]=list(range(10))
        tr_tasks.append(datasets.MNIST(root, True , tfs[t], download=True))
        te_tasks.append(datasets.MNIST(root, False, tfs[t], download=True))
    return tr_tasks, te_tasks, cmap

# ----------------------------------------------------------------- Wrapper
class RemapDataset(Dataset):
    def __init__(self, sub:Dataset, remap:torch.Tensor):
        self.sub, self.remap = sub, remap.long()
    def __len__(self): return len(self.sub)
    def __getitem__(self,i):
        x,y=self.sub[i]
        return x, int(self.remap[int(y)])

def make_loaders(tr_tasks, te_tasks, cmap, scn, bs=128, nw=0):
    outs=[]
    for t,(tr,te) in enumerate(zip(tr_tasks,te_tasks)):
        if scn.shared_head: remap=torch.arange(10)
        else:
            remap=torch.full((10,),-1)
            for j,c in enumerate(cmap[t]): remap[c]=j
        outs.append((DataLoader(RemapDataset(tr,remap),bs,True ,num_workers=nw,pin_memory=True),
                     DataLoader(RemapDataset(te,remap),bs,False,num_workers=nw,pin_memory=True)))
    return outs

# ----------------------------------------------------------------- Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Flatten(), nn.Linear(28*28,256), nn.ReLU())
    def forward(self,x): 
        return self.f(x)

class CLNet(nn.Module):
    def __init__(self, scn:ScenarioConfig):
        super().__init__()
        self.enc=MLP()
        self.scn=scn
        self.cls=nn.Linear(256,10) if scn.shared_head else nn.ModuleDict()
    def add_head(self,tid,out): 
        self.cls[f"t{tid}"]=nn.Linear(256,out)
    def forward(self,x,tid=None):
        h=self.enc(x)
        return self.cls(h) if self.scn.shared_head else self.cls[f"t{tid}"](h)

# ----------------------------------------------------------------- Grad utils
def flat_grad(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
def assign_flat_grad(model, vec):
    i=0
    for p in model.parameters():
        if p.grad is None: continue
        n=p.numel()
        p.grad.copy_(vec[i:i+n].view_as(p)); i+=n

# ----------------------------------------------------------------- Fisher utils (EWC)
def fisher_diag(model, loader, device, tid, max_samp=1024):
    model.eval(); torch.set_grad_enabled(True)
    Fdict={n:torch.zeros_like(p,device='cpu') for n,p in model.named_parameters() if p.requires_grad}
    seen=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); model.zero_grad(set_to_none=True)
        loss=F.nll_loss(F.log_softmax(model(x,None if model.scn.shared_head else tid),1), y)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None: Fdict[n]+=p.grad.detach().cpu()**2
        seen+=1
        if seen* x.size(0)>=max_samp: break
    for n in Fdict: Fdict[n]/=seen
    return Fdict

def ewc_penalty(model, pstar, Fstar, device):
    loss=torch.tensor(0.,device=device)
    for n,p in model.named_parameters():
        if n in pstar:
            diff=p-pstar[n].to(device)
            loss+=(Fstar[n].to(device)*diff.pow(2)).sum()
    return loss

# ----------------------------------------------------------------- Train loop
def train_inc(tasks:List[Tuple[DataLoader,DataLoader]], model:CLNet,opt,scn,
              epochs=1,
              use_replay=True,
              use_ewc=False, lam_ewc=10.,
              use_kd=False , lam_kd=1., T_kd=2.,
              use_proj=False, m_proj=128,
              device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    K=len(tasks); acc=torch.full((K,K),float('nan'))

    # replay memory
    mem_x, mem_y = [], []; MEM=2000; quota=MEM//K if K else 0
    # EWC stats
    pstar,Fstar={},{}
    # KD teacher
    prev_model=None
    kl=nn.KLDivLoss(reduction='batchmean')

    for k,(ld_tr,_) in enumerate(tasks):
        # add new head (TIL)
        if not scn.shared_head and f"t{k}" not in model.cls:
            model.add_head(k,out=len(tasks[k][0].dataset.sub.dataset.classes))
            opt.add_param_group({"params":model.cls[f"t{k}"].parameters()})

        model.train()
        for _ in range(epochs):
            for x,y in ld_tr:
                x,y=x.to(device),y.to(device)

                # replay mix
                if use_replay and mem_x:
                    sel=torch.randperm(len(mem_x))[:x.size(0)]
                    rx=torch.stack([mem_x[i] for i in sel]).to(device)
                    ry=torch.tensor([mem_y[i] for i in sel],device=device)
                    x=torch.cat([x,rx]); y=torch.cat([y,ry])

                # --------- 参考梯度 g_ref (仅投影时) ----------
                if use_proj and mem_x:
                    with torch.no_grad():
                        sel=random.sample(range(len(mem_x)), min(m_proj, len(mem_x)))
                        bx=torch.stack([mem_x[i] for i in sel]).to(device)
                        by=torch.tensor([mem_y[i] for i in sel],device=device)
                    model.zero_grad(set_to_none=True)
                    loss_ref=F.cross_entropy(model(bx,None if scn.shared_head else k), by)
                    loss_ref.backward()
                    g_ref=flat_grad(model).clone()
                    model.zero_grad(set_to_none=True)
                # ---------------------------------------------

                # forward current batch
                logits=model(x,None if scn.shared_head else k)
                loss=F.cross_entropy(logits,y)

                # KD
                if use_kd and prev_model is not None and scn.shared_head:
                    with torch.no_grad(): t_logits=prev_model(x)
                    kd=kl(F.log_softmax(logits/T_kd,1),
                          F.softmax(t_logits/T_kd,1))*(T_kd*T_kd)
                    loss+=lam_kd*kd

                # EWC
                if use_ewc and pstar:
                    loss+= (lam_ewc/2.)*ewc_penalty(model,pstar,Fstar,device)

                # backward current loss
                model.zero_grad(set_to_none=True)
                loss.backward()
                g_cur=flat_grad(model)

                # A‑GEM projection
                if use_proj and mem_x:
                    dot=torch.dot(g_cur, g_ref)
                    if dot<0:
                        proj=g_cur - (dot / g_ref.pow(2).sum())*g_ref
                        assign_flat_grad(model, proj)

                opt.step()

        # update memory
        if use_replay:
            seen=0
            for bx,by in ld_tr:
                for xx,yy in zip(bx,by):
                    if len(mem_x)<(k+1)*quota and seen<quota:
                        mem_x.append(xx.cpu()); mem_y.append(int(yy)); seen+=1
                    if seen>=quota: break
                if seen>=quota: break

        # update EWC
        if use_ewc:
            Fstar=fisher_diag(model,ld_tr,device,None if scn.shared_head else k)
            pstar={n:p.detach().cpu().clone() for n,p in model.named_parameters()}

        # update teacher for KD
        if use_kd:
            prev_model=copy.deepcopy(model).cpu().eval()
            for p in prev_model.parameters(): p.requires_grad=False

        # eval
        model.eval()
        with torch.no_grad():
            for j in range(k+1):
                cor=tot=0
                for x,y in tasks[j][1]:
                    x,y=x.to(device),y.to(device)
                    out=model(x,None if scn.shared_head else j)
                    cor+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
                acc[k,j]=cor/tot
    return acc.cpu()

# ----------------------------------------------------------------- Metrics
def metrics(A:torch.Tensor):
    K=A.size(0); final=A[-1,:K]
    return dict(avg_acc=final.mean().item(),
                BWT=((final[:-1]-torch.diag(A)[:-1]).mean().item() if K>1 else 0.))

# ----------------------------------------------------------------- CLI
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--scenario",choices=["CIL","TIL","DIL"],default="CIL")
    pa.add_argument("--tasks",type=int,default=5)
    pa.add_argument("--epochs",type=int,default=1)
    pa.add_argument("--batch",type=int,default=128)
    # replay
    g=pa.add_mutually_exclusive_group()
    g.add_argument("--replay",dest="replay",action="store_true")
    g.add_argument("--no-replay",dest="replay",action="store_false")
    pa.set_defaults(replay=True)
    # EWC
    g2=pa.add_mutually_exclusive_group()
    g2.add_argument("--regularization",dest="reg",action="store_true")
    g2.add_argument("--no-regularization",dest="reg",action="store_false")
    pa.set_defaults(reg=False)
    pa.add_argument("--ewc-lambda",type=float,default=10.)
    # KD
    g3=pa.add_mutually_exclusive_group()
    g3.add_argument("--distill",dest="distill",action="store_true")
    g3.add_argument("--no-distill",dest="distill",action="store_false")
    pa.set_defaults(distill=False)
    pa.add_argument("--kd-lambda",type=float,default=1.)
    pa.add_argument("--kd-temp",type=float,default=2.)
    # Projection
    g4=pa.add_mutually_exclusive_group()
    g4.add_argument("--proj",dest="proj",action="store_true",
                    help="enable A‑GEM‑style gradient projection")
    g4.add_argument("--no-proj",dest="proj",action="store_false")
    pa.set_defaults(proj=False)
    pa.add_argument("--m-proj",type=int,default=128,
                    help="reference batch size for projection")
    args=pa.parse_args()

    scn=SCENARIOS[args.scenario]
    split=split_mnist_by_domain if args.scenario=="DIL" else split_mnist_by_class
    tr,te,cmap=split(args.tasks)
    loaders=make_loaders(tr,te,cmap,scn,bs=args.batch)
    model=CLNet(scn); opt=torch.optim.SGD(model.parameters(),lr=0.1)

    A=train_inc(loaders,model,opt,scn,
                epochs=args.epochs,
                use_replay=args.replay,
                use_ewc=args.reg , lam_ewc=args.ewc_lambda,
                use_kd=args.distill, lam_kd=args.kd_lambda, T_kd=args.kd_temp,
                use_proj=args.proj , m_proj=args.m_proj)
    print("Acc matrix\n",A); print("Metrics",metrics(A))

if __name__=="__main__": main()
