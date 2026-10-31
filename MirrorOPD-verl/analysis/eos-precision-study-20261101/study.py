"""Fixed-input precision/cost study. No optimizer, training, or run-config edits."""
import argparse, contextlib, hashlib, importlib.util, json, math, statistics, subprocess, sys, time, types
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
WORK=ROOT/'.worktrees/eos-full-vocab'
sys.path.insert(0,str(WORK))
from verl.trainer.distillation.termination import TerminationActionMap
from verl.trainer.distillation.full_vocab_triton import fused_mirror_logits

# Freeze the existing numerical algorithm; precision is varied on identical logits.
source=subprocess.check_output(['git','-C',str(ROOT),'show','9fd276af:verl/trainer/distillation/numerics.py'],text=True)
def load_math(name,pure=False):
 mod=types.ModuleType(name);sys.modules[name]=mod;exec(compile(source,name,'exec'),mod.__dict__)
 if pure:
  def diff(x,y,pivot,*,anchor=None,eta=1.):
   rx,ry=mod.relative(x,pivot),mod.relative(y,pivot)
   return rx-ry if anchor is None else (rx-mod.relative(anchor,pivot))-eta*(ry-mod.relative(anchor,pivot))
  mod.relative_difference=diff
 return mod
stable=load_math('precision_stable');pure=load_math('precision_pure')
METHODS=['torch_fp64','torch_fp32_pure','torch_fp32_stable','fused_fp32','fused_compensated']

def evaluate(z,q,t,kappa,eta,method,actions,diagnostics=True):
 if method=='torch_fp64':
  wide=[x.double() for x in (z,q,t)]
  co=[torch.cat((x.index_select(1,actions.keep),x.index_select(1,actions.stop).logsumexp(-1,keepdim=True)),-1) for x in wide];mod=stable;correction=None
 elif method=='fused_compensated':
  pairs=[actions.coarsen(x,compensated=True) for x in (z,q,t)]
  co=[p[0] for p in pairs];correction=tuple(p[1] for p in pairs)
 else:
  co=[torch.cat((x.index_select(1,actions.keep),x.index_select(1,actions.stop).logsumexp(-1,keepdim=True)),-1) for x in (z,q,t)]
  mod=pure if method=='torch_fp32_pure' else stable;correction=None
 if method.startswith('fused'):
  spec=SimpleNamespace(objective='mdpo',kappa=kappa,eta=eta,diagnostics=diagnostics)
  value,g,info=fused_mirror_logits(*co,torch.float32,spec=spec,corrections=correction)
  diag={k:info[:,i] for i,k in enumerate(['target_kl','target_log_z','anchor_kl','teacher_kl'])} if diagnostics else {}
 else:
  value,g,diag=mod.mirror_logits_rows(*co,objective='mdpo',kappa=kappa,eta=eta,diagnostics=diagnostics)
 if method=='torch_fp32_pure':
  expanded=g.new_empty(z.shape)
  expanded.index_copy_(1,actions.keep,g[:,:-1])
  expanded.index_copy_(1,actions.stop,g[:,-1:]*z.index_select(1,actions.stop).softmax(-1))
  pivot=z.argmax(-1,keepdim=True);other=expanded.scatter(-1,pivot,0.)
  expanded=other.scatter(-1,pivot,-other.sum(-1,keepdim=True))
 else:
  expanded=actions.expand_gradient(g,z)
 return value,expanded,diag

def metrics(actual,reference,z):
 value,g,info=actual;rv,rg,ri=reference
 g,rg=g.double(),rg.double();norm=rg.norm(dim=-1);err=(g-rg).norm(dim=-1);nonzero=norm>0
 rel=err[nonzero]/norm[nonzero]
 denominator=g.norm(dim=-1)*norm;has_direction=denominator>0
 cos=(g*rg).sum(-1)[has_direction]/denominator[has_direction]
 dom=z.argmax(-1,keepdim=True);gd=g.gather(1,dom).squeeze();rd=rg.gather(1,dom).squeeze()
 def stats(x):
  return {'median':x.median().item(),'p95':torch.quantile(x,.95).item(),'max':x.max().item()} if x.numel() else None
 return {'rows':len(z),'nonzero_reference_rows':int(nonzero.sum()),'gradient_relative':stats(rel),'gradient_absolute_l2':stats(err),'gradient_reference_l2':stats(norm),'gradient_rel_over_1e3':int((rel>1e-3).sum()),'gradient_rel_over_1e2':int((rel>1e-2).sum()),'zeroed_nonzero_rows':int(((g.norm(dim=-1)==0)&nonzero).sum()),'dominant_zeroed':int(((gd==0)&(rd!=0)).sum()),'cosine_min':cos.min().item() if cos.numel() else None,'loss_abs_max':(value.double()-rv.double()).abs().max().item(),'target_kl_abs_max':(info['target_kl'].double()-ri['target_kl'].double()).abs().max().item(),'target_kl_reference_max':ri['target_kl'].abs().max().item(),'negative_target_kl':int((info['target_kl']<0).sum()),'finite':bool(torch.isfinite(value).all() and torch.isfinite(g).all())}

def oracle(z,q,t,kappa,eta,ids):
 import mpmath as mp
 with mp.workdps(80):
  def probs(v):
   a=[mp.exp(mp.mpf(float(x))) for x in v];s=sum(a);return [x/s for x in a]
  p,aq,at=map(probs,(z,q,t));keep=[i for i in range(len(z)) if i not in ids]
  pc,qc,tc=[[x[i] for i in keep]+[sum(x[i] for i in ids)] for x in (p,aq,at)]
  e,k=mp.mpf(eta),mp.mpf(kappa)
  b=[(1-e)*mp.log(a)+e*mp.log(c) if kappa==0 else mp.log((1-e)*a**k+e*c**k)/k for a,c in zip(qc,tc)]
  logz=mp.log(sum(mp.exp(x) for x in b));h=[mp.log(a)-bb for a,bb in zip(pc,b)];mu=sum(a*x for a,x in zip(pc,h));gc=[a*(x-mu) for a,x in zip(pc,h)]
  g=[mp.mpf(0)]*len(z)
  for i,j in enumerate(keep):g[j]=gc[i]
  for j in ids:g[j]=gc[-1]*p[j]/pc[-1]
  return torch.tensor([float(mu)],device='cuda',dtype=torch.float64),torch.tensor([[float(x) for x in g]],device='cuda',dtype=torch.float64),{'target_kl':torch.tensor([float(mu+logz)],device='cuda',dtype=torch.float64),'target_log_z':torch.tensor([float(logz)],device='cuda',dtype=torch.float64)}

def audit():
 records=[]
 datasets={'r1-real':ROOT/'outputs/opd-numerical-audit-20261025/fixed_logits.pt'}
 if (OUT/'qwen-fixed-logits.pt').exists():datasets['qwen-real']=OUT/'qwen-fixed-logits.pt'
 for dataset,path in datasets.items():
  data=torch.load(path,map_location='cpu',weights_only=True);q0,t0=data['student'],data['teacher']
  for cohort,alpha,progress in [('initial',1.,0.),('toward_teacher',1.,.05),('agreement_1e2',1e-2,0.),('agreement_1e4',1e-4,0.),('agreement_1e6',1e-6,0.),('self',0.,0.)]:
   q=q0.cuda();t=(q0+alpha*(t0-q0)).cuda();z=(q0+progress*(t0-q0)).cuda();actions=TerminationActionMap.create(z.shape[1],(151643,151645),z.device)
   for kappa in [0.,.5,1.]:
    eta=.1;reference=evaluate(z,q,t,kappa,eta,'torch_fp64',actions)
    for method in METHODS[1:]:
     result=evaluate(z,q,t,kappa,eta,method,actions)
     row={'dataset':dataset,'cohort':cohort,'kappa':kappa,'eta':eta,'method':method,**metrics(result,reference,z)};records.append(row)
     print('audit',dataset,cohort,kappa,method,row['gradient_relative'],flush=True)
     del result
    del reference
   del q,t,z,data;torch.cuda.empty_cache()
   data=torch.load(path,map_location='cpu',weights_only=True)
  (OUT/'real-audit.json').write_text(json.dumps(records,indent=2)+'\n')
 boundary=[]
 cases=[]
 for offset in [0.,32.,128.]:
  for gap in [2.,8.,16.,24.]:
   for delta in [1e-1,1e-3,1e-5]:
    q=torch.full((1,17),-24.+offset,device='cuda');q[0,1]=offset;q[0,4]=offset-gap
    t=q.clone();t[0,4]+=delta
    cases.append((f'eos-gap{gap}-delta{delta}-offset{offset}',q,q,t))
 for mass in [.01,.5,.99]:
  for gap in [2.,8.,16.]:
   for delta in [.1,.001,.00001]:
    for offset in [0.,32.,128.]:
     background=math.log((1-mass)/mass)+math.log1p(math.exp(-gap))-math.log(15)
     q=torch.full((1,17),background+offset,device='cuda');q[0,1]=offset;q[0,4]=offset-gap
     t=q.clone();t[0,4]+=delta
     cases.append((f'eos-mass{mass}-gap{gap}-delta{delta}-offset{offset}',q,q,t))
 for eta in [1e-6,.1,1.]:
  q=torch.tensor([[32.,22.,20.,-1.,24.]],device='cuda');t=q+q.new_tensor([[0.,3e-5,-2e-5,1e-5,2e-5]])
  cases.append((f'near-target-eta{eta}',q,q,t))
 for name,z,q,t in cases:
  eta=float(name.split('eta')[-1]) if name.startswith('near-target') else .1
  for kappa in [0.,.5,1.]:
   ids=(1,4);actions=TerminationActionMap.create(z.shape[1],ids,z.device);ref=oracle(z[0].tolist(),q[0].tolist(),t[0].tolist(),kappa,eta,ids)
   for method in METHODS:
    r=evaluate(z,q,t,kappa,eta,method,actions)
    boundary.append({'case':name,'kappa':kappa,'eta':eta,'method':method,**metrics(r,ref,z)})
 print('boundary',len(boundary),flush=True)
 (OUT/'boundary-audit.json').write_text(json.dumps(boundary,indent=2)+'\n')
 return records,boundary

def benchmark():
 data=torch.load(OUT/'qwen-fixed-logits.pt',map_location='cpu',weights_only=True) if (OUT/'qwen-fixed-logits.pt').exists() else torch.load(ROOT/'outputs/opd-numerical-audit-20261025/fixed_logits.pt',map_location='cpu',weights_only=True)
 rows=[]
 for n in [32,128,512]:
  index=torch.arange(n)%len(data['student']);q=data['student'][index].cuda();t=data['teacher'][index].cuda();z=q;actions=TerminationActionMap.create(z.shape[1],(151643,151645),z.device)
  for kappa in [0.,.5,1.]:
   timings={m:[] for m in METHODS};peak={}
   for method in METHODS:
    for _ in range(2):evaluate(z,q,t,kappa,.1,method,actions)
   torch.cuda.synchronize()
   for repeat in range(5):
    for method in METHODS if repeat%2==0 else list(reversed(METHODS)):
     torch.cuda.synchronize();start=time.perf_counter();result=evaluate(z,q,t,kappa,.1,method,actions);torch.cuda.synchronize();timings[method].append((time.perf_counter()-start)*1000);del result
   for method in METHODS:
    torch.cuda.empty_cache();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();result=evaluate(z,q,t,kappa,.1,method,actions);torch.cuda.synchronize();peak[method]=(torch.cuda.max_memory_allocated()-before)/2**20;del result
   rows.append({'tokens':n,'vocab':z.shape[1],'kappa':kappa,'eta':.1,'timing_ms':timings,'median_ms':{m:statistics.median(v) for m,v in timings.items()},'extra_peak_mib':peak})
   print('bench',n,kappa,rows[-1]['median_ms'],flush=True)
  (OUT/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['audit','benchmark','all'],default='all');args=p.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 meta={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'methods':METHODS,'reference':'Frozen 9fd276af stable FP64 Torch on the same FP32 logits; boundary cases use independent 80-digit probability-space oracle.','scope':'EOS row algebra + gradient scatter; excludes projections, backbone, optimizer and distributed communication. All diagnostics enabled.','gradient_basis':'Unscaled d(eta*L)/d logits; global-token normalization and 1/eta are excluded equally for all methods.','note':'FP32-stable and fused still retain historical local FP64 differences. No claim of pure-FP32 instructions for them. Real agreement/progress cohorts interpolate logits; they are not trained checkpoints.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidate_sha256':{f:hashlib.sha256((WORK/f).read_bytes()).hexdigest() for f in ['verl/trainer/distillation/full_vocab_triton.py','verl/trainer/distillation/termination.py']}}
 (OUT/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
 if args.mode in ['audit','all']:audit()
 if args.mode in ['benchmark','all']:benchmark()
