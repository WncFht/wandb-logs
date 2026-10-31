"""Same-input head projections plus EOS loss and native BF16 dH/dW products."""
import json,statistics,time
from pathlib import Path
import torch
import study as base

out=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
data=torch.load(out/'qwen-fixed-logits.pt',map_location='cpu',weights_only=True)
ws=torch.load(out/'student-head.pt',map_location='cpu',weights_only=True).cuda()
wt=torch.load(out/'teacher-head.pt',map_location='cpu',weights_only=True).cuda()
methods=['torch_fp64','torch_fp32_pure','fused_fp32','fused_compensated']
rows=[]
for n in [32,128,512]:
 ix=torch.arange(n)%len(data['student_hidden']);hs=data['student_hidden'][ix].cuda();ht=data['teacher_hidden'][ix].cuda()
 actions=base.TerminationActionMap.create(len(ws),(151643,151645),'cuda')
 for kappa in [0.,.5,1.]:
  def run(method):
   z=torch.mm(hs,ws.T,out_dtype=torch.float32)
   q=torch.mm(hs,ws.T,out_dtype=torch.float32)
   t=torch.mm(ht,wt.T,out_dtype=torch.float32)
   v,g,di=base.evaluate(z,q,t,kappa,.1,method,actions)
   dz=g.to(torch.bfloat16)
   dh=torch.mm(dz,ws,out_dtype=torch.float32)
   dw=torch.mm(dz.T,hs,out_dtype=torch.float32)
   return v,dh,dw
  ref=run('torch_fp64');errors={}
  for method in methods:
   actual=run(method);errors[method]={}
   for key,a,b in zip(['hidden','head'],actual[1:],ref[1:]):
    diff=(a-b).double().norm();norm=b.double().norm();errors[method][key+'_relative_l2']=(diff/norm).item()
   del actual
  del ref
  timings={m:[] for m in methods};peak={}
  for m in methods:
   for _ in range(2):run(m)
  torch.cuda.synchronize()
  for iteration in range(5):
   for m in methods if iteration%2==0 else list(reversed(methods)):
    torch.cuda.synchronize();start=time.perf_counter();r=run(m);torch.cuda.synchronize();timings[m].append((time.perf_counter()-start)*1000);del r
  for m in methods:
   torch.cuda.empty_cache();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();r=run(m);torch.cuda.synchronize();peak[m]=(torch.cuda.max_memory_allocated()-before)/2**20;del r
  row={'tokens':n,'kappa':kappa,'eta':.1,'median_ms':{m:statistics.median(v) for m,v in timings.items()},'timing_ms':timings,'extra_peak_mib':peak,'gradient_errors':errors}
  rows.append(row);(out/'head-benchmark.json').write_text(json.dumps({'scope':'Three native BF16 head projections with FP32 outputs, EOS row algebra and BF16-operand FP32-output dH/dW. No backbone, optimizer, FSDP or normalization across ranks.','rows':rows},indent=2)+'\n')
  print(n,kappa,row['median_ms'],errors,flush=True)
