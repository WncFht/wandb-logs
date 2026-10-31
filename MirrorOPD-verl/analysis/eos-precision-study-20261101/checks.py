import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
import study

out=Path(__file__).resolve().parent
torch.set_num_threads(1)
class Dtypes(TorchDispatchMode):
 def __init__(self):self.fp64=[]
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  r=func(*args,**(kwargs or {}))
  for x in tree_flatten(r)[0]:
   if isinstance(x,torch.Tensor) and x.dtype==torch.float64:self.fp64.append(str(func))
  return r
z=torch.tensor([[0.,-2.,-4.,-1.,-2.]],device='cuda');t=z+z.new_tensor([[0.,.001,-.001,.002,.001]])
a=study.TerminationActionMap.create(5,(1,4),'cuda');counter=Dtypes()
with counter:r=study.evaluate(z,z,t,.5,.1,'torch_fp32_pure',a)
assert not counter.fp64,counter.fp64
rows=[]
for k in [0.,.5,1.]:
 q=torch.full((1,17),-2.,device='cuda');q[0,1]=0.;q[0,4]=-2.;t=q.clone();t[0,4]+=2**-10
 a=study.TerminationActionMap.create(17,(1,4),'cuda')
 ref=study.oracle(q[0].tolist(),q[0].tolist(),t[0].tolist(),k,.1,(1,4))
 for m in study.METHODS:
  first=study.evaluate(q,q,t,k,.1,m,a)
  swapped=t.clone();swapped[:,[1,4]]=t[:,[4,1]]
  second=study.evaluate(q,q,swapped,k,.1,m,a)
  assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1]),(m,k)
  for shift in [0.,32.,128.]:
   got=study.evaluate(q+shift,q+shift,t+shift,k,.1,m,a)
   rows.append({'kappa':k,'method':m,'shift':shift,**study.metrics(got,ref,q+shift)})
(out/'additional-checks.json').write_text(json.dumps({'pure_fp32_created_fp64_tensor_ops':counter.fp64,'teacher_eos_swap_bitwise_equal':True,'gauge_cases':rows},indent=2)+'\n')
print('Pure FP32 tensor path and teacher EOS-swap identity passed; gauge errors recorded.')
