import gc,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
bank=root/'runs/qwen-prefix-six-20261031/banks/S0/bank.json'
rows=json.loads(bank.read_text())
# Spread across the available bank, preserving its original tokens.
selected=[rows[i] for i in torch.linspace(0,len(rows)-1,8).long().tolist()]
inputs=[]
for r in selected:
 response=r['response_ids'][:2048]
 positions=torch.linspace(0,len(response)-1,16).long().tolist()
 inputs.append({'id':r['id'],'prompt_ids':r['prompt_ids'],'response_ids':response,'positions':positions})
results={}
metadata={'bank':str(bank),'bank_sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),'rows':[{'id':r['id'],'prompt_tokens':len(r['prompt_ids']),'response_tokens':len(r['response_ids']),'positions':r['positions']} for r in inputs],'precision':'BF16 model operands, original FP32 RoPE buffers, FP32 head outputs','scope':'Initial Qwen3-1.7B-Base and Qwen3-30B-A3B-Instruct-2507 on existing chat-bank tokens; not E55-E57 trained checkpoints or their rollout distribution.'}
torch.set_num_threads(4)
for role,name in [('student','Qwen3-1.7B-Base'),('teacher','Qwen3-30B-A3B-Instruct-2507')]:
 start=time.monotonic();model_path=root.parent/'model'/name
 print('Loading',role,name,flush=True)
 model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'}).eval()
 for n,b in model.named_buffers():
  if 'inv_freq' in n: assert b.dtype==torch.float32,(n,b.dtype)
 values=[];states=[]
 with torch.inference_mode():
  for r in inputs:
   seq=r['prompt_ids']+r['response_ids'][:-1]
   h=model.get_decoder()(input_ids=torch.tensor([seq],device='cuda'),use_cache=False).last_hidden_state[0]
   index=torch.tensor([len(r['prompt_ids'])-1+x for x in r['positions']],device='cuda')
   h=h.index_select(0,index)
   z=torch.mm(h,model.get_output_embeddings().weight.T,out_dtype=torch.float32)
   values.append(z.cpu());states.append(h.cpu())
   print(role,r['id'],len(seq),flush=True)
 results[role]=torch.cat(values);results[role+'_hidden']=torch.cat(states)
 # Keep the native head for the small same-input end-to-end head VJP benchmark.
 torch.save(model.get_output_embeddings().weight.detach().cpu(),out/(role+'-head.pt'))
 metadata[role]={'model':str(model_path),'config_sha256':hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),'seconds':time.monotonic()-start}
 del model,h,z,values,states;gc.collect();torch.cuda.empty_cache()
 torch.save(results,out/'qwen-fixed-logits.partial.pt')
torch.save(results,out/'qwen-fixed-logits.pt');(out/'qwen-inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved', {k:tuple(v.shape) for k,v in results.items()},flush=True)
