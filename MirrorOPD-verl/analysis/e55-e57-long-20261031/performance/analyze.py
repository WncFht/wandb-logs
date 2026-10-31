import hashlib,json,subprocess
from pathlib import Path
repo=Path.cwd();logs=repo.parent/'wandb-logs'
source=repo/'examples/mirroropd/docs/infra/model-compatibility/qwen-e54/figures/e55-e57-20261031/data.json'
data=json.loads(source.read_text());rows=[]
for run in data['runs']:
 decoded=next((logs/'derived/MirrorOPD-verl').glob('*'+run['run_id']))
 history=[json.loads(line) for line in (decoded/'history.jsonl').read_text().splitlines()]
 outer=[r for r in history if 'rollout/response_length/mean' in r]
 def aggregate(values):
  return {'outers':len(values),'start':values[0]['step/outer'],'end':values[-1]['step/outer'],**{name:sum(r[key] for r in values)/3600 for name,key in [('train_h','perf/timing/outer_step_seconds'),('actor_h','perf/timing/update_actor_seconds'),('rollout_h','perf/timing/rollout_batch_seconds')]}}
 all_training=aggregate(outer);last=aggregate(outer[-10:]);val=[]
 for step in run['validation_outers']:
  v=[r for r in run['validation'] if r['outer']==step]
  record=next(r for r in history if r.get('step/outer')==step and 'eval/math500/acc/observed_mean' in r)
  tokens=sum(r['length']*r['responses'] for r in v);capped=sum(r['cap_count'] for r in v)*run['config']['val_cap'];seconds=record['perf/timing/validation_rollout_seconds']
  val.append({'outer':step,'responses':sum(r['responses'] for r in v),'output_tokens':tokens,'capped_output_tokens':capped,'capped_token_fraction':capped/tokens,'generation_h':seconds/3600,'generation_tokens_per_s':tokens/seconds,'post_generation_s':record['perf/timing/validation_seconds']-seconds})
 total=all_training['train_h']+sum(r['generation_h'] for r in val)
 common=[r for r in outer if 26<=r['step/outer']<=35];actor=sum(r['perf/timing/update_actor_seconds'] for r in common);tokens=sum(r['rollout/total_tokens'] for r in common)
 rows.append({'name':run['name'],'run_id':run['run_id'],'source_sha256':run['sha256'],'training':all_training,'last10':last,'validation':val,'covered_h':total,'validation_fraction':sum(r['generation_h'] for r in val)/total,'actor_fraction':all_training['actor_h']/total,'static_speedup_if_actor_2x':total/(total-all_training['actor_h']/2),'last10_actor_fraction':last['actor_h']/last['train_h'],'common_outer26_35':{'input_tokens_per_outer':tokens/len(common),'actor_minutes_per_outer':actor/len(common)/60,'input_token_passes_per_gpu_second':tokens*16/8/actor,'teacher_backbone_seconds_per_outer':sum(r['perf/timing/teacher_forward_seconds'] for r in common)/len(common)}})
result={'logs_revision':data['logs_head'],'training_code_revision':data['runs'][0]['code_commit'],'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'validation_time_basis':'perf/timing/validation_rollout_seconds; no inferred missing/startup time','runs':rows}
output=logs/'MirrorOPD-verl/analysis/e55-e57-long-20261031/performance';output.mkdir(parents=True,exist_ok=True)
(output/'current-timing.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
