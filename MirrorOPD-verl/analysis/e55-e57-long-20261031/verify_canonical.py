"""Sync immutable complete-record copies and verify their cloud histories."""
import hashlib
import importlib.util
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess
import time

work = Path(__file__).resolve().parent
logs = work.parents[2] / 'wandb-logs'
module_spec = importlib.util.spec_from_file_location('wandb_local', logs / 'tools/wandb_local.py')
decoder = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(decoder)
ids = ['j5uaal3j', 'zsnhqzts', '6j5knkmk']
credentials = logs.parent / '.secrets/netrc'
auth = netrc.netrc(credentials).authenticators('api.wandb.ai')
assert auth
sync_env = os.environ.copy()
sync_env.update(NETRC=str(credentials), WANDB_API_KEY=auth[2], WANDB_BASE_URL='https://api.wandb.ai', WANDB_MODE='online')
for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME'):
    sync_env.pop(key, None)
import wandb
api = wandb.Api(api_key=auth[2], timeout=90)
verifications = []
keys = ['_step', 'step/outer', 'step/update', 'step/optimizer', 'distill/mdpo/objective', 'distill/target_kl', 'distill/teacher_kl', 'rollout/response_length/mean', 'rollout/finish_reason/length_count']
for ds in ['math500', 'amc23', 'aime24', 'aime25', 'aime26']:
    keys += [f'eval/{ds}/acc/observed_mean', f'eval/{ds}/generation/response_length/mean', f'eval/{ds}/generation/finish_reason/length_count']
for rid in ids:
    directory = next((logs / 'derived/MirrorOPD-verl').glob('*-' + rid))
    local = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
    run = api.run('wncfht/MirrorOPD-verl/' + rid)
    # scan_history(keys=...) intersects sparse keys; project after reading all rows.
    cloud = list(run.scan_history(keys=['_step','step/outer'], page_size=5000, use_cache=False, max_step=max(r['_step'] for r in local)+1))
    by_step = {row['_step']: dict(row) for row in cloud}
    local_by_step = {row['_step']: row for row in local}
    duplicate_conflicts = []
    signatures = {tuple(key for key in keys if key in row) for row in local}
    for signature in sorted(signatures):
        if signature == ('_step',):
            continue
        for row in run.scan_history(keys=list(signature), page_size=5000, use_cache=False, max_step=max(r['_step'] for r in local)+1):
            expected = local_by_step.get(row['_step'], {})
            for key, value in row.items():
                if key not in keys or key not in expected:
                    continue
                other = expected[key]
                equal = value == other or (isinstance(value, (int,float)) and isinstance(other, (int,float)) and abs(value-other)<=1e-7*max(1,abs(other)))
                if not equal:
                    duplicate_conflicts.append({'step':row['_step'],'key':key,'cloud':value,'local':other})
            by_step.setdefault(row['_step'], {}).update(row)
    problems = []
    for row in local:
        other = by_step.get(row['_step'], {})
        for key in keys:
            if key in row and other.get(key) != row[key]:
                a,b = row[key],other.get(key)
                if not (isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b) <= 1e-7*max(1,abs(a))):
                    problems.append({'step':row['_step'],'key':key,'local':a,'cloud':b})
    item = {'run_id':rid, 'url':run.url, 'cloud_state':run.state, 'local_rows':len(local), 'cloud_rows':len(cloud), 'unique_cloud_steps':len(by_step), 'missing_steps':sorted(set(r['_step'] for r in local)-by_step.keys()), 'extra_steps':sorted(by_step.keys()-set(r['_step'] for r in local)), 'metric_mismatches':problems, 'keys_checked':keys}
    item['replayed_prefix_rows'] = len(cloud)-len(set(r['_step'] for r in cloud))
    item['replay_metric_conflicts'] = duplicate_conflicts
    item['passed'] = len(local)==len(by_step) and not item['missing_steps'] and not item['extra_steps'] and not problems and not duplicate_conflicts
    verifications.append(item)
    (work / 'cloud-verification-canonical.json').write_text(json.dumps(verifications,indent=2)+'\n')
    print('Verified',rid,'passed',item['passed'],'local/cloud',len(local),len(cloud),'mismatches',len(problems),flush=True)
assert all(v['passed'] for v in verifications)
