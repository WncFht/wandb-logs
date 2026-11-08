"""Run the SDK's legacy sync with one final metadata flush, preserving every record.

SDK 0.28.1 sends a config request for every metric definition. Defer those
requests in this process only; final flush contains the same merged config.
Final summary writes are also batched; history records are unchanged.
No synthetic exit is created.
"""
import netrc, os, json, sys, time
from pathlib import Path
from wandb.sdk.internal.sender import SendManager
from wandb.cli.cli import cli
work=Path(__file__).resolve().parent
logs=work.parents[2]/'wandb-logs'
key=netrc.netrc(logs.parent/'.secrets/netrc').authenticators('api.wandb.ai')[2]
os.environ.update(WANDB_API_KEY=key, NETRC=str(logs.parent/'.secrets/netrc'), WANDB_MODE='online',WANDB_BASE_URL='https://api.wandb.ai')
for name in ['WANDB_RUN_ID','WANDB_RESUME','WANDB_NAME']: os.environ.pop(name,None)
original_debounce=SendManager._debounce_config
original_finish=SendManager.finish
original_summary=SendManager._update_summary

def defer_config(self):
    self._config_needs_debounce=True

def defer_summary(self):
    summary=self._cached_summary.copy()
    summary.pop("_wandb",None)
    if self._metadata_summary:
        summary["_wandb"]=self._metadata_summary
    self._consolidated_summary.update(summary)

def finish_with_config(self):
    original_summary(self)
    if self._config_needs_debounce:
        original_debounce(self)
    original_finish(self)

SendManager._debounce_config=defer_config
SendManager.finish=finish_with_config
SendManager._update_summary=defer_summary
os.chdir(work)
rid=sys.argv[1]
target=next((work/'sync-copies').glob('*-'+rid))
cli.main(args=['sync','--legacy','--no-mark-synced','--no-sync-tensorboard','--entity','wncfht','--project','MirrorOPD-verl',str(target)],standalone_mode=False)
print('Batched legacy sync returned successfully:',rid,flush=True)
