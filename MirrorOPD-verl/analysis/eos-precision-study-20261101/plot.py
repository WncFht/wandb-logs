"""EOS precision study: observed latency and errors on fixed inputs.

Semantic ledger: each method retains one color across panels. Real-data curves
are maxima across kappa, not training trajectories. Boundary dots are deliberately
constructed cases, not frequencies in the training distribution. Latency intervals
are min/max of five interleaved repeats, not confidence intervals.
Geometry ledger: logarithmic positive axes; identical error scales within facets.
"""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
root=Path(__file__).resolve().parents[2];data=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'dotfiles/.agents/skills/figure-viz/assets/mpl'))
import fvstyle as fv
fv.apply_rc('paper',fontsize=12)
out=root/'examples/mirroropd/docs/infra/performance/full-vocab/figures/eos-precision-20261101';out.mkdir(parents=True,exist_ok=True)
methods=['torch_fp64','torch_fp32_pure','fused_fp32','fused_compensated']
names={'torch_fp64':'FP64 Torch','torch_fp32_pure':'FP32 Torch','fused_fp32':'FP32 fused','fused_compensated':'Fused + EOS correction'}
colors=dict(zip(methods,[fv.HUES['vgray'],fv.HUES['vorange'],fv.HUES['vteal'],fv.HUES['vviolet']]))
def save(fig,name):
 fig.canvas.draw();issues=fv.audit(fig,ignore=['facet-ylim-mismatch'])
 (out/(name+'-audit.json')).write_text(json.dumps(issues)+'\n')
 if any(fv.problem_severity(i)=='error' for i in issues):raise RuntimeError(issues)
 for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=180)
 print(name,issues);plt.close(fig)

row=json.loads((data/'benchmark.json').read_text());head=json.loads((data/'head-benchmark.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(left=.22,right=.97,bottom=.30,top=.83,wspace=.65)
for ax,rows,title in zip(axes,[row,head],['EOS row algebra','Head projections + loss + dH/dW']):
 r=next(x for x in rows if x['tokens']==512 and x['kappa']==.5)
 for i,m in enumerate(methods):
  v=r['median_ms'][m];raw=r['timing_ms'][m]
  ax.plot([min(raw),v,max(raw)],[3-i]*3,color=colors[m],marker='o',markevery=[1],ms=7,label=names[m])
 ax.set(xscale='log',xlim=(8,600),ylim=(-.5,3.5),yticks=[3,2,1,0],yticklabels=[names[m] for m in methods],xlabel='Milliseconds (log scale)',ylabel='Implementation',title=title)
 ax.grid(axis='x',alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,frameon=False,fontsize=11)
fig.text(.5,.025,'H800; 512 positions, vocab 151,936; kappa=0.5, eta=0.1; diagnostics on.\nMedian and min/max of five repeats; excludes backbone, optimizer and FSDP.',ha='center',fontsize=11,color=fv.NEUTRALS['vink'])
save(fig,'latency')

boundary=json.loads((data/'boundary-audit.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(10,7.2));fig.subplots_adjust(left=.1,right=.97,bottom=.2,top=.91,wspace=.28,hspace=.45)
for ax,m in zip(axes.flat,methods):
 rows=[r for r in boundary if r['method']==m and r['gradient_relative']]
 ax.scatter([r['gradient_reference_l2']['max'] for r in rows],[max(r['gradient_relative']['max'],1e-10) for r in rows],color=colors[m],s=16,alpha=.5)
 ax.axhline(.001,color=fv.NEUTRALS['vink'],lw=.8,ls='--')
 ax.set(xscale='log',yscale='log',xlim=(1e-27,1e-2),ylim=(1e-11,10),xlabel='Reference gradient L2 norm',ylabel='Relative gradient error',title=names[m])
 ax.set_xticks([1e-24,1e-16,1e-8]);ax.set_yticks([1e-9,1e-6,1e-3,1]);ax.grid(alpha=.2)
fig.text(.5,.055,'Constructed EOS stress cases; 80-digit reference; dashed line = 0.1% relative error.\nErrors below 1e-10 use a display floor. These dots do not estimate real failure frequency.',ha='center',fontsize=11,color=fv.NEUTRALS['vink'])
save(fig,'boundary')

real=json.loads((data/'real-audit.json').read_text());cohorts=['initial','agreement_1e2','agreement_1e4','agreement_1e6']
fig,axes=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(left=.1,right=.98,bottom=.31,top=.86,wspace=.3)
for ax,ds,title in zip(axes,['r1-real','qwen-real'],['R1 / JustRL logits','Qwen3 Base / MoE logits']):
 for m,ls in [('torch_fp32_pure',':'),('fused_fp32','--'),('fused_compensated','-')]:
  y=[max(x['gradient_relative']['max'] for x in real if x['dataset']==ds and x['cohort']==c and x['method']==m) for c in cohorts]
  ax.plot([1,1e-2,1e-4,1e-6],y,color=colors[m],ls=ls,marker='o',ms=4,label=names[m])
 ax.axhline(.001,color=fv.NEUTRALS['vink'],lw=.8,ls='--')
 ax.set(xscale='log',yscale='log',xlim=(2,5e-7),ylim=(1e-7,3e-3),title=title,xlabel='Teacher logit displacement multiplier',ylabel='Worst relative gradient error')
 ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.115),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.025,'128 fixed positions per pair; maximum across kappa=0, 0.5, 1; eta=0.1.\nMultipliers below 1 construct near-agreement cases; they are not trained checkpoints.',ha='center',fontsize=11,color=fv.NEUTRALS['vink'])
save(fig,'real-inputs')
