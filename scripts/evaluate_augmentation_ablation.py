#!/usr/bin/env python3
"""Evaluate finished seed-42 ablations on validation only and save auditable numbers."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
RUNS={
 'baseline':'baseline_yolo11n_320',
 'conservative':'yolo11n_320_conservative_aug',
 'hsv_only':'yolo11n_320_hsv_only_seed42',
 'geometry_only':'yolo11n_320_geometry_only_seed42',
 'flip_only':'yolo11n_320_flip_only_seed42',
}
def complete(folder:Path)->bool:
 p=folder/'results.csv';w=folder/'weights/best.pt'
 if not (p.is_file() and w.is_file()):return False
 import csv
 with p.open(newline='') as f:return sum(1 for _ in csv.DictReader(f))>=60
def main():
 records={}
 for label,name in RUNS.items():
  folder=ROOT/'experiments'/name
  if not complete(folder):
   records[label]={'status':'INCOMPLETE','directory':str(folder.relative_to(ROOT))};continue
  print('VALIDATE',label,flush=True)
  m=YOLO(str(folder/'weights/best.pt')).val(data=str(ROOT/'configs/neu_det.yaml'),split='val',imgsz=320,device='mps',plots=False,verbose=False)
  records[label]={'status':'COMPLETED','directory':str(folder.relative_to(ROOT)),
   'precision':float(m.box.mp),'recall':float(m.box.mr),'map50':float(m.box.map50),'map50_95':float(m.box.map),
   'classes':{m.names[i]:{'precision':float(m.box.p[i]),'recall':float(m.box.r[i]),'ap50':float(m.box.ap50[i]),'ap50_95':float(m.box.ap[i])} for i in range(6)}}
 out=ROOT/'reports/augmentation_ablation_metrics.json';out.write_text(json.dumps(records,ensure_ascii=False,indent=2));print(out)
 ready=[(k,v) for k,v in records.items() if v['status']=='COMPLETED']
 if ready:
  fig,ax=plt.subplots(figsize=(8,4));names=[k for k,_ in ready];x=range(len(names))
  ax.bar([i-.18 for i in x],[v['map50_95'] for _,v in ready],width=.36,label='mAP50-95')
  ax.bar([i+.18 for i in x],[v['classes']['crazing']['ap50'] for _,v in ready],width=.36,label='crazing AP50')
  ax.set_xticks(list(x),names,rotation=20);ax.set_ylim(0,1);ax.set_ylabel('Validation score');ax.set_title('NEU-DET augmentation ablation, seed 42');ax.legend();fig.tight_layout();fig.savefig(ROOT/'reports/figures/augmentation_ablation.png',dpi=180)
if __name__=='__main__':main()
