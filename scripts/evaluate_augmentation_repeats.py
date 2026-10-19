#!/usr/bin/env python3
"""Validation-only evaluation of fixed-split seed 42/43/44 repeats."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/augmentation_ablation_metrics.json'
def complete(folder:Path)->bool:
 p=folder/'results.csv';w=folder/'weights/best.pt'
 if not (p.is_file() and w.is_file()):return False
 with p.open(newline='') as f:return sum(1 for _ in csv.DictReader(f))>=60
def main():
 initial=json.loads(BASE.read_text());out={}
 for mode in ('baseline','conservative'):
  out[mode]={}
  first=initial[mode]
  out[mode]['42']={'map50':first['map50'],'map50_95':first['map50_95'],'precision':first['precision'],'recall':first['recall'],'classes':first['classes']}
  for seed in (43,44):
   name=f'yolo11n_320_{mode}_seed{seed}';folder=ROOT/'experiments'/name
   if not complete(folder):raise SystemExit(f'Incomplete experiment: {folder}')
   print('VALIDATE',name,flush=True)
   m=YOLO(str(folder/'weights/best.pt')).val(data=str(ROOT/'configs/neu_det.yaml'),split='val',imgsz=320,device='mps',plots=False,verbose=False)
   out[mode][str(seed)]={'map50':float(m.box.map50),'map50_95':float(m.box.map),'precision':float(m.box.mp),'recall':float(m.box.mr),
    'classes':{m.names[i]:{'ap50':float(m.box.ap50[i]),'ap50_95':float(m.box.ap[i]),'precision':float(m.box.p[i]),'recall':float(m.box.r[i])} for i in range(6)}}
  out[mode]['summary']={k:{'mean':statistics.mean(out[mode][str(s)][k] for s in (42,43,44)),
    'std_sample':statistics.stdev(out[mode][str(s)][k] for s in (42,43,44))} for k in ('map50','map50_95','precision','recall')}
 path=ROOT/'reports/augmentation_repeat_metrics.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(path)
if __name__=='__main__':main()
