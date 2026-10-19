#!/usr/bin/env python3
"""Run three predeclared 60-epoch ablations serially without overwriting results."""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
JOBS=[('hsv_only','yolo11n_320_hsv_only_seed42'),('geometry_only','yolo11n_320_geometry_only_seed42'),('flip_only','yolo11n_320_flip_only_seed42')]
def epochs_done(path:Path)->int:
 p=path/'results.csv'
 if not p.exists():return 0
 with p.open(newline='') as f:return sum(1 for _ in csv.DictReader(f))
def main():
 for mode,name in JOBS:
  target=ROOT/'experiments'/name
  done=epochs_done(target)
  if done>=60 and (target/'weights/best.pt').is_file():
   print(f'SKIP {name}: completed',flush=True);continue
  if target.exists() and done==0:
   raise SystemExit(f'Existing incomplete directory without results: {target}; inspect manually.')
  args=[sys.executable,str(ROOT/'scripts/train.py'),'--model','yolo11n.pt','--imgsz','320','--epochs','60','--batch','8','--device','mps','--seed','42','--name',name,'--augmentation',mode]
  if target.exists():args+=['--resume']
  print('START',name,'completed_epochs',done,flush=True)
  with (ROOT/'experiments'/f'{name}.train.log').open('a') as log:
   result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise SystemExit(f'{name} failed ({result.returncode}); inspect its train log and checkpoint.')
  print('DONE',name,flush=True)
if __name__=='__main__':main()
