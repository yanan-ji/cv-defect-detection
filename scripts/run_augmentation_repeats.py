#!/usr/bin/env python3
"""Repeat baseline and a frozen validation-selected candidate on the fixed split."""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def complete(folder:Path)->bool:
 p=folder/'results.csv';w=folder/'weights/best.pt'
 if not (p.is_file() and w.is_file()):return False
 with p.open(newline='') as f:return sum(1 for _ in csv.DictReader(f))>=60
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,choices=['conservative','hsv_only','geometry_only','flip_only']);a=p.parse_args()
 for seed in (43,44):
  for mode in ('default',a.candidate):
   label='baseline' if mode=='default' else mode
   name=f'yolo11n_320_{label}_seed{seed}';target=ROOT/'experiments'/name
   if complete(target):print('SKIP',name,flush=True);continue
   args=[sys.executable,str(ROOT/'scripts/train.py'),'--model','yolo11n.pt','--imgsz','320','--epochs','60','--batch','8','--device','mps','--seed',str(seed),'--augmentation',mode,'--name',name]
   if target.exists():args.append('--resume')
   print('START',name,flush=True)
   with (ROOT/'experiments'/f'{name}.train.log').open('a') as log:
    rc=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   if rc:raise SystemExit(f'{name} failed ({rc}); inspect log and checkpoint.')
   print('DONE',name,flush=True)
if __name__=='__main__':main()
