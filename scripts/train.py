#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='yolo11n.pt');p.add_argument('--imgsz',type=int,default=320);p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',default='8');p.add_argument('--device',default='mps');p.add_argument('--name',default='baseline_yolo11n_320');p.add_argument('--augmentation',choices=['default','conservative','hsv_only','geometry_only','flip_only'],default='default');p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true');a=p.parse_args()
 device=a.device
 try:
  import torch
  if device=='mps' and not torch.backends.mps.is_available(): device='cpu'; print('MPS unavailable; falling back to CPU')
 except ImportError: pass
 augments={
  'default':{},
  'conservative':{'hsv_h':0.0,'hsv_s':0.12,'hsv_v':0.15,'translate':0.03,'scale':0.15,'fliplr':0.0},
  'hsv_only':{'hsv_h':0.0,'hsv_s':0.12,'hsv_v':0.15},
  'geometry_only':{'translate':0.03,'scale':0.15},
  'flip_only':{'fliplr':0.0},
 }
 save_dir=ROOT/'experiments'/a.name
 if save_dir.exists() and not a.resume:
  raise SystemExit(f'Experiment exists: {save_dir}. Choose another name or use --resume.')
 checkpoint=save_dir/'weights/last.pt'
 if a.resume and not checkpoint.is_file():
  raise SystemExit(f'Cannot resume without checkpoint: {checkpoint}')
 model=YOLO(str(checkpoint) if a.resume else a.model)
 model.train(data=str(ROOT/'configs/neu_det.yaml'),imgsz=a.imgsz,epochs=a.epochs,batch=int(a.batch),device=device,seed=a.seed,deterministic=True,project=str(ROOT/'experiments'),name=a.name,resume=str(checkpoint) if a.resume else False,plots=True,**augments[a.augmentation])
if __name__=='__main__': main()
