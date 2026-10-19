#!/usr/bin/env python3
"""Sequential, resumable experiment launcher; intentionally never runs parallel training."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=60);p.add_argument('--device',default='mps');p.add_argument('--include-heavy',action='store_true');a=p.parse_args()
jobs=[('yolo11n_320', 'yolo11n.pt',320),('yolo11n_416','yolo11n.pt',416)]
if a.include_heavy: jobs += [('yolo11n_640','yolo11n.pt',640),('yolo11s_640','yolo11s.pt',640)]
for name,model,size in jobs: subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--name',name,'--model',model,'--imgsz',str(size),'--epochs',str(a.epochs),'--device',a.device],check=True)
