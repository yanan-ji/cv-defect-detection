#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description='Run single-image or directory inference.');p.add_argument('weights',type=Path);p.add_argument('source',type=Path);p.add_argument('--imgsz',type=int,default=320);p.add_argument('--conf',type=float,default=.25);p.add_argument('--device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'demo/output');a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
results=YOLO(a.weights).predict(source=str(a.source),imgsz=a.imgsz,conf=a.conf,device=a.device,save=True,project=str(a.out),name='predictions',exist_ok=False)
for r in results:
 for c,cf in zip(r.boxes.cls.tolist(),r.boxes.conf.tolist()): print(f'{Path(r.path).name}: {r.names[int(c)]} {cf:.3f}')
