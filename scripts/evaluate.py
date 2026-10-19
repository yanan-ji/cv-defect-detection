#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('--imgsz',type=int,default=320);p.add_argument('--split',default='test');p.add_argument('--device',default='mps');a=p.parse_args()
metrics=YOLO(a.weights).val(data=str(ROOT/'configs/neu_det.yaml'),split=a.split,imgsz=a.imgsz,device=a.device,plots=True)
print({'precision':metrics.box.mp,'recall':metrics.box.mr,'map50':metrics.box.map50,'map50_95':metrics.box.map})
