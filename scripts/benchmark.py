#!/usr/bin/env python3
"""Measure end-to-end Ultralytics inference (preprocess + model + NMS) after warmup."""
from __future__ import annotations
import argparse, statistics, time
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('image',type=Path);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--runs',type=int,default=50);p.add_argument('--warmup',type=int,default=10);p.add_argument('--device',default='mps');a=p.parse_args();model=YOLO(a.weights)
def sync():
 try:
  import torch
  if a.device == 'mps' and torch.backends.mps.is_available(): torch.mps.synchronize()
 except Exception: pass
for _ in range(a.warmup):model.predict(str(a.image),imgsz=a.imgsz,device=a.device,verbose=False)
sync()
times=[]
for _ in range(a.runs):
 sync();t=time.perf_counter();model.predict(str(a.image),imgsz=a.imgsz,device=a.device,verbose=False);sync();times.append((time.perf_counter()-t)*1000)
times.sort(); p95=times[min(len(times)-1, int(.95*len(times))-1)]
print({'runs':a.runs,'end_to_end_ms_mean':round(statistics.mean(times),2),'std':round(statistics.stdev(times),2),'p50':round(statistics.median(times),2),'p95':round(p95,2),'throughput_img_s':round(1000/statistics.mean(times),2),'includes':'preprocess + inference + NMS'})
