#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--dynamic',action='store_true');a=p.parse_args()
path=YOLO(a.weights).export(format='onnx',imgsz=a.imgsz,dynamic=a.dynamic,simplify=True)
import onnx
m=onnx.load(path); onnx.checker.check_model(m)
out=ROOT/'reports'/f'onnx_export_{a.weights.parent.parent.name}.json'
out.write_text(json.dumps({'path':str(path),'status':'validated by onnx.checker','imgsz':a.imgsz,'dynamic':a.dynamic},indent=2)+'\n')
print(path)
