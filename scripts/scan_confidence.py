#!/usr/bin/env python3
"""Validation-only confidence scan with fixed NMS and one-to-one IoU matching."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]; NAMES=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def main():
 p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('--imgsz',type=int,default=320);p.add_argument('--device',default='mps');p.add_argument('--nms-iou',type=float,default=.7);p.add_argument('--match-iou',type=float,default=.5);a=p.parse_args()
 thresholds=np.round(np.arange(.05,.801,.05),2); ims=sorted((ROOT/'data/neu_det/images/val').glob('*')); all_gt=[]; model=YOLO(a.weights)
 results=list(model.predict([str(x) for x in ims],imgsz=a.imgsz,conf=.001,iou=a.nms_iou,device=a.device,verbose=False,stream=True))
 samples=[]
 for im,r in zip(ims,results):
  h,w=cv2.imread(str(im)).shape[:2]; seen=set(); gt=[]
  for line in (ROOT/'data/neu_det/labels/val'/f'{im.stem}.txt').read_text().splitlines():
   q=tuple(line.split())
   if q in seen: continue
   seen.add(q); c,x,y,bw,bh=map(float,q);gt.append((int(c),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
  pr=[(int(c),box.tolist(),float(cf)) for c,box,cf in zip(r.boxes.cls.cpu().numpy(),r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())]
  samples.append((gt,pr))
 rows=[]
 for t in thresholds:
  tp=np.zeros(6,int);fp=np.zeros(6,int);fn=np.zeros(6,int)
  for gt,allp in samples:
   pr=[x for x in allp if x[2]>=t]; used=set()
   for gc,gb in gt:
    cand=[(iou(gb,pb),j,pc) for j,(pc,pb,_) in enumerate(pr) if j not in used and iou(gb,pb)>=a.match_iou]
    if cand:
     _,j,pc=max(cand);used.add(j)
     if pc==gc:tp[gc]+=1
     else: fn[gc]+=1;fp[pc]+=1
    else:fn[gc]+=1
   for j,(pc,_,_) in enumerate(pr):
    if j not in used:fp[pc]+=1
  for c,n in enumerate(NAMES):
   P=tp[c]/max(1,tp[c]+fp[c]);R=tp[c]/max(1,tp[c]+fn[c]);F=2*P*R/max(1e-9,P+R)
   rows.append({'confidence':float(t),'class':n,'tp':int(tp[c]),'fp':int(fp[c]),'fn':int(fn[c]),'precision':P,'recall':R,'f1':F})
 out=ROOT/'reports'/'confidence_scan';out.mkdir(exist_ok=True)
 with (out/'metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with (out/'protocol.json').open('w') as f:json.dump({'split':'val','prediction_candidate_conf':.001,'nms_iou':a.nms_iou,'match_iou':a.match_iou,'gt_deduplication':'exact duplicate YOLO label lines removed'},f,indent=2)
 for metric in ['precision','recall','f1']:
  plt.figure(figsize=(7,4));
  for n in NAMES:
   q=[r for r in rows if r['class']==n];plt.plot([r['confidence'] for r in q],[r[metric] for r in q],label=n)
  plt.xlabel('confidence threshold');plt.ylabel(metric);plt.title(f'Validation {metric}: fixed NMS IoU={a.nms_iou}, match IoU={a.match_iou}');plt.legend(fontsize=7,ncol=2);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out/f'{metric}.png',dpi=160);plt.close()
 print(out/'metrics.csv')
if __name__=='__main__':main()
