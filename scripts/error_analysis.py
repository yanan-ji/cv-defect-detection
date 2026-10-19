#!/usr/bin/env python3
"""Validation-only error analysis. Never uses the held-out test set to select changes."""
from __future__ import annotations
import argparse, csv, json, shutil
from collections import Counter
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
NAMES=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
def iou(a,b):
 x1,y1,_,_=np.maximum(a[:4],b[:4]); _,_,x2,y2=np.minimum(a[:4],b[:4]); inter=max(0,x2-x1)*max(0,y2-y1); return inter/(max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))
def main():
 p=argparse.ArgumentParser();p.add_argument('weights',type=Path);p.add_argument('--split',choices=['val'],default='val');p.add_argument('--imgsz',type=int,default=320);p.add_argument('--conf',type=float,default=.25);p.add_argument('--device',default='mps');a=p.parse_args()
 out=ROOT/'reports'/'baseline_error_analysis';out.mkdir(parents=True,exist_ok=True);cases=out/'cases';shutil.rmtree(cases,ignore_errors=True);cases.mkdir()
 ims=sorted((ROOT/'data/neu_det/images'/a.split).glob('*')); model=YOLO(a.weights); cm=np.zeros((7,7),int); rows=[]; area=[]; details=[]
 for im,r in zip(ims,model.predict([str(x) for x in ims],imgsz=a.imgsz,conf=a.conf,device=a.device,verbose=False,stream=True)):
  image=cv2.imread(str(im));h,w=image.shape[:2]; labs=[]
  seen=set()
  for line in (ROOT/'data/neu_det/labels'/a.split/f'{im.stem}.txt').read_text().splitlines():
   if line in seen: continue
   seen.add(line); c,x,y,bw,bh=map(float,line.split()); labs.append((int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])))
  preds=[(int(c),b.astype(float),float(cf)) for c,b,cf in zip(r.boxes.cls.cpu().numpy(),r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())]
  matched=set(); errs=[]; image_details={'image':im.name,'ground_truth':[],'predictions':[],'errors':[]}
  for tc,tb in labs: image_details['ground_truth'].append({'class':NAMES[tc],'xyxy':[round(float(v),1) for v in tb]})
  for pc,pb,cf in preds: image_details['predictions'].append({'class':NAMES[pc],'confidence':round(cf,4),'xyxy':[round(float(v),1) for v in pb]})
  for tc,tb in labs:
   candidates=[(iou(tb,pb),j,pc) for j,(pc,pb,_) in enumerate(preds) if j not in matched and iou(tb,pb)>=.5]
   if candidates:
    _,j,pc=max(candidates);matched.add(j);cm[tc,pc]+=1
    if tc!=pc: errs.append('confusion')
   else: cm[tc,6]+=1;errs.append('fn')
   area.append({'class':NAMES[tc],'relative_area':float((tb[2]-tb[0])*(tb[3]-tb[1])/(w*h)),'matched':bool(candidates)})
  for j,(pc,pb,cf) in enumerate(preds):
   if j not in matched: cm[6,pc]+=1;errs.append('fp')
  if errs:
   image_details['errors']=sorted(set(errs))
   for tc,tb in labs:
    cv2.rectangle(image,tuple(tb[:2].astype(int)),tuple(tb[2:].astype(int)),(0,255,0),1)
    cv2.putText(image,'GT:'+NAMES[tc],tuple(tb[:2].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,0),1)
   for pc,pb,cf in preds:
    cv2.rectangle(image,tuple(pb[:2].astype(int)),tuple(pb[2:].astype(int)),(0,0,255),1)
    cv2.putText(image,f'P:{NAMES[pc]} {cf:.2f}',tuple(pb[:2].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,255),1)
   rows.append({'image':im.name,'errors':','.join(sorted(set(errs))),'count':len(errs)})
   image_details['case_file']=f'{len(rows):03d}_{im.name}'
   details.append(image_details)
   cv2.imwrite(str(cases/f'{len(rows):03d}_{im.name}'),image)
 summary={'split':'val','confidence':a.conf,'iou_match':.5,'labels':NAMES+['background'],'matrix':cm.tolist(),'error_images':len(rows),'note':'green=ground truth; red=prediction'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));
 with (out/'errors.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=['image','errors','count']).writerows(rows)
 with (out/'size_records.json').open('w') as f: json.dump(area,f)
 (out/'case_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
