#!/usr/bin/env python3
"""Apply the Ultralytics training operators to a real NEU-DET training image."""
from __future__ import annotations
import copy, json, random
from pathlib import Path
import cv2, numpy as np
from ultralytics.data.augment import RandomHSV, RandomPerspective, RandomFlip
from ultralytics.utils.instance import Instances

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/figures/augmentation_samples';OUT.mkdir(parents=True,exist_ok=True)
image_path=ROOT/'data/neu_det/images/train/crazing_1.jpg'
if not image_path.exists():image_path=sorted((ROOT/'data/neu_det/images/train').glob('crazing*.jpg'))[0]
label_path=ROOT/'data/neu_det/labels/train'/f'{image_path.stem}.txt'
img=cv2.imread(str(image_path)); boxes=[]; classes=[]
for line in label_path.read_text().splitlines():
 c,*b=map(float,line.split());classes.append(int(c));boxes.append(b)
def fresh():return {'img':img.copy(),'cls':np.array(classes).reshape(-1,1),'instances':Instances(np.array(boxes,dtype=np.float32),segments=np.zeros((0,0,2),dtype=np.float32),bbox_format='xywh',normalized=True)}
def draw(labels):
 x=labels['img'].copy();ins=copy.deepcopy(labels['instances']);ins.convert_bbox('xyxy');
 if ins.normalized:ins.denormalize(x.shape[1],x.shape[0])
 for box in ins.bboxes:
  x1,y1,x2,y2=map(int,box);cv2.rectangle(x,(x1,y1),(x2,y2),(0,0,255),1)
 return x
np.random.seed(42);random.seed(42)
variants={'original':fresh()}
variants['hsv_baseline']=RandomHSV(.015,.7,.4)(fresh())
variants['hsv_conservative']=RandomHSV(0,.12,.15)(fresh())
variants['geometry_baseline']=RandomPerspective(translate=.1,scale=.5)(fresh())
variants['geometry_conservative']=RandomPerspective(translate=.03,scale=.15)(fresh())
variants['flip_baseline_forced']=RandomFlip(p=1,direction='horizontal')(fresh())
variants['flip_conservative']=RandomFlip(p=0,direction='horizontal')(fresh())
summary={'source':str(image_path.relative_to(ROOT)),'training_operators':['RandomHSV','RandomPerspective','RandomFlip'],'note':'Isolated operators on one training image; full training also includes mosaic.'}
for name,labels in variants.items():
 cv2.imwrite(str(OUT/f'{name}.jpg'),draw(labels)); x=labels['img'];ins=copy.deepcopy(labels['instances']);ins.convert_bbox('xyxy');
 if ins.normalized:ins.denormalize(x.shape[1],x.shape[0])
 summary[name]={'gray_pixel_fraction':round(float(np.mean((x[:,:,0]==x[:,:,1]) & (x[:,:,1]==x[:,:,2]))),4),'mean_intensity':round(float(x.mean()),2),'mean_absolute_pixel_change':round(float(np.abs(x.astype(np.int16)-img.astype(np.int16)).mean()),2),'num_boxes':len(ins),'boxes_xyxy':np.round(ins.bboxes,1).tolist()}
(OUT/'measurements.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
