#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/neu_det');a=p.parse_args(); issues=[]; counts={}
 for split in ('train','val','test'):
  ims=sorted((a.data/'images'/split).glob('*')); nobj=0
  for im in ims:
   image=cv2.imread(str(im)); lab=a.data/'labels'/split/f'{im.stem}.txt'
   if image is None: issues.append(f'corrupt: {im}')
   if not lab.exists(): issues.append(f'missing label: {im}')
   else:
    for line in lab.read_text().splitlines():
     q=line.split()
     if len(q)!=5 or not 0<=int(q[0])<6 or any(not 0 < float(v) <= 1 for v in q[1:]): issues.append(f'invalid label: {lab}: {line}')
     nobj+=1
  counts[split]={'images':len(ims),'objects':nobj}
 result={'counts':counts,'issues':issues}; (ROOT/'reports'/'dataset_validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); raise SystemExit(bool(issues))
if __name__=='__main__': main()
