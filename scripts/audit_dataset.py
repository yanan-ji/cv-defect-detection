#!/usr/bin/env python3
"""Non-mutating integrity and split-leakage audit for processed NEU-DET."""
from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/neu_det'; NAMES=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def phash(p):
 x=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
 if x is None:return None
 x=cv2.resize(x,(32,32)); d=cv2.dct(np.float32(x)); z=d[:8,:8];return ''.join('1' if v>np.median(z) else '0' for v in z.flat)
def main():
 rows=[]; exact=defaultdict(list); hashes={}; counts={}; invalid=[]
 for split in ('train','val','test'):
  ic=Counter(); oc=Counter(); n=0
  for im in sorted((DATA/'images'/split).glob('*')):
   label=DATA/'labels'/split/f'{im.stem}.txt'; x=cv2.imread(str(im));
   if x is None or not label.exists():invalid.append(str(im));continue
   classes=[]
   for line in label.read_text().splitlines():
    q=line.split()
    try:
     c=int(q[0]);vals=list(map(float,q[1:]));assert len(q)==5 and 0<=c<6 and all(0<v<=1 for v in vals)
     classes.append(c);oc[NAMES[c]]+=1
    except Exception:invalid.append(f'{label}: {line}')
   for c in set(classes):ic[NAMES[c]]+=1
   digest=sha(im); exact[digest].append((split,im.name)); hashes[f'{split}/{im.name}']=phash(im);n+=1
  counts[split]={'images':n,'images_per_class':dict(ic),'instances_per_class':dict(oc)}
 cross_exact=[v for v in exact.values() if len({x[0] for x in v})>1]
 keys=list(hashes); near=[]
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if a.split('/')[0]==b.split('/')[0]:continue
   d=sum(x!=y for x,y in zip(hashes[a],hashes[b]))
   if d<=4:near.append({'a':a,'b':b,'hamming':d})
 report={'dataset_hashes':{'split_manifest.json':sha(DATA/'split_manifest.json'),'neu_det.yaml':sha(ROOT/'configs/neu_det.yaml')},'split_counts':counts,'invalid_or_missing':invalid,'cross_split_exact_duplicates':cross_exact,'cross_split_near_duplicates_phash_hamming_le_4':near,'method':'SHA-256 exact duplicate detection; 64-bit DCT perceptual hash threshold <=4 across different splits.'}
 (ROOT/'reports'/'dataset_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
