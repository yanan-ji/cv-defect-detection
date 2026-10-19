#!/usr/bin/env python3
"""Convert Pascal VOC NEU-DET annotations to YOLO format with a stratified 80/10/10 split."""
from __future__ import annotations
import argparse, shutil, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import CLASSES, IMAGE_SUFFIXES, image_files, set_seed, write_json, xyxy_to_yolo

def find_image(xml: Path, image_index: dict[str, Path]) -> Path | None:
    for p in [xml.with_suffix(ext) for ext in IMAGE_SUFFIXES]:
        if p.exists(): return p
    stem = xml.stem
    return image_index.get(stem)

def parse_xml(path: Path):
    root = ET.parse(path).getroot(); size = root.find('size')
    width, height = int(size.findtext('width')), int(size.findtext('height'))
    rows=[]
    for obj in root.findall('object'):
        name=obj.findtext('name','').strip()
        if name not in CLASSES: raise ValueError(f'{path}: unsupported class {name!r}')
        b=obj.find('bndbox'); box=[float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f'{path}: invalid bbox {box} for {width}x{height}')
        rows.append((CLASSES.index(name), xyxy_to_yolo(box,width,height)))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,default=ROOT/'data/raw'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); set_seed(args.seed)
    xmls=sorted(args.raw.rglob('*.xml'))
    if not xmls: raise SystemExit('No VOC XML files found under data/raw. Supply the NEU-DET detection release, not a classification-only dataset.')
    image_index = {p.stem: p for p in image_files(args.raw)}
    records=[]; by_class=defaultdict(list)
    for x in xmls:
        image=find_image(x, image_index)
        if not image: continue
        labels=parse_xml(x)
        if labels:
            primary=labels[0][0]; records.append((x,image,labels)); by_class[primary].append((x,image,labels))
    import random; rng=random.Random(args.seed); splits={k:[] for k in ('train','val','test')}
    for group in by_class.values():
        rng.shuffle(group); n=len(group); a=round(n*.8); b=round(n*.9)
        for split, items in [('train',group[:a]),('val',group[a:b]),('test',group[b:])]: splits[split].extend(items)
    dst=ROOT/'data/neu_det'
    for split in splits:
        for kind in ('images','labels'):
            target=dst/kind/split
            if target.exists(): shutil.rmtree(target)
            target.mkdir(parents=True)
        for _, image, labels in splits[split]:
            shutil.copy2(image,dst/'images'/split/image.name)
            (dst/'labels'/split/f'{image.stem}.txt').write_text('\n'.join(f'{c} '+ ' '.join(f'{v:.6f}' for v in box) for c,box in labels)+'\n')
    manifest=[{'split':s,'image':str(i.relative_to(args.raw)),'objects':len(l)} for s,rs in splits.items() for _,i,l in rs]
    write_json(dst/'split_manifest.json',manifest); write_json(dst/'prepare_summary.json',{'seed':args.seed,'images':len(manifest),'by_split':{k:len(v) for k,v in splits.items()},'by_class':Counter(CLASSES[l[0][0]] for _,_,l in records)})
    print(f'Prepared {len(manifest)} images at {dst}')
if __name__=='__main__': main()
