"""After actual human verification, create records compatible with Step 8 teacher_adapter.py.

B: reviewer must check the proposed class and every coordinate; this CLI only checks schema.
A: a human-verified full-resolution binary mask must be provided separately.
"""
import argparse
import json
from pathlib import Path
from PIL import Image
from distillation.teacher_adapter import sha256
from distillation.vlm_teacher_client import parse_teacher_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--reviewer',required=True)
    p.add_argument('--teacher-root',default='datasets/teacher_approved')
    p.add_argument('--mask',help='For Task A: path to REVIEWED binary image mask')
    p.add_argument('--approve',action='store_true',help='Affirm manual review of image, labels and boxes/mask')
    args=p.parse_args()
    if not args.approve or not args.reviewer.strip():
        raise ValueError('Explicit --approve and named --reviewer required after manual visual review')
    candidate_path=Path(args.candidate)
    record=json.loads(candidate_path.read_text(encoding='utf-8'))
    image=Path(record['image_path']); task=record['task']; split=record['split']
    if task not in ('A','B') or split not in ('train','val','test'):
        raise ValueError('Unknown task/split')
    if not image.is_file() or sha256(image)!=record['image_sha256']:
        raise ValueError('Image missing or SHA-256 mismatch')
    out={'task':task,'source_model':record['source_model'],'reviewer':args.reviewer,
         'approved':True,'image_sha256':record['image_sha256'],
         'annotation_source':'human_reviewed_vlm_candidate',
         'candidate_path':str(candidate_path)}
    if task=='A':
        if not args.mask:raise ValueError('Task A requires a manually reviewed --mask')
        mask=Path(args.mask).resolve()
        with Image.open(image) as im: image_size=im.size
        with Image.open(mask) as m:
            if m.size != image_size:raise ValueError('Mask resolution must match input image')
        out['mask_path']=str(mask)
        out['mask_sha256']=sha256(mask)
    else:
        objects=[]
        for obj in record['objects']:
            c=obj['bbox_xyxy_normalized'];x1,y1,x2,y2=c
            if type(obj['class_id']) is not int or not (0<=x1<x2<=1 and 0<=y1<y2<=1):
                raise ValueError('Bad object coordinate/class')
            objects.append({'class_id':obj['class_id'],
                            'bbox_xywh_normalized':[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]})
        out['objects']=objects
    destination=Path(args.teacher_root)/task/split/(image.stem+'.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(f'Approved record exists: {destination}')
    destination.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Approved record written:',destination)

if __name__=='__main__':main()
