"""UrbanSound8K milestone pipeline. Run python pipeline.py --help."""
import argparse,csv,hashlib,json,os,sys,platform,time
from pathlib import Path
from collections import Counter,defaultdict
from project_paths import ROOT, DATA_DIR, DATA_RESULTS, FEATURES_DIR, TRAIN_RESULTS
LABELS=['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
CONFIG={'sample_rate':22050,'n_mfcc':20,'n_fft':2048,'hop_length':512,'n_mels':128,'seed':42}

def save_json(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2),encoding='utf-8')

def manifest(index):
 import re
 rows=[]
 for key,value in json.loads(Path(index).read_text())['clips'].items():
  rel,md5=value['audio'];stem=Path(rel).stem;label=int(stem.split('-')[1]);fold=int(re.search(r'fold(\d+)',rel)[1])
  rows.append({'clip_id':key,'path':rel,'fsID':stem.split('-')[0],'fold':fold,'classID':label,'class':LABELS[label],'md5':md5})
 rows.sort(key=lambda r:(r['fold'],r['path']))
 if len(set(r['clip_id'] for r in rows))!=len(rows):raise ValueError('Duplicate clip IDs')
 groups=defaultdict(set)
 for r in rows:groups[r['fsID']].add(r['fold'])
 # Preserve official folds, but report source-recording overlap explicitly.
 if set(r['fold'] for r in rows)!=set(range(1,11)):raise ValueError('Expected ten official folds')
 return rows

def read_rows(file):
 with open(file,newline='',encoding='utf-8') as f:return [{**r,'fold':int(r['fold']),'classID':int(r['classID'])} for r in csv.DictReader(f)]

def audit(args):
 rows=manifest(args.index);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 with open(out/'manifest.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 stats={'source':'Soundata UrbanSound8K index 1.0 on Zenodo','samples':len(rows),'class_counts':dict(Counter(r['class'] for r in rows)),'fold_counts':dict(Counter(r['fold'] for r in rows)),'recording_groups':len({r['fsID'] for r in rows}),'recording_groups_crossing_folds':{g:sorted({r['fold'] for r in rows if r['fsID']==g}) for g in {r['fsID'] for r in rows} if len({r['fold'] for r in rows if r['fsID']==g})>1},'audio_status':'Manifest only; audio availability checked separately'}
 save_json(out/'data_audit.json',stats);print(json.dumps(stats,indent=2))

def label_baseline(args):
 import numpy as np
 from sklearn.dummy import DummyClassifier
 from sklearn.metrics import accuracy_score,f1_score
 rows=read_rows(args.manifest);y=np.array([r['classID'] for r in rows]);fold=np.array([r['fold'] for r in rows]);scores=[]
 for k in range(1,11):
  train=fold!=k;test=fold==k;m=DummyClassifier(strategy='most_frequent').fit(np.zeros((train.sum(),1)),y[train]);pred=m.predict(np.zeros((test.sum(),1)))
  scores.append({'test_fold':k,'train_n':int(train.sum()),'test_n':int(test.sum()),'predicted_class':LABELS[int(pred[0])],'accuracy':accuracy_score(y[test],pred),'macro_f1':f1_score(y[test],pred,labels=list(range(10)),average='macro',zero_division=0)})
 result={'scope':'Full official label manifest. No audio features or learned audio classifier used. Fixed majority baseline; not used for tuning.','folds':scores,'mean_accuracy':float(np.mean([r['accuracy'] for r in scores])),'std_accuracy':float(np.std([r['accuracy'] for r in scores],ddof=1)),'mean_macro_f1':float(np.mean([r['macro_f1'] for r in scores])),'std_macro_f1':float(np.std([r['macro_f1'] for r in scores],ddof=1))}
 save_json(args.out,result);print(json.dumps(result,indent=2))

def feature_vector(file):
 import numpy as np,librosa
 y,sr=librosa.load(file,sr=CONFIG['sample_rate'],mono=True)
 if not len(y) or not np.isfinite(y).all() or np.mean(y*y)<=0:raise ValueError('Empty, nonfinite or silent recording')
 if len(y)<CONFIG['n_fft']:y=np.pad(y,(0,CONFIG['n_fft']-len(y)))
 mf=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=CONFIG['n_mfcc'],n_fft=CONFIG['n_fft'],hop_length=CONFIG['hop_length'],n_mels=CONFIG['n_mels'])
 v=np.concatenate([mf.mean(axis=1),mf.std(axis=1)]).astype('float32')
 if not np.isfinite(v).all():raise ValueError('Nonfinite features')
 return v

def extract(args):
 import numpy as np,soundfile as sf
 rows=read_rows(args.manifest);audio=Path(args.audio);cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
 selected=[r for r in rows if (audio/r['path']).is_file()]
 if args.limit:selected=selected[:args.limit]
 if not selected:raise ValueError('No audio files found. Point --audio to UrbanSound8K root containing audio/fold1.')
 old=cache/'config.json'
 if old.exists() and json.loads(old.read_text())!=CONFIG:raise ValueError('Cache settings changed; use a new cache directory')
 save_json(old,CONFIG);features=[];ids=[];headers=[];errors=[]
 for i,r in enumerate(selected):
  path=audio/r['path'];dest=cache/(r['clip_id']+'.npy')
  try:
   digest=hashlib.md5(path.read_bytes()).hexdigest()
   if digest!=r['md5']:raise ValueError('Audio checksum mismatch')
   info=sf.info(path)
   if dest.exists():v=np.load(dest,allow_pickle=False)
   else:v=feature_vector(path);np.save(dest,v)
   if v.shape!=(40,) or not np.isfinite(v).all():raise ValueError('Invalid cached feature')
   features.append(v);ids.append(r['clip_id']);headers.append({'clip_id':r['clip_id'],'sample_rate':info.samplerate,'channels':info.channels,'duration':info.duration,'fold':r['fold'],'class':r['class']})
  except Exception as e:errors.append({'clip_id':r['clip_id'],'error':str(e)})
  if (i+1)%25==0:print(f'Extracted {i+1}/{len(selected)}',flush=True)
 if features:np.savez_compressed(cache/'features.npz',X=np.stack(features),ids=np.array(ids))
 save_json(cache/'audio_audit.json',{'expected':len(rows),'available':sum((audio/r['path']).is_file() for r in rows),'processed':len(selected),'succeeded':len(ids),'failed':errors,'audio_headers':headers,'scope':'Full dataset' if len(ids)==len(rows) else 'Partial audio pipeline check; not a representative benchmark'})
 print(f'{len(ids)} successful, {len(errors)} failed. Feature dimension: 40',flush=True)
 if errors:raise RuntimeError('Some clips failed; inspect audio_audit.json')

def train(args):
 import numpy as np
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.svm import SVC
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.dummy import DummyClassifier
 from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
 rows=read_rows(args.manifest);data=np.load(args.features,allow_pickle=False);lookup={r['clip_id']:r for r in rows}
 ids=data['ids'].tolist();X=data['X'];
 if len(ids)!=len(set(ids)) or len(X)!=len(ids):raise ValueError('Duplicate IDs or feature length mismatch')
 meta=[lookup[k] for k in ids];y=np.array([r['classID'] for r in meta]);fold=np.array([r['fold'] for r in meta])
 # Development run reserves fold 10 completely. Fold 9 is validation, folds 1-8 training.
 if args.mode=='development':
  mask=fold<=9;X=X[mask];y=y[mask];fold=fold[mask];meta=[r for r in meta if r['fold']<=9]
  expected={r['clip_id'] for r in rows if r['fold']<=9}
 else:expected=set(lookup)
 if {r['clip_id'] for r in meta}!=expected:raise ValueError('Incomplete audio features for this mode. Refusing a misleading partial-dataset score.')
 if args.mode=='development':
  cases=[(0,fold<=8,fold==9,None)]
 else:
  cases=[]
  for k in range(1,11):
   v=k%10+1;cases.append((k,(fold!=k)&(fold!=v),fold==v,fold==k))
 scores=[]
 groups=np.array([r['fsID'] for r in meta])
 for k,tr,va,te in cases:
  # Purge recording IDs shared with validation/test without moving test samples.
  blocked=set(groups[va]) | (set(groups[te]) if te is not None else set())
  tr=tr & ~np.isin(groups,list(blocked))
  if te is not None:va=va & ~np.isin(groups,groups[te])
  assert not (tr&va).any()
  if te is not None:assert not (te&(tr|va)).any()
  configs={'svm':[{'C':c,'gamma':'scale'} for c in [.1,1,10]],'random_forest':[{'max_depth':d,'min_samples_leaf':2} for d in [None,20]]}
  makers={'svm':lambda c:make_pipeline(StandardScaler(),SVC(**c)),'random_forest':lambda c:RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=2,**c)}
  fitted={};chosen={}
  for name,grid in configs.items():
   best=(-1,None,None)
   for conf in grid:
    m=makers[name](conf).fit(X[tr],y[tr]);f=f1_score(y[va],m.predict(X[va]),labels=list(range(10)),average='macro',zero_division=0)
    if f>best[0]:best=(f,m,conf)
   fitted[name]=best[1];chosen[name]=best[2]
  fitted['majority']=DummyClassifier(strategy='most_frequent').fit(X[tr],y[tr]);chosen['majority']={}
  for name,m in fitted.items():
   if te is None:ev=va
   else:
    ev=te;refit=(~te)&~np.isin(groups,groups[te]);m.fit(X[refit],y[refit])
   pred=m.predict(X[ev]);scores.append({'model':name,'test_fold':k if k else None,'validation_fold':9 if not k else k%10+1,'scope':'validation, not independent test' if te is None else 'outer test','parameters':chosen[name],'n':int(ev.sum()),'accuracy':accuracy_score(y[ev],pred),'macro_f1':f1_score(y[ev],pred,labels=list(range(10)),average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y[ev],pred,labels=list(range(10))).tolist()})
   print(name,k,scores[-1]['accuracy'],flush=True)
 save_json(args.out,{'mode':args.mode,'protocol':'Official test folds, with source-recording overlap purged from training/validation; not identical to unpurged literature protocol','config':CONFIG,'python':platform.python_version(),'results':scores})

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('audit');a.add_argument('--index',default=DATA_DIR/'official_index.json');a.add_argument('--out',default=DATA_RESULTS);a.set_defaults(func=audit)
 a=sub.add_parser('baseline');a.add_argument('--manifest',default=DATA_RESULTS/'manifest.csv');a.add_argument('--out',default=TRAIN_RESULTS/'legacy_majority_baseline.json');a.set_defaults(func=label_baseline)
 a=sub.add_parser('extract');a.add_argument('--manifest',default=DATA_RESULTS/'manifest.csv');a.add_argument('--audio',required=True);a.add_argument('--cache',default=FEATURES_DIR);a.add_argument('--limit',type=int);a.set_defaults(func=extract)
 a=sub.add_parser('train');a.add_argument('--manifest',default=DATA_RESULTS/'manifest.csv');a.add_argument('--features',default=FEATURES_DIR/'features.npz');a.add_argument('--mode',choices=['development','final'],default='development');a.add_argument('--out',default=TRAIN_RESULTS/'legacy_model_results.json');a.set_defaults(func=train)
 args=p.parse_args();args.func(args)
if __name__=='__main__':main()
