"""Exploratory severity models. 60/10/30 stratified split seed 40.
Choose using validation only, report untouched test for selected model and original baseline.
CPU only, no pretrained transformer. Research experiment, not author's reproduction.
"""
import zipfile,io,time,json
import numpy as np,pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
out=Path(__file__).resolve().parents[1] / 'outputs'
out.mkdir(parents=True, exist_ok=True)
archive=Path(__file__).resolve().parents[1] / 'FINCORP-main.zip'
with zipfile.ZipFile(archive) as z: df=pd.read_csv(io.BytesIO(z.read('FINCORP-main/Database/FINCORP.csv')))
texts=df['Complaint/ Opinion'].fillna('').astype(str).str.replace(r'http\S+',' URL ',regex=True).str.replace(r'@\w+',' USER ',regex=True)
y=df['Severity level'].astype(str).values
idx=np.arange(len(df)); trval,te=train_test_split(idx,test_size=.3,random_state=40,stratify=y)
tr,va=train_test_split(trval,test_size=1/7,random_state=40,stratify=y[trval])
print('SPLIT',len(tr),len(va),len(te),'COUNTS',pd.Series(y).value_counts().to_dict(),flush=True)
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True,strip_accents='unicode')
char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=45000,sublinear_tf=True,dtype=np.float32)
Wtr=word.fit_transform(texts.iloc[tr]);Wva=word.transform(texts.iloc[va]);Wte=word.transform(texts.iloc[te]);Ctr=char.fit_transform(texts.iloc[tr]);Cva=char.transform(texts.iloc[va]);Cte=char.transform(texts.iloc[te]);
Xtr={};Xva={};Xte={}
for cw in [.4,1.,2.]:
 Xtr[cw]=hstack([Wtr,Ctr*cw],format='csr');Xva[cw]=hstack([Wva,Cva*cw],format='csr');Xte[cw]=hstack([Wte,Cte*cw],format='csr')
configs=[]
for cw in [.4,1.,2.]:
 for C in [.1,.3,1.]:
  for balance in [None,'balanced']:
   configs.append(('svc',cw,C,balance))
rows=[]; t0=time.time()
for i,(kind,cw,C,balance) in enumerate(configs):
 if kind=='svc': m=LinearSVC(C=C,class_weight=balance,max_iter=2500,random_state=40)
 else: m=LogisticRegression(C=C,class_weight=balance,max_iter=400,solver='lbfgs')
 m.fit(Xtr[cw],y[tr]);pred=m.predict(Xva[cw]);a=accuracy_score(y[va],pred);f=f1_score(y[va],pred,average='macro',zero_division=0)
 rows.append({'model':kind,'char_weight':cw,'C':C,'class_weight':str(balance),'val_accuracy':a,'val_macro_f1':f})
 if i%8==0: print('CHECK',i,'score',round(f,4),'elapsed',round(time.time()-t0,1),flush=True)
res=pd.DataFrame(rows).sort_values(['val_macro_f1','val_accuracy'],ascending=False)
res.to_csv(out/'FINCORP_exploratory_validation_search.csv',index=False)
print('TOP 10\n',res.head(10).to_string(index=False),flush=True)
best=res.iloc[0];kind=best['model'];cw=float(best['char_weight']);C=float(best['C']);bal=None if best['class_weight']=='None' else 'balanced'
selected=LinearSVC(C=C,class_weight=bal,max_iter=2500,random_state=40) if kind=='svc' else LogisticRegression(C=C,class_weight=bal,max_iter=400,solver='lbfgs')
selected.fit(Xtr[cw],y[tr]); pred=selected.predict(Xte[cw]);
standard=LinearSVC(C=1,class_weight=None,max_iter=2500,random_state=40);standard.fit(Xtr[1.],y[tr]);base=standard.predict(Xte[1.])
rows_test=[]
for name,p in [('original_standard_svm',base),('new_validation_selected',pred)]:
 r={'model':name,'test_accuracy':accuracy_score(y[te],p),'test_macro_f1':f1_score(y[te],p,average='macro',zero_division=0)};rows_test.append(r);print('TEST',r,flush=True)
 labels=sorted(set(y));pd.DataFrame(confusion_matrix(y[te],p,labels=labels),index=labels,columns=labels).to_csv(out/f'FINCORP_{name}_confusion.csv')
 (out/f'FINCORP_{name}_classification.txt').write_text(classification_report(y[te],p,labels=labels,zero_division=0))
 pd.DataFrame({'original_row':te,'true':y[te],'predicted':p}).to_csv(out/f'FINCORP_{name}_predictions.csv',index=False)
pd.DataFrame(rows_test).to_csv(out/'FINCORP_exploratory_test_results.csv',index=False)
(out/'FINCORP_exploratory_selected.json').write_text(json.dumps({'kind':kind,'char_weight':cw,'C':C,'class_weight':str(bal),'val_macro_f1':float(best['val_macro_f1'])},indent=2))
print('total_seconds',time.time()-t0,flush=True)
