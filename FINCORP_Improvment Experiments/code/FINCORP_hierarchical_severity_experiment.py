"""Supplementary FINCORP CPU experiment: hierarchical complaint -> severity classifier.
Same reproducible 60/10/30 stratified split as previous independent CPU baseline.
Hyperparameters chosen by validation. Test is supplementary/exploratory because the same
held-out sample has already been consulted in the previous CPU comparison.
"""
import io, zipfile, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.sparse import hstack
out=Path(__file__).resolve().parents[1] / 'outputs'
out.mkdir(parents=True, exist_ok=True)
archive=Path(__file__).resolve().parents[1] / 'FINCORP-main.zip'
with zipfile.ZipFile(archive) as z:
    d=pd.read_csv(io.BytesIO(z.read('FINCORP-main/Database/FINCORP.csv')))
txt=d['Complaint/ Opinion'].fillna('').astype(str).str.replace(r'http\S+', ' URL ',regex=True).str.replace(r'@\w+',' USER ',regex=True)
y=d['Severity level'].astype(str).to_numpy(); idx=np.arange(len(d))
trainval, test=train_test_split(idx,test_size=.3,random_state=40,stratify=y)
train,val=train_test_split(trainval,test_size=1/7,random_state=40,stratify=y[trainval])
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True,strip_accents='unicode')
char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=45000,sublinear_tf=True,dtype=np.float32)
wtr=word.fit_transform(txt.iloc[train]);wv=word.transform(txt.iloc[val]);wt=word.transform(txt.iloc[test]);
ctr=char.fit_transform(txt.iloc[train]);cv=char.transform(txt.iloc[val]);ct=char.transform(txt.iloc[test]);
results=[]; configs=[]
def evaluate(name, pv):
    a=accuracy_score(y[val],pv);f=f1_score(y[val],pv,average='macro');results.append({'model':name,'val_accuracy':a,'val_macro_f1':f});print(name,round(a,4),round(f,4),flush=True)
# hierarchical SVM, train 1st on binary complaint, 2nd on severity only for complaints
for cw in [0.5,1.0,1.5,2.0]:
  for c1 in [0.3,1.0,3.0]:
    for c2 in [0.1,0.3,1.0,3.0]:
      # optimize reuse later if needed
      configs.append((cw,c1,c2))
for cw in [0.5,1.0,1.5,2.0]:
  Xtr=hstack([wtr,ctr*cw],format='csr');Xv=hstack([wv,cv*cw],format='csr')
  binary_models={}; sev_models={}
  for c1 in [0.3,1.0,3.0]:
    b=LinearSVC(C=c1,random_state=40,max_iter=3000);b.fit(Xtr,y[train]!='Non-Complaint');binary_models[c1]=b
  complaint_mask=(y[train]!='Non-Complaint')
  for c2 in [0.1,0.3,1.0,3.0]:
    m=LinearSVC(C=c2,class_weight='balanced',random_state=40,max_iter=3000);m.fit(Xtr[complaint_mask],y[train][complaint_mask]);sev_models[c2]=m
  for c1,b in binary_models.items():
    pred_binary=b.predict(Xv)
    for c2,m in sev_models.items():
      pv=np.where(pred_binary,m.predict(Xv),'Non-Complaint')
      evaluate(f'hier_char{cw}_bin{c1}_sev{c2}',pv)
# direct model variations weight tuning
for cw in [0.35,0.7,1.3,1.7]:
 Xtr=hstack([wtr,ctr*cw],format='csr'); Xv=hstack([wv,cv*cw],format='csr')
 for c in [.1,.3,1,3]:
  m=LinearSVC(C=c,class_weight='balanced',random_state=40,max_iter=3000);m.fit(Xtr,y[train]);p=m.predict(Xv)
  evaluate(f'direct_char{cw}_C{c}',p)
res=pd.DataFrame(results);res.sort_values('val_macro_f1',ascending=False).to_csv(out/'FINCORP_hierarchical_validation_results.csv',index=False)
best=res.sort_values(['val_macro_f1','val_accuracy'],ascending=False).iloc[0]
print('BEST',best.to_dict(),flush=True)
# Test once for selected config (other tested algorithms on same split; exploratory only)
name=best['model']; parts=name.split('_')
if name.startswith('hier_'):
 cw=float(parts[1][4:]);c1=float(parts[2][3:]);c2=float(parts[3][3:])
 Xtr=hstack([wtr,ctr*cw],format='csr');Xt=hstack([wt,ct*cw],format='csr')
 b=LinearSVC(C=c1,random_state=40,max_iter=3000).fit(Xtr,y[train]!='Non-Complaint')
 mask=y[train]!='Non-Complaint'
 m=LinearSVC(C=c2,class_weight='balanced',random_state=40,max_iter=3000).fit(Xtr[mask],y[train][mask])
 pred=np.where(b.predict(Xt),m.predict(Xt),'Non-Complaint')
else:
 cw=float(parts[1][4:]);c=float(parts[2][1:])
 Xtr=hstack([wtr,ctr*cw],format='csr');Xt=hstack([wt,ct*cw],format='csr')
 m=LinearSVC(C=c,class_weight='balanced',random_state=40,max_iter=3000).fit(Xtr,y[train]);pred=m.predict(Xt)
report={'model':name,'test_accuracy':accuracy_score(y[test],pred),'test_macro_f1':f1_score(y[test],pred,average='macro')}
print('TEST',report,flush=True)
pd.DataFrame([report]).to_csv(out/'FINCORP_hierarchical_selected_test.csv',index=False)
labels=sorted(set(y));pd.DataFrame(confusion_matrix(y[test],pred,labels=labels),index=labels,columns=labels).to_csv(out/'FINCORP_hierarchical_selected_confusion.csv')
(out/'FINCORP_hierarchical_selected_report.txt').write_text(classification_report(y[test],pred,labels=labels,zero_division=0))
