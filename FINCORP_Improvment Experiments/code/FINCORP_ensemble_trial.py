from pathlib import Path
import zipfile,io
import numpy as np,pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score,classification_report
with zipfile.ZipFile(Path(__file__).resolve().parents[1] / 'FINCORP-main.zip') as z: df=pd.read_csv(io.BytesIO(z.read('FINCORP-main/Database/FINCORP.csv')))
t=df['Complaint/ Opinion'].fillna('').astype(str).str.replace(r'http\S+',' URL ',regex=True).str.replace(r'@\w+',' USER ',regex=True);y=df['Severity level'].values
idx=np.arange(len(y));tv,te=train_test_split(idx,test_size=.3,random_state=40,stratify=y);tr,va=train_test_split(tv,test_size=1/7,random_state=40,stratify=y[tv])
w=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True,strip_accents='unicode');c=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=45000,sublinear_tf=True,dtype=np.float32)
a=w.fit_transform(t.iloc[tr]);b=w.transform(t.iloc[va]);e=w.transform(t.iloc[te]);d=c.fit_transform(t.iloc[tr]);f=c.transform(t.iloc[va]);g=c.transform(t.iloc[te]);models=[]
for cw,C,weight in [(1,.3,'balanced'),(1,.1,'balanced'),(2,.1,None),(.4,1,'balanced')]:
 m=LinearSVC(C=C,class_weight=weight,random_state=40,max_iter=2000).fit(hstack([a,d*cw],format='csr'),y[tr]);
 v=m.decision_function(hstack([b,f*cw],format='csr'));x=m.decision_function(hstack([e,g*cw],format='csr'))
 # standardized margins by validation aggregate scale
 scale=np.std(v)
 models.append((v/scale,x/scale));print('single',cw,C,weight, f1_score(y[va],m.classes_[v.argmax(1)],average='macro'),flush=True)
cls=m.classes_;candidates=[]
for p in [0,.25,.5,.75,1]:
 for j in [1,2,3]:
  v=(1-p)*models[0][0]+p*models[j][0]
  for boost in [-.15,0,.15,.3,.45]:
   # class-specific calibration: boost Accusation, penalty for noncomplaint
   vv=v.copy(); vv[:,list(cls).index('Accusation')]+=boost
   pred=cls[vv.argmax(1)];candidates.append((f1_score(y[va],pred,average='macro'),accuracy_score(y[va],pred),p,j,boost))
candidates.sort(reverse=True);print('BEST VAL',candidates[:5],flush=True)
macro,acc,p,j,boost=candidates[0]
x=(1-p)*models[0][1]+p*models[j][1];x[:,list(cls).index('Accusation')]+=boost
pred=cls[x.argmax(1)];print('TEST',accuracy_score(y[te],pred),f1_score(y[te],pred,average='macro'));print(classification_report(y[te],pred,zero_division=0))
