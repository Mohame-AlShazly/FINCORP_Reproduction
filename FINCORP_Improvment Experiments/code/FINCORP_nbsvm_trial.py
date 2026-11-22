import zipfile,io,time
import numpy as np,pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,accuracy_score
with zipfile.ZipFile('/mnt/data/FINCORP-main.zip') as z: df=pd.read_csv(io.BytesIO(z.read('FINCORP-main/Database/FINCORP.csv')))
txt=df['Complaint/ Opinion'].fillna('').astype(str).str.replace(r'http\S+',' URL ',regex=True).str.replace(r'@\w+',' USER ',regex=True)
y=df['Severity level'].values
idx=np.arange(len(y));tv,te=train_test_split(idx,test_size=.3,random_state=40,stratify=y);tr,va=train_test_split(tv,test_size=1/7,random_state=40,stratify=y[tv]);labels=np.unique(y)
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True,strip_accents='unicode');char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=45000,sublinear_tf=True,dtype=np.float32)
WT=word.fit_transform(txt.iloc[tr]);WV=word.transform(txt.iloc[va]);WTs=word.transform(txt.iloc[te]);CT=char.fit_transform(txt.iloc[tr]);CV=char.transform(txt.iloc[va]);CTs=char.transform(txt.iloc[te])
for cw in [0.4,1.,2.]:
 X=hstack([WT,CT*cw],format='csr');V=hstack([WV,CV*cw],format='csr');T=hstack([WTs,CTs*cw],format='csr')
 for C in [.05,.2,1.]:
  for b in [False,True]:
   out=[];t0=time.time()
   for l in labels:
    z=(y[tr]==l)
    # common NB-SVM log-count ratio, one-vs-rest
    p=(X[z]>0).sum(axis=0)+1;n=(X[~z]>0).sum(axis=0)+1
    p=np.asarray(p).ravel()/float(z.sum()+1);n=np.asarray(n).ravel()/float((~z).sum()+1)
    r=np.log(p/n)
    m=LinearSVC(C=C,class_weight='balanced' if b else None,max_iter=1000,random_state=40)
    m.fit(X.multiply(r).tocsr(),z.astype(int))
    out.append(m.decision_function(V.multiply(r).tocsr()))
   pred=labels[np.asarray(out).argmax(axis=0)]
   acc=accuracy_score(y[va],pred);f1=f1_score(y[va],pred,average='macro',zero_division=0)
   print('VAL',cw,C,b,round(acc,4),round(f1,4),'time',round(time.time()-t0,1),flush=True)
