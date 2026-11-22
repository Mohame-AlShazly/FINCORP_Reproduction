"""Independent CPU benchmark: separate from the authors' RoBERTa reproduction.
Split 60/10/30 stratified, seed 40. Select by validation macro-F1 only.
"""
import json, zipfile, io, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.sparse import hstack
out=Path('/mnt/data')
with zipfile.ZipFile(out/'FINCORP-main.zip') as z:
    df=pd.read_csv(io.BytesIO(z.read('FINCORP-main/Database/FINCORP.csv')))
texts=df['Complaint/ Opinion'].fillna('').astype(str).str.replace(r'http\S+', ' URL ',regex=True).str.replace(r'@\w+',' USER ',regex=True)
y=df['Severity level'].astype(str)
all_idx=np.arange(len(df))
trainval_idx, test_idx=train_test_split(all_idx,test_size=.3,random_state=40,stratify=y)
train_idx, val_idx=train_test_split(trainval_idx,test_size=1/7,random_state=40,stratify=y.iloc[trainval_idx])
print('split',len(train_idx),len(val_idx),len(test_idx), flush=True)
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=35000,sublinear_tf=True,strip_accents='unicode')
char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=45000,sublinear_tf=True,dtype=np.float32)
Wtr=word.fit_transform(texts.iloc[train_idx]);Wval=word.transform(texts.iloc[val_idx]);Wtest=word.transform(texts.iloc[test_idx]); print('word',Wtr.shape, flush=True)
Ctr=char.fit_transform(texts.iloc[train_idx]);Cval=char.transform(texts.iloc[val_idx]);Ctest=char.transform(texts.iloc[test_idx]);print('char',Ctr.shape, flush=True)
Xtr=hstack([Wtr,Ctr],format='csr'); Xval=hstack([Wval,Cval],format='csr'); Xtest=hstack([Wtest,Ctest],format='csr');
configs=[('standard_svm_C1', 1.0,None),('balanced_svm_C1',1.0,'balanced'),('balanced_svm_C0.3',.3,'balanced'),('balanced_svm_C3',3.0,'balanced'),('standard_svm_C3',3.0,None)]
rows=[]
for name,C,weight in configs:
    t=time.time();model=LinearSVC(C=C,class_weight=weight,random_state=40,max_iter=4000)
    model.fit(Xtr,y.iloc[train_idx]);p=model.predict(Xval)
    r={'model':name,'val_accuracy':accuracy_score(y.iloc[val_idx],p),'val_macro_f1':f1_score(y.iloc[val_idx],p,average='macro'),'seconds':round(time.time()-t,1)}
    print(r,flush=True); rows.append(r)
selected=max(rows,key=lambda x:(x['val_macro_f1'],x['val_accuracy']))
# Evaluating untouched test only for original default and selected (two prespecified reported comparisons)
testrows=[]
for name,C,weight in [configs[0],next(x for x in configs if x[0]==selected['model'])]:
    m=LinearSVC(C=C,class_weight=weight,random_state=40,max_iter=4000)
    m.fit(Xtr,y.iloc[train_idx]); pred=m.predict(Xtest)
    r={'model':name,'test_accuracy':accuracy_score(y.iloc[test_idx],pred),'test_macro_f1':f1_score(y.iloc[test_idx],pred,average='macro')}
    print('TEST',r,flush=True);testrows.append(r)
    labels=sorted(y.unique())
    (out/f'FINCORP_{name}_classification_report.txt').write_text(classification_report(y.iloc[test_idx],pred,labels=labels,zero_division=0))
    pd.DataFrame(confusion_matrix(y.iloc[test_idx],pred,labels=labels),index=labels,columns=labels).to_csv(out/f'FINCORP_{name}_confusion_matrix.csv')
    pd.DataFrame({'row_index':test_idx,'true_severity':y.iloc[test_idx].values,'predicted_severity':pred}).to_csv(out/f'FINCORP_{name}_test_predictions.csv',index=False)
pd.DataFrame(rows).to_csv(out/'FINCORP_severity_validation_comparison.csv',index=False)
pd.DataFrame(testrows).to_csv(out/'FINCORP_severity_test_results.csv',index=False)
print('SELECTED',selected['model'],flush=True)
