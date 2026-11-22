Jin & Aletras (2021) complaint-severity CSV, prepared for FINCORP transfer learning.

Original uploaded CSV has NO HEADER; parsed using header=None.
Source dataset: https://archive.org/details/complaint_severity_data
Official paper: https://aclanthology.org/2021.naacl-main.180/
Original code: https://github.com/mali726/Complaint-Severity

Files:
- jin_aletras_all_labeled.csv: all 3,449 tweets, including 2,217 noncomplaints
- jin_aletras_complaints_only.csv: 1,232 labeled complaints, for stage 1 4-class training
- source original CSV in the zip for traceability

Columns: source_tweet_id, text, complaint_label, severity_code, domain, Severity level
Severity label mapping: 0 Non-Complaint; 1 No explicit reproach; 2 Disapproval; 3 Accusation; 4 Blame.

IMPORTANT: Before training, normalize texts and check exact/near duplicates against ALL FINCORP partitions, excluding any overlapping external records; preserve FINCORP train/val/test exactly. Do not train on FINCORP validation or test data. Verify mapping against original authors' documentation and sample annotations.

In Kaggle, attach this package as a new dataset. Read it from /kaggle/input/... rather than external HTTP requests.
