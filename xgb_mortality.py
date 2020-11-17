import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve, auc
#!pip install -q xgboost

df = pd.read_csv('https://api.vitaldb.net/cases')
df['sex'] = (df['sex'] == 'M')

# remove categorical variables
df = df.drop(['department','optype', 'dx', 'opname', 'approach', 'position', 'ane_type', 'cormack', 'airway', 'tubesize', 'dltubesize', 'lmasize', 'lmasize', 'preop_ecg', 'preop_pft', 'iv1', 'iv2', 'aline1', 'aline2', 'cline1', 'cline2'], axis=1)
df = df.astype(float)

# remove outcome variables
y = df['death_inhosp'].values.flatten()
y_old = df['asa'].values.flatten()
df = df.drop(['death_inhosp','caseid','asa','casestart','caseend','anestart','aneend','opstart','opend','adm','dis','devices','tracks','casedur','filesize'], axis=1)

# input variables
x = df.values

nsamp = len(x)
ntest = int(nsamp * 0.1)
ntrain = nsamp - ntest
x_test = x[-ntest:, :]
y_test = y[-ntest:]
y_old = y_old[-ntest:]
x_train = x[:-ntest, :]
y_train = y[:-ntest]

print('{} (event {:.1f}%) training, {} testing (event {:.1f} %) samples'.format(ntrain, np.mean(y_train) * 100 , ntest, np.mean(y_test) * 100))

model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=50, max_depth=5).fit(x_train, y_train)
y_pred = model.predict_proba(x_test)[:, 1].ravel()

# test the final model
fpr, tpr, thvals = roc_curve(y_test, y_pred)
auroc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auprc = auc(recall, precision)
optimal_idx = np.argmax(tpr - fpr)
thval = thvals[optimal_idx]

print('optimal thval: {}'.format(thval))
y_pred = y_pred > thval
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('new auroc: {:.3f}, auprc: {:.3f}\tacc: {:.3f}\tf1: {:.3f}\tTN {}\tfp {}\tfn {}\tTP {}'.format(auroc, auprc, acc, f1, tn, fp, fn, tp))

old_y_valid_mask = ~np.isnan(y_old)
fpr_old, tpr_old, _ = roc_curve(y_test[old_y_valid_mask], y_old[old_y_valid_mask])
auroc_old = auc(fpr_old, tpr_old)
precision_old, recall_old, _ = precision_recall_curve(y_test[old_y_valid_mask], y_old[old_y_valid_mask])
auprc_old = auc(recall_old, precision_old)

print('old auroc: {:.3f}, auprc: {:.3f}'.format(auroc_old, auprc_old), flush=True)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label='NEW AUROC = {:0.3f}'.format(auroc))
plt.plot(fpr_old, tpr_old, label='OLD AUROC = {:0.3f}'.format(auroc_old))
plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
