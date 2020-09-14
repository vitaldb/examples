import pandas as pd
import numpy as np
import vitaldb

df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보
df_trks = pd.read_csv('https://api.vitaldb.net/v2/trks')  # 트랙 목록

caseids = list(
    set(df_trks[df_trks['tname'] == 'Solar8000/ART_MBP']['caseid']) & 
    set(df_trks[df_trks['tname'] == 'BIS/BIS']['caseid']) & 
    set(df_cases[df_cases['department'] == 'General surgery']['caseid']) &
    set(df_cases[df_cases['emop'] == 'Y']['caseid'])
)
#caseids = caseids[:10]
print('Total {} cases found'.format(len(caseids)))

df = pd.DataFrame(columns=['mbp_under_65', 'bis_under_20', 'death'])
for caseid in caseids:
    print('loading {}...'.format(caseid), flush=True, end='')

    tid_mbp = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'Solar8000/ART_MBP')]['tid'].values[0]
    tid_bis = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'BIS/BIS')]['tid'].values[0]

    vals = vitaldb.load_trks([tid_mbp, tid_bis], 60)

    valid_mask = (vals[:,0] > 20) & (vals[:,0] < 150) & (vals[:,1] > 0) & (vals[:,1] < 80)
    vals = vals[valid_mask]

    if len(vals) < 10:
        print()
        continue
    
    mbp_under_65 = np.nanmean(vals[:,0] < 65) * 100
    bis_under_20 = np.nanmean(vals[:,1] < 20) * 100
    double_low = np.nanmean((vals[:,0] < 65) & (vals[:,1] < 20)) * 100

    death = df_cases[df_cases['caseid'] == caseid]['death_inhosp'].values[0] == 'Y'

    df = df.append({'mbp_under_65':mbp_under_65, 'bis_under_20':bis_under_20, 'double_low':double_low, 'death':death}, ignore_index=True)

    print('mbp<65 {:.1f}%, {}'.format(mbp_under_65, 'death' if death else ''))

df['death'] = df['death'].astype(bool)

# group comparison
print(df.groupby('death').mean())

# univariate logistic regression
import statsmodels.api as sm
for c in ['mbp_under_65', 'bis_under_20', 'double_low']:
    df['intercept'] = 1
    model = sm.Logit(df['death'], df[['intercept', c]])
    res = model.fit()
    b = res.params[c]
    pval = res.pvalues[c]
    print('{}\tb={:.3f}, exp(b)={:.3f}, pval={:.3f}'.format(c, b, np.exp(b), pval))
