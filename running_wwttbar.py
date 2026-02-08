import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

path='./Data/'
# plt.style.use('/Users/camwilson/Documents/plot_styles/custom_plot_style.txt')


def run_bdt(classifier, new_df, feature_names):

    if 'Scale Factor' in new_df.keys():
        new_df['Weight'] = new_df['Scale Factor'] * new_df['Reweight']
    x_input = new_df[feature_names]
    
    probabilities = classifier.predict_proba(x_input)
    bdt_scores = probabilities[:, 1]

    results_df = new_df.copy()
    results_df['bdt_score'] = bdt_scores
    
    return results_df

def find_best_significance_gt(dataframe, step=0.01):
    
    # Defining how many thresholds there should be in the bdt scores
    list_of_thresholds = np.arange(0, 1.0, step)
    significances = []

    # Iterating through each threshold
    for cut in list_of_thresholds:
        # Defining events which pass this threshold iteration
        passed = dataframe[dataframe['bdt_score'] > cut]

        # Summing up the weights of pass/fail this threshold iteration
        s = passed[passed['Signal'] == 1]['Weight'].sum()
        b = passed[passed['Signal'] == 0]['Weight'].sum()

        # Significance formula (with safety if dividing by zero)
        significance = s / np.sqrt(s + b) if (s + b) > 0 else 0
        significances.append(significance)

    # Take the best significance out of all the significances and get that threshold value
    max_significance = max(significances)
    best_cut = list_of_thresholds[np.argmax(significances)]

    return best_cut, max_significance, list_of_thresholds, significances

ww_tree = uproot.open(path+"WW_MG5_NLO_wID.root:OS_MuE_Reco")
ww_df = ww_tree.arrays(library='pd')

ttbar_tree = uproot.open(path+"ttbar_Pythia_LO_rwgt_wID.root:OS_MuE_Reco")
ttbar_df = ttbar_tree.arrays(library='pd')
ttbar_df['Weight'] = ttbar_df['Scale Factor'] * ttbar_df['Reweight']

kinematic_keys = ["_PT","_ETA","_PHI","DeltaRLL","DeltaPhiLL","DeltaEtaLL","DiLeptonpT","DiLeptonMass","Weight"]

list_of_keys_needed = []
for key1 in ww_tree.keys():
    for name in kinematic_keys:
        if name in key1:
            list_of_keys_needed.append(key1)

ww_df = ww_df[list_of_keys_needed]
ttbar_df = ttbar_df[list_of_keys_needed]

ww_df['Signal'], ttbar_df['Signal'] = 1, 0

ww_ttbar_tot = pd.concat([ww_df, ttbar_df])

from pickle import load
with open('WWDFDY_trained_bdt.pkl', "rb") as f:
    imported_trained_bdt = load(f)

wwdfdy_model = imported_trained_bdt["Model"]
required_features = imported_trained_bdt["Features"]

results_from_wwdfdy = run_bdt(wwdfdy_model, ww_ttbar_tot, required_features)

ww = results_from_wwdfdy[results_from_wwdfdy['Signal']==1]
ttbar = results_from_wwdfdy[results_from_wwdfdy['Signal']==0]
num_bins = 100

best_cut_wwdfdy_tree, max_significance, list_of_thresholds, wwdfdy_significances = find_best_significance_gt(results_from_wwdfdy)
passed_wwdfdy_tree = results_from_wwdfdy[results_from_wwdfdy['bdt_score'] >= best_cut_wwdfdy_tree]


plt.figure(figsize=(8, 5))
plt.plot(list_of_thresholds, wwdfdy_significances, color='dodgerblue', lw=2)
plt.axvline(best_cut_wwdfdy_tree, color='red', linestyle='--', label=f'Best Cut: {best_cut_wwdfdy_tree:.2f}')
plt.xlabel('BDT Score', labelpad=10)
plt.ylabel('Significance $\;S / \sqrt{S+B}$')
plt.title('WW; DFDY Trained Tree', pad=10)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0,1)
plt.ylim(0,10)
plt.savefig('./Plots/best_significance_wwdfdyBDT_wwttbar.png', dpi=200)

plt.figure(figsize=(12,6))
plt.hist(ww['bdt_score'], bins=num_bins, density=True, color='dodgerblue', alpha=0.5, label=r'$WW$')
plt.hist(ttbar['bdt_score'], bins=num_bins, density=True, color='red', alpha=0.5, label=r'$t\overline{t}$')
plt.axvline(best_cut_wwdfdy_tree, color='k', linestyle='--', label=f'Best Cut = {best_cut_wwdfdy_tree}')

plt.xlim(0,1)
plt.xlabel('BDT Score')
plt.ylabel('Normalized Event Count', labelpad=10)
plt.title(r'$WW$; $t\overline{t}$ Results from $WW$ and $Z\rightarrow\tau\tau$ Trained BDT', pad=10)
plt.legend()
plt.xlim(0,1)
plt.savefig('./Plots/wwttbar_results_through_wwdfdy_bdt.png', dpi=200)

from pickle import load
with open('WWttbar_trained_bdt.pkl', "rb") as f:
    imported_trained_bdt = load(f)

wwttbar_model = imported_trained_bdt["Model"]
required_features = imported_trained_bdt["Features"]

results_from_wwttbar_bdt = run_bdt(wwttbar_model, ww_ttbar_tot, required_features)

best_cut_wwttbar_tree, max_significance, list_of_thresholds, wwttbar_significances = find_best_significance_gt(results_from_wwttbar_bdt)
results_from_wwttbar_bdt = results_from_wwttbar_bdt[results_from_wwttbar_bdt['bdt_score']>best_cut_wwttbar_tree]

plt.figure(figsize=(8, 5))
plt.plot(list_of_thresholds, wwttbar_significances, color='dodgerblue', lw=2)
plt.axvline(best_cut_wwttbar_tree, color='red', linestyle='--', label=f'Best Cut: {best_cut_wwttbar_tree:.2f}')
plt.xlabel('BDT Score')
plt.ylabel('Significance $\;S / \sqrt{S+B}$')
plt.title('WW; ttbar Trained Tree', pad=10)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0,1)
plt.ylim(0,10)
plt.savefig('./Plots/best_significance_wwttbarBDT_wwttbar.png', dpi=200)

ww_fin = results_from_wwttbar_bdt[results_from_wwttbar_bdt['Signal']==1]
ttbar_fin = results_from_wwttbar_bdt[results_from_wwttbar_bdt['Signal']==0]

plt.figure(figsize=(12,6))
plt.hist(ww_fin['bdt_score'], bins=num_bins, density=True, color='dodgerblue', alpha=0.5, label=r'$WW$')
plt.hist(ttbar_fin['bdt_score'], bins=num_bins, density=True, color='red', alpha=0.5, label=r'$t\overline{t}$')
plt.axvline(best_cut_wwttbar_tree, color='k', linestyle='--', label=f'Best Cut = {best_cut_wwttbar_tree:.2f}')

plt.xlim(0,1)
plt.xlabel('BDT Score')
plt.ylabel('Normalized Event Count', labelpad=10)
plt.title(r'$WW$; $t\overline{t}$ Results from $WW$ and $t\overline{t}$ Trained BDT', pad=10)
plt.legend()
plt.savefig('./Plots/wwttbar_passed_results_through_wwttbar_bdt.png', dpi=200)