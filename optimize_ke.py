import pandas as pd
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(__file__), "output", "reports", "batch_summary_detailed.csv")
df = pd.read_csv(csv_path)

# Let's optimize the weights for net_disp and max_acc
best_acc = 0
best_w1 = 1.5
best_w2 = 0.5
best_threshold = 550

print("Testing different weight configurations and thresholds...")
for w1 in [1.0, 1.5, 2.0, 2.5]:
    for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        df['ke'] = df['net_disp'] * w1 + df['max_acc'] * w2
        
        # Test thresholds
        min_ke = df['ke'].min()
        max_ke = df['ke'].max()
        
        for threshold in np.linspace(min_ke, max_ke, 200):
            # simulate logic
            # if has_contact is True? We don't have has_contact in CSV directly.
            # but we can assume predicted == REAL_HIT right now means it passed the threshold 
            # wait, our decision_engine code uses:
            # if has_contact: REAL_HIT
            # else: if (disp*1.5 + acc*0.5) > 550 -> REAL_HIT else GHOST_HIT
            
            # Since has_contact is already applied, we can just look at the raw kinetic energy
            # Wait, if has_contact was true, the label is REAL_HIT regardless.
            # But we can just see if we can separate True_label using just KE
            
            # For Ghost Hits, we want ke < threshold (so it predicts GHOST_HIT)
            # For Real Hits, we want ke > threshold (so it predicts REAL_HIT)
            
            gh_correct = len(df[(df['true_label'] == 'Ghost_Hit') & (df['ke'] <= threshold)])
            rh_correct = len(df[(df['true_label'] == 'Real_Hit') & (df['ke'] > threshold)])
            
            total_acc = (gh_correct + rh_correct) / 120.0
            
            if total_acc > best_acc:
                best_acc = total_acc
                best_w1 = w1
                best_w2 = w2
                best_threshold = threshold

print(f"\nBest Overall Configuration:")
print(f"Weight 1 (net_disp): {best_w1}")
print(f"Weight 2 (max_acc): {best_w2}")
print(f"Threshold: {best_threshold:.1f}")
print(f"Accuracy: {best_acc*100:.1f}%")

# Apply best config to show exact counts
df['best_ke'] = df['net_disp'] * best_w1 + df['max_acc'] * best_w2
gh_correct = len(df[(df['true_label'] == 'Ghost_Hit') & (df['best_ke'] <= best_threshold)])
rh_correct = len(df[(df['true_label'] == 'Real_Hit') & (df['best_ke'] > best_threshold)])

print(f"\nBreakdown:")
print(f"Ghost Hit Accuracy: {gh_correct}/60 ({(gh_correct/60)*100:.1f}%)")
print(f"Real Hit Accuracy: {rh_correct}/60 ({(rh_correct/60)*100:.1f}%)")

