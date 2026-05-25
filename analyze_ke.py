import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), "output", "reports", "batch_summary_detailed.csv")
df = pd.read_csv(csv_path)

# Ghost Hits
gh = df[df['true_label'] == 'Ghost_Hit']
print("=== GHOST HITS ===")
print("Mean Net Disp:", gh['net_disp'].mean())
print("Max Net Disp:", gh['net_disp'].max())
print("Min Net Disp:", gh['net_disp'].min())
print("Mean Max Acc:", gh['max_acc'].mean())
print("Max Max Acc:", gh['max_acc'].max())

# Real Hits
rh = df[df['true_label'] == 'Real_Hit']
print("\n=== REAL HITS ===")
print("Mean Net Disp:", rh['net_disp'].mean())
print("Max Net Disp:", rh['net_disp'].max())
print("Min Net Disp:", rh['net_disp'].min())
print("Mean Max Acc:", rh['max_acc'].mean())
print("Max Max Acc:", rh['max_acc'].max())

# We want to separate the two clusters. 
print("\nTop 10 Ghost Hits by Net Disp:")
print(gh.nlargest(10, 'net_disp')[['video', 'net_disp', 'max_acc']])

print("\nBottom 10 Real Hits by Net Disp:")
print(rh.nsmallest(10, 'net_disp')[['video', 'net_disp', 'max_acc']])
