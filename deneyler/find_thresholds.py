import json
import glob
import os
import pandas as pd

results = []
for file in glob.glob("output/analysis/*_analysis.json"):
    video_name = os.path.basename(file).replace("_analysis.json", "")
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    true_label = "Real_Hit" if video_name.startswith("RH") else "Ghost_Hit"
    
    cd = data.get("contact_details", {})
    kd = data.get("kinematic_details", {})
    
    has_contact = cd.get("has_any_contact", False)
    min_dist = cd.get("min_distance", 999)
    if not has_contact and min_dist <= 15.0:
        has_contact = True
        
    net_disp = kd.get("net_displacement", 0.0)
    max_acc = kd.get("max_acceleration", 0.0)
    max_overlap = cd.get("max_overlap", 0)
    
    results.append({
        "video": video_name,
        "true_label": true_label,
        "has_contact": has_contact,
        "net_disp": net_disp,
        "max_acc": max_acc,
        "max_overlap": max_overlap
    })

df = pd.DataFrame(results)

# Let's find the best rules!
for threshold_contact in [100, 120, 140, 150]:
    for threshold_nocontact in [150, 160, 170, 180, 200]:
        correct = 0
        for _, row in df.iterrows():
            if row["has_contact"]:
                is_real = row["net_disp"] > threshold_contact
            else:
                is_real = row["net_disp"] > threshold_nocontact
                
            if is_real and row["true_label"] == "Real_Hit": correct += 1
            elif not is_real and row["true_label"] == "Ghost_Hit": correct += 1
        
        acc = correct / len(df)
        if acc > 0.8:
            print(f"Contact Thresh: {threshold_contact}, No-Contact Thresh: {threshold_nocontact} => Accuracy: {acc*100:.1f}%")
