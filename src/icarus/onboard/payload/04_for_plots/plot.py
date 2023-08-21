import numpy as np
import pylab as plt
import json

input_json = "grid_results_cor2_from_50.json"
title_str = "PSNR x Compression rates - Cor2"

# input_json = "grid_results_cor1_from_50.json"
# title_str = "PSNR x Compression rates - Cor1"
with open(input_json, 'r') as fp:
    grid_results = json.load(fp)

image_indices = sorted([int(i) for i in grid_results.keys()])

settings_keys = grid_results[str(image_indices[0])].keys()
print(settings_keys)

#baseline_settings_params = [int(float(s.split("_")[1])) for s in settings_keys if "base_" in s]
baseline_settings = [s for s in settings_keys if "base_" in s]

print(baseline_settings)
baseline_settings = baseline_settings[0:14]
print(baseline_settings)

#ai_settings_params = [int(float(s.split("_")[1])) for s in settings_keys if "ai_" in s]
ai_settings = [s for s in settings_keys if "ai_" in s]
print(ai_settings)

annotation_on = True

### Plot PSNR x Compression ratios
def data_from_series(series):
    data = []
    names = []
    for k in series:
        psnrs = []
        comp_rates = []
        for image_i in image_indices:
            result = grid_results[str(image_i)][k]
            psnrs.append(result["psnr"])
            comp_rates.append(result["comp_rate"])
        # means
        data.append( [np.mean(np.asarray(psnrs)), np.mean(np.asarray(comp_rates))] )

        name = str(int(float(k.split("_")[-1]))) # k
        names.append(name)


    data = np.asarray(data)
    # print(data)
    # print(names)
    return data, names

base_data, base_names = data_from_series(baseline_settings)
ai_data, ai_names = data_from_series(ai_settings)

fig, ax = plt.subplots()

plt.title(title_str)
#plt.scatter(x=data[:,0], y=data[:,1])
plt.plot(base_data[:,0], base_data[:,1], '-o', label="Baseline J2K (compression)")
plt.plot(ai_data[:,0], ai_data[:,1], '-o', label="CompressAI (quality)")

if annotation_on:
    for i, txt in enumerate(range(len(base_data[:,0]))):
        ax.annotate("  "+base_names[i], (base_data[i,0], base_data[i,1]))

    for i, txt in enumerate(range(len(ai_data[:,0]))):
        ax.annotate("  "+ai_names[i], (ai_data[i,0], ai_data[i,1]))

plt.legend()
plt.xlabel("PSNR")
plt.ylabel("Compression rate (x)")
plt.show()