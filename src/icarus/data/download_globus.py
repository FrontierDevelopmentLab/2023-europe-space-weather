import os
import urllib.request

from tqdm import tqdm

for lat in ['040N']:
    for lon in tqdm(range(20, 380, 20)):
        for step in range(5, 80):
            # pB
            path_pB = f'data_fits/dcmer_{lon:03d}W_bang_{lat}_pB/stepnum_{step:03d}.fits'
            dir_pB = os.path.split(f"/mnt/ground-data/{path_pB}")[0]
            # tB
            path_tB = f'data_fits/dcmer_{lon:03d}W_bang_{lat}_tB/stepnum_{step:03d}.fits'
            dir_tB = os.path.split(f"/mnt/ground-data/{path_tB}")[0]
            if os.path.exists(dir_pB) and os.path.exists(dir_tB):
                continue
            # pB
            os.makedirs(dir_pB, exist_ok=True)
            urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_pB}', f"/mnt/ground-data/{path_pB}")
            # tB
            os.makedirs(dir_tB, exist_ok=True)
            urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_tB}', f"/mnt/ground-data/{path_tB}")

    urllib.request.urlretrieve('https://g-824449.7a577b.6fbd.data.globus.org/sample_dens_stepnum_43.sav', '/mnt/ground-data/sample_dens_stepnum_43.sav')