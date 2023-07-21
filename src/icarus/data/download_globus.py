import glob
import os
import urllib.request

from tqdm import tqdm

import shutil

def download_data(lat, lon, step):
    # pB
    path_pB = f'data_fits/dcmer_{lon}_bang_{lat}_pB/stepnum_{step:03d}.fits'
    # tB
    path_tB = f'data_fits/dcmer_{lon}_bang_{lat}_tB/stepnum_{step:03d}.fits'
    # create dirs
    if not os.path.exists(f"/mnt/ground-data/{path_pB}"):
        urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_pB}', f"/mnt/ground-data/{path_pB}")
    if not os.path.exists(f"/mnt/ground-data/{path_tB}"):
        urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_tB}', f"/mnt/ground-data/{path_tB}")

if __name__ == '__main__':

    for lat in ['040N', '040W']:
        for lon in tqdm(range(20, 380, 20)):
            lon = f"{lon:03d}W" if lon > 0 else "0000"
            dir_pB = f"/mnt/ground-data/data_fits/dcmer_{lon}_bang_{lat}_pB"
            dir_tB = f"/mnt/ground-data/data_fits/dcmer_{lon}_bang_{lat}_tB"
            os.makedirs(dir_pB, exist_ok=True)
            os.makedirs(dir_tB, exist_ok=True)
            try:
                for step in range(5, 80):
                    download_data(lat, lon, step)
            except:
                shutil.rmtree(dir_pB)
                shutil.rmtree(dir_tB)
                print(f'INVALID URLs: lat {lat}, lon {lon}')
        print(f'download lat: {lat} finished')

        urllib.request.urlretrieve('https://g-824449.7a577b.6fbd.data.globus.org/sample_dens_stepnum_43.sav', '/mnt/ground-data/sample_dens_stepnum_43.sav')