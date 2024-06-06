import argparse
import os
import urllib.request

from tqdm import tqdm


def download_data(lat, lon, step, download_dir):
    # pB
    path_pB = os.path.join(download_dir, f'dcmer_{lon}_bang_{lat}_pB/stepnum_{step:03d}.fits')
    # tB
    path_tB = os.path.join(download_dir, f'dcmer_{lon}_bang_{lat}_tB/stepnum_{step:03d}.fits')
    # create dirs
    if not os.path.exists(path_pB):
        urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_pB}',
                                   f"/glade/work/rjarolim/data/sunerf-cme/hao/{path_pB}")
    if not os.path.exists(path_tB):
        urllib.request.urlretrieve(f'https://g-824449.7a577b.6fbd.data.globus.org/{path_tB}',
                                   f"/glade/work/rjarolim/data/sunerf-cme/hao/{path_tB}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download data from Globus')
    parser.add_argument('--download_dir', type=str)

    args = parser.parse_args()
    download_dir = args.download_dir
    os.makedirs(download_dir, exist_ok=True)

    for lat in ['040N', '040W']:
        for lon in tqdm(range(20, 380, 20)):
            lon = f"{lon:03d}W" if lon > 0 else "0000"
            try:
                for step in range(5, 80):
                    download_data(lat, lon, step, download_dir)
            except:
                print(f'INVALID URLs: lat {lat}, lon {lon}')
        print(f'download lat: {lat} finished')

        urllib.request.urlretrieve('https://g-824449.7a577b.6fbd.data.globus.org/sample_dens_stepnum_43.sav',
                                   '/glade/work/rjarolim/data/sunerf-cme/hao/sample_dens_stepnum_43.sav')
