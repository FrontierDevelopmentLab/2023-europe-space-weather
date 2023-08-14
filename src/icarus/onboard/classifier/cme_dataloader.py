# dataloader for getting background subtracted difference images for c1 with polarisation 0, 120, 240
# TODO: update normalisation constants
# TODO: better constrain end of cme labels using second catalogue

import copy
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


def extract_images_within_time_range(events, image_paths):
    selected_images = []

    # Organize image paths by date for efficient checking
    image_paths_by_date = {}
    for image_path in image_paths:
        image_date = datetime.strptime(
            os.path.basename(image_path).split("_")[0], "%Y%m%d"
        ).date()
        image_paths_by_date.setdefault(image_date, []).append(image_path)

    for event in events:
        if not event["visible"]:
            continue

        if event["faint"]:
            continue

        event_start_time = datetime.strptime(event["datetime"], "%Y/%m/%d %H:%M")
        event_stop_time = datetime.strptime(event["event_stop_time"], "%Y%m%d_%H%M%S")
        event_date = event_start_time.date()

        # Check images within the event's date
        if event_date in image_paths_by_date:
            paths = image_paths_by_date[event_date]

            next_day = event_date + timedelta(days=1)
            if next_day in image_paths_by_date:
                paths.extend(image_paths_by_date[next_day])

            for image_path in paths:
                image_timestamp = datetime.strptime(
                    "_".join(os.path.basename(image_path).split("_")[:2]),
                    "%Y%m%d_%H%M%S",
                )
                if event_start_time <= image_timestamp <= event_stop_time:
                    selected_images.append(image_path)

    return selected_images


class CMEDataset(Dataset):
    def __init__(
        self,
        root,
        events=[],
        stereo_a_file_list=[],
        stereo_b_file_list=[],
        pol="all",
        size=512,
    ):
        self.cache_dir = os.path.join(root, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.events = events
        self.stereo_a = stereo_a_file_list
        self.stereo_b = stereo_b_file_list

        self.pol = pol

        self.images = []

        if pol == "all":
            self.mean = 2691.3037070368546
            self.std = 2579.566574917962
            # self.images.extend([im for im in self.stereo_a if os.path.basename(os.path.dirname(im)) != '1001.0'])
            # self.images.extend([im for im in self.stereo_b if os.path.basename(os.path.dirname(im)) != '1001.0'])
            self.images.extend(
                [
                    im
                    for im in self.stereo_a
                    if os.path.basename(os.path.dirname(im))
                    in ["0.0", "120.0", "240.0"]
                ]
            )
            self.images.extend(
                [
                    im
                    for im in self.stereo_b
                    if os.path.basename(os.path.dirname(im))
                    in ["0.0", "120.0", "240.0"]
                ]
            )
        elif pol == "sum":
            self.images.extend([im for im in self.stereo_a if "n4" in im])
            self.images.extend([im for im in self.stereo_b if "n4" in im])
            self.mean = 3658.224788149089
            self.std = 3399.0258091444553
        else:
            self.mean = 2691.3037070368546
            self.std = 2579.566574917962

            self.images.extend(
                [
                    im
                    for im in self.stereo_a
                    if os.path.basename(os.path.dirname(im)) == pol
                ]
            )
            self.images.extend(
                [
                    im
                    for im in self.stereo_b
                    if os.path.basename(os.path.dirname(im)) == pol
                ]
            )

        # filter size
        for image in self.images:
            if fits.getdata(image).shape != (size, size):
                print(f"Removing {image}")
                self.images.remove(image)

        self.images = sorted(self.images)
        self.images_for_date = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.dates = set()
        self.background = defaultdict(lambda: defaultdict(lambda: defaultdict()))

        for image_path in self.images:
            image_date = self._image_date(image_path)
            sat, angle = self._image_info(image_path)

            self.images_for_date[image_date][sat][angle].append(image_path)
            self.dates.add(image_date)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to PyTorch tensor
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),  # Normalize using mean and std
            ]
        )

        self._get_labels()
        self._gen_background()

    def _image_date(self, image):
        return datetime.strptime(os.path.basename(image).split("_")[0], "%Y%m%d").date()

    def _image_info(self, image_path):
        if "stereo_a" in image_path:
            sat = "stereo_a"
        elif "stereo_b" in image_path:
            sat = "stereo_b"
        if "/0.0/" in image_path:
            angle = "0.0"
        elif "/120.0/" in image_path:
            angle = "120.0"
        elif "/240.0/" in image_path:
            angle = "240.0"
        elif "/1001.0/" in image_path:
            angle = "1001.0"
        else:
            print(image_path)

        return sat, angle

    def _get_labels(self):
        self.positive_labels = set(
            extract_images_within_time_range(self.events["stereo_a"], self.stereo_a)
        )
        self.positive_labels |= set(
            extract_images_within_time_range(self.events["stereo_b"], self.stereo_b)
        )
        self.cme_images = set([im for im in self.images if im in self.positive_labels])

    def _gen_background(self, filter_cme=False):
        """
        Generate backgrounds for each satellite, each polarisation angle
        Need to delete cache if want to overwrite
        """
        cache_root = os.path.join(self.cache_dir, "background")
        os.makedirs(cache_root, exist_ok=True)
        print("Cache:", cache_root)

        for date in tqdm(sorted(list(self.dates))):
            image_date = date - timedelta(days=1)
            date_string = date.strftime("%Y%m%d")

            if image_date not in self.images_for_date:
                image_date = date

            for sat in ["stereo_a", "stereo_b"]:
                for angle in self.images_for_date[image_date][sat].keys():
                    cache_path = os.path.join(
                        cache_root, f"{date_string}_{sat}_{angle}.npy"
                    )

                    if os.path.exists(cache_path):
                        self.background[date][sat][angle] = np.load(cache_path)
                    else:
                        imgs = self.images_for_date[image_date][sat][angle]

                        if filter_cme:
                            imgs = np.array(
                                [
                                    fits.getdata(im)
                                    for im in imgs
                                    if im not in self.cme_images
                                ]
                            )
                        else:
                            imgs = np.array([fits.getdata(im) for im in imgs])

                        self.background[date][sat][angle] = np.median(imgs, axis=0)
                        np.save(cache_path, self.background[date][sat][angle])

    def _get_background(self, image_path):
        """
        given image path, retrieve correct background from dict
        """
        image_date = self._image_date(image_path)
        sat, angle = self._image_info(image_path)

        bg = self.background[image_date][sat][angle]

        return bg

    def _get_difference_image(self, i):
        """
        will not give viable results for the first image
        (and first day will use same-day background)

        TODO: ensure differencing over same sat, same polar with i-1
        """
        # read raw images for current and previous
        if i == 0:
            j = 0
        else:
            j = i - 1
        raw_img_i = fits.getdata(self.images[i]).astype(np.float32)
        raw_img_j = fits.getdata(self.images[j]).astype(np.float32)

        # get background from day before
        bg = self._get_background(self.images[i])

        # get difference image with subtracted background
        diff_img = (raw_img_i - bg) - (raw_img_j - bg)

        return diff_img

    def __getitem__(self, i):
        data = self._get_difference_image(i)
        label = int(self.images[i] in self.cme_images)

        return data, label  # self.transform(data), label

    def __len__(self):
        return len(self.images)


class CMEDataModule(LightningDataModule):
    """
    CMEDataModule:
        Class incorporating the dataloader for CMEData.
        This class enables the use of a consistent selection of data splits, saving the precise data in its member variables.

        Recommended Methods:
            train_dataloader() -> Dataloader: Returns the dataloader for the selected training data in the module
            test_dataloader() -> Dataloader: Returns the dataloader for the selected testing data in the module
            val_dataloader() -> Dataloader: Returns the validation dataloader
    """

    def __init__(self, config: dict = None):
        """__init__

        Args:
            config (dict): Dictionary encompassing the parameters passed in
                Dictionary Keys:
                    "Batch Size" - Default 32: Batchsize to use in the dataloaders
                    "num workers" - Default: 4 - Number of workers for the dataset
                    "train percentage" - Default: 0.85 - Percentage of collected data from Cor1, Cor2 to be used for training.
                                                         Each File will have an associated integer on whether an event is present (1) or not (0)
                    "test percentage" - Default: 0.1 - Percentage of collected data from Cor1, Cor2 to be used for testing.
                                                      Each File will have an associated integer on whether an event is present (1) or not (0)
                    "val percentage" - Default: 0.05 - Percentage of collected data from Cor1, Cor2 to be used for validation.
                                                       Each File will have an associated integer on whether an event is present (1) or not (0)

        """
        super().__init__()
        logging.info("Load data")

        # TODO note previous data_loader also loaded cme label

        self.config = config

        self.train_pct = config["train_pct"]
        self.valid_pct = config["valid_pct"]
        self.test_pct = 1 - (self.train_pct + self.valid_pct)
        if (self.train_pct + self.valid_pct + self.test_pct) != 1:
            print("Percentages do not add to 1!")

        self.num_workers = config["num_workers"]

    def _get_events(self):
        with open(self.config["events_path"], "r") as fp:
            events = json.load(fp)
        return events

    def _get_file_lists(self):
        root = self.config["root_path"]
        observation_days = sorted(glob(os.path.join(root, "*")))
        fnames_splits = random_split(
            observation_days, [self.train_pct, self.valid_pct, self.test_pct]
        )
        stereo_a_splits = []
        stereo_b_splits = []
        for fnames in fnames_splits:
            fname_list_a = []
            fname_list_b = []
            for fname in fnames:
                fname_list_a.extend(
                    sorted(glob(os.path.join(fname, "cor1", "stereo_a", "*", "*.fts")))
                )

                fname_list_b.extend(
                    sorted(glob(os.path.join(fname, "cor1", "stereo_b", "*", "*.fts")))
                )
            stereo_a_splits.append(fname_list_a)
            stereo_b_splits.append(fname_list_b)
        return stereo_a_splits, stereo_b_splits

    def setup(self, stage: str):
        """setup
        Method setting up the internals of the system
        including loading the relevant events (events_path)
        Dataloaders are created at the end of the method

        Args:
            stage (str): Stage for the data, encapsulating what this configuration is for - Example: Training, Testing, Validation
                         required by pytorch lightning
        """

        events = self._get_events()

        stereo_a_splits, stereo_b_splits = self._get_file_lists()
        print("LEN train", len(stereo_a_splits[0]))

        print("LEN val", len(stereo_a_splits[1]))

        print("LEN test", len(stereo_a_splits[2]))

        self.train_data = CMEDataset(
            self.config["root_path"],
            events=events,
            stereo_a_file_list=stereo_a_splits[0],
            stereo_b_file_list=stereo_b_splits[0],
            pol=self.config["pol"],
            size=self.config["image_size"],
        )
        self.val_data = CMEDataset(
            self.config["root_path"],
            events=events,
            stereo_a_file_list=stereo_a_splits[1],
            stereo_b_file_list=stereo_b_splits[1],
            pol=self.config["pol"],
            size=self.config["image_size"],
        )
        self.test_data = CMEDataset(
            self.config["root_path"],
            events=events,
            stereo_a_file_list=stereo_a_splits[2],
            stereo_b_file_list=stereo_b_splits[2],
            pol=self.config["pol"],
            size=self.config["image_size"],
        )

        print("Datasets set up")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, num_workers=None) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.test_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers=None) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.val_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    # from event scraper of:
    # https://cor1.gsfc.nasa.gov/catalog/cme/2014/Daniel_Hong_COR1_preliminary_event_list_2014-02.html
    with open("events_201402.json", "r") as fp:
        events = json.load(fp)

    # 7 mins without cache, 2-3 with cache
    ds = CMEDataset(root="/mnt/onboard_data/classifier", pol="all", events=events)

    print(ds.background.keys())
    for k in ds.background.keys():
        print(ds.background[k].keys())
        print(ds.background[k]["stereo_a"].keys())
        print(ds.background[k]["stereo_a"]["0.0"])
        break
