"""
    File: Download_satellite_data.py
    Author: Martin Sanner
    Created: 14.7.2023 - 15:14 CEST

    Data Downloader creating the required folder structure locally to save data to for COR1, COR2 data for specified times.
    Times are specified in the .ephemeris files, created using the Nasa HORIZONS tool. (.ephemeris are renamed text files)

    Steps:
        1: Load Ephemeris data
        2: Estimate when 60° angles are hit
        3: Gather data from COR1, COR2 on SECCHI
        4: Create folder structure for COR1, COR2
        5: Download data to that folder
"""

import logging
import os
import time
from datetime import date, datetime, timedelta
from functools import reduce
from glob import glob
from importlib.resources import files
from typing import List

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import sscws
import sunpy
import yaml
from rich.progress import Progress
from sunpy.net import Fido
from sunpy.net import attrs as a

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ephemeris_data(fname: str) -> list:
    """
    Loads data from horizon ephemeris files, passed as argument.
    """
    with open(fname) as f:
        data = f.readlines()

    start_position = 0
    end_position = 0
    for i, line in enumerate(data):
        if "$$SOE" in line:
            start_position = i + 1
        if "$$EOE" in line:
            end_position = i

    return data[start_position:end_position]


def data_to_vectors(data, Time: list = None):
    """
    Takes the data returned by the load_data function and turns it into the required vectors
    """
    time = Time if Time is not None else []
    x = []
    y = []
    z = []
    r = []

    for line in data:
        t, p = line.strip("\n").split("     ")
        t2 = t.split(" ")[1] + " " + t.split(" ")[2]
        if Time is not None:
            assert (
                datetime.strptime(t2, "%Y-%b-%d %H:%M") in Time
            ), "New Time found in line {}".format(line)
        else:
            time.append(datetime.strptime(t2, "%Y-%b-%d %H:%M"))
        space_split = [i for i in p.split(" ") if i != ""]
        RA = float(space_split[0]) * np.pi / 180  # Radian for numpy
        DEC = float(space_split[1]) * np.pi / 180  # Radian for numpy

        cr = float(space_split[2])  # in km
        # dont use rdot here
        cx = cr * np.cos(DEC) * np.cos(RA * np.cos(DEC))  # km
        cy = cr * np.cos(DEC) * np.sin(RA * np.cos(DEC))  # km
        cz = cr * np.sin(DEC)  # km

        x.append(cx)
        y.append(cy)
        z.append(cz)
        r.append(cr)
    return time, x, y, z, r


def download_batch(batch: List, folder: str) -> None:
    filenames = []
    try:
        os.makedirs(folder, exist_ok=True, mode=0o777)
    except OSError as e:
        logging.error("Error when creating Folder = {} - {}".format(folder, e))

    try:
        filenames = Fido.fetch(
            batch, path="{}/".format(folder), progress=False, overwrite=True, max_conn=2
        )

        if len(filenames) > 0:
            Fido.fetch(
                filenames,
                path="{}/".format(folder),
                progress=False,
                overwrite=True,
                max_conn=2,
            )
    except KeyboardInterrupt:
        break
    except Exception as e:
        logging.error("Error encountered in downloading batch: {}".format(e))

    return


def get_events(min_time, max_time):
    """
    Get events from HEK
    """
    time_requested_for_batch = a.Time(str(min_time), str(max_time))
    events = Fido.search(time_requested_for_batch, a.hek.EventType(event_type))
    return events


def get_images(min_time, max_time) -> dict:  #
    time_requested_for_batch = a.Time(str(min_time), str(max_time))
    images_cor1 = Fido.search(
        time_requested_for_batch, a.Instrument("SECCHI"), a.Detector("COR1")
    )
    images_cor2 = Fido.search(
        time_requested_for_batch, a.Instrument("SECCHI"), a.Detector("COR2")
    )

    return {"cor1": images_cor1, "cor2": images_cor2}


if __name__ == "__main__":
    ephem_dir = os.path.dirname(__file__)
    l5_positions_fname = os.path.join(ephem_dir, "L5_positions.ephemeris")
    stereoA_positions_fname = os.path.join(ephem_dir, "StereoA_positions.ephemeris")
    stereoB_positions_fname = os.path.join(ephem_dir, "StereoB_positions.ephemeris")
    SOHO_positions_fname = os.path.join(ephem_dir, "SOHO_positions.ephemeris")

    # Get data
    l5_positions_data = load_ephemeris_data(l5_positions_fname)
    stereoA_positions_data = load_ephemeris_data(stereoA_positions_fname)
    stereoB_positions_data = load_ephemeris_data(stereoB_positions_fname)
    SOHO_positions_data = load_ephemeris_data(SOHO_positions_fname)
    # TODO: Create a method to find the dataframe without requiring equal time
    Time, L5x, L5y, L5z, L5r = data_to_vectors(l5_positions_data)
    _, SAx, SAy, SAz, SAr = data_to_vectors(stereoA_positions_data, Time)
    _, SBx, SBy, SBz, SBr = data_to_vectors(stereoB_positions_data, Time)
    _, Sohox, Sohoy, Sohoz, Sohor = data_to_vectors(SOHO_positions_data, Time)

    initial_colnames = [
        "L5 x [km]",
        "L5 y [km]",
        "L5 z [km]",
        "L5 r [km]",
        "SA x [km]",
        "SA y [km]",
        "SA z [km]",
        "SA r [km]",
        "SB x [km]",
        "SB y [km]",
        "SB z [km]",
        "SB r [km]",
        "SOHO x [km]",
        "SOHO y [km]",
        "SOHO z [km]",
        "SOHO r [km]",
    ]
    initial_data = (
        np.asarray(
            [
                L5x,
                L5y,
                L5z,
                L5r,
                SAx,
                SAy,
                SAz,
                SAr,
                SBx,
                SBy,
                SBz,
                SBr,
                Sohox,
                Sohoy,
                Sohoz,
                Sohor,
            ]
        )
    ).T

    df = pd.DataFrame(initial_data, index=Time, columns=initial_colnames)
    df["Distance L5 Stereo A [km]"] = np.sqrt(
        (df["L5 x [km]"] - df["SA x [km]"]) ** 2
        + (df["L5 y [km]"] - df["SA y [km]"]) ** 2
        + (df["L5 z [km]"] - df["SA z [km]"]) ** 2
    )
    df["Distance L5 Stereo B [km]"] = np.sqrt(
        (df["L5 x [km]"] - df["SB x [km]"]) ** 2
        + (df["L5 y [km]"] - df["SB y [km]"]) ** 2
        + (df["L5 z [km]"] - df["SB z [km]"]) ** 2
    )
    df["Stereo AB Angle [deg]"] = (
        np.arccos(
            (
                df["SA x [km]"] * df["SB x [km]"]
                + df["SA y [km]"] * df["SB y [km]"]
                + df["SA z [km]"] * df["SB z [km]"]
            )
            / (df["SA r [km]"] * df["SB r [km]"])
        )
        * 180
        / np.pi
    )
    df["Stereo A Soho Angle [deg]"] = (
        np.arccos(
            (
                df["SA x [km]"] * df["SOHO x [km]"]
                + df["SA y [km]"] * df["SOHO y [km]"]
                + df["SA z [km]"] * df["SOHO z [km]"]
            )
            / (df["SA r [km]"] * df["SOHO r [km]"])
        )
        * 180
        / np.pi
    )
    df["Stereo B Soho Angle [deg]"] = (
        np.arccos(
            (
                df["SB x [km]"] * df["SOHO x [km]"]
                + df["SB y [km]"] * df["SOHO y [km]"]
                + df["SB z [km]"] * df["SOHO z [km]"]
            )
            / (df["SB r [km]"] * df["SOHO r [km]"])
        )
        * 180
        / np.pi
    )

    # Parameters to define L5 geometry
    earth_l5_angle_degrees = 60
    error_range_degrees = 5
    required_distance_to_l5_km = 50000000  # km
    max_angle_between_crafts_deg = earth_l5_angle_degrees + error_range_degrees
    min_angle_between_crafts_deg = earth_l5_angle_degrees - error_range_degrees

    approx_date_last_B_contact = date(2016, 9, 1)  # some time September 2016

    df_angles_fit = df.query(
        "`Stereo AB Angle [deg]` >= {} & `Stereo AB Angle [deg]` <= {}".format(
            min_angle_between_crafts_deg, max_angle_between_crafts_deg
        )
    )
    df_angles_perfect_fit = df.query(
        "abs(`Stereo AB Angle [deg]` - {}) <= {}".format(earth_l5_angle_degrees, 1e-6)
    )
    df_stereoA_angles_fit = df.query(
        "`Stereo A Soho Angle [deg]` >= {} & `Stereo A Soho Angle [deg]` <= {}".format(
            min_angle_between_crafts_deg, max_angle_between_crafts_deg
        )
    )
    df_stereoB_angles_fit = df.query(
        "`Stereo B Soho Angle [deg]` >= {} & `Stereo B Soho Angle [deg]` <= {}".format(
            min_angle_between_crafts_deg, max_angle_between_crafts_deg
        )
    )

    df_StereoB_close = df.query(
        "`Distance L5 Stereo B [km]` <= {}".format(required_distance_to_l5_km)
    )
    df_StereoA_close = df.query(
        "`Distance L5 Stereo A [km]` <= {}".format(required_distance_to_l5_km)
    )

    # get times from df_{}.index
    # Main point: Do angles fit? - more important than distance for now.
    angle_AB_index = df_angles_fit.index
    angle_ASOHO_index = df_stereoA_angles_fit.index
    angle_BSOHO_index = df_stereoB_angles_fit.index
    # Overlap?
    overlap_ABASOHO = np.intersect1d(angle_AB_index, angle_ASOHO_index)  # empty
    overlap_ABBSOHO = np.intersect1d(angle_AB_index, angle_BSOHO_index)  # empty
    overlap_ASOHOBSOHO = np.intersect1d(angle_ASOHO_index, angle_BSOHO_index)
    # 24% of times, both angles A-SUN-SOHO, B-SUN-SOHO are within 55 to 65 degrees
    # AB and Soho are never in a 60° pairing - ie no scenario of 60° on both angles between A,B and SOHO from either one
    # in other words, never happens when either A, B are close to earth
    union_index = reduce(
        np.union1d, (angle_AB_index, angle_ASOHO_index, angle_BSOHO_index)
    )
    # Need to recover consecutive groups - ie, each group only consists of elements that are no more than 1 day removed from each other.
    time_differences = union_index[1:] - union_index[:-1]

    endpoints = [-1, len(union_index)]
    for time_diff_jumps_idx, td in enumerate(time_differences):
        if td != 86400000000000:  # = 1 day in ns
            endpoints.append(time_diff_jumps_idx)
    endpoints = np.sort(endpoints)
    timeseries_batches = [
        union_index[endpoints[i - 1] + 1 : endpoints[i]]
        for i in range(1, len(endpoints))
    ]

    # Create Folders to save data to
    config_path = "/home/josh/code/2023-europe-space-weather/config"

    with open(os.path.join(config_path, "onboard.yaml"), "r") as f:
        data_path = yaml.load(f, Loader=yaml.Loader)["drive_locations"]["datapath"]

    cor1_folder = os.path.join(data_path, "data", "cor1")
    cor2_folder = os.path.join(data_path, "data", "cor2")
    event_folder = os.path.join(data_path, "data", "events")

    try:
        os.makedirs(event_folder, exist_ok=True, mode=0o777)
    except OSError as e:
        logging.error("Error when creating Folder = {} - {}".format(event_folder, e))

    logger.info("Searching for timeseries")
    event_type = "CE"
    # Can download SECCHI data with help from FIDO

    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    progress_table = Table.grid()

    # Left (all batches)
    overall_progress = Progress()
    overall_task = overall_progress.add_task(
        "All Jobs", total=int(len(timeseries_batches))
    )
    progress_table.add_row(
        Panel.fit(
            overall_progress,
            title="Overall Progress",
            border_style="green",
            padding=(2, 2),
        ),
        Panel.fit(job_progress, title="[b]Jobs", border_style="red", padding=(1, 2)),
    )

    with Live(progress_table, refresh_per_second=10):
        for i, batch in enumerate(timeseries_batches):
            for j in range(len(batch) - 1):
                min_time = batch[j]
                max_time = batch[j + 1]

                logger.info("Searching between {} and {}".format(min_time, max_time))
                events = get_events(min_time, max_time)

                # Download data
                logger.info("Starting Cor1 Download")
                download_batch(res["cor1"], cor1_folder)
                logger.info("Starting Cor2 Download")
                download_batch(res["cor2"], cor2_folder)

                min_time_str = pd.to_datetime(min_time).strftime("%Y_%m_%d_%H_%M_%S")
                max_time_str = pd.to_datetime(max_time).strftime("%Y_%m_%d_%H_%M_%S")
                filename = os.path.join(
                    event_folder, f"events_{min_time_str}_{max_time_str}.csv"
                )
                # only saving event start and end times for now, some events are void
                if len(res["events"]["hek"]):
                    res["events"]["hek"]["event_starttime", "event_endtime"].write(
                        filename, format="csv", overwrite=True
                    )

                # job_progress.advance(job.id)

            overall_progress.overall_task(job.id, completed=i)
