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
from datetime import date, datetime, timedelta
from functools import reduce
from glob import glob

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import sscws
import sunpy
from sunpy.net import Fido
from sunpy.net import attrs as a

#



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


if __name__ == "__main__":
    l5_positions_fname = "./L5_positions.ephemeris"
    stereoA_positions_fname = "./StereoA_positions.ephemeris"
    stereoB_positions_fname = "./StereoB_positions.ephemeris"
    SOHO_positions_fname = "./SOHO_positions.ephemeris"
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
        "`Stereo AB Angle [deg]`== {}".format(earth_l5_angle_degrees)
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

    event_type = "CE"
    # Can download SECCHI data with help from FIDO
    event_batches = []
    cor1_batches = []
    cor2_batches = []
    
    for batch in timeseries_batches:
        min_time = np.min(batch)
        max_time = np.max(batch)
        time_requested_for_batch = a.Time(str(min_time), str(max_time))
        resulting_events_for_batch = Fido.search(
            time_requested_for_batch, a.hek.EventType(event_type)
        )
        resulting_images_for_batch_cor1 = Fido.search(
            time_requested_for_batch, a.Instrument("SECCHI"), a.Detector("COR1")
        )
        resulting_images_for_batch_cor2 = Fido.search(
            time_requested_for_batch, a.Instrument("SECCHI"), a.Detector("COR2")
        )
        cor1_batches.append(resulting_images_for_batch_cor1)
        cor2_batches.append(resulting_images_for_batch_cor2)
        event_batches.append(resulting_events_for_batch)
    

    # Create Folders to save data to
    # Current Working Directory
    cwd = os.getcwd()
    cor1_folder = os.path.join(cwd, "Data", "Cor1")
    cor2_folder = os.path.join(cwd, "Data", "Cor2")
    try:
        os.makedirs(cor1_folder, exist_ok=True, mode=0o777)
    except OSError as e:
        logging.error(
            "Error when creating Cor1 Folder = {} - {}".format(cor1_folder, e)
        )

    try:
        os.makedirs(cor2_folder, exist_ok=True, mode=0o777)
    except OSError as e:
        logging.error(
            "Error when creating Cor2 Folder = {} - {}".format(cor2_folder, e)
        )
    # Download data
    filenames = []
    for cor1_batch in cor1_batches:
        try:
            first_batch_downloads = Fido.fetch(cor1_batch, path = "{}/".format(cor1_folder))
            filenames.append(first_batch_downloads)
        except Exception as e:
            logging.error("Error encountered in downloading batch for Cor1: {}".format(e))
    for cor2_batch in cor2_batches:
        try:
            second_batch_downloads = Fido.fetch(cor2_batch, path = "{}/".format(cor2_folder))
            filenames.append(second_batch_downloads)
        except Exception as e:
            logging.error("Error encountered in downloading batch for Cor1: {}".format(e))
    # To find data of level: fits.open as f: f[0].header