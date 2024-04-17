import os
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging
from bkg_functions import (
    get_google_sheets_service,
    read_data_from_sheet,
    read_root_file_from_url,
    eq_sim_time,
    calculate_rate,
    seconds_to_years,
    energy_histogram,
    plot_energy_histogram,
    process_isotope,
    total_rate_within_interval
)

SPREADSHEET_ID = "1OE5g3B6f43WHYlp49HuKK3h8Ac9ZMRpqrXVqkBuU7VQ"

detector_masses = {
    "CuLayer_0": 1852.24,
    "CuLayer_1": 1972.05,
    "CuLayer_2": 2095.38,
    "CuLayer_3": 2222.07,
    "CuLayer_4": 2352.19,
}

#'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Co58', 'Co57', 'Co56', 'Mn54', 'Be7'],
#'excel_cells': ['Activity!D138', 'Activity!D137', 'Activity!D140', 'Activity!D136', 'Activity!D141', 'Activity!D142', 'Activity!D143', 'Activity!D144', 'Activity!D145', 'Activity!D146', 'Activity!D147', 'Activity!D148'],

detector_excel_cells_isotopes = {
    'CuLayer_0': {
        'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Co58', 'Co57', 'Co56', 'Mn54', 'Be7'],
        'excel_cells': ['Activity!D138', 'Activity!D137', 'Activity!D140', 'Activity!D136', 'Activity!D141', 'Activity!D142', 'Activity!D143', 'Activity!D144', 'Activity!D145', 'Activity!D146', 'Activity!D147', 'Activity!D148'],
        'copper_reference': 'Schrieber-SABRE'
    }
    # 'CuLayer_1': {
    #     'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Co58', 'Co57', 'Co56', 'Mn54', 'Be7'],
    #     'excel_cells': ['Activity!D138', 'Activity!D137', 'Activity!D140', 'Activity!D136', 'Activity!D141', 'Activity!D142', 'Activity!D143', 'Activity!D144', 'Activity!D145', 'Activity!D146', 'Activity!D147', 'Activity!D148'],
    #     'copper_reference': 'Schrieber-SABRE'
    # },
    # 'CuLayer_2': {
    #     'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Ag108m', 'Bi207', 'Pb210'],
    #     'excel_cells': ['Activity!D103', 'Activity!D102', 'Activity!D104', 'Activity!D105', 'Activity!D107', 'Activity!D108', 'Activity!D109', 'Activity!D110', 'Activity!D111', 'Activity!D112'],
    #     'copper_reference': 'OPERA'
    # },
    # 'CuLayer_3': {
    #     'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Ag108m', 'Bi207', 'Pb210'],
    #     'excel_cells': ['Activity!D103', 'Activity!D102', 'Activity!D104', 'Activity!D105', 'Activity!D107', 'Activity!D108', 'Activity!D109', 'Activity!D110', 'Activity!D111', 'Activity!D112'],
    #     'copper_reference': 'OPERA'
    # },
    # 'CuLayer_4': {
    #     'isotopes': ['U238', 'Ra226', 'U235', 'Th232', 'K40', 'Cs137', 'Co60', 'Ag108m', 'Bi207', 'Pb210'],
    #     'excel_cells': ['Activity!D103', 'Activity!D102', 'Activity!D104', 'Activity!D105', 'Activity!D107', 'Activity!D108', 'Activity!D109', 'Activity!D110', 'Activity!D111', 'Activity!D112'],
    #     'copper_reference': 'OPERA'
    # }
}

# Function to set up logging
def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='rates.log', level=logging.INFO, format=log_format)

def main():
    # Set up logging
    setup_logging()

    # Initialize variables to accumulate data from all detectors
    all_events_per_year_per_keV = []
    all_errorbars_per_year_per_keV = []  # New list to collect error bars
    max_length = 3001  # Maximum length of histograms (0 to 3000 keV)
    energy_min = 1
    energy_max = 20

    # Iterate over detectors and their corresponding isotopes and Excel cell references
    for detector, data in detector_excel_cells_isotopes.items():
        copper_reference = data.get('copper_reference', 'No copper reference specified')
        logging.info(f"Analyzing part '{detector}' with copper reference '{copper_reference}'")
        isotopes = data["isotopes"]
        excel_cells = data["excel_cells"]

        # Initialize variables to accumulate data from all isotopes for the current detector
        detector_events_per_year_per_keV = np.zeros(max_length)
        detector_errorbars_per_year_per_keV = np.zeros(max_length)  # Initialize error bars array

        # Get the detector mass
        detector_mass = detector_masses.get(detector, 0.0)

        # Loop through isotopes and corresponding Excel cells for the current detector
        for isotope, excel_cell in zip(isotopes, excel_cells):
            print(f"Analyzing isotope '{isotope}' for part '{detector}'")
            energy_dep, time = process_isotope(SPREADSHEET_ID, detector_mass, isotope, detector, excel_cell)
            if energy_dep is not None:
                e_dep_tot = np.array(energy_dep[:, 0])
                e_dep_NR = np.array(energy_dep[:, 1])
                events_per_year_per_keV_isotope, errorbars_per_year_per_keV_isotope = energy_histogram(e_dep_tot, time)

                if events_per_year_per_keV_isotope is not None:
                    isotope_rate = total_rate_within_interval(events_per_year_per_keV_isotope, energy_min, energy_max)
                    isotope_error = total_rate_within_interval(errorbars_per_year_per_keV_isotope, energy_min, energy_max)
                    logging.info(f"Total Rate for isotope '{isotope}' within the energy interval [{energy_min}, {energy_max}] keV from {detector}: {isotope_rate} ± {isotope_error} events per year. Activity: {excel_cell}")

                # Print the total rate for the current isotope
                # if events_per_year_per_keV_isotope is not None:
                #     rate = np.sum(events_per_year_per_keV_isotope)
                #     print(f"Rate for isotope '{isotope}': {rate} events/year")

                # Plot the spectrum for the current isotope
                if events_per_year_per_keV_isotope is not None:
                    plot_energy_histogram(np.arange(max_length), events_per_year_per_keV_isotope, errorbars_per_year_per_keV_isotope, title=f'EnergyDep in {detector} for {isotope}', bin_width=5.0)

                # Add histogram of the current isotope to the detector's accumulator if it's not empty
                if events_per_year_per_keV_isotope is not None:
                    detector_events_per_year_per_keV += events_per_year_per_keV_isotope
                    detector_errorbars_per_year_per_keV += errorbars_per_year_per_keV_isotope

        # Calculate the total rate for events within the energy interval [1, 20] keV from the current detector
        part_rate = total_rate_within_interval(detector_events_per_year_per_keV, energy_min, energy_max)
        part_error = total_rate_within_interval(detector_errorbars_per_year_per_keV, energy_min, energy_max)  # Calculate error
        print(f"Total Rate for events within the energy interval [{energy_min}, {energy_max}] keV from {detector}: {part_rate} ± {part_error} events per year")
        logging.info(f"Total Rate for events within the energy interval [{energy_min}, {energy_max}] keV from {detector}: {part_rate} ± {part_error} events per year")

        # Accumulate events per year per keV from all detectors
        all_events_per_year_per_keV.append(detector_events_per_year_per_keV)
        all_errorbars_per_year_per_keV.append(detector_errorbars_per_year_per_keV)

    # Sum up histograms and error bars from all detectors
    sum_histogram = np.sum(all_events_per_year_per_keV, axis=0)
    sum_errorbars = np.sqrt(np.sum(np.square(all_errorbars_per_year_per_keV), axis=0))  # Combine error bars

    # Calculate the total rate for events within the energy interval [1, 20] keV from all detectors
    total_rate = total_rate_within_interval(sum_histogram, energy_min, energy_max)
    total_error = total_rate_within_interval(sum_errorbars, energy_min, energy_max)  # Calculate error
    print(f"Total Rate for events within the energy interval [{energy_min}, {energy_max}] keV from all detectors: {total_rate} ± {total_error} events per year")
    logging.info(f"Total Rate for events within the energy interval [{energy_min}, {energy_max}] keV from all detectors: {total_rate} ± {total_error} events per year")
    
    # Plot the combined histogram of summed events per year per keV from all detectors
    plot_energy_histogram(np.arange(max_length), sum_histogram, sum_errorbars, bin_width=5.0)

if __name__ == "__main__":
    main()
