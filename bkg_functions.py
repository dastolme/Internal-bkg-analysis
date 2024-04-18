import os
import numpy as np
import pandas as pd
import uproot
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt
import time
from functools import wraps
from socket import error as ClientOSError
import tempfile
import shutil
import urllib.request
import boto3

def retry(max_attempts=3, initial_delay=1, backoff_factor=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt == max_attempts:
                        raise  # Re-raise the exception if all attempts fail
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_google_sheets_service():
    """Create a Google Sheets API service client."""
    creds = None
    token_file = "token.json"  # You may adjust this filename as needed

    # Load existing credentials from the token file if it exists
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    # Build the Google Sheets API service
    service = build("sheets", "v4", credentials=creds)
    return service

def read_data_from_sheet(service, spreadsheet_id, range_):
    """Read data from a Google Sheet."""
    try:
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_)
            .execute()
        )
        values = result.get("values", [])
        return values
    except HttpError as e:
        print(f"Error reading data: {e}")
        return None

def read_root_file(file_path):
    """Read a root file and extract 'energyDep' and 'energyDep_NR' branches."""
    try:
        # Open the root file
        with uproot.open(file_path) as file:
            # Get the 'EnergyDep' and 'EnergyDep_NR' arrays
            energy_dep = file["tree2"]["e_dep"].array()
            energy_dep_nr = file["tree2"]["e_dep_NR"].array()

            # Calculate the total number of generated events
            events = file["tree1"]["Events"].array()
            total_gen_events = len(events)

            # Combine the arrays into a 2D array
            data_array = np.column_stack((energy_dep, energy_dep_nr))

            return data_array

    except Exception as e:
        print(f"Error reading root file: {e}")
        return None

@retry(max_attempts=5, initial_delay=10, backoff_factor=2, exceptions=(ClientOSError,))
def read_root_file_from_url(url):
    """
    Read a ROOT file from a URL and extract data from 'tree1' and all entries from 'tree2'.

    Parameters:
    - url (str): URL of the ROOT file.

    Returns:
    - total_gen_events (int): Total number of generated events from 'tree1'.
    - data_array_tree2 (numpy.ndarray): Extracted data from all entries of 'tree2'.
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Download the file to the temporary directory
        temp_file_path = os.path.join(temp_dir, "temp.root")
        urllib.request.urlretrieve(url, temp_file_path)

        # Open the ROOT file from the downloaded file
        print("Attempting to open downloaded file:", temp_file_path)
        file = uproot.open(temp_file_path)
        print("File opened successfully.")

        # Get the 'row_wise_branch' of 'tree1'
        row_wise_branch_tree1 = file["tree1"]["row_wise_branch"]

        # Count the number of entries in 'tree1' to get the total number of generated events
        total_gen_events = len(row_wise_branch_tree1.array(library="np"))

        # Get the 'row_wise_branch' of 'tree2'
        row_wise_branch_tree2 = file["tree2"]["row_wise_branch"]

        # Extract all entries from 'tree2'
        entries = row_wise_branch_tree2.array(library="np")

        # Initialize lists to store 'e_dep' and 'e_dep_NR'
        e_dep_list = []
        e_dep_NR_list = []

        # Extract 'e_dep' and 'e_dep_NR' for each entry
        for entry in entries:
            e_dep_list.append(entry[2])  # Third element
            e_dep_NR_list.append(entry[3])  # Fourth element

        # Convert lists to numpy arrays
        e_dep = np.array(e_dep_list)
        e_dep_NR = np.array(e_dep_NR_list)

        # Combine the arrays into a 2D array
        data_array_tree2 = np.column_stack((e_dep, e_dep_NR))

        return total_gen_events, data_array_tree2

    except ClientOSError as ce:
        print("ClientOSError:", ce)
        raise ce  # Re-raise the exception to trigger the retry mechanism

    except Exception as e:
        print("Error:", e)
        return None, None

    finally:
        # Cleanup: Delete the temporary directory and its contents
        shutil.rmtree(temp_dir, ignore_errors=True)


def eq_sim_time(gen_events, mass, specific_activity):
    """
    Calculate equivalent simulation time per mass.

    Parameters:
    - gen_events: Number of generated events
    - mass: Mass
    - specific_activity: Specific activity

    Returns:
    - Equivalent simulation time per mass
    """
    # Your implementation to calculate equivalent simulation time
    if (mass == 0).any():
        return 0  # Avoid division by zero

    return gen_events / (mass * specific_activity)  # sec

def seconds_to_hours(seconds):
    """
    Convert seconds to hours.

    Parameters:
    - seconds: Time duration in seconds.

    Returns:
    - Equivalent time duration in hours.
    """
    hours = seconds / 3600  # 1 hour = 3600 seconds
    return hours

def calculate_rate(num_events, time_duration):
    """
    Calculate the rate given a quantity and a time duration.

    Parameters:
    - quantity: The quantity or count.
    - time_duration: The time duration in seconds.

    Returns:
    - The rate, i.e., quantity per unit time.
    """
    if time_duration <= 0:
        print("Error: Time duration should be a positive value.")
        return None
    
    rate = num_events / time_duration
    return rate # Hz

def seconds_to_years(seconds):
    """
    Convert seconds to years.

    Parameters:
    - seconds: Time in seconds.

    Returns:
    - Time in years.
    """
    return seconds / 60 / 60 / 24 / 365.25

def energy_histogram(energy_dep, exposure_time_seconds=3600):
    """
    Calculate the histogram of the 'energyDep' data.

    Parameters:
    - energy_dep: 1D array containing the 'energyDep' data.
    - exposure_time_seconds: Exposure time in seconds (default is 3600 seconds or 1 hour).

    Returns:
    - bin_centers: Centers of the bins.
    - events_per_year_per_keV: Events per year per keV for each bin.
    """
    # Check if energy_dep is empty
    if len(energy_dep) == 0:
        print("energy_dep is empty. Unable to calculate histogram.")
        return None, None
    
    energy_min = 0
    energy_max = 3000

    num_bins = energy_max - energy_min + 1  # One bin per keV

    counts, bin_edges = np.histogram(energy_dep, bins=num_bins, range=(energy_min, energy_max))
    bin_centers = np.arange(energy_min, energy_max + 1, dtype=int)  # Integer bin centers
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Convert counts to events per year per keV
    if bin_width != 0:  # Avoid division by zero
        events_per_year_per_keV = counts / (seconds_to_years(exposure_time_seconds) * bin_width)
        errorbar_per_year_per_keV = np.sqrt(counts) / (seconds_to_years(exposure_time_seconds) * bin_width)
    else:
        print("Bin width is zero. Unable to calculate events_per_year_per_keV.")
        return None, None

    return events_per_year_per_keV, errorbar_per_year_per_keV

def plot_energy_histogram(bin_centers, events_per_year_per_keV, errorbars_per_year_per_keV, xlabel='Energy (keV)', ylabel='Events per yr/bin', title='EnergyDep Histogram', bin_width=1.0):
    """
    Plot a histogram of the energy data with error bars.

    Parameters:
    - bin_centers: Centers of the bins.
    - events_per_year_per_keV: Events per year per keV for each bin.
    - errorbars_per_year_per_keV: Error bars per year per keV for each bin.
    - xlabel: Label for the x-axis (default is 'Energy (keV)').
    - ylabel: Label for the y-axis (default is 'Events per yr/keV').
    - title: Title of the histogram plot (default is 'EnergyDep Histogram').
    - bin_width: Width of each bin (default is 1.0).
    """
    # Calculate the new bin edges
    half_bin_width = bin_width / 2.0
    new_bin_edges = np.arange(bin_centers[0] - half_bin_width, bin_centers[-1] + half_bin_width + 1e-10, bin_width)
    
    # Compute new bin centers by taking the mean of adjacent bin edges
    new_bin_centers = (new_bin_edges[1:] + new_bin_edges[:-1]) / 2.0
    
    # Aggregate events_per_year_per_keV for adjacent bins
    new_events_per_year_per_keV = []
    new_errorbars_per_year_per_keV = []
    idx = 0
    for i in range(len(new_bin_edges) - 1):
        events = 0
        errorbars = 0
        count = 0
        while idx < len(bin_centers) and bin_centers[idx] < new_bin_edges[i + 1]:
            events += events_per_year_per_keV[idx]
            errorbars += errorbars_per_year_per_keV[idx]
            count += 1
            idx += 1
        if count > 0:
            new_events_per_year_per_keV.append(events / count)
            new_errorbars_per_year_per_keV.append(errorbars / count)
        else:
            new_events_per_year_per_keV.append(0)
            new_errorbars_per_year_per_keV.append(0)
    
    new_bin_centers = np.array(new_bin_centers)
    new_events_per_year_per_keV = np.array(new_events_per_year_per_keV)
    new_errorbars_per_year_per_keV = np.array(new_errorbars_per_year_per_keV)
    
    # Plot the histogram with error bars
    plt.bar(new_bin_centers, new_events_per_year_per_keV, width=bin_width, edgecolor='crimson', alpha=0.6, label='Histogram')
    plt.errorbar(new_bin_centers, new_events_per_year_per_keV, yerr=new_errorbars_per_year_per_keV, fmt='.', color='crimson', markersize=5, label='Error Bars')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    
    plt.show()

# def plot_energy_histogram(bin_centers, events_per_year_per_keV, xlabel='Energy (keV)', ylabel='Events per yr/keV', title='EnergyDep Histogram'):
#     """
#     Plot a histogram of the energy data with error bars.

#     Parameters:
#     - bin_centers: Centers of the bins.
#     - events_per_year_per_keV: Events per year per keV for each bin.
#     - xlabel: Label for the x-axis (default is 'Energy (keV)').
#     - ylabel: Label for the y-axis (default is 'Events per yr/keV').
#     - title: Title of the histogram plot (default is 'EnergyDep Histogram').
#     """
#     plt.bar(bin_centers, events_per_year_per_keV, width=1.0, edgecolor='crimson', alpha=0.6, label='Histogram')
    
#     # Calculate errors as the square root of events_per_year_per_keV
#     errors = np.sqrt(events_per_year_per_keV)
    
#     plt.errorbar(bin_centers, events_per_year_per_keV, yerr=errors, fmt='.', color='crimson', markersize=5, label='Error Bars')
    
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.yscale("log")
    
#     plt.show()

# Define a function to process each isotope, detector part, and corresponding Excel cell
def process_isotope(spreadsheet_id, detector_mass, isotope, detector_part, excel_cell):
    try:
        # Create the service client
        sheets_service = get_google_sheets_service()

        # Read data from ROOT file named after the isotope
        url = f"https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-sim/CYGNO_04_MC/{isotope}_{detector_part}.root"
        print(url)
        total_events, data_root = read_root_file_from_url(url)

        if data_root is None:
            print(f"Error reading ROOT file for isotope {isotope} and detector part {detector_part}. Skipping.")
            return None, None, None

        # Read data from Google Sheets
        data_google_sheets = read_data_from_sheet(sheets_service, spreadsheet_id, excel_cell)
        data_google_sheets = float(data_google_sheets[0][0])

        # Match entries
        data = {'mass': [detector_mass],
                'specific_activity': data_google_sheets}
        matched_data = pd.DataFrame(data)

        # Calculate equivalent simulation time per mass
        time = eq_sim_time(total_events, matched_data["mass"][0], matched_data["specific_activity"][0])

        return data_root, time

    except Exception as e:
        print(f"Error processing isotope {isotope} and detector part {detector_part}: {e}")
        return None, None

def total_rate_within_interval(events_per_year_per_keV, energy_min, energy_max):
    total_rate = 0.0
    for energy, count in enumerate(events_per_year_per_keV):
        if energy_min <= energy <= energy_max:
            total_rate += count
    return total_rate

def sum_errors_in_quadrature_within_interval(errors, energy_min, energy_max):
    """
    Sum errors in quadrature between energy_min and energy_max.

    Parameters:
    errors (array-like): Array of errors corresponding to energy bins.
    energy_min (int): Minimum energy bin index for the range.
    energy_max (int): Maximum energy bin index for the range.

    Returns:
    float: Quadrature sum of errors within the specified energy range.
    """

    # Ensure energy_min and energy_max are within the range of errors
    if energy_min < 0 or energy_max >= len(errors):
        raise ValueError("Energy range exceeds the bounds of the errors array.")

    # Extract errors within the specified energy range
    errors_range = errors[energy_min:energy_max+1]

    # Sum errors in quadrature
    sum_squared_errors = np.sum(np.square(errors_range))

    return np.sqrt(sum_squared_errors)