import numpy as np
import pandas as pd
from datetime import datetime
import os
import re

import source_files.UtilityFunctions as UF
import source_files.InitializationFunctions as IF

p_PMMA = 1.185
p_SAN = 1.08

def collapse_all_measurements(df, noise_floor = 1e-6):
    """Applying the per-sample 'group' collapsing in collapse_group to the entire dataset"""
    # Compute grouping key (i.e. sample name)
    df_copy = df.copy()
    df_copy["group_key"] = df_copy.apply(get_group_key, axis=1)

    # Group by the new grouping key (i.e. aggregating same-type samples via averaging)
    collapsed = (
        df_copy.groupby("group_key", sort = False)
          .apply(lambda g: collapse_group(g, noise_floor = noise_floor), include_groups = False)
          .reset_index(drop=True)
    )

    return collapsed

def collapse_group(g, noise_floor = 1e-6):
    """Collapse replicate measurements for a particular sample into a single measurement and variance"""

    iter_val = g["Iteration"].iloc[0]
    
    # Name handling
    if iter_val == "init":
        sample = g["Sample Name"].iloc[0].split("_")[0]
    else:
        sample = g["Sample Name"].iloc[0]

    x_val = g["x (wt. frac. PMMA)"].iloc[0]
    T_val = g["T (C) "].iloc[0]

    # CASE A — sample not visible
    if (g["Sample Visible?"] == 0).any():
        return pd.Series({
            "Iteration": iter_val,
            "Sample": sample,
            "x": x_val,
            "T": T_val,
            "measurement": 0.0,
            "variance": noise_floor
        })

    # CASE B — sample visible
    measurement = g["cloudiness"].mean()

    vars_ = g["variance_single"].values    
    variance = (vars_[0] + vars_[1]) / 4.0
    
    return pd.Series({
        "Iteration": iter_val,
        "Sample": sample,
        "x": x_val,
        "T": T_val,
        "measurement": measurement,
        "variance": variance
    })

def get_group_key(row):
    """Extracting the iteration and sample name corresponding to a particular sample."""
    iter_val = row["Iteration"]
    name_val = row["Sample Name"]

    if iter_val == "init":
        # Example raw names: SAN150_0 → base = SAN150
        base_name = name_val.split("_")[0]
        return ("init", base_name)
    else:
        # Example: (0, NH0), (12, NH59), etc.
        return (int(iter_val), name_val)

def get_ind_cloudiness_score_and_variance(df_row, cloudiness_params, noise_floor = 1e-6, a = 0.05):
    """Application of the cloudiness scoring protocol to a single sample condition"""
    #Cloudiness
    p = cloudiness_params["p"]
    h = cloudiness_params["h"]
    k = cloudiness_params["k"]
    
    if df_row["Sample Visible?"] == 0: #sample not visible
        cloudiness = 0
        variance = noise_floor

    else: #sample is visible
        diff = df_row["Diff = I_S - I_B"]
        dist = df_row["Dist = total dist to source (cm)"]
        
        cloudiness = 0.5 * (
            1 + np.tanh(h * ((diff * (dist**p)) - k))
        )
        ##########################
        #Variance
        variance = a * (cloudiness ** 2)

    return cloudiness, variance    

def get_all_cummulative_data(logging_filepath = "FINALComposto_BO_diagnostics_log.xlsx"):
    """Parse through logging spreadsheet and return all relevant data, with each list entry corresponding to an iteration"""
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names
  
    latest_sheet_name = sheet_names[-1]
  
    df = pd.read_excel(dataframe_from_excel, sheet_name = latest_sheet_name)
    cummulative_datapoints = [parse_array_string(x) for x in df["Current Samples"].to_list() if not pd.isna(x)]
    cummulative_measurements = [y for y in df["Current Measurements"].to_list() if not pd.isna(y)] #good
    cummulative_variances = [v for v in df["Current Variances"].to_list() if not pd.isna(v)]
    current_hyperparameters = [h for h in df["Current Hyperparameters"].to_list() if not pd.isna(h)]

    return cummulative_datapoints, cummulative_measurements, cummulative_variances, current_hyperparameters

def parse_npfloat64_tuple(s): #relevant for parsing the phase boundaries, since those x's and T's are saved as np.float64 strings
    """Parse stringified np.float64 tuples into Python float tuples."""
    matches = re.findall(r'np\.float64\((.*?)\)', s)
    result = []
    for val in matches:
        if val.strip().lower() == 'nan':
            result.append(np.nan)
        else:
            result.append(float(val))
    return tuple(result)

def parse_array_string(s):
    """Parse stringified arrays into lists"""
    if pd.isna(s):
        return None
    s = s.strip("[]")  # remove brackets
    parts = s.strip().split()  # split on whitespace
    return [float(p) for p in parts if p]  # ignore empty splits

def get_all_HPs(logging_filepath = "FINALComposto_BO_diagnostics_log.xlsx"):
    """Retrieve hyperparameters from all iterations recorded in the logging spreadsheet"""
    HP_logging_list = []
    
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names

    for curr_name in sheet_names:
        curr_df = pd.read_excel(dataframe_from_excel, sheet_name = curr_name)
        current_HPs = [h for h in curr_df["Current Hyperparameters"].to_list() if not pd.isna(h)]
        HP_logging_list.append(current_HPs)
    
    return HP_logging_list

def get_latest_recommended_points(logging_filepath = "Composto_BO_diagnostics_log.xlsx"):
    """Retrieve the most recent set of recommended points recorded in the logging spreadsheet"""
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names
    
    latest_sheet_name = sheet_names[-1]
    
    df = pd.read_excel(dataframe_from_excel, sheet_name = latest_sheet_name)

    selected_points = [parse_array_string(x) for x in df["Selected Point (x,T)"].to_list() if not pd.isna(x)]

    return selected_points

def get_latest_data_and_HPs(logging_filepath = "Composto_BO_diagnostics_log.xlsx"):
    """Retrieve the cummulative datapoints and associated measurements, as well as the current hyperparameters recorded in the logging spreadsheet"""
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names
    
    latest_sheet_name = sheet_names[-1]
    
    df = pd.read_excel(dataframe_from_excel, sheet_name = latest_sheet_name)
    cummulative_datapoints = [parse_array_string(x) for x in df["Current Samples"].to_list() if not pd.isna(x)]
    cummulative_measurements = [y for y in df["Current Measurements"].to_list() if not pd.isna(y)] #good
    current_hyperparameters = [h for h in df["Current Hyperparameters"].to_list() if not pd.isna(h)]
    
    return cummulative_datapoints, cummulative_measurements, current_hyperparameters

def get_all_boundary_locations(parameter_bounds = [[0., 1.], [150., 200.]], n_grid_plotting = 200, logging_filepath = "Composto_BO_diagnostics_log.xlsx"):
    """Get the list of phase boundary points from each iteration, returning a list of lists"""
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names
    
    boundary_points_list = []

    for temp_sheet_name in sheet_names:
        temp_df = pd.read_excel(dataframe_from_excel, sheet_name = temp_sheet_name)

        if "Current Boundary Points" in temp_df.columns:
            boundary_points = temp_df["Current Boundary Points"].apply(parse_npfloat64_tuple).tolist()
            boundary_points_list.append(boundary_points)
        else:
            compositionValues_plotting = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_grid_plotting)
            temperatureValues_plotting = [parameter_bounds[1][0]] * len(compositionValues_plotting)
            boundary_points = [(a,b) for a,b in zip(compositionValues_plotting, temperatureValues_plotting)]
      
            boundary_points_list.append(boundary_points)

    return boundary_points_list

def next_volfracs(next_weightfracs):
    """Convert weight fraction to volume fraction based on the specified PMMA and SAN densities"""
    next_vs = []

    for w in next_weightfracs:
        temp_v = 1 / (1 + ((1-w)/w) * (p_PMMA/p_SAN))
        next_vs.append(temp_v)

    return next_vs

def next_weightfracs(next_volfracs):
    """Convert volume fraction to weight fraction based on the specified PMMA and SAN densities"""
    return [(p_PMMA*v) / (p_PMMA*v + p_SAN*(1 - v)) for v in next_volfracs]

def next_component_volumes(next_volume_fracs, sample_vol=60, vol_dec_places=1):
    """Convert volume fraction of each component to the absolute volume of each based on the specified total volume of solution"""
    next_PMMA_vol = [round(v * sample_vol, vol_dec_places) for v in next_volume_fracs]
    next_SAN_vol = [round(sample_vol - pmma, vol_dec_places) for pmma in next_PMMA_vol]

    return {
        "PMMA": next_PMMA_vol,
        "SAN": next_SAN_vol
    }

def log_to_excel_new_sheet(diagnostic_log, curr_inputdata, my_GP_opt, current_phase_boundary, generating_new_points = True,
                           filename="Composto_BO_diagnostics_log.xlsx",  n_initialization = 4):
    """Verbose logging of information from the current experimental iteration to the corresponding excel sheet file destination"""
    # Timestamped sheet name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sheet_name = f"BO_Log_{timestamp}"

    
    current_optimizer_data_dict = my_GP_opt.get_data()
    
    # Unpack with fallbacks 
    temps = diagnostic_log.get("possible_temperatures", [])
    temp_acqs = diagnostic_log.get("temperature_acquisitions", [])
    points = diagnostic_log.get("candidate_points", [])
    point_acqs = diagnostic_log.get("candidate_acquisitions", [])
    best_pts = diagnostic_log.get("best_points", [])
    best_evals = diagnostic_log.get("best_acquisitions", [])
    
    # Compute padding length once
    max_len = max(len(temps), len(points), len(best_pts))
    
    def pad(lst): return lst + [None] * (max_len - len(lst))
    
    if generating_new_points == True:
        #calculating vol fracs
        next_xs = [point[0] for point in best_pts]
        next_vs = next_volfracs(next_xs)
    
        next_PMMA_SAN_vols = next_component_volumes(next_vs, sample_vol = 60, vol_dec_places = 1)
        next_PMMA_vol = next_PMMA_SAN_vols["PMMA"]
        next_SAN_vol = next_PMMA_SAN_vols["SAN"]
    
        #calculating annealing time
        next_temp = best_pts[0][1]
        annealing_time_str = UF.time_disp(UF.time_temperature_correction(next_temp))
    
    
        #sample naming
        start_index = len(curr_inputdata)-n_initialization
        sample_names = [f"NH_{i}" for i in range(start_index, start_index + len(best_pts))]
    
        
        # Build DataFrame
        df = pd.DataFrame({
            "Current Samples": pad(list(current_optimizer_data_dict['x data'])),
            "Current Measurements": pad(list(current_optimizer_data_dict['y data'])),
            "Current Variances": pad(list(current_optimizer_data_dict['measurement variances'])),
            "Current Hyperparameters": pad(list(current_optimizer_data_dict['hyperparameters'])),
            "Current Boundary Points": pad(current_phase_boundary),
            "Temperature (°C)": pad(temps),
            "Avg Acquisition @ Temp": pad(temp_acqs),
            "Candidate Point (x,T)": pad(points),
            "Candidate Acquisition": pad(point_acqs),
            "Sample Name": pad(sample_names),
            "Selected Point (x,T)": pad(best_pts),
            "Selected Acquisition": pad(best_evals),
            "PMMA Volume (uL)": pad(next_PMMA_vol),
            "SAN Volume (uL)": pad(next_SAN_vol),
            "Annealing Temp (C)": pad([next_temp]),
            "Annealing Set Point (C)": pad([next_temp + 2]),
            "Annealing Time": pad([annealing_time_str]),
        })

    elif generating_new_points == False:
        
        df = pd.DataFrame({
            "Current Samples": pad(list(current_optimizer_data_dict['x data'])),
            "Current Measurements": pad(list(current_optimizer_data_dict['y data'])),
            "Current Variances": pad(list(current_optimizer_data_dict['measurement variances'])),
            "Current Hyperparameters": pad(list(current_optimizer_data_dict['hyperparameters'])),
            "Current Boundary Points": pad(current_phase_boundary),
        })
        
    # Write
    if os.path.exists(filename):
        with pd.ExcelWriter(filename, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return sheet_name