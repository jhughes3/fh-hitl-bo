import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpcam.gp_optimizer import GPOptimizer
from scipy.spatial import KDTree
import copy
import cv2

import source_files.InitializationFunctions as IF
import source_files.ExperimentalFunctions as EF

kb = 1.38E-23 #boltzman constant
#####################################################
_config_forFHPrior = None

def set_FH_prior_config(config):
    """Initialize the global scope config to pass to the FH prior"""
    global _config_forFHPrior
    _config_forFHPrior = config
    
def flory_huggins_prior_mean(x, hyperparams):
    """Flory-Huggins-informed prior mean for GP surrogate modelling, required parameter config file set via helper set_FH_prior_config"""
    if _config_forFHPrior == None:
        raise RuntimeError(
            "parameter config for FH prior not set. Call set_FH_prior_config(config) before use."
        )

    return _flory_huggins_prior_mean_priv(x, hyperparams, _config_forFHPrior)
        
def _flory_huggins_prior_mean_priv(x, hyperparams, config):
    """
    Compute a Flory-Huggins-informed prior mean compatible with GP surrogate modeling 

    """    
    x_list = x[:,0]
    x_list = EF.next_volfracs(x_list) #converting weight fraction (original x_list) to volume fraction
  
    T_list = x[:,1] #units of C
    A_chi = hyperparams[3]
    B_chi = hyperparams[4]

    
    chi_list = chi(T_list, A_chi, B_chi)

    N_PMMA = config["experimental_parameters"]["material_parameters"]["PMMA"]["degree_of_polymerization"]
    
    N_SAN = config["experimental_parameters"]["material_parameters"]["SAN"]["degree_of_polymerization"]

    d2_list = []

    for i in range(len(x_list)):
        curr_T = T_list[i] + 273.15   # converts temperature (in C) to K 
        
        curr_x = x_list[i] 
        curr_chi = chi_list[i]

        if curr_x == 0:
            curr_d2 = 2e6
        elif curr_x == 1:
            curr_d2 = 2e6
        else:
            curr_d2 = (kb * curr_T) * ((1/(N_PMMA * curr_x)) + (1/(N_SAN * (1-curr_x))) - (2 * curr_chi))
        
        
        d2_list.append(curr_d2)

    #Now, transform second derivative to "immiscibility"                                   
    a = config["general_parameters"]["immiscibility_sharpness"]
        
    immiscibility_transform = [0 if d2F > 1e6 else 1 / (1 + np.exp(a * (d2F))) for d2F in d2_list]
    return np.array(immiscibility_transform)


#####################################################
def plot_avg_acquisition_surface(
        objs,
    ):
    """Plot the posterior mean and acquisition surfaces for a given optimizer state"""
    x_bounds = (0.0, 1.0)
    T_bounds = (150.0, 200.0)
    
    n_grid = 60
    cmap = plt.colormaps.get_cmap("viridis")
    point_size = 25

    # 1. Create grid
    x_grid = np.linspace(*x_bounds, n_grid)
    T_grid = np.linspace(*T_bounds, n_grid)
    xx, TT = np.meshgrid(x_grid, T_grid)
    pts = np.column_stack([xx.ravel(), TT.ravel()])

    # 2. Accumulate acquisition and posterior values
    acq_vals_all = []
    post_mean_vals_all = []

    for obj in objs:
        acq_vals, _, _ = acquisition_function_main(pts, obj, intermediate_acq=1)
        acq_vals_all.append(acq_vals.reshape(xx.shape))

        post_mean = obj.posterior_mean(pts)["f(x)"]
        post_mean_vals_all.append(post_mean.reshape(xx.shape))

    # 3. Compute averages
    acq_vals_avg = np.mean(acq_vals_all, axis=0)
    post_mean_avg = np.mean(post_mean_vals_all, axis=0)
    post_mean_avg = np.clip(post_mean_avg, 0.0, 1.0)  # Clip to [0, 1]

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Posterior Mean Plot
    cs1 = ax1.contourf(xx, TT, post_mean_avg, levels=np.linspace(0,1,100), cmap=cmap, vmin=0, vmax=1)
    for obj in objs:
        ax1.scatter(obj.x_data[:, 0], obj.x_data[:, 1], 
                    c=obj.y_data, edgecolor='k', linewidths=1, s=point_size)
    fig.colorbar(cs1, ax=ax1, label="Posterior Mean")
    ax1.set_xlabel("Weight Fraction of PMMA")
    ax1.set_ylabel("Temperature $T$ (°C)")
    ax1.set_title("Average Posterior Mean Surface")

    # Acquisition Function Plot
    cs2 = ax2.contourf(xx, TT, acq_vals_avg, levels=np.linspace(0,1,100), cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(cs2, ax=ax2, label="Acquisition Value")
    ax2.set_xlabel("Weight Fraction of PMMA")
    ax2.set_ylabel("Temperature $T$ (°C)")
    ax2.set_title("Average Acquisition Surface")

    ax1.set_xlim(x_bounds)
    ax2.set_xlim(x_bounds)
    ax1.set_ylim(T_bounds)
    ax2.set_ylim(T_bounds)
    
    fig.tight_layout()
    plt.show()

def get_all_acquisition_values(logging_filepath = "FINALComposto_BO_diagnostics_log.xlsx"):
    """Retrieve all acquisition values (for points at the time that they were selected) over all iterations recorded in the logging spreadsheet"""
    dataframe_from_excel = pd.ExcelFile(logging_filepath)
    sheet_names = dataframe_from_excel.sheet_names
    acq_logger = []
    
    for curr_sheet_name in sheet_names:
        curr_df = pd.read_excel(dataframe_from_excel, sheet_name = curr_sheet_name)
        curr_acqs = [A for A in curr_df["Selected Acquisition"].to_list() if not pd.isna(A)]
        acq_logger.append(curr_acqs)

    return acq_logger

def compute_mean_and_std_lists(log_of_interest):
    """For a given log of values, compute the mean and standard deviations of the log_10 of the list values"""
    max_len = max(len(lst) for lst in log_of_interest)
    stack = np.full((len(log_of_interest), max_len), np.nan, dtype = float)
    for i, lst in enumerate(log_of_interest):
        stack[i, :len(lst)] = np.log10(lst)

    mean = np.nanmean(stack, axis=0)
    std  = np.nanstd(stack,  axis=0)
    x    = np.arange(max_len)

    return (x, mean, std)

def plot_acquisition_surface(
        obj,
        x_bounds=(0.0, 1.0),
        T_bounds=(150.0, 200.0),
        n_grid=60,
        *,
        selected_points=None,
        dx=0.025,
        dT=1.0,
        normalized_cutoff=1,
        write_local = False,
        graph_title = "",
        show_curr_data = True,
        ax1 = None, ax2 = None,
    ):
    """
    Visualize the acquisition surface and posterior mean, with optional selected points corresponding to measurements at subsequent iteration.
    """
    cmap = plt.colormaps.get_cmap("viridis")

    point_size = 25
    
    # ---------- 1. build grid ----------
    x_grid  = np.linspace(*x_bounds, n_grid)
    T_grid  = np.linspace(*T_bounds, n_grid)
    xx, TT  = np.meshgrid(x_grid, T_grid)
    pts     = np.column_stack([xx.ravel(), TT.ravel()])

    #Function evaluations
    acq_vals, _, _ = acquisition_function_main(pts, obj, intermediate_acq=1)
    acq_vals = acq_vals.reshape(xx.shape)
    post_mean_vals = obj.posterior_mean(pts)["f(x)"].reshape(xx.shape)

    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    else:
        fig = ax1.figure

    # -- Posterior Mean Surface --
    post_mean_vals = np.clip(post_mean_vals, 0.0, 1.0) #clipping s.t. the data falls between 0 and 1. hard cutoff
    cs1 = ax1.contourf(xx, TT, post_mean_vals, levels=np.linspace(0,1,100), cmap=cmap, vmin = 0, vmax = 1)

    prev_xT_data = obj.x_data
    prev_meas_data = obj.y_data

    if show_curr_data == True:
        scat1 = ax1.scatter(prev_xT_data[:,0], prev_xT_data[:,1], 
                            c = prev_meas_data, edgecolor = 'k', linewidths = 1, s = point_size, label = "Measurements")
        
    cbar1 = fig.colorbar(cs1, ax=ax1, label="Posterior Mean")
    cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar1.set_ticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax1.set_xlabel("Weight Fraction of PMMA")
    ax1.set_ylabel("Temperature $T$ (°C)")
    ax1.set_title(f"Posterior Mean Surface (Post-{graph_title})")    
    
    # -- Acquisition Surface --
    cs2 = ax2.contourf(xx, TT, acq_vals, levels=np.linspace(0,1,100), cmap=cmap, vmin=0, vmax=1)
    cbar2 = fig.colorbar(cs2, ax=ax2, label="Acquisition value")
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar2.set_ticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax2.set_ylabel("Temperature $T$ (°C)")
    ax2.set_title(f"2-D Acquisition Surface (Post-{graph_title})")
    
    if selected_points is not None and len(selected_points):
        sel_pts = np.asarray(selected_points)
        ax2.scatter(sel_pts[:, 0], sel_pts[:, 1],
                    color="red", edgecolor="k", s=point_size, zorder=3, label="selected")

        ax2.legend(loc="upper center")

    ax1.set_xlim(x_bounds)
    ax2.set_xlim(x_bounds)
    ax1.set_ylim(T_bounds)
    ax2.set_ylim(T_bounds)

    fig.tight_layout()

    if write_local == True:
        plt.savefig(f"{graph_title}_{cv2.getTickCount()}.jpg", dpi = 300)

def compute_LCST_from_HPs(AB_tuple, N_PMMA = 843.987, N_SAN = 665.375):
    """Calculate the LCST from the fitted A, B values and the critical Flory-Huggins interaction parameter."""
    A = AB_tuple[0]
    B = AB_tuple[1]

    chi_C = 0.5 * (((1/np.sqrt(N_PMMA)) + (1/np.sqrt(N_SAN)))**2)

    return (B / (chi_C - A)) - 273.15

def copy_constructor(my_GP_opt, FH_flag = True):
    """
    Return a copy of the gpCAM GP optimizer based on the current state
    """  
    if FH_flag == True:
        return GPOptimizer(x_data = copy.deepcopy(my_GP_opt.x_data),
                          y_data = copy.deepcopy(my_GP_opt.y_data),
                          init_hyperparameters = copy.deepcopy(my_GP_opt.hyperparameters),
                          noise_variances = copy.deepcopy(my_GP_opt.get_data()["measurement variances"]),
                          gp_kernel_function = IF.custom_kernel,
                          gp_mean_function = lambda x, hps: flory_huggins_prior_mean(x, hps),)
    
    else: #not using the FH-informed prior
        return GPOptimizer(x_data = copy.deepcopy(my_GP_opt.x_data),
                          y_data = copy.deepcopy(my_GP_opt.y_data),
                          init_hyperparameters = copy.deepcopy(my_GP_opt.hyperparameters),
                          noise_variances = copy.deepcopy(my_GP_opt.get_data()["measurement variances"]),
                          gp_kernel_function = IF.custom_kernel,)

def compute_conditions_list(my_GP_opt, hps_log, clip_first_iter = 1):
    """Computing the condition of the K_tot = K + Sigma matrix given the current hyperparameters"""

    test_dict = my_GP_opt.get_data()

    #formatting hps log
    if clip_first_iter == 1:
        cropped_hps_log = hps_log[1:]
    else:
        cropped_hps_log = hps_log
    #formatting x data
    x_data = test_dict['x data']
    initial = 4
    step = 5
    total = len(x_data)
    
    segmented_x_data = [x_data[:initial + i * step] for i in range((total - initial) // step + 1)]
        
    #formatting measurement variances
    meas_var = test_dict['measurement variances']

    segmented_meas_var = [meas_var[:initial + i * step] for i in range((total - initial) // step + 1)]
    
    assert len(cropped_hps_log) == len(segmented_x_data) == len(segmented_meas_var), \
        f"Input length mismatch: got {len(cropped_hps_log)} hps, {len(segmented_x_data)} x segments, {len(segmented_meas_var)} var segments"

    condition_list = []
    
    for ind in range(len(cropped_hps_log)):
        current_hps = cropped_hps_log[ind]
        current_x_data_segment = segmented_x_data[ind]
        current_meas_var_segment = segmented_meas_var[ind]

        K = IF.custom_kernel(current_x_data_segment, current_x_data_segment, current_hps)

        Sigma = np.diag(current_meas_var_segment)

        K_tot = K + Sigma

        current_cond_num = np.linalg.cond(K_tot)
        condition_list.append(current_cond_num)
    
    return condition_list

def ground_truth(x,y):
    """Empirical synthetic phase diagram generation for simulation benchmarking and workflow validation, not experimental results"""
    h = 0.55
    k = 160
    a_pos = 1750
    a_neg = 600
  
    scale = 2
    sharpness = 50
    scale_factor = 1

    binodal = np.where(x >= h, a_pos * (x - h)**4 + k, a_neg * (x - h)**4 + k)

    spinodal = scale * binodal

    z = scale_factor * (0.5 * (1 + np.tanh(sharpness * (y - binodal) / (spinodal - binodal))))

    return z

def acquisition_function_main(x, obj, intermediate_acq = 1):
    """
    Using the GP surrogate state encoded in obj, evaluate the acquisition function at x 
    using the specified acquisition function type (intermediate or gradient)
    """
    if intermediate_acq == 1:
        #Weighting Parameters
        a_cov = 1
        a_targ = 0.5
        
        #"Exploration" term
        mean = obj.posterior_mean(x)["f(x)"]
        
        target = 0.5
        
        target_arr = target * np.ones_like(mean)
        diff = np.abs(target_arr - mean)
        
        h = 0.1
        scale = 0.5
        targ_eval = np.tanh(h * np.reciprocal(diff)) * scale
        
        targ_term = (a_targ * targ_eval)
        
        #"Exploitation" term
        cov = obj.posterior_covariance(x)["v(x)"]
        cov_term = (a_cov * cov)
        
        #Adding together and returning
        _eval = cov_term + targ_term
        
        return _eval, cov_term, targ_term
    
    else: #doing the gradient-based method (incomplete)
        a_cov = 1 # keep at 1 and adjust targ
        a_grad = 0.1 #0.04, 4, 1
        
        mean = []
        cov = []
        
        for xi in x:
            mean.append(obj.posterior_mean(xi)["f(x)"])
            cov.append(obj.posterior_covariance(xi)["v(x)"])
        
        ######################################
        #calculating the gradient term (exploitation)
        acq_tree = KDTree(x)
        gradients = np.zeros((len(x), 2)) #len(x) should just be the number of points passed
        #into the function
        
        for i, (xi, yi) in enumerate(x):
            dists, indices = acq_tree.query([xi, yi], k = 5) # take gradients w.r.t. 5-1=4 NN
        
            neighbors = list(indices[1:])
            
            neighbor_points = np.array([x[n_i] for n_i in neighbors])
            
            dx = neighbor_points[:, 0] - xi
            dy = neighbor_points[:, 1] - yi
            dz = np.array([mean[ni] - mean[i] for ni in neighbors])
            
            # Compute gradient using least squares
            A = np.column_stack((dx, dy))
            gradient, _, _, _ = np.linalg.lstsq(A, dz, rcond=None)
            gradients[i] = gradient.flatten()
        
        gradient_magnitudes = np.linalg.norm(gradients, axis = 1)
        grad_term = [a_grad * grad_mag for grad_mag in gradient_magnitudes]
        
        ######################################
        #calculating covariance term (exploration)
        cov_term = [a_cov * cov_i for cov_i in cov]
        ######################################
        _eval = [a+b for a,b in zip(cov_term, grad_term)]
        
        return _eval, cov_term, grad_term

def chi(T_C, A, B):
    """Computation of Flory-Huggins interaction parameter using fitted A, B values. Takes in temperature in C and converts to K before using"""
    T_K = T_C + 273.15
    X = A + (B / T_K)
    return X


def generate_random_float(min_val, max_val):
    """Generate a random float within a specified interval"""
    return min_val + np.random.random() * (max_val - min_val)

def lhs_np(n, samples):
    """
    Latin-Hypercube-like implementation for generation of candidates within a 0-1 interval for each dim
    Later can be scaled to desired units of interval of interest 
    """
    intervals = np.linspace(0, 1, samples + 1)
    result = np.zeros((samples, n))

    for i in range(n):
        # Shuffle the intervals for each variable
        perm = np.random.permutation(samples)
        result[:, i] = intervals[perm] + np.random.rand(samples) * (1.0 / samples)

    return result
      
def gen_acq_func_points(parameter_bounds, n = 20):
    """Generate a specified number of points within the parameter bounds via lhs_np for feeding into the acquisition function"""
    x_range = parameter_bounds[0]
    y_range = parameter_bounds[1]
    
    lhs_samples = lhs_np(2, n) 
    
    x_vals = x_range[0] + lhs_samples[:,0] * (x_range[1] - x_range[0])
    y_vals = y_range[0] + lhs_samples[:,1] * (y_range[1] - y_range[0])
    
    points = [np.array([x,y]) for x,y in zip(x_vals, y_vals)]
    
    return points

def get_next_point_s(my_GP_opt, prec_comp, prec_temp, parameter_bounds, n_acq = 20, curr_intermediate_acq = 1, 
                     n_points_per_stack=1, multi_point = True, distance_weight = 0.0,
                    use_point_exclusion = True, point_exclusion_params = [0.025, 0.1],
                    normalized_exclusion_cutoff = 1, verbose_diagnostics = False):
    """ 
    Selects the next n_points_per_stack number of points for sample preparation based on a custom UCB-like acquisition function
    and accompanying decision policy, subject to parameter bounds, rounding / precision constraints, and point proximity / exclusion
    zone constraints.
    """

    if multi_point == 0: #for single point generation
        max_attempts = 5
        curr_attempt = 0
        found_valid = False

        while curr_attempt < max_attempts and not found_valid:

            temp_points = gen_acq_func_points(parameter_bounds, n = n_acq) #n_acq-long list of 2x1 np arrays
            
            temp_points = [np.array([round(a[0], prec_comp), round(a[1], prec_temp)]) for a in temp_points]
            
            temp_acq_func_evals = []
            temp_cov_term = []
            temp_grad_term = []
            
            if curr_intermediate_acq == 1:
                for cand in temp_points:
                    temp_acq, temp_cov, temp_grad = acquisition_function_main(cand.reshape(1,-1), my_GP_opt, intermediate_acq = 1)
                    temp_acq_func_evals.append(temp_acq[0])
                    temp_cov_term.append(temp_cov[0])
                    temp_grad_term.append(temp_grad[0])
            elif curr_intermediate_acq == 0:
                X_batch = np.array([np.ravel(pt) for pt in temp_points])
                temp_acq_func_evals, temp_cov_term, temp_grad_term = acquisition_function_main(X_batch, my_GP_opt, intermediate_acq = 0)
            
            if use_point_exclusion == True:
                temp_validity_check = point_exclusion_validity_check(temp_points, 
                                                                     my_GP_opt, 
                                                                     dx = point_exclusion_params[0], 
                                                                     dT = point_exclusion_params[1],
                                                                     ddist=0.0, 
                                                                     iterative_xT_list=[], 
                                                                     norm_dist_cutoff = normalized_exclusion_cutoff)
                masked_temp_acq_func_evals = [a * b for a,b in zip(temp_validity_check, temp_acq_func_evals)]
            else:
                masked_temp_acq_func_evals = temp_acq_func_evals
            
            if max(masked_temp_acq_func_evals) == 0:
                curr_attempt += 1
            else:
                ind_max_eval = masked_temp_acq_func_evals.index(max(masked_temp_acq_func_evals))
                next_point = temp_points[ind_max_eval]
                found_valid = True

        if not found_valid:
            ind_max_eval = temp_acq_func_evals.index(max(temp_acq_func_evals))
            next_point = temp_points[ind_max_eval]

        acq_ranking = sorted(enumerate(temp_acq_func_evals),
                             key=lambda x: -x[1]) # sort descending by acquisition score
        acquisition_rank = next(rank for rank, (idx, _) in enumerate(acq_ranking) if idx == ind_max_eval)
        
        return ([next_point], temp_acq_func_evals[ind_max_eval], 
                {"used_fallback": not found_valid,
                "attempts": curr_attempt if not found_valid else curr_attempt + 1,
                "acquisition_rank": acquisition_rank})

    elif multi_point == 1: #for multi-point generation (i.e. calculate useful temperature, then calculate useful points at said temperature)
    #Calculating the best temperature to make measurement at
        T_low = parameter_bounds[1][0]
        T_high = parameter_bounds[1][1]
        T_step = prec_temp + 1
        
        poss_temps = np.arange(T_low, T_high, T_step).astype(float)
        
        avg_acq_vals = []
        for temp in poss_temps:
            x_range = parameter_bounds[0]
            lhs_samples_1D = lhs_np(1,n_acq)
            temp_comps = list(x_range[0] + lhs_samples_1D[:,0] * (x_range[1] - x_range[0]))
            
            temp_comps = [round(c,prec_comp) for c in temp_comps]
            temp_points = [np.array([c, temp]) for c in temp_comps]
            
            
            if curr_intermediate_acq == 1:      
                temp_acq_func_evals = []
                
                for cand in temp_points:
                    temp_acq, temp_cov, temp_grad = acquisition_function_main(cand.reshape(1,-1), my_GP_opt, intermediate_acq = 1)
                    temp_acq_func_evals.append(temp_acq[0])
            elif curr_intermediate_acq == 0:
                X_batch = np.array([np.ravel(pt) for pt in temp_points])
                temp_acq_func_evals, _, _ = acquisition_function_main(X_batch, my_GP_opt, intermediate_acq = 0)
            
            
            
            #applying the point exlcusion - assumption = at least one point exists
            if use_point_exclusion == True:
                temp_validity_check = point_exclusion_validity_check(temp_points, 
                                                                     my_GP_opt, 
                                                                     dx = point_exclusion_params[0], 
                                                                     dT = point_exclusion_params[1],
                                                                     ddist=0.0, 
                                                                     iterative_xT_list=[], 
                                                                     norm_dist_cutoff = normalized_exclusion_cutoff)
                masked_temp_acq_func_evals = [a * b for a,b in zip(temp_validity_check, temp_acq_func_evals)]
            else:
                masked_temp_acq_func_evals = temp_acq_func_evals
            
            avg_acq_vals.append(np.mean(masked_temp_acq_func_evals))

        best_val = max(avg_acq_vals)
        tied_idx = [i for i, v in enumerate(avg_acq_vals) if v == best_val]
        if len(tied_idx) >= 2:
            mid = len(tied_idx)//2
            ind_max_eval = tied_idx[mid]
        else:
            ind_max_eval = tied_idx[0]
        next_temp = poss_temps[ind_max_eval]

        ###############################################
        #Calculating the best compositions to take measurements at
        max_attempts = 5
        curr_attempt = 0
        found_valid = False

        while curr_attempt < max_attempts and not found_valid:
            if verbose_diagnostics == True: 
                print(f"Current attempt: {curr_attempt}")
            dx = point_exclusion_params[0] #at a given temperature, compositions must be different by at least this much
            const_T_param_bounds = np.array([[parameter_bounds[0][0], parameter_bounds[0][1]], [next_temp, next_temp]])


            temp_points = gen_acq_func_points(const_T_param_bounds, n = n_acq) #n_acq-long list of 2x1 np arrays

            temp_points = [np.array([round(a[0], prec_comp), round(a[1], prec_temp)]) for a in temp_points]

            if curr_intermediate_acq == 1:
                temp_acq_func_evals = []
                temp_cov_term = []
                temp_grad_term = []

                for cand in temp_points:
                    temp_acq, temp_cov, temp_grad = acquisition_function_main(cand.reshape(1,-1), my_GP_opt, intermediate_acq = 1)
                    temp_acq_func_evals.append(temp_acq[0])
                    temp_cov_term.append(temp_cov[0])
                    temp_grad_term.append(temp_grad[0])
            elif curr_intermediate_acq == 0:
                X_batch = np.array([np.ravel(pt) for pt in temp_points])
                temp_acq_func_evals, temp_cov_term, temp_grad_term = acquisition_function_main(X_batch, my_GP_opt, intermediate_acq = 0)

            if use_point_exclusion == True:
                temp_validity_check = point_exclusion_validity_check(temp_points,
                                                                     my_GP_opt, 
                                                                     dx = point_exclusion_params[0], 
                                                                     dT = point_exclusion_params[1],
                                                                     ddist=0.0, 
                                                                     iterative_xT_list=[], 
                                                                     norm_dist_cutoff = normalized_exclusion_cutoff)
                masked_acq_func_evals = [a*b for a,b in zip(temp_validity_check, temp_acq_func_evals)]
            else:
                masked_acq_func_evals = temp_acq_func_evals

            sorted_indices = np.argsort(masked_acq_func_evals)  # Sort indices based on acquisition function values
            best_indices = [sorted_indices[-1]]  # Start with the highest evaluation index
            best_points = [temp_points[best_indices[0]]]
            best_evals = [masked_acq_func_evals[best_indices[0]]]
            best_cov = [temp_cov_term[best_indices[0]]]
            best_grad = [temp_grad_term[best_indices[0]]]
            
            
            remaining_indices = sorted_indices[:-1] 
            remaining_points = [temp_points[i] for i in remaining_indices]
            remaining_evals = [masked_acq_func_evals[i] for i in remaining_indices]
            remaining_cov = [temp_cov_term[i] for i in remaining_indices]
            remaining_grad = [temp_grad_term[i] for i in remaining_indices]

            while len(best_points) < n_points_per_stack and remaining_points:
                best_comp_list = [point[0] for point in best_points]
                combined_scores = []

                for i, point in enumerate(remaining_points):
                    point_array = np.array(point)
                    
                    tot_dist = sum(np.linalg.norm(np.array(selected_point) - point_array) for selected_point in best_points)
                    score = remaining_evals[i] + distance_weight * tot_dist
                    
                    combined_scores.append(score)

                max_score_index = np.argmax(combined_scores)
                selected_point = remaining_points[max_score_index]
                selected_x = selected_point[0]

                if use_point_exclusion == True:
                    if point_exclusion_validity_check([selected_point], 
                                                      my_GP_opt, 
                                                      dx = point_exclusion_params[0], 
                                                      dT = point_exclusion_params[1], 
                                                      ddist = 0.0, 
                                                      iterative_xT_list = best_points, 
                                                      norm_dist_cutoff = normalized_exclusion_cutoff)[0] == True: #if the newly proposed point works ...
                        best_points.append(selected_point)
                        best_evals.append(remaining_evals[max_score_index])
                        best_cov.append(remaining_cov[max_score_index])
                        best_grad.append(remaining_grad[max_score_index])

                    remaining_points.pop(max_score_index)
                    remaining_evals.pop(max_score_index)
                    remaining_cov.pop(max_score_index)
                    remaining_grad.pop(max_score_index)
                else:
                    best_points.append(selected_point)
                    best_evals.append(remaining_evals[max_score_index])
                    best_cov.append(remaining_cov[max_score_index])
                    best_grad.append(remaining_grad[max_score_index])
                    
                    remaining_points.pop(max_score_index)
                    remaining_evals.pop(max_score_index)
                    remaining_cov.pop(max_score_index)
                    remaining_grad.pop(max_score_index)


            #if we have all that we want, then we exit. if not, try again up to the number of max retries
            if len(best_points) < n_points_per_stack: #if we don't pick enough points, then try again, up to the number of allowable attempts
                curr_attempt += 1
            else: #if we successfully generate our list of points
                found_valid = True

        if not found_valid: #if, even after the max number of retries, we aren't able to generate a valid set of points ... ignore the mask
            const_T_param_bounds = np.array([[parameter_bounds[0][0], parameter_bounds[0][1]], [next_temp, next_temp]])
            temp_points = gen_acq_func_points(const_T_param_bounds, n=n_acq)
            temp_points = [np.array([round(a[0], prec_comp), round(a[1], prec_temp)]) for a in temp_points]

            if curr_intermediate_acq == 1:
                temp_acq_func_evals = []
                temp_cov_term = []
                temp_grad_term = []

                for cand in temp_points:
                    temp_acq, temp_cov, temp_grad = acquisition_function_main(cand.reshape(1, -1), my_GP_opt, intermediate_acq=1)
                    temp_acq_func_evals.append(temp_acq[0])
                    temp_cov_term.append(temp_cov[0])
                    temp_grad_term.append(temp_grad[0])
            elif curr_intermediate_acq == 0:
                X_batch = np.array([np.ravel(pt) for pt in temp_points])
                temp_acq_func_evals, temp_cov_term, temp_grad_term = acquisition_function_main(X_batch, my_GP_opt, intermediate_acq = 0)
            
            sorted_indices = np.argsort(temp_acq_func_evals)[::-1]  # Descending
            best_indices = [sorted_indices[0]]
            best_points = [temp_points[best_indices[0]]]
            best_evals = [temp_acq_func_evals[best_indices[0]]]
            best_cov = [temp_cov_term[best_indices[0]]]
            best_grad = [temp_grad_term[best_indices[0]]]
            
            remaining_indices = sorted_indices[1:]
            remaining_points = [temp_points[i] for i in remaining_indices]
            remaining_evals = [temp_acq_func_evals[i] for i in remaining_indices]
            remaining_cov = [temp_cov_term[i] for i in remaining_indices]
            remaining_grad = [temp_grad_term[i] for i in remaining_indices]

            while len(best_points) < n_points_per_stack and remaining_points:
                best_comp_list = [pt[0] for pt in best_points]
                combined_scores = []

                for i, pt in enumerate(remaining_points):
                    tot_dist = sum(np.linalg.norm(pt - other_pt) for other_pt in best_points)
                    score = remaining_evals[i] + distance_weight * tot_dist
                    combined_scores.append(score)

                max_score_index = np.argmax(combined_scores)

                best_points.append(remaining_points[max_score_index])
                best_evals.append(remaining_evals[max_score_index])
                best_cov.append(remaining_cov[max_score_index])
                best_grad.append(remaining_grad[max_score_index])
                
                # Remove selected
                remaining_points.pop(max_score_index)
                remaining_evals.pop(max_score_index)
                remaining_cov.pop(max_score_index)
                remaining_grad.pop(max_score_index)
        else: #meaning we found a set of valid points using the mask
        #yay
            pass

        #calculating / logging the sum deviation from ideality
        acq_ranking = sorted(enumerate(masked_acq_func_evals), key = lambda x: -x[1])
        index_to_rank = {idx: rank for rank, (idx, _) in enumerate(acq_ranking)}
        selected_indices = [i for pt in best_points for i, cand in enumerate(temp_points) if np.array_equal(pt, cand)]
        actual_ranks = [index_to_rank[idx] for idx in selected_indices]

        ideal_ranks = list(range(len(actual_ranks)))
        acquisition_rank_deviation = sum(act - ideal for act, ideal in zip(sorted(actual_ranks), ideal_ranks))

        if verbose_diagnostics == True: 
            print(f"Acquisition rank deviation: {acquisition_rank_deviation}")
        return (best_points, 
                best_evals,
                {"used_fallback": not found_valid,
                "attempts": curr_attempt if not found_valid else curr_attempt + 1,
                "acquisition_rank_deviation": acquisition_rank_deviation},
                {"possible_temperatures": list(poss_temps),
                "temperature_acquisitions": avg_acq_vals,
                "candidate_points": [list(pt) for pt in temp_points],
                "candidate_acquisitions": masked_acq_func_evals,
                "best_points": best_points,
                "best_acquisitions": best_evals})
    else:

        pass

def point_exclusion_validity_check(new_xT_list, obj, dx=0.025, dT=0.1, ddist=0.0, iterative_xT_list=[], norm_dist_cutoff = 1):
    """
    Check if a list of new_xT values are valid based on exclusion zones.
    Returns a list of boolean values, one for each point in new_xT_list.
    """
    valid_states = []
    prev_xT_list = list(obj.get_data()['x data'])
    
    prev_xT_list.extend(iterative_xT_list)

    if isinstance(new_xT_list, np.ndarray) and new_xT_list.shape == (2,):
        new_xT_list = [new_xT_list]
    elif isinstance(new_xT_list, list) and all(isinstance(x, (float, int)) for x in new_xT_list):
        new_xT_list = [np.array(new_xT_list)]
    
    if len(prev_xT_list) == 0:
        return [True] * len(new_xT_list)  # All valid if there are no previous points

    new_xT_list = list(new_xT_list)
    
    for new_xT in new_xT_list:
        valid_state = True
        for prev_xT in prev_xT_list:
            x_dist = new_xT[0] - prev_xT[0]
            T_dist = new_xT[1] - prev_xT[1]

            if dx != 0:
              norm_x_dist = x_dist / dx
            else:
              norm_x_dist = 0

            if dT != 0:
              norm_T_dist = T_dist / dT
            else:
              norm_T_dist = 0

            total_dist = np.sqrt(norm_x_dist**2 + norm_T_dist**2)
            total_dist += ddist

            if total_dist < norm_dist_cutoff:
                valid_state = False
                break

        valid_states.append(valid_state)

    return valid_states

def time_disp(t):
    """Helper function to convert seconds into days, hours, minutes, seconds. Useful for visualizing required annealing times"""
    days = t // 86400
    hours = (t % 86400) // 3600
    minutes = (t % 3600) // 60
    seconds = int(t % 60)
    return f"{days}d, {hours}h, {minutes}m, {seconds}s"

def first_local_max(arr):
    """Get the first instance of a local maximum in a list / array, return None if no such instance found"""
    for i in range(1, len(arr) - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:
            return i  # Return the index of the first local maximum
    return None  # Return None if no local maximum is found

def time_temperature_correction(T3, limited = 0):
    """ 
    Calculate the required annealing time at a given temperature based on Arrhenius-scaling calibrated from reference conditions
    """
    ################## 
    
    
    T3 = T3 + 2
    t_lim = (72 * 3600) + (0 * 60) + (0) #upper limit for annealing time
    
    #Calibration Data
    #calibration done w.r.t. set point, so need to increment the desired "felt" temp by 2 (T3+2) to convert to setpoint-space
    T1 = 195 #actual temperature
    t1 = (0 * 3600) + (30 * 60) + (0)
    
    T2 = 170 #actual temperature
    t2 = (9 * 3600) + (0 * 60) + (0)
    
    ##################
    #Calculation of new time t3 for given temperature T3
    
    #converting everything to K from C
    T1 = T1 + 273.15
    T2 = T2 + 273.15
    T3 = T3 + 273.15
    
    #calculating the arrhenius-based temperature changes, assuming equivalent
    #diffusion lengths
    num = kb*np.log(t1/t2)
    den = (1/T1) - (1/T2)
    Ea = num / den
    
    inner = (1/T3) - (1/T2)
    inner = (Ea/kb) * inner
    
    t3 = t2 * np.exp(inner)
    
    if limited == 1:
        return np.min([t3, t_lim])
    else:
        return t3

def get_post_mean(my_GP_opt, parameter_bounds, n_side_plotting):
    """
    Evaluates the posterior mean of the provided GP optimizer instance over a grid 
    with resolution defined by n_side_plotting and bounds defined by parameter_bounds
    """
    
    compositionValues_plotting = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_side_plotting)
    tempValues_plotting = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], n_side_plotting)
    
    compositionMesh_plotting, tempMesh_plotting = np.meshgrid(compositionValues_plotting, tempValues_plotting)
    analysis_points = np.array(np.vstack([compositionMesh_plotting.flatten(), tempMesh_plotting.flatten()]).T.tolist())
    
    post_mean_guess = my_GP_opt.posterior_mean(analysis_points)["f(x)"].reshape(compositionMesh_plotting.shape)
    
    return post_mean_guess

def get_post_acq(my_GP_opt, parameter_bounds, n_side_plotting):
    """
    Evaluates the acquisition function using the provided GP optimizer instance over a grid 
    with resolution defined by n_side_plotting and bounds defined by parameter_bounds
    """
    
    compositionValues_plotting = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_side_plotting)
    tempValues_plotting = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], n_side_plotting)
    
    compositionMesh_plotting, tempMesh_plotting = np.meshgrid(compositionValues_plotting, tempValues_plotting)
    analysis_points = np.array(np.vstack([compositionMesh_plotting.flatten(), tempMesh_plotting.flatten()]).T.tolist())
    
    temp_eval, temp_cov_term, temp_targ_term = acquisition_function_main(analysis_points, my_GP_opt, intermediate_acq = 1)
    
    return temp_eval.reshape(n_side_plotting, n_side_plotting), temp_cov_term.reshape(n_side_plotting, n_side_plotting), temp_targ_term.reshape(n_side_plotting, n_side_plotting)

def sigmoid(z):
    """Sigmoid function for normalizing z input between 0 (negative z) and 1 (positive z)"""
    return 1 / (1+np.exp(-z))

def gen_FH_plot(parameter_bounds, n_side_plotting, hps):
    """
    Evaluate the prior mean based on specified hyperparameters A and B over a grid
    with resolution defined by n_side_plotting and bounds defined by parameter_bounds, and return the grid of prior mean values
    """
    compositionValues_plotting = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_side_plotting)
    tempValues_plotting = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], n_side_plotting)
    
    compositionMesh_plotting, tempMesh_plotting = np.meshgrid(compositionValues_plotting, tempValues_plotting)
    analysis_points = np.array(np.vstack([compositionMesh_plotting.flatten(), tempMesh_plotting.flatten()]).T.tolist())
    
    FH_output = flory_huggins_prior_mean(analysis_points, hps)
    FH_output = FH_output.reshape(compositionMesh_plotting.shape)
    
    plt.figure()
    contour = plt.contourf(compositionMesh_plotting, tempMesh_plotting, FH_output, levels = 100, cmap = 'viridis')
    plt.xlabel("Composition (wt frac. PMMA)")
    plt.ylabel("Temperature (°C)")
    cbar = plt.colorbar(contour, label="Prior Mean (Immiscibility)")
    # colorbar_tick_list = np.arange(0, 1.05, 0.2)
    # cbar.set_ticks(colorbar_tick_list)
    plt.title(f"Flory–Huggins-Inspired Prior Mean Over (x, T): A = {round(hps[3],3)}, B = {round(hps[4],1)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return FH_output

def iteration_end_indices(num_post_init_iters):
    """Return cummulative data indices marking the end of each iteration assuming 4 initial samples and 5 samples per subsequent iteration"""
    return [4 + 5*i for i in range(num_post_init_iters + 1)]

def split_into_iterations(data):
    """Split data into iteration-sized subsets assuming 4 initial samples and 5 samples per subsequent iteration"""
    subsets = []
    idx = 0

    # First iteration: 4 points
    subsets.append(data[idx:idx+4])
    idx += 4

    # Remaining iterations: 5 points each
    while idx + 5 <= len(data):
        subsets.append(data[idx:idx+5])
        idx += 5

    return subsets

def get_data_up_to_iter(data, iter_num):
    """Get all accumulated data up to a specified iteration"""
    if iter_num == 0:
        return data[:4]
    else:
        return data[:4 + 5 * iter_num]

def get_HPs_for_iter(HP_list, iter_num):
    """
    Return the hyperparameter vector corresponding to a particular iteration
    
    HP_list = list of HPs for a given iteration
    iter_num = iteration (see below - will use matching indices)
    lst[0] = manual init
    lst[1] = post-training
    lst[2] = post-first iter
    lst[3] = post-second iter
    ...
    """
    ignore_manual = True
    
    if ignore_manual == False:
        return HP_list[iter_num]
    elif ignore_manual == True:
        return HP_list[iter_num + 1]

def comp_red(lst):
    """Compute the number of completed iterations, including initialization, assuming 4 initial samples and 5 sample per iteration"""
    curr_lst_len = len(lst)

    a = curr_lst_len - 4
    num_iters = a/5

    return int(num_iters + 1)

def filter_valid_points(curr_boundary):
    """Filter out phase boundary points with undefined (NaN) temperature values"""
    return [pt for pt in curr_boundary if not np.isnan(pt[1])]

def get_LCST(curr_valid_bound):
    """Extract the LCST as the minimum temperature along a phase boundary"""
    min_phi, min_T = min(curr_valid_bound, key = lambda pt: pt[1])
    return min_T

def pad_to_max(arr_list, max_len):
    """Pad each list in a nested list with NaNs to a common length and return as a 2D array"""
    return np.array([x + [np.nan] * (max_len - len(x)) for x in arr_list])