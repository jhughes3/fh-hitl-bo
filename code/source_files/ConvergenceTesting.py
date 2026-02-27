import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt

from source_files.UtilityFunctions import acquisition_function_main

def nan_aware_gaussian_blur_1d(y_vals, sigma):
    """Application of 1D Gaussian smoothing while ignoring NaN values"""
    y_vals = np.asarray(y_vals)
    mask = ~np.isnan(y_vals)

    # Replace NaNs with zero for convolution
    filled = np.where(mask, y_vals, 0.0)

    # Convolve values and weights separately
    blurred = gaussian_filter1d(filled, sigma=sigma)
    weights = gaussian_filter1d(mask.astype(float), sigma=sigma)

    # Normalize only where weights are non-zero
    with np.errstate(invalid='ignore', divide='ignore'):
        result = blurred / weights
        result[weights < 1e-6] = np.nan  # retain NaN where there's no valid input

    return result


def phase_bound_gen(my_GP_opt, parameter_bounds, n_grid_plotting, gradient_gen = False, crop_flag = 0, show_plot_flag = 0, gauss_blur = 0):
    """Given a current state of the GP surrogate, construct a phase boundary with resolution defined by n_grid_plotting"""
    
    compositionValues_plotting = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_grid_plotting)
    tempValues_plotting = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], n_grid_plotting)
    
    compositionMesh_plotting, tempMesh_plotting = np.meshgrid(compositionValues_plotting, tempValues_plotting)
    analysis_points = np.array(np.vstack([compositionMesh_plotting.flatten(), tempMesh_plotting.flatten()]).T.tolist())
    
    post_mean_guess = my_GP_opt.posterior_mean(analysis_points)["f(x)"]
    
    Z = post_mean_guess.reshape(compositionMesh_plotting.shape)

    if gradient_gen == True:
        #original gradient-based method
        # Compute the phase boundary
        boundary_x = compositionValues_plotting
        boundary_y = np.zeros_like(boundary_x)
        boundary_points = []
        
        for i in range(n_grid_plotting):
            temp_comp = compositionValues_plotting[i]
            temp_misc = Z[:, i]
            gradient = np.gradient(temp_misc)
            boundary_index = np.argmax(np.abs(gradient))
            boundary_y[i] = tempValues_plotting[boundary_index]
        
            # Store the boundary points
            boundary_points.append((temp_comp, boundary_y[i]))
    elif gradient_gen == False:
        boundary_x = compositionValues_plotting
        boundary_y = np.full_like(boundary_x, np.nan)
        boundary_points = []
        #new intermediate value-based method (more in line with acquisition function construction)
        for i in range(n_grid_plotting):
            temp_comp = compositionValues_plotting[i]
            temp_misc = Z[:,i]
            
            #checking to see whether a phase boundary exists here or not
            if np.any(temp_misc < 0.5) and np.any(temp_misc > 0.5):
                boundary_index = np.argmin(np.array([abs(a-0.5) for a in temp_misc]))
                boundary_y[i] = tempValues_plotting[boundary_index]
                boundary_points.append((temp_comp, boundary_y[i]))
            else:
                boundary_y[i] = np.nan
                boundary_points.append((temp_comp, np.nan))
            
            if gauss_blur != 0:
                temp_boundary_x, temp_boundary_y = zip(*boundary_points)
                temp_boundary_x = np.array(temp_boundary_x)
                temp_boundary_y = np.array(temp_boundary_y)
            
            
                boundary_y_smoothed = nan_aware_gaussian_blur_1d(temp_boundary_y, sigma=gauss_blur)
                boundary_points = list(zip(temp_boundary_x, boundary_y_smoothed))

    
    if crop_flag == 1:
        Z[Z < 0] = 0
        Z[Z > 1] = 1
    
    final_dataset_plot = np.zeros((n_grid_plotting, Z.shape[0]))
    threshold = 0.5

    if gradient_gen == True: 
        for i in range(n_grid_plotting):
            gradient = np.gradient(Z[:,i])
            
            # Find the index where the gradient is maximized (phase boundary)
            boundary_index = np.argmax(np.abs(gradient))
            
            # Initialize an array for classification and apply Gaussian filter in one step
            temp_red_misc_smoothed = np.zeros_like(Z[:, i])
            temp_red_misc_smoothed[boundary_index:] = 1
            temp_red_misc_smoothed = gaussian_filter(temp_red_misc_smoothed, sigma=gauss_blur)
            
            # Reclassify smoothed data to binary (1-phase or 2-phase)
            final_dataset_plot[i] = np.where(temp_red_misc_smoothed > threshold, 1, 0)
            
        np_final_dataset_plot = np.array(final_dataset_plot).T
    elif gradient_gen == False:
        for i in range(n_grid_plotting):
            column = Z[:,i]
            
            if np.any(column < threshold) and np.any(column > threshold):
                boundary_index = np.argmax(column > threshold)
            
                if column[boundary_index] <= threshold and np.all(column <= threshold):
                    boundary_index = len(column)-1
                
                temp_red_misc_smoothed = np.zeros_like(column)
                temp_red_misc_smoothed[boundary_index:] = 1
                temp_red_misc_smoothed = gaussian_filter(temp_red_misc_smoothed, sigma=gauss_blur)
                
                # Reclassify smoothed data to binary (1-phase or 2-phase)
                final_dataset_plot[i] = np.where(temp_red_misc_smoothed > threshold, 1, 0)
            else:
                phase_label = 1 if np.mean(column) > threshold else 0
                final_dataset_plot[i] = np.full_like(column, phase_label)
        np_final_dataset_plot = np.array(final_dataset_plot).T
    
    if show_plot_flag == 1:
    
        scatter = plt.imshow(Z, extent=(parameter_bounds[0][0], parameter_bounds[0][1], parameter_bounds[1][0], parameter_bounds[1][1]), cmap='viridis', origin='lower', aspect='auto')
        
        plt.plot(boundary_x, boundary_y, 'r-', label='Phase Boundary', linewidth=2)  # Plot the phase boundary
        
        # Add labels, title, and legend
        plt.xlabel('Composition')
        plt.ylabel('Temperature')
        plt.colorbar(scatter, label='Posterior Mean')
        plt.title('Predicted Phase Diagram')
        plt.legend()
        
        plt.gca().set_aspect(0.01)
    # plt.gca().set_aspect('equal')  
    
    return np_final_dataset_plot, analysis_points, Z, boundary_points

def compute_boundary_distances(boundary_points_list):
    """Compute difference between successive phase boundaries via mean pointwise Euclidian distnaces, ignoring NaN entries"""
    distances = []
    for i in range(len(boundary_points_list) - 1):
        x1, y1 = zip(*boundary_points_list[i])
        x2, y2 = zip(*boundary_points_list[i + 1])

        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)

        # Build mask where both y1 and y2 are valid (not NaN)
        valid_mask = ~np.isnan(y1) & ~np.isnan(y2)

        if np.any(valid_mask):
            dx = x2[valid_mask] - x1[valid_mask]
            dy = y2[valid_mask] - y1[valid_mask]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(np.mean(dist))
        else:
            distances.append(np.nan)  # No valid points to compare

    return distances


def last_smoothed_pct_change(data, window_size):
    """Computes percent change between two adjacent windows of size `window_size` at the end of the data list. This smooths both numerator and denominator."""
    if len(data) < 2 * window_size:
        return 1.0  # Not enough data for both windows

    prev_window = data[-(2 * window_size):-window_size]
    curr_window = data[-window_size:]

    mean_prev = np.mean(prev_window)
    mean_curr = np.mean(curr_window)

    if abs(mean_prev) > 1e-8:
        pct_change = abs(mean_curr - mean_prev) / abs(mean_prev)
    else:
        pct_change = 0.0

    return pct_change

def smooth_series(data, window_size):
    """Applies a simple moving average over the last `window_size` entries. Output length matches input length by padding the start with original values."""
    if window_size <= 1 or len(data) < 2:
        return data[:]

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        smoothed.append(sum(window) / len(window))
    return smoothed

def relative_variation_calc(smoothed_log):
    """
    Returns True if the last `window_size` smoothed values are relatively flat,
    defined as (max - min) / mean < `tolerance_pct`.
    
    `tolerance_pct` should be a fraction (e.g., 0.05 for 5% variation).
    """
    if len(smoothed_log) < 2:
        return 0.0

    prev_val = smoothed_log[-2]
    curr_val = smoothed_log[-1]

    relative_variation = curr_val - prev_val

    return relative_variation

def has_converged_flat(smoothed_log, window=5, flat_thresh=0.05):
    """Checks if the last `window` values are within `flat_thresh` range, returns True if metric has converged."""
    if len(smoothed_log) < window:
        return False, None

    recent = smoothed_log[-window:]
    if np.max(recent) - np.min(recent) < flat_thresh:
        return True, len(smoothed_log) - window  # return convergence start index
    else:
        return False, None