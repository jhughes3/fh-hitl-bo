import numpy as np
import warnings
from gpcam.gp_optimizer import GPOptimizer

import source_files.UtilityFunctions as UF

def custom_kernel(x1, x2, hps):
    """Matern-3/2 kernel formulation with separate composition and temperature length scales"""
    var_f, l_x, l_T = hps[:3]

    x1 = np.atleast_2d(x1)  # (N1, 2)
    x2 = np.atleast_2d(x2)  # (N2, 2)

    dx = (x1[:, None, 0] - x2[None, :, 0]) / l_x  # shape (N1, N2)
    dT = (x1[:, None, 1] - x2[None, :, 1]) / l_T

    r = np.sqrt(dx**2 + dT**2)

    return var_f * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
  


def Initialization_HITL(simulation_flag = True,
                       default_variance = True, 
                       scale_vars = 0.1,
                       min_noise = 1e-6,
                       init_training_method = 'global',
                       include_FH = False,
                       random_params = False,
                       only_FH_random = True, #if True, only the FH Hps would be random, else both normal and FH
                       prec_comp = 3,
                       prec_temp = 0,
                       comp_min = 0,
                       comp_max = 1,
                       temp_min = 150., 
                       temp_max = 200.,
                       n_grid = 2,
                       base_HP_guess = [5e-1, 1e-2, 1e0],
                       base_HP_lims = [ [1e-2, 1e1], [1e-3, 1e0], [1e-1, 1e2] ],
                       FH_HP_guess = [0.15, -50.],
                       FH_HP_lims = [[0.00005, 1], [-200, -1]],
                       experimental_initial_points = [],
                       experimental_initial_measurements = [],
                       experimental_initial_variances = [],
                       retrain_now = False,
                       use_specified_HP_init = True,
                       specified_kernel = custom_kernel, 
                       ):
    """
    Initialize and optionally train a gpCAM GP optimizer for simulated or experimental workflows.
    
    This unified wrapper handles optional prior-mean integration, initialization with either experimental 
    or simulated data, and optional hyperparameter training for both simulation and experimental workflows.
    
    Returns the initialized GP optimizer along with parameter bounds, hyperparameter bounds, and a log of 
    hyperparameter values from before and after training
    """
    
    ## Hyperparameter and Parameter Optimization
    d_comp = 0.0 #d_comp = 0 would be original condition
    parameter_bounds = np.array([[float(comp_min + d_comp), float(comp_max-d_comp)], [temp_min, temp_max]])
    
    if include_FH == True: 
        hps_bounds = np.array(base_HP_lims + FH_HP_lims)
        if random_params == True: #assume this means random normal and FH HPs
            if only_FH_random == True:
                FH_HP_guess = [UF.generate_random_float(FH_HP_lims[0][0], FH_HP_lims[0][1]),
                               UF.generate_random_float(FH_HP_lims[1][0], FH_HP_lims[1][1]),]
                hps_init_guess = np.array(base_HP_guess + FH_HP_guess)
            elif only_FH_random == False:
                base_HP_guess = [UF.generate_random_float(base_HP_lims[0][0], base_HP_lims[0][1]),
                                 UF.generate_random_float(base_HP_lims[1][0], base_HP_lims[1][1]),
                                 UF.generate_random_float(base_HP_lims[2][0], base_HP_lims[2][1]),]
                FH_HP_guess = [UF.generate_random_float(FH_HP_lims[0][0], FH_HP_lims[0][1]),
                               UF.generate_random_float(FH_HP_lims[1][0], FH_HP_lims[1][1]),]
                hps_init_guess = np.array(base_HP_guess + FH_HP_guess)
        
        elif random_params == False:
            hps_init_guess = np.array(base_HP_guess + FH_HP_guess)
        
    elif include_FH == False:
        hps_bounds = np.array(base_HP_lims)
        
        if random_params == True:
            HP_guess = [UF.generate_random_float(base_HP_lims[0][0], base_HP_lims[0][1]),
                        UF.generate_random_float(base_HP_lims[1][0], base_HP_lims[1][1]),
                        UF.generate_random_float(base_HP_lims[2][0], base_HP_lims[2][1]),]
            hps_init_guess = np.array(HP_guess)
        
        elif random_params == False:
            hps_init_guess = np.array(base_HP_guess)

##################################################################################################################

    if simulation_flag == True: #for simulation initialization
        ## Collecting Initial Measurements, based on grid initialization method
        compositionValues_init = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], n_grid)
        tempValues_init = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], n_grid)
        compositionMesh, tempMesh = np.meshgrid(compositionValues_init, tempValues_init)
        initial_points = np.vstack([compositionMesh.flatten(), tempMesh.flatten()]).T.tolist()
        
        initial_points = [np.array([round(a[0],prec_comp), round(a[1],prec_temp)]) for a in initial_points]
        init_inputs = [{'x': subarray} for subarray in initial_points]
        curr_inputdata = []
        curr_outputdata = []
        
        for init_index in range(len(initial_points)):
            temp_x = initial_points[init_index]
            
            temp_sim_instrument_output = UF.ground_truth(temp_x[0], temp_x[1])
            curr_inputdata.append(initial_points[init_index])
            curr_outputdata.append(temp_sim_instrument_output)
            
        warnings.filterwarnings("ignore", category=UserWarning)
        
        try:
            del my_GP_opt
        except: 
            pass
        
        if default_variance == True:
            if include_FH == True:
                my_GP_opt = GPOptimizer(x_data = np.array(curr_inputdata),
                                        y_data = np.array(curr_outputdata),
                                        init_hyperparameters = hps_init_guess,
                                        gp_kernel_function = specified_kernel,
                                        gp_mean_function = lambda x, hps: UF.flory_huggins_prior_mean(x, hps))
            elif include_FH == False: 
                my_GP_opt = GPOptimizer(x_data = np.array(curr_inputdata),
                                        y_data = np.array(curr_outputdata),
                                        init_hyperparameters = hps_init_guess,
                                        gp_kernel_function = specified_kernel,)
        
        elif default_variance == False:
            curr_variances = [max(min_noise, scale_vars * (meas**2)) for meas in curr_outputdata]
            if include_FH == True:
                my_GP_opt = GPOptimizer(x_data = np.array(curr_inputdata),
                                        y_data = np.array(curr_outputdata),
                                        init_hyperparameters = hps_init_guess,
                                        noise_variances = np.array(curr_variances),
                                        gp_kernel_function = specified_kernel,
                                        gp_mean_function = lambda x, hps: UF.flory_huggins_prior_mean(x, hps))
            elif include_FH == False:
                my_GP_opt = GPOptimizer(x_data = np.array(curr_inputdata),
                                        y_data = np.array(curr_outputdata),
                                        init_hyperparameters = hps_init_guess,
                                        noise_variances = np.array(curr_variances),
                                        gp_kernel_function = specified_kernel,)
        
        hps_log = []
        
        
        
        hps_log.append(my_GP_opt.hyperparameters)
        
        # warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
        
        if use_specified_HP_init == False:
            my_GP_opt.train(hyperparameter_bounds = hps_bounds, method = init_training_method)
        else:
            my_GP_opt.train(hyperparameter_bounds = hps_bounds, 
                            init_hyperparameters = my_GP_opt.hyperparameters, 
                            method = init_training_method)
        
        hps_log.append(my_GP_opt.hyperparameters)
        
        return my_GP_opt, parameter_bounds, hps_bounds, hps_log
    ###############################################################
  
    elif simulation_flag == False: #for experimental initialization
        try:
            del my_GP_opt
        except: 
            pass
        
        if default_variance == True:
            if include_FH == True:
                my_GP_opt = GPOptimizer(x_data = np.array(experimental_initial_points),
                                        y_data = np.array(experimental_initial_measurements),
                                        init_hyperparameters = hps_init_guess,
                                        gp_kernel_function = specified_kernel,
                                        gp_mean_function = lambda x, hps: UF.flory_huggins_prior_mean(x, hps),)
            elif include_FH == False:
                my_GP_opt = GPOptimizer(x_data = np.array(experimental_initial_points),
                                        y_data = np.array(experimental_initial_measurements),
                                        init_hyperparameters = hps_init_guess,
                                        gp_kernel_function = specified_kernel,)
        elif default_variance == False:
            if len(experimental_initial_variances) == 0:
                curr_variances = [max(min_noise, scale_vars * (meas**2)) for meas in experimental_initial_measurements]
            else:
                curr_variances = experimental_initial_variances
            
            if include_FH == True: #BRANCH USED FOR EXPERIMENTS
                my_GP_opt = GPOptimizer(x_data = np.array(experimental_initial_points),
                                        y_data = np.array(experimental_initial_measurements),
                                        init_hyperparameters = hps_init_guess,
                                        noise_variances = np.array(curr_variances),
                                        gp_kernel_function = specified_kernel,
                                        gp_mean_function = lambda x, hps: UF.flory_huggins_prior_mean(x, hps),)
            elif include_FH == False:
                my_GP_opt = GPOptimizer(x_data = np.array(experimental_initial_points),
                                        y_data = np.array(experimental_initial_measurements),
                                        init_hyperparameters = hps_init_guess,
                                        noise_variances = np.array(curr_variances),
                                        gp_kernel_function = specified_kernel,)
        hps_log = []
        
        hps_log.append(my_GP_opt.hyperparameters)
        
        if retrain_now == True: 
            if use_specified_HP_init == False:
                my_GP_opt.train(hyperparameter_bounds = hps_bounds, method = init_training_method)
            else:
                my_GP_opt.train(hyperparameter_bounds = hps_bounds, 
                                init_hyperparameters = my_GP_opt.hyperparameters, 
                                method = init_training_method)
        
            hps_log.append(my_GP_opt.hyperparameters)
        else:
            pass
        
        return my_GP_opt, parameter_bounds, hps_bounds, hps_log