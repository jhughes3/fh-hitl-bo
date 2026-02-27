# FH-BO-PolymerPhaseDiagram
This repository contains the source code, notebooks, and associated data for **"Using Flory-Huggins-Informed Human-in-the-Loop Bayesian Optimization to Map the Phase Diagram of Polymer Blends"**

## Requirements 
This code was developed and tested using **Python 3.10.11**

Install the required python dependencies:
```
pip install -r requirements.txt 
```

## Project Structure
The hierarchical layout of the repository is shown below. 
```
fh-hitl-bo/
├── code/
│   ├── notebooks/
│   │   ├── HITL_ExperimentalControl.ipynb
│   │   │   # Full human-in-the-loop Bayesian optimization workflow used in experiments to analyze data and recommend points
│   │   │
│   │   ├── Simulation_Validation.ipynb
│   │   │   # Simulation validation of Flory-Huggins-informed prior integration into the Bayesian optimization workflow 
│   │   │
│   │   └── Example_GPReconstruction.ipynb
│   │       # Demonstration of constructing a GP optimizer using gpcam with the Flory-Huggins-informed prior mean    
│   │
│   ├── source_files/
│   │    # Python source files used by the notebooks
│   │
│   └── param_config.json
│       # Configuration file to encode physical parameters and other experimental constants
│
├── data/
│   ├── calibrations.xlsx
│   │   # Data used for calibrating analysis-related parameters, as described in the Supplementary Information
│   ├── raw_measurements.xlsx
│   │   # Experimental cloudiness measurements used by the BO loop
│   └── sample_images/
│       # Example colored images of samples. Examples of high (NH3), medium (NH86), and low (NH43) cloudiness samples are shown.
│
└── hardware/
    └── ImagingStation.STEP
        # CAD/STEP file for the imaging system used to collect cloudiness data
```

## Temperature Units
Temperature is represented in degrees Celsius within the surrogate modeling framework. For thermodynamic calculations of the Flory-Huggins free energy (used in the prior mean formulation), temperature is converted to Kelvin, consistent with the standard form of the Flory-Huggins free energy equation. This unit conversion is automatic and internal, so no unit-selection flag is exposed to users. 

## Additional Note: 
The results in the manuscript were generated without enforcing a fixed random seed. An optional NumPy seed (np_RNG_seed) is used in the notebooks to enable deterministic behavior for reproducibility and debugging. 