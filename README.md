# covid-vaccine-evaluation

## Installation

The infection dynamics simulation code is included as a submodule. Clone this repository and initialize submodules with
```bash
git clone git@github.com:akekic/covid-vaccine-policy.git
git submodule init
git submodule update 
```
The inferred parameters for the infection dynamics simulation is stored here. 
Download the data and place it in the corresponding data folder with
```bash
cd covid-vaccine-dynamics/data/traces
wget https://gin.g-node.org/jdehning/covid-vaccine-dynamics-data/raw/488d5b6235be00e37c872ced080af713cdf92d1d/traces/run-begin=2020-12-20-end=2021-12-19-C_mat=70-V1_eff=70-V2_eff=90-V3_eff=95-influx=0.5-draws=500.pkl
wget https://gin.g-node.org/jdehning/covid-vaccine-dynamics-data/raw/488d5b6235be00e37c872ced080af713cdf92d1d/traces/run-begin=2020-12-20-end=2021-12-19-C_mat=80-V1_eff=70-V2_eff=90-V3_eff=95-influx=0.5-draws=500.pkl
wget https://gin.g-node.org/jdehning/covid-vaccine-dynamics-data/raw/488d5b6235be00e37c872ced080af713cdf92d1d/traces/run-begin=2020-12-20-end=2021-12-19-C_mat=90-V1_eff=70-V2_eff=90-V3_eff=95-influx=0.5-draws=500.pkl
cd ../../..
``` 
Install the submodule with
```bash
pip install -e ./covid-vaccine-dynamics
```
Install the main package with
```bash
pip install -e .
```

Convert all notebook files into `.ipynb`

```bash
jupytext notebooks/*.py --to .ipynb
jupytext data_preprocessing/*.py --to .ipynb
```

## Data preparation

The raw input data is stored under `data/input-data`.
The data is prepared with the script `data_preprocessing/israel_data_processing.py`.
The script is run with
```bash
cd data_preprocessing
python israel_data_processing.py
```
This creates preprocessed data in `data/preprocessed_data`.

We then estimate the severity factorisation
```bash
python severity_factorisation.py
cd ..
```

This creates the severity factorisation for different assumed waning timescales:
- regular waning: `data/factorisation-with-fit`
- fast waning: `data/factorisation-with-fit-fast-waning`
- no waning: `data/factorisation-with-fit-no-waning`.



## Hello world

To run a scenario run the main script with
```bash 
python main.py --config=configs/<VACCINE_ALLOCATION_STRATEGY>.yml
```
where `<VACCINE_ALLOCATION_STRATEGY>` is one of the following:
- `observed`,
- `uniform`,
- `young-first`,
- `elderly-first`.

This saves the results in `run/YYYY-MM-DD_HH-MM-SS_<VACCINE_ALLOCATION_STRATEGY>/`.

## Output files

```bash
run_output_directory
├── result.npy
├── result_samples.npy
├── factorisation_data
│   ├── observed_infection_data.csv
│   ├── observed_severity_data.csv
│   ├── population_data.csv
│   ├── risk_factors.csv
│   ├── time_dependence.csv
│   ├── vaccination_data.csv
│   ├── vaccine_acceptance_data.csv
│   ├── vaccine_efficacy_waning_data.csv
│   └── waning_curve.csv
├── factorisation_data
│   ├── age_group_names.npy
│   ├── age_groups.npy
│   ├── D_a.npy
│   ├── observed_vaccinations.npy
│   ├── P_a.npy
│   ├── P_t.npy
│   ├── vaccination_statuses.npy
│   ├── week_dates.npy
│   ├── week_dates_scenario.npy
│   ├── weeks.npy
│   ├── weeks_extended.npy
│   ├── weeks_scenario.npy
│   └── weeks_scenario_extended.npy
├── severity_factorisation
│   ├── f_0.npy
│   ├── f_1.npy
│   ├── g.npy
│   ├── infection_dynamics.csv
│   ├── infection_dynamics_samples.npy
│   ├── median_weekly_base_R_t.npy
│   ├── median_weekly_eff_R_t.npy
│   └── vaccine_efficacy_params.npy
└── vaccination_policy
    ├── U_2.npy
    ├── U_2_full.npy
    ├── u_3.npy
    └── u_3_full.npy

```






