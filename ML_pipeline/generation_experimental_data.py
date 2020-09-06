
'''
AUTHOR: Eva Serrano-Candelas
DATA: 20190821

Generation of a single txt file taht contains the SMILES and known y

· Elimination of duplicated SMILES in the input dataset
· Generation of dictionary with Input smiles and experimental dataset

INPUT:

    INPUT_DS: [file] Specify the input data obtained as a result of
    "Preprocessing_inputdataset.py" script (named as _v1) , named as:
    model_name-exp.txt

'''

import pandas as pd



# MELTING POINT
INPUT_DS = pd.read_csv('./data/physico-chemical/Melting_Point/v2/MP_dragon_descriptors_sanitized.csv', sep=';')
model_name = "melting_point"

# SELF_IGNITION
INPUT_DS = pd.read_csv('./data/physico-chemical/self_ignition/v2/self_ignition_dataset_ACS_sanitized.csv', sep=';')
model_name = "self_ignition"

# VAPOUR PRESSURE
INPUT_DS = pd.read_csv('./data/physico-chemical/VP/v2/VP_smiles_logVP_sanitized.csv', sep=';')
model_name = "vapour_pressure"

# LOGKOW
INPUT_DS = pd.read_csv('./data/physico-chemical/LogKow/v2/LOGP_descriptors_sanitized.csv', sep=';')
model_name = "log_kow"

# SURFACE TENSION
INPUT_DS = dataset = pd.read_csv('./data/physico-chemical/surface_tension/v2/ST_descriptors_sanitized.csv', sep=';')
model_name = "surface_tension"

# WATER SOLUBILITY
INPUT_DS = pd.read_csv('./data/physico-chemical/Water_solubility/v2/WS_sanitized.csv', sep=';', low_memory=False)
model_name = "water_solubility"

# BOILING POINT
INPUT_DS = pd.read_csv('./data/physico-chemical/boiling_point/v2/boiling_point_sanitized.csv', sep=';')
model_name = "boiling_point"

# SORPTION
INPUT_DS = pd.read_csv('./data/Ecotox/sorption/v2/sorption_dataset_sanitized.csv', sep=';')
model_name = "sorption"

# SLUDGE INHIBITION
INPUT_DS = pd.read_csv('./data/Ecotox/sludge/sludge_inhibition_qualitative_sanitized.csv', sep=';')
model_name = "sludge_inhibition"

# ALGAE GROWTH INHIBITION
INPUT_DS = pd.read_csv('./data/Ecotox/Algae_growth_inhibition/v1/Algae_growth_inhibition_sanitized.csv', sep=';')
model_name = "Algae_growth_inhibition"

# DAPHNIA ACUTE TOXICITY
INPUT_DS = pd.read_csv('./data/Ecotox/daphnia/v2/dataset_DAPHNIA_EC50_sanitized.csv', sep=';')
model_name = "daphnia_acute_toxicity"

# FISH ACUTE TOXICITY
INPUT_DS = pd.read_csv('./data/Ecotox/fish_acute/v2/fish_acute_toxicity_sanitized.csv', sep = ';')
model_name = "fish_acute_toxicity"

# REPEATED DOSE 28 DAYS
INPUT_DS = pd.read_csv('./data/Toxicology/repeated_dose_28_days/v2/repeated_dose_28_days_sanitized.txt', sep=';')
model_name = "repeated_dose_28_days"

# ACUTE ORAL TOXICITY
INPUT_DS = pd.read_csv('./data/Toxicology/Acute_Oral/v2/Acute_Oral_sanitized.csv', sep=';')
model_name = "acute_oral_toxicity"

# REPEATED DOSE 90 DAYS
INPUT_DS = pd.read_csv('./data/Toxicology/repeated_dose_90_days/v2/repeated_dose_90_days_echachem_sanitized.csv', sep=';')
model_name = "repeated_dose_90_days"

# READY BIODEGRADABILITY
INPUT_DS = pd.read_csv('./data/Ecotox/biodegradability/v2/biodegradability_sanitized.csv', sep=';')
model_name = "ready_biodegradability"

# SKIN SENSITIZATION (IVCAM)
INPUT_DS = pd.read_csv('./data/Toxicology/Skin_sens/v2/skin_sensit_sanitized.csv', sep=';')
model_name = "skin_sensitisation_icvam"

# CHROMOSOMAL ABERRATION
INPUT_DS = pd.read_csv('./data/Toxicology/Chrom_aberr/v2/Chrom_aberr_sanitized.csv', sep=';')
model_name = "chromosomal_aberration"

# DEVELOPMENTAL TOXICITY
INPUT_DS = pd.read_csv('./data/Toxicology/Developmental_tox/v2/DEVTOX_sanitized.csv', sep=';')
model_name = "developmental_toxicity"

# EYE IRRITATION
INPUT_DS = pd.read_csv('./data/Toxicology/cosmetics_eye/v2/cosmetics_eye_sanitized.csv', sep=';')
model_name = "eye_irritation"

# GENOTOXICITY (MICRONUCLEOUS)
INPUT_DS = pd.read_csv('./data/Toxicology/in_vivo_micronuclei/v2/in_vivo_micronuclei_sanitized.csv', sep=';')
model_name = "genotoxicity_micronucleus"

# SKIN IRRITATION
INPUT_DS = pd.read_csv('./data/Toxicology/skin_irr/v2/skin_irritation_sanitized.csv', sep=';')
model_name = "skin_irritation"

# MUTAGENICITY (ames)
INPUT_DS = pd.read_csv('./data/Toxicology/mutagenicity_ames/v2/mutagenicity_ames_sanitized.csv', sep=';')
model_name = "mutagenicity_ames"

# CARCINOGENICITY
INPUT_DS = pd.read_csv('./data/Toxicology/Carcinogenicity/v2/Carcinogenicity_sanitized.csv', sep=';')
model_name = "carcinogenicity"

#BIOACUMULATING
INPUT_DS = pd.read_csv('./data/Ecotox/bioconcentration_factor/v2/bioconcentration_factor_sanitized.csv', sep = ';')
model_name = "bioconcentration_factor"

#PERSISTENCE SEDIMENTS
INPUT_DS = pd.read_csv('./data/Ecotox/persist_sed/v1/PERS_SED_sanitized.csv', sep=';')
model_name = "persistence_sediment"

#PERSISTENCE SOIL
INPUT_DS = pd.read_csv('./data/Ecotox/persist_soil/v1/PERS_SOIL_sanitized.csv', sep=';')
model_name = "persistence_soil"

#PERSISTENCE WATER
INPUT_DS = pd.read_csv('./data/Ecotox/persis_water/v1/PERS_WATER_sanitized.csv', sep=';')
model_name = "persistence_water"

#DAPHNIA CHRONIC
INPUT_DS = pd.read_csv('./data/Ecotox/daphnia_chronic/daphnia_JMoe_chronic_sanitized_log_fuera_outliers.csv', sep=';')
model_name = "daphnia_chronic_toxicity"

#EARTHWORM ACUTE TOXICITY
INPUT_DS = pd.read_csv('./data/Ecotox/earthworm_acute_toxicity/earthworm_cleaned_RAW_calculated_with_y.txt', sep=',')
model_name = "earthworm_acute_toxicity"

#AVIAN ACUTE TOXICITY
INPUT_DS = pd.read_csv('./data/Ecotox/avian_reproduction_toxicity/bobwhite_filtered_by_units_sanitized.csv', sep = ';')
model_name = "avian_reproduction_toxicity"

#PHOTOTOXICITY
INPUT_DS = pd.read_csv('./data/Models_not_in_REACH/phototoxicity/phototoxicity_sanitized.csv', sep=';')
model_name = "phototoxicity"

#INHALED TOXICITY
INPUT_DS = pd.read_csv('./data/Toxicology/inhaled_toxicity/RAT_inhaled_final_sanitized.csv', sep=';')
model_name = "inhaled_toxicity"


INPUT_DS.shape

ONLY_EXP_df = INPUT_DS[["SMILES","y"]]

ONLY_EXP_df.shape

ONLY_EXP_df.to_csv('{}-exp.txt'.format(model_name), sep='\t', index=False)
