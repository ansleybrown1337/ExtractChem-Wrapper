'''
Created by A.J. Brown
09/20/19
Script for creating ExtractChem batch files (.xcb) from a .csv file with inputs,
in this case, my custom file named, "AJBatch.csv"


updated 01/06/2020 to include NO3
'''
import glob
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def create_df(dir, input='AJBatch', type='.csv'):
    #creates directories and filepaths for a computers with windows OS
    dir = Path(dir)
    df = pd.read_csv(str(dir) + '\\' + input + type)
    return df

def batch_file(dir, initial_type='Paste', output='batchResult', input='AJBatch',
    water_change=False, gypsum=True, calcite=True, ion_mult=1):
    type=initial_type
    '''
    Creates batch file (.xcb) for ExtractChem model from .csv made from balanced
    input file generated in xcel with the following order of column headings:
     "Year, Field, Depth, BulkD, ssatGWC, pasteGWC, Na, Ca, Mg, K, HCO3, SO4,
      Cl, NO3, CaCO3, Gypsum, Gapon Ca/Mg, Gapon Ca/Na, Gapon Ca/K,
      CEC meq/100g, CEC meq/kg"

    dir = directory folder where input file is, and where output file will be
          stored
    type = whether sample is Paste extract or soil solution access tube (ssat)
           extract (i.e. pore water or ECpw) (must be 'Paste' or 'ssat')
    output = name of output file without file extension (assumed .xcb)
    input = name of input file without file extension (assumed .csv)

    '''
    #creates directories and filepaths for a computers with windows OS
    dir = Path(dir)
    output_file=str(dir)+'\\'+output+f'_{type}'+'.xcb'

    #This lets you change wether the initial and final water contents are the
    #same, and which direction it changes.
    if water_change==False and type=='Paste':
        type1 = type
        print('Final EC = ECe, and water content is unchanged')
    elif water_change==False and type=='ssat':
        type1 = type
        print('Final EC = ECpw, and water content is unchanged')
    elif water_change==True and type=='Paste':
        type1 = 'ssat'
        print('Final EC = ECpw, and water content is changed')
    elif water_change==True and type=='ssat':
        type1 = 'Paste'
        print('Final EC = ECe, and water content is changed')

    #read in input file .csv
    df1 = create_df(dir, input=input, type='.csv')
    #drop columns that are identifiers (i.e. Year, Field, Depth)
    df = df1.drop(df1.columns[0:3], axis=1)
    #assign values for variables in .xcb that are constant
    (df['B'], df['unknown'], df['KomCa/Na'], df['KomCa/Mg'], df['KomCa/K'],
    df['SA'], df['OC'], df['IOC'], df['Al'], df['NaX'], df['CaX'], df['MgX'],
    df['BX'], df['KX'], df['Logic1'], df['Logic2'], df['MinCrit'], df['OrgC'],
    df['T'])=(0,2,0.24,10,0.36,100,7.6,3.6,0.7,0,0,0,0,0,1,1,0.00001,0, 25)

    # 'Paste CO2 = 0.10kPa', 'field CO2 = 1.3kPa', 'default CO2=0.01atm'
    #df['CO2']=0.01
    if type=='Paste':
        #df['CO2'] = 0.1*0.00986923 #convert kPa into atm; originally 0.1
        df['CO2'] = ((0.02*df['HCO3']**2.3532)+0.01)*0.00986923
        #^^^ optimized CO2 curve based on HCO3 solution concentrations
        #df['PasteGaponCaMg'] = df['PasteGaponCaMg']*1.1
        #df['PasteGaponCaNa'] = df['PasteGaponCaNa']*0.75
        #df['PasteGaponCaK'] = df['PasteGaponCaK']*1.1
        #df['PasteGWC']=df['PasteGWC']*1.1
        df['Na']=df['Na']*ion_mult
        df['Ca']=df['Ca']*ion_mult
        df['SO4']=df['SO4']*ion_mult
        df['Cl']=df['Cl']*ion_mult
        print(f'ions at {ion_mult*100}%')

    else:
        #df['CO2'] = 7.5*0.00986923 #convert kPa into atm; originally 1.3
        df['CO2'] = ((0.035*df['HCO3']**2.05)+0.01)*0.00986923
        df['Na']=df['Na']*ion_mult
        df['Ca']=df['Ca']*ion_mult
        df['SO4']=df['SO4']*ion_mult
        df['Cl']=df['Cl']*ion_mult
        print(f'ions at {ion_mult*100}%')
        '''
        df['Na']=df['Na']*0.90
        df['Ca']=df['Ca']*0.92
        df['SO4']=df['SO4']*0.915
        df['Cl']=df['Cl']*0.915
        '''

    #convert gypsum and calcite and CEC into meq/kg (my lab reports in meq/100g)
    if gypsum==True:
        df['Gypsum'] = df['Gypsum']*10
        print('Gypsum is included in batch file')
    else:
        df['Gypsum'] = 0
        print('Gypsum is NOT included in batch file')
    if calcite==True:
        df['CaCO3'] = df['CaCO3']*10
        print('Calcite is included in batch file')
    else:
        df['CaCO3'] = 0
        print('Calcite is NOT included in batch file')
    df['CEC'] = df['CEC']*10

    #re-arrange columns for proper export into .xcb file
    df=df[['Na', 'Ca', 'Mg', 'B', 'K', 'HCO3', 'SO4', 'Cl', 'NO3', 'NaX',
    'CaX', 'MgX', 'BX', 'KX', 'Logic1', 'Logic2', 'CEC', 'OrgC', 'BulkD', 'CO2',
    'T', f'{type}GWC', f'{type1}GWC', 'unknown', 'MinCrit', f'{type}GaponCaMg',
    f'{type}GaponCaNa', f'{type}GaponCaK','KomCa/Na', 'KomCa/Mg', 'KomCa/K',
    'CaCO3', 'Gypsum', 'SA', 'OC', 'IOC', 'Al']]

    with open(output_file, 'w') as outfile:
        outfile.write('ExtractChem,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'+'\n')
        outfile.write('2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'+'\n')
        rows=len(df.index)
        outfile.write(f'{rows},,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'+'\n')
        df.to_csv(outfile, mode='a', header=False, index=False)

    return df

def merge_out(dir, output="result_all.csv", subfolder_layers=0, csv=True):
    '''
    Takes data from .xco files (output of ExtractChem for 1 run), labels each
    value, and merges it into one master .csv where each row is a single run
    from ExtractChem. NO underscores allowed in .xco file names!!!

    dir: working directory containing .xco (or subfolders if applicable) files
         are contained.

    output: name of optional output .csv
            subfolder_layers: number of subfolder layers to sort through until
            .xco files are found
    csv: if True, then .csv file will be generated as output, otherwise df will
         be only return

    Note: .xco file names must take the following format for proper labelling:
          "Dataloggername_Depth"
    '''
    dir = Path(dir)

    read_files = glob.glob(str(dir)+subfolder_layers*'\\*'+'\\*.xco')
    read_extra_files = glob.glob(str(dir)+subfolder_layers*'\\*'+'\\*.xcm')
    cols = ['Sodium Adsorption Ratio','pH','OP_mmH2O','Initial EC','Final EC',
    'Initial GWC','Final GWC','drop1','drop2','Calcium_result_solution',
    'Magnesium_result_solution','Sodium_result_solution',
    'Potassium_result_solution','Bicarbonate_result_solution',
    'Sulphate_result_solution','Chloride_result_solution',
    'Nitrate_result_solution','Boron_result_solution',
    'drop3','Sodium_result_exchange','Calcium_result_exchange',
    'Magnesium_result_exchange','Boron_result_exchange',
    'Potassium_result_exchange','Sodium_equillibrium_solution',
    'Calcium_equillibrium_solution','Magnesium_equillibrium_solution',
    'Boron_equillibrium_solution','Potassium_equillibrium_solution',
    'Bicarbonate_equillibrium_solution','Sulphate_equillibrium_solution',
    'Chloride_equillibrium_solution','Nitrate_equillibrium_solution',
    'Sodium_equillibrium_exchange','Calcium_equillibrium_exchange',
    'Magnesium_equillibrium_exchange','Boron_equillibrium_exchange',
    'Potassium_equillibrium_exchange']
    #list of delimiters in the file name for listing later
    #example file name: "1_Batch_Paste.xco"
    delimiters = ['_', '\\','.']
    #might need to change delimiters for other OS platforms
    regexPattern = '|'.join(map(re.escape, delimiters))
    id_list = []
    type_list = []
    df_master = pd.DataFrame(columns=cols)
    for f in read_files:
        df = pd.read_csv(str(f), skiprows=2, index_col=False, names=cols)
        id_list.append(re.split(regexPattern, str(f))[-4])
        type_list.append(re.split(regexPattern, str(f))[-2])
        df_master = df_master.append(df)
    df_master['ID'] = id_list
    df_master['type'] = type_list
    df_master['ID'] = df_master['ID'].astype('int64')
    df_master = df_master.drop(['drop1', 'drop2', 'drop3'], axis=1)

    if csv == True:
        df_master.set_index(df_master['ID'],drop=True).to_csv(
        str(dir)+'\\'+output)

    for file in read_files:
        os.remove(file)
    for file in read_extra_files:
        os.remove(file)

    return df_master.set_index(df_master['ID'])

def final_result(directory, batch='AJBatch.csv', result='result_all.csv', final_type='Paste'):
    batch_df = pd.read_csv(directory+'\\'+batch)
    result_df = pd.read_csv(directory+'\\'+result)
    df = batch_df.merge(result_df, on='ID')
    if final_type=='Paste':
        df['CO2'] = ((0.02*df['HCO3']**2.3532)+0.01)*0.00986923
        df['Rel_diff_ECe'] = (df['Final EC'] - df['ECe'])/df['ECe']*100
    else:
        df['CO2'] = ((0.0102*batch_df['HCO3']**2.3532)+0.01)*0.00986923
        df['Rel_diff_ECpw'] = (df['Final EC'] - df['ECpw'])/df['ECpw']*100
    return df
