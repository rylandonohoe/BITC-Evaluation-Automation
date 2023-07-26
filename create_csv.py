import os
import pandas as pd

import diagnostics.line_crossing_template as diag_LineC_T
import diagnostics.line_crossing as diag_LineC
import diagnostics.letter_cancellation as diag_LetC
import diagnostics.star_cancellation as diag_StarC
import diagnostics.star_copying as diag_CopyStar
import diagnostics.diamond_copying as diag_CopyDiamond
import diagnostics.flower_copying as diag_CopyFlower
import diagnostics.line_bisection as diag_LineB

pathname = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation" # path to cloned BIT-Screening-Automation directory
data = []

# iterate through templates subdirectory to determine template constants
templates_folder_path = os.path.join(pathname, "templates")
for file_name in os.listdir(templates_folder_path):
    file_path = os.path.join(templates_folder_path, file_name)
    if os.path.isfile(file_path):
        if file_name == "LineC_T.png":
             LineC_T_C1 = diag_LineC_T.process_image(file_path) # line cancellation task template, constant 1: coordinates of centroids of non-central lines
                    
# iterate through patients subdirectory to determine patient scores
patients_folder_path = os.path.join(pathname, "patients")
for folder_name in os.listdir(patients_folder_path):
    folder_path = os.path.join(patients_folder_path, folder_name)
    if os.path.isdir(folder_path):
        row_data = {'Pseudonym': folder_name,
                    'LineC': None, # line cancellation task: number of lines crossed
                    'LineC_SV': None, # line cancellation task: standard value of number of lines crossed
                    'LetC': None, # letter cancellation task: number of letters crossed
                    'LetC_SV': None, # letter cancellation task: standard value of number of letters crossed
                    'StarC': None, # star cancellation task: number of stars crossed
                    'StarC_SV': None, # star cancellation task: standard value of number of stars crossed
                    'CopyStar': None, # star copying task: final score
                    'CopyStar_SV': None, # star copying task: standard value of final score
                    'CopyDiamond': None, # diamond copying task: final score
                    'CopyDiamond_SV': None, # diamond copying task: standard value of final score
                    'CopyFlower': None, # flower copying task: final score
                    'CopyFlower_SV': None, # flower copying task: standard value of final score
                    'LineB': None, # line bisection task: final score
                    'LineB_SV': None, # line bisection task: standard value of final score
                    'Total': None, # total score (only calculated if patient file contains all 5 completed tasks)
                    'Total_SV': None, # standard value of total score (only calculated if patient file contains all 5 completed tasks)
                    'LineC_LS': None, # line cancellation task: number of lines crossed on left side
                    'LineC_RS': None, # line cancellation task: number of lines crossed on right side
                    'LineC_HCoC': None, # line cancellation task: horizontal centre of cancellation
                    'LineC_VCoC': None, # line cancellation task: vertical centre of cancellation
                    'LetC_LS': None, # letter cancellation task: number of letters crossed on left side
                    'LetC_RS': None, # letter cancellation task: number of letters crossed on right side
                    'LetC_HCoC': None, # letter cancellation task: horizontal centre of cancellation
                    'LetC_VCoC': None, # letter cancellation task: vertical centre of cancellation
                    'StarC_LS': None, # star cancellation task: number of stars crossed on left side
                    'StarC_RS': None, # star cancellation task: number of stars crossed on right side
                    'StarC_HCoC': None, # star cancellation task: horizontal centre of cancellation
                    'StarC_VCoC': None, # star cancellation task: vertical centre of cancellation
                    'CopyStar_F': None, # star copying task: form score
                    'CopyStar_D': None, # star copying task: detail score
                    'CopyStar_A': None, # star copying task: arrangement score
                    'CopyDiamond_F': None, # diamond copying task: form score
                    'CopyDiamond_D': None, # diamond copying task: detail score
                    'CopyDiamond_A': None, # diamond copying task: arrangement score
                    'CopyFlower_F': None, # flower copying task: form score
                    'CopyFlower_D': None, # flower copying task: detail score
                    'CopyFlower_A': None, # flower copying task: arrangement score
                    'LineB_T': None, # line bisection task: score of top line on L or R side
                    'LineB_M': None, # line bisection task: score of middle line on L or R side
                    'LineB_B': None, # line bisection task: score of bottom line on L or R side
                    'LineB_HCoC': None} # line bisection task: horizontal centre of cancellation
        
        # iterate through files in pseudonym folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_name in ["LineC.png", "LineC.jpg", "LineC.jpeg"]:
                    row_data['LineC_LS'], row_data['LineC_RS'], row_data['LineC'], row_data['LineC_SV'], row_data['LineC_HCoC'], row_data['LineC_VCoC'] = diag_LineC.process_image(file_path, LineC_T_C1)
                elif file_name in ["LetC.png", "LetC.jpg", "LetC.jpeg"]:
                    row_data['LetC_LS'], row_data['LetC_RS'], row_data['LetC'], row_data['LetC_SV'], row_data['LetC_HCoC'], row_data['LetC_VCoC'] = diag_LetC.process_image(file_path, folder_name)
                elif file_name in ["StarC.png", "StarC.jpg", "StarC.jpeg"]:
                    row_data['StarC_LS'], row_data['StarC_RS'], row_data['StarC'], row_data['StarC_SV'], row_data['StarC_HCoC'], row_data['StarC_VCoC'] = diag_StarC.process_image(file_path, folder_name)
                elif file_name in ["Copy.png", "Copy.jpg", "Copy.jpeg"]:
                    row_data['CopyStar_F'], row_data['CopyStar_D'], row_data['CopyStar_A'], row_data['CopyStar'], row_data['CopyStar_SV'] = diag_CopyStar.process_image(file_path)
                    row_data['CopyDiamond_F'], row_data['CopyDiamond_D'], row_data['CopyDiamond_A'], row_data['CopyDiamond'], row_data['CopyDiamond_SV'] = diag_CopyDiamond.process_image(file_path)
                    row_data['CopyFlower_F'], row_data['CopyFlower_D'], row_data['CopyFlower_A'], row_data['CopyFlower'], row_data['CopyFlower_SV'] = diag_CopyFlower.process_image(file_path)
                elif file_name in ["LineB.png", "LineB.jpg", "LineB.jpeg"]:
                    row_data['LineB_T'], row_data['LineB_M'], row_data['LineB_B'], row_data['LineB'], row_data['LineB_SV'], row_data['LineB_HCoC'] = diag_LineB.process_image(file_path)
                else:
                    continue
        
        # calculate total score and standard value of total score if all tasks have been scored
        scores = ['LineC', 'LineC_SV', 'LetC', 'LetC_SV', 'StarC', 'StarC_SV', 'CopyStar', 'CopyStar_SV', 'CopyDiamond', 'CopyDiamond_SV', 'CopyFlower', 'CopyFlower_SV', 'LineB', 'LineB_SV']
        if all(row_data[score] is not None for score in scores):
            row_data['Total'] = row_data['LineC'] + row_data['LetC'] + row_data['StarC'] + row_data['CopyStar'] + row_data['CopyDiamond'] + row_data['CopyFlower'] + row_data['LineB']
            row_data['Total_SV'] = round((row_data['LineC_SV'] + row_data['LetC_SV'] + row_data['StarC_SV'] + row_data['CopyStar_SV'] + row_data['CopyDiamond_SV'] + row_data['CopyFlower_SV'] + row_data['LineB_SV']) / 7, 1)
        
        data.append(row_data)

df = pd.DataFrame(data)
df.to_csv("patients.csv")