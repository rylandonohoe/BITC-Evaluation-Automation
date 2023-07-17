import os
import pandas as pd

import diagnostics.line_cancellation_template as diag_LineC_T
import diagnostics.line_cancellation as diag_LineC
import diagnostics.letter_cancellation as diag_LetC
import diagnostics.star_cancellation as diag_StarC
import diagnostics.star_drawing as diag_DrawStar
import diagnostics.diamond_drawing as diag_DrawDiamond
import diagnostics.line_bisection as diag_LineB

pathname = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation" # path to directory containing your added patients in the patient folder
data = []

# iterate through templates folder to determine template constants
templates_folder_path = os.path.join(pathname, "templates")
for file_name in os.listdir(templates_folder_path):
    file_path = os.path.join(templates_folder_path, file_name)
    if os.path.isfile(file_path):
        if file_name == "LineC_T.png":
             LineC_T_C1 = diag_LineC_T.process_image(file_path) # line cancellation test template, constant 1: coordinates of centroids of non-central lines
                    
# iterate through patients folder to determine patient scores
patients_folder_path = os.path.join(pathname, "patients")
for folder_name in os.listdir(patients_folder_path):
    folder_path = os.path.join(patients_folder_path, folder_name)
    if os.path.isdir(folder_path):
        row_data = {'Pseudonym': folder_name,
                    'BIT_LineC': None, # line cancellation test: number of lines crossed
                    'BIT_LineC_SV': None, # line cancellation test: standard value of number of lines crossed
                    'BIT_LetC': None, # letter cancellation test: number of letters crossed
                    'BIT_LetC_SV': None, # letter cancellation test: standard value of number of letters crossed
                    'BIT_StarC': None, # star cancellation test: number of stars crossed
                    'BIT_StarC_SV': None, # star cancellation test: standard value of number of stars crossed
                    'BIT_DrawStar': None, # star drawing test: final score
                    'BIT_DrawStar_SV': None, # star drawing test: standard value of final score
                    'BIT_DrawDiamond': None, # diamond drawing test: final score
                    'BIT_DrawDiamond_SV': None, # diamond drawing test: standard value of final score
                    'BIT_DrawFlower': None,
                    'BIT_DrawFlower_SV': None,
                    'BIT_LineB': None, # line bisection test: final score
                    'BIT_LineB_SV': None, # line bisection test: standard value of final score
                    'BIT_DrawClock': None,
                    'BIT_DrawClock_SV': None,
                    'BIT_LineC_LS': None, # line cancellation test: number of lines crossed on left side
                    'BIT_LineC_RS': None, # line cancellation test: number of lines crossed on right side
                    'BIT_LineC_HCoC': None, # line cancellation test: horizontal centre of cancellation
                    'BIT_LineC_VCoC': None, # line cancellation test: vertical centre of cancellation
                    'BIT_LetC_LS': None, # letter cancellation test: number of letters crossed on left side
                    'BIT_LetC_RS': None, # letter cancellation test: number of letters crossed on right side
                    'BIT_LetC_HCoC': None, # letter cancellation test: horizontal centre of cancellation
                    'BIT_LetC_VCoC': None, # letter cancellation test: vertical centre of cancellation
                    'BIT_StarC_LS': None, # star cancellation test: number of stars crossed on left side
                    'BIT_StarC_RS': None, # star cancellation test: number of stars crossed on right side
                    'BIT_StarC_HCoC': None, # star cancellation test: horizontal centre of cancellation
                    'BIT_StarC_VCoC': None, # star cancellation test: vertical centre of cancellation
                    'BIT_DrawStar_F': None, # star drawing test: form score
                    'BIT_DrawStar_D': None, # star drawing test: detail score
                    'BIT_DrawStar_A': None, # star drawing test: arrangement score
                    'BIT_DrawDiamond_F': None, # diamond drawing test: form score
                    'BIT_DrawDiamond_D': None, # diamond drawing test: detail score
                    'BIT_DrawDiamond_A': None, # diamond drawing test: arrangement score
                    'BIT_LineB_T': None, # line bisection test: score of top line on L or R side
                    'BIT_LineB_M': None, # line bisection test: score of middle line on L or R side
                    'BIT_LineB_B': None, # line bisection test: score of bottom line on L or R side
                    'BIT_LineB_HCoC': None} # line bisection test: horizontal centre of cancellation
                    
        # iterate through files in pseudonym folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_name == "LineC.png":
                    row_data['BIT_LineC_LS'], row_data['BIT_LineC_RS'], row_data['BIT_LineC'], row_data['BIT_LineC_SV'], row_data['BIT_LineC_HCoC'], row_data['BIT_LineC_VCoC'] = diag_LineC.process_image(file_path, LineC_T_C1)
                elif file_name == "LetC.png":
                    row_data['BIT_LetC_LS'], row_data['BIT_LetC_RS'], row_data['BIT_LetC'], row_data['BIT_LetC_SV'], row_data['BIT_LetC_HCoC'], row_data['BIT_LetC_VCoC'] = diag_LetC.process_image(file_path, folder_name)
                elif file_name == "StarC.png":
                    row_data['BIT_StarC_LS'], row_data['BIT_StarC_RS'], row_data['BIT_StarC'], row_data['BIT_StarC_SV'], row_data['BIT_StarC_HCoC'], row_data['BIT_StarC_VCoC'] = diag_StarC.process_image(file_path, folder_name)
                elif file_name == "Draw.png":
                    row_data['BIT_DrawStar_F'], row_data['BIT_DrawStar_D'], row_data['BIT_DrawStar_A'], row_data['BIT_DrawStar'], row_data['BIT_DrawStar_SV'] = diag_DrawStar.process_image(file_path)
                    row_data['BIT_DrawDiamond_F'], row_data['BIT_DrawDiamond_D'], row_data['BIT_DrawDiamond_A'], row_data['BIT_DrawDiamond'], row_data['BIT_DrawDiamond_SV'] = diag_DrawDiamond.process_image(file_path)
                    row_data['BIT_DrawFlower'], row_data['BIT_DrawFlower_SV'] = ..., ...
                elif file_name == "LineB.png":
                    row_data['BIT_LineB_T'], row_data['BIT_LineB_M'], row_data['BIT_LineB_B'], row_data['BIT_LineB'], row_data['BIT_LineB_SV'], row_data['BIT_LineB_HCoC'] = diag_LineB.process_image(file_path)
                elif file_name == "DrawClock.png":
                    row_data['BIT_DrawClock'], row_data['BIT_DrawClock_SV'] = ..., ...
                else:
                    continue
        data.append(row_data)

df = pd.DataFrame(data)
df.to_csv("patients.csv")