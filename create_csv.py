import os
import pandas as pd
import diagnostics.line_cancellation as diag_LineC
import diagnostics.diamond_drawing as diag_DrawDiamond

pathname = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients" # path to directory containing pseudonym folders
data = []

# iterate through the pseudonym folders
for folder_name in os.listdir(pathname):
    folder_path = os.path.join(pathname, folder_name)
    if os.path.isdir(folder_path):
        row_data = {'Pseudonym': folder_name,
                    'BIT_LineC': None, # line cancellation test: number of lines crossed
                    'BIT_LineC_SV': None, # line cancellation test: standard value of number of lines crossed
                    'BIT_LineC_LS': None, # line cancellation test: number of lines crossed on left side
                    'BIT_LineC_RS': None, # line cancellation test: number of lines crossed on right side
                    'BIT_LetC': None,
                    'BIT_LetC_SV': None,
                    'BIT_StarC': None,
                    'BIT_StarC_SV': None,
                    'BIT_DrawStar': None,
                    'BIT_DrawStar_SV': None,
                    'BIT_DrawDiamond': None, # diamond drawing test: final score
                    'BIT_DrawDiamond_SV': None, # diamond drawing test: standard value of final score
                    'BIT_DrawFlower': None,
                    'BIT_DrawFlower_SV': None,
                    'BIT_LineB': None,
                    'BIT_LineB_SV': None,
                    'BIT_DrawClock': None,
                    'BIT_DrawClock_SV': None}
        # iterate through files in pseudonym folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_name == "LineC.png":
                    row_data['BIT_LineC'] = diag_LineC.process_image(file_path)[0]
                    row_data['BIT_LineC_SV'] = diag_LineC.process_image(file_path)[1]
                    row_data['BIT_LineC_LS'] = ...
                    row_data['BIT_LineC_RS'] = ...
                elif file_name == "LetC.png":
                    row_data['BIT_LetC'] = ...
                    row_data['BIT_LetC_SV'] = ...
                elif file_name == "StarC.png":
                    row_data['BIT_StarC'] = ...
                    row_data['BIT_StarC_SV'] = ...
                elif file_name == "DrawStar.png":
                    row_data['BIT_DrawStar'] = ...
                    row_data['BIT_DrawStar_SV'] = ...
                elif file_name == "DrawDiamond.png":
                    row_data['BIT_DrawDiamond'] = ...
                    row_data['BIT_DrawDiamond_SV'] = ...
                elif file_name == "DrawFlower.png":
                    row_data['BIT_DrawFlower'] = ...
                    row_data['BIT_DrawFlower_SV'] = ...
                elif file_name == "LineB.png":
                    row_data['BIT_LineB'] = ...
                    row_data['BIT_LineB_SV'] = ...
                elif file_name == "DrawClock.png":
                    row_data['BIT_DrawClock'] = ...
                    row_data['BIT_DrawClock_SV'] = ...
                else:
                    continue
        data.append(row_data)

df = pd.DataFrame(data)
df.to_csv(os.path.basename(pathname) + ".csv")