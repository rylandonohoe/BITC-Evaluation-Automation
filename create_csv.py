import os
import pandas as pd
import line_crossing.line_crossing as LineC
#import diamond_drawing.diamond_drawing as DrawDiamond

pathname = "/Users/rylandonohoe/Documents/GitHub/RISE_Germany_2023/BIT-Screening-Automation/patients" # path to the directory containing pseudonym folders
data = []

# iterate through the pseudonym folders
for folder_name in os.listdir(pathname):
    folder_path = os.path.join(pathname, folder_name)
    if os.path.isdir(folder_path):
        row_data = {'Pseudonym': folder_name,
                    'BIT_LineC': None,
                    'BIT_LineC_SV': None,
                    'BIT_LetC': None,
                    'BIT_LetC_SV': None,
                    'BIT_StarC': None,
                    'BIT_StarC_SV': None,
                    'BIT_DrawStar': None,
                    'BIT_DrawStar_SV': None,
                    'BIT_DrawDiamond': None,
                    'BIT_DrawDiamond_SV': None,
                    'BIT_DrawFlower': None,
                    'BIT_DrawFlower_SV': None,
                    'BIT_LineB': None,
                    'BIT_LineB_SV': None,
                    'BIT_DrawClock': None,
                    'BIT_DrawClock_SV': None}
        # iterate through the files in the pseudonym folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_name == "LineC.png":
                    row_data['BIT_LineC'] = LineC.process_image(file_path)[0]
                    row_data['BIT_LineC_SV'] = LineC.process_image(file_path)[1]
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