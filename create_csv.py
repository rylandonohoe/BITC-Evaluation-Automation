import os
import pandas as pd

import diagnostics.line_crossing_template as diag_LineC_T
import diagnostics.line_crossing as diag_LineC
import diagnostics.letter_cancellation as diag_LetC
import diagnostics.star_cancellation as diag_StarC
import diagnostics.star_copying as diag_CopyStar
import diagnostics.diamond_copying as diag_CopyDiamond
import diagnostics.line_bisection as diag_LineB

path = "/Users/rylandonohoe/Documents/GitHub/BITC-Evaluation-Automation" # path to cloned BITC-Evaluation-Automation directory
data = []

# process template of line cancellation task to compute constant
template_path = os.path.join(path, "templates/LineC_T.png")
if os.path.isfile(template_path):
    LineC_Const = diag_LineC_T.process_image(template_path) # coordinates of centroids of non-central lines

# iterate through "patients" subdirectory to determine patient scores
patients_path = os.path.join(path, "patients")
for folder in os.listdir(patients_path):
    folder_path = os.path.join(patients_path, folder)
    if os.path.isdir(folder_path):
        row_data = {"Pseudonym": folder,
                    "LineC": None, # line cancellation task: number of lines crossed
                    "LineC_SV": None, # line cancellation task: standard value of number of lines crossed
                    "LetC": None, # letter cancellation task: number of letters crossed
                    "LetC_SV": None, # letter cancellation task: standard value of number of letters crossed
                    "StarC": None, # star cancellation task: number of stars crossed
                    "StarC_SV": None, # star cancellation task: standard value of number of stars crossed
                    "CopyStar": None, # star copying task: final score
                    "CopyStar_SV": None, # star copying task: standard value of final score
                    "CopyDiamond": None, # diamond copying task: final score
                    "CopyDiamond_SV": None, # diamond copying task: standard value of final score
                    "LineB": None, # line bisection task: final score
                    "LineB_SV": None, # line bisection task: standard value of final score
                    "Total": None, # total score (only calculated if patient file contains all 5 completed tasks)
                    "Total_SV": None, # standard value of total score (only calculated if patient file contains all 5 completed tasks)
                    "LineC_LS": None, # line cancellation task: number of lines crossed on left side
                    "LineC_RS": None, # line cancellation task: number of lines crossed on right side
                    "LineC_HCoC": None, # line cancellation task: horizontal centre of cancellation
                    "LineC_VCoC": None, # line cancellation task: vertical centre of cancellation
                    "LetC_LS": None, # letter cancellation task: number of letters crossed on left side
                    "LetC_RS": None, # letter cancellation task: number of letters crossed on right side
                    "LetC_HCoC": None, # letter cancellation task: horizontal centre of cancellation
                    "LetC_VCoC": None, # letter cancellation task: vertical centre of cancellation
                    "StarC_LS": None, # star cancellation task: number of stars crossed on left side
                    "StarC_RS": None, # star cancellation task: number of stars crossed on right side
                    "StarC_HCoC": None, # star cancellation task: horizontal centre of cancellation
                    "StarC_VCoC": None, # star cancellation task: vertical centre of cancellation
                    "CopyStar_S": None, # star copying task: shape score
                    "CopyStar_D": None, # star copying task: detail score
                    "CopyStar_A": None, # star copying task: arrangement score
                    "CopyStar_AIA": None, # star copying task: all internal angles (clockwise from top-right)
                    "CopyStar_AVA": None, # star copying task: absolute vertical angle
                    "CopyStar_AHA": None, # star copying task: absolute horzontal angle
                    "CopyStar_LA": None, # star copying task: left area
                    "CopyStar_RA": None, # star copying task: right area
                    "CopyStar_TA": None, # star copying task: top area
                    "CopyStar_BA": None, # star copying task: bottom area
                    "CopyStar_AL": None, # star copying task: all lengths (clockwise from top-right)
                    "CopyDiamond_S": None, # diamond copying task: shape score
                    "CopyDiamond_D": None, # diamond copying task: detail score
                    "CopyDiamond_A": None, # diamond copying task: arrangement score
                    "CopyDiamond_TLA": None, # diamond copying task: top left angle
                    "CopyDiamond_TRA": None, # diamond copying task: top right angle
                    "CopyDiamond_BLA": None, # diamond copying task: bottom left angle
                    "CopyDiamond_BRA": None, # diamond copying task: bottom right angle
                    "CopyDiamond_AVA": None, # diamond copying task: absolute vertical angle
                    "CopyDiamond_AHA": None, # diamond copying task: absolute horzontal angle
                    "CopyDiamond_LAR": None, # diamond copying task: left area ratio
                    "CopyDiamond_RAR": None, # diamond copying task: right area ratio
                    "CopyDiamond_TLLR": None, # diamond copying task: top left length ratio
                    "CopyDiamond_TRLR": None, # diamond copying task: top right length ratio
                    "CopyDiamond_BLLR": None, # diamond copying task: bottom left length ratio
                    "CopyDiamond_BRLR": None, # diamond copying task: bottom right length ratio
                    "CopyDiamond_VHLR": None, # diamond copying task: vertical horizontal length ratio
                    "LineB_T": None, # line bisection task: score of top line on L or R side
                    "LineB_M": None, # line bisection task: score of middle line on L or R side
                    "LineB_B": None, # line bisection task: score of bottom line on L or R side
                    "LineB_HCoC": None} # line bisection task: horizontal centre of cancellation
        
        # iterate through files in folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file in ["LineC.png", "LineC.jpg", "LineC.jpeg"]:
                    row_data["LineC_LS"], row_data["LineC_RS"], row_data["LineC"], row_data["LineC_SV"], row_data["LineC_HCoC"], row_data["LineC_VCoC"] = diag_LineC.process_image(file_path, LineC_Const)
                elif file in ["LetC.png", "LetC.jpg", "LetC.jpeg"]:
                    row_data["LetC_LS"], row_data["LetC_RS"], row_data["LetC"], row_data["LetC_SV"], row_data["LetC_HCoC"], row_data["LetC_VCoC"] = diag_LetC.process_image(file_path, folder)
                elif file in ["StarC.png", "StarC.jpg", "StarC.jpeg"]:
                    row_data["StarC_LS"], row_data["StarC_RS"], row_data["StarC"], row_data["StarC_SV"], row_data["StarC_HCoC"], row_data["StarC_VCoC"] = diag_StarC.process_image(file_path, folder)
                elif file in ["Copy.png", "Copy.jpg", "Copy.jpeg"]:
                    row_data["CopyStar_S"], row_data["CopyStar_D"], row_data["CopyStar_A"], row_data["CopyStar"], row_data["CopyStar_SV"], row_data["CopyStar_AIA"], row_data["CopyStar_AVA"], row_data["CopyStar_AHA"], row_data["CopyStar_LA"], row_data["CopyStar_RA"], row_data["CopyStar_TA"], row_data["CopyStar_BA"], row_data["CopyStar_AL"] = diag_CopyStar.process_image(file_path)
                    row_data["CopyDiamond_S"], row_data["CopyDiamond_D"], row_data["CopyDiamond_A"], row_data["CopyDiamond"], row_data["CopyDiamond_SV"], row_data["CopyDiamond_TLA"], row_data["CopyDiamond_TRA"], row_data["CopyDiamond_BLA"], row_data["CopyDiamond_BRA"], row_data["CopyDiamond_AVA"], row_data["CopyDiamond_AHA"], row_data["CopyDiamond_LAR"], row_data["CopyDiamond_RAR"], row_data["CopyDiamond_TLLR"], row_data["CopyDiamond_TRLR"], row_data["CopyDiamond_BLLR"], row_data["CopyDiamond_BRLR"], row_data["CopyDiamond_VHLR"] = diag_CopyDiamond.process_image(file_path)
                elif file in ["LineB.png", "LineB.jpg", "LineB.jpeg"]:
                    row_data["LineB_T"], row_data["LineB_M"], row_data["LineB_B"], row_data["LineB"], row_data["LineB_SV"], row_data["LineB_HCoC"] = diag_LineB.process_image(file_path)
                else:
                    continue
        
        # calculate total score and standard value of total score if all tasks have been scored
        scores = ["LineC", "LineC_SV", "LetC", "LetC_SV", "StarC", "StarC_SV", "CopyStar", "CopyStar_SV", "CopyDiamond", "CopyDiamond_SV", "LineB", "LineB_SV"]
        if all(row_data[score] is not None for score in scores):
            row_data["Total"] = row_data["LineC"] + row_data["LetC"] + row_data["StarC"] + row_data["CopyStar"] + row_data["CopyDiamond"] + row_data["LineB"]
            row_data["Total_SV"] = round((row_data["LineC_SV"] + row_data["LetC_SV"] + row_data["StarC_SV"] + row_data["CopyStar_SV"] + row_data["CopyDiamond_SV"] + row_data["LineB_SV"]) / 6, 1)
        
        data.append(row_data)

df = pd.DataFrame(data)
df.to_csv("patients.csv")