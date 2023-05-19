import pandas as pd
import line_crossing.line_crossing as LineC

data = {'Pseudonym': [None], 
        'BIT_LineC': [LineC.lines_crossed], 
        'BIT_LineC_SV': [LineC.lines_crossed_SV], 
        'BIT_LetC': [None], 
        'BIT_LetC_SV': [None], 
        'BIT_StarC': [None], 
        'BIT_StarC_SV': [None], 
        'BIT_DrawStar': [None], 
        'BIT_DrawStar_SV': [None], 
        'BIT_DrawDiamond': [None], 
        'BIT_DrawDiamond_SV': [None], 
        'BIT_DrawFlower': [None], 
        'BIT_DrawFlower_SV': [None], 
        'BIT_LineB': [None], 
        'BIT_LineB_SV': [None], 
        'BIT_DrawClock': [None], 
        'BIT_DrawClock_SV': [None]}

df = pd.DataFrame(data)
df.to_csv('data.csv')