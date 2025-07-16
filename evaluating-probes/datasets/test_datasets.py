import pandas as pd, numpy as np

df = pd.DataFrame({'prompt': np.random.choice(['A', 'B'], size=2000),})
df['prompt_len'] = 1
df['target'] = (df['prompt'] == 'A').astype(int)

df.to_csv("./cleaned/92_letter.csv", index=False)