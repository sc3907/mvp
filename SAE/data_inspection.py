import pandas as pd

path = "~/Dropbox/missense_pred/data/Ben/input_data.HS.csv"
df = pd.read_csv(path)

count = df['target'].value_counts()

print(count)
