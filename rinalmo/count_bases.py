import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('C:/Users/aquar/PycharmProjects/PTR_project/PTR_ratio_prediction/data/paper/Table_EV4.tsv', sep='\t')

# Calculate lengths
utr5_lengths = df['UTR5_Sequence'].str.len()
utr3_lengths = df['UTR3_Sequence'].str.len()

count_5p = (df['UTR5_Sequence'].str.len() > 10000).sum()
count_3p = (df['UTR3_Sequence'].str.len() > 10000).sum()

print(f"5' UTR sequences >10000 nt: {count_5p}")
print(f"3' UTR sequences >10000 nt: {count_3p}")

'''
5' UTR sequences >5000 nt: 0
3' UTR sequences >5000 nt: 585
'''
'''
5' UTR sequences >7000 nt: 0
3' UTR sequences >7000 nt: 241
'''

'''
5' UTR sequences >10000 nt: 0
3' UTR sequences >10000 nt: 68
'''


'''
plt.figure()
plt.hist(utr5_lengths, bins=50, alpha=0.5, label="5' UTR")
plt.hist(utr3_lengths, bins=50, alpha=0.5, label="3' UTR")
plt.xlabel("UTR Length (nt)")
plt.ylabel("Frequency")
plt.title("Combined Distribution of 5' and 3' UTR Lengths")
plt.legend()
plt.tight_layout()
plt.show()
'''