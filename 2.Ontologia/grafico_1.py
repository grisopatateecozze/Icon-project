import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dei dati
data = pd.read_csv("TCGA_InfoWithGrade.csv")

""""
# Seleziona le colonne desiderate per le domande e is_autistic
selected_columns = ["Grade", "Gender", "Age_at_diagnosis", "Race", "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC",
                    "MUC16", "PIK3CA", "NF1", "PIK3R1", "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A",
                    "IDH2", "FAT4", "PDGFRA"]

# Filtra il dataset con le colonne selezionate
filtered_data = data[selected_columns]
"""

# Creazione della matrice di correlazione
corr_matrix = data.corr()

# Creazione della heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heat Map of Dataset")
plt.show()
