from Bio import SeqIO
import pandas as pd

# Define empty lists to store data
labels = []
sequences = []

# Read the fasta file
# for record in SeqIO.parse("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-train.fa", "fasta"):
for record in SeqIO.parse("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.fa", "fasta"):
    # Add the label and sequence to the lists
    labels.append(record.id)
    sequences.append(str(record.seq))

# Create a dataframe
df = pd.DataFrame({
    'label': labels,
    'Seq': sequences
})

# Write the dataframe to a CSV file
df.to_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.csv", index=False)
