from Bio import SeqIO
import pandas as pd

# 定义空列表来存储数据
labels = []
sequences = []

# 读取fasta文件
# for record in SeqIO.parse("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-train.fa", "fasta"):
for record in SeqIO.parse("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.fa", "fasta"):
    # 将标签和序列添加到列表中
    labels.append(record.id)
    sequences.append(str(record.seq))

# 创建一个数据框
df = pd.DataFrame({
    'label': labels,
    'Seq': sequences
})

# 将数据框写入CSV文件
df.to_csv("/root/autodl-tmp/myDNAPredict/program 1.1/data/Y-test.csv", index=False)
