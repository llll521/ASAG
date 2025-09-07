import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def get_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for i in file:
            elem_list = i.split("$")
            if len(elem_list) < 4:
                print(f"警告：行 {i} 列数不足（需要4列，实际{len(elem_list)}列）")
                print(f"问题行内容: {i[:100]}...")
                continue
            q, a, l1, l2 = elem_list
            l1 = l1.strip()
            l2 = l2.strip()
            if l2 == 'Off-topic':
                l2 = 0
            elif l2 == 'partly off-topic':
                l2 = 1
            else:
                l2 = 2
            data.append((q, a, l1, l2))
    # print(data)
    return data


class QADataset(Dataset):
    def __init__(self, path, tokenizer_name="bert-base", max_length=128):
        self.data = get_data(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_q, sentence_a, label1, label2 = self.data[idx]
        encoding_q = self.tokenizer(
            sentence_q,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding_a = self.tokenizer(
            sentence_a,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding_q = {key: val.squeeze(0) for key, val in encoding_q.items()}
        encoding_a = {key: val.squeeze(0) for key, val in encoding_a.items()}
        nsolabel_ter1 = torch.tensor(float(label1), dtype=torch.float32)
        nsolabel_ter2 = torch.tensor(int(label2), dtype=torch.long)
        return encoding_q, encoding_a, nsolabel_ter1, nsolabel_ter2


# Example Usage:

if __name__ == "__main__":
    get_data()
