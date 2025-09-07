import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import KernelPCA
from transformers import AutoModel, AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GatedAdaptiveResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        # 主干特征变换
        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # 更平滑的非线性
            nn.Dropout(dropout),

            # 自适应归一化层（带可学习参数）
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()  # 输出[0,1]门控值
        )

        # 残差连接适配器
        self.res_adapter = (nn.Linear(in_dim, hidden_dim))

    def forward(self, x):
        residual = self.res_adapter(x)
        # 特征变换 + 门控调制
        features = self.feature_net(x)
        gates = self.gate_net(x)
        # 门控特征融合
        modulated = features * gates
        # 残差连接
        return modulated + residual


class EnhancedRegressionHead(nn.Module):
    def __init__(self, in_dim=2304, hidden_dims=[768, 256], dropout=0.2):
        super().__init__()
        self.block=nn.Sequential(
        GatedAdaptiveResBlock(in_dim, hidden_dims[0], dropout),
        # 第二级特征压缩
        GatedAdaptiveResBlock(hidden_dims[0], hidden_dims[1], dropout),
        # 最终输出层
        nn.Linear(hidden_dims[1], 1))
        # 输出范围约束（假设目标在[0,5]）
        self.scale = nn.Parameter(torch.tensor(5.0))  # 可学习缩放因子
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        layer1_out = self.block[0](x)  # 第一层输出 (768维)
        layer2_out = self.block[1](layer1_out)  # 第二层输出 (256维)
        final_out = self.block[2](layer2_out)  # 最终输出 (1维)

        return torch.sigmoid(final_out)* self.scale+self.bias,layer2_out # 输出约束到[0,10]
    def get_final_layer_weights(self):
        """获取最终输出层的权重"""
        return self.block[2].weight.to(device)

class EnhancedClassificationHead(nn.Module):
    def __init__(self, in_dim=2304, hidden_dims=[768, 256],
                 dropout=0.2, num_classes=3):
        super().__init__()
        self.block= nn.Sequential(
        GatedAdaptiveResBlock(in_dim, hidden_dims[0], dropout),
            # 第二级特征压缩
        GatedAdaptiveResBlock(hidden_dims[0], hidden_dims[1], dropout),
            # 最终输出层
        nn.Linear(hidden_dims[1], num_classes))

    def forward(self, x):
        layer1_out = self.block[0](x)  # 第一层输出 (768维)
        layer2_out = self.block[1](layer1_out)  # 第二层输出 (256维)
        final_out = self.block[2](layer2_out)  # 最终输出 (1维)

        return final_out,layer2_out

class Subnet(nn.Module):
    def __init__(self, capsule_params=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained("m-roberta")
        self.norm = nn.LayerNorm(768)  # 2 * 384 = 768

    def forward(self, x):
        bert_outputs = self.bert(**x).last_hidden_state
        bert_output=self.norm(bert_outputs)

        return bert_output

class SNN(nn.Module):
    def __init__(self, subnet):
        super(SNN, self).__init__()
        self.subnet = subnet
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_norm = nn.LayerNorm( 2304)
        self.layernorm = nn.LayerNorm( 2304)
        self.dropout = nn.Dropout(0.5)
        self.regression_head=EnhancedRegressionHead()
        self.classification_head=EnhancedClassificationHead()
        self.att = nn.MultiheadAttention(
            embed_dim=768,  # 输入特征维度
            num_heads=6,  # 注意力头数
            dropout=0.2,  # Dropout概率
            batch_first=True  # 输入格式为 [batch_size, seq_len, embed_dim]

        )

    def forward(self, q, a):
        bert_out_q = self.subnet(q)# [batch, seq_len, 768], [batch, seq_len, 768]
        bert_out_a= self.subnet(a)
        cross_att,_= self.att(bert_out_q, bert_out_a, bert_out_a)
        feature = torch.cat([bert_out_q, bert_out_a,cross_att], dim=-1)  # [batch, seq_len, 3*768]
        feature = self.feature_norm(feature)
        feature_out = feature.transpose(1, 2)  # [batch, 3*768, seq_len]
        pooling_out=self.global_pool(feature_out) #[batch, 3*768, 1]
        pooling_out=np.squeeze(pooling_out) #[batch, 3*768]
        layer_out = self.layernorm(pooling_out)  # [batch, 3*768]
        dropout_out = self.dropout(layer_out)  # [batch, 3*768]

        #分类约束
        regression_out,_= self.regression_head(dropout_out)
        classification_out1,_= self.classification_head(dropout_out)
        target_tensor = torch.tensor([0, 1, 2], device=regression_out.device, dtype=regression_out.dtype)
        regression_out1 = target_tensor - regression_out
        regression_out2 = sum(abs(num) for num in regression_out1)
        regression_out3 = torch.softmax(1 - (regression_out + regression_out1) / regression_out2, dim=1)
        classification_out=torch.softmax(classification_out1+regression_out3,dim=1)
        return regression_out ,classification_out

def test_forward_propagation():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    sentences_q = ["你好，今天的天气如何？", "机器学习是一门有趣的学科。"]
    sentences_a = ["今天天气晴朗，适合出门。", "是的，特别是在数据科学领域。"]

    encoding_q = tokenizer(
        sentences_q,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoding_a = tokenizer(
        sentences_a,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )


    subnet = Subnet(capsule_params=None)
    model = SNN(subnet).cuda()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoding_q = {k: v.to(device) for k, v in encoding_q.items()}
    encoding_a = {k: v.to(device) for k, v in encoding_a.items()}


    model.eval()

    with torch.no_grad():
        output = model(encoding_q, encoding_a)

    print("Sample Output Probabilities:")
    print(output)
    print(
        f"Encoding_q.input_ids shape: {encoding_q['input_ids'].shape}"
    )
    print(
        f"Encoding_a.input_ids shape: {encoding_a['input_ids'].shape}"
    )
if __name__ == "__main__":
    test_forward_propagation()