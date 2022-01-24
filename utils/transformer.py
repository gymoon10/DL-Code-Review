# 7장 자연어 처리에 의한 감정 분석(Transformer)
# 구현에서 참조
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


# 필요한 패키지 import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext


class Embedder(nn.Module):
    '''id로 표시된 단어를 벡터로 변환합니다'''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        # freeze=True에 의해 역전파로 갱신되지 않고, 변하지 않습니다

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec


class PositionalEncoder(nn.Module):
    '''입력된 단어의 위치를 나타내는 벡터 정보를 부가'''

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 단어 벡터의 차원수

        # 단어 순서(pos)와 내장 벡터의 차원 위치(i)에 의해 고유하게 정해지는 값의 표를 pe로 작성
        pe = torch.zeros(max_seq_len, d_model)

        # GPU가 사용 가능하면 GPU에 전달
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))

                # 오탈자 수정_200510 #76
                # pe[pos, i + 1] = math.cos(pos /
                #                          (10000 ** ((2 * (i + 1))/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # 표 pe의 선두에서, 미니 배치 차원을 더함
        self.pe = pe.unsqueeze(0)

        # 경사를 계산하지 않음
        self.pe.requires_grad = False

    def forward(self, x):

        # 입력 x와 Positonal Encoding을 더함
        # x가 pe보다 작으므로 크게 한다
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


class Attention(nn.Module):
    '''Transformer는 사실상 멀티 헤드 Attention이지만, 
    쉽게 이해되도록 우선 싱글 Attention로 구현합니다'''

    def __init__(self, d_model=300):
        super().__init__()

        # SAGAN에서는 1dConv를 사용했지만, 이번에는 전결합층에서 특징량을 변환
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 출력 시에 사용할 전결합층
        self.out = nn.Linear(d_model, d_model)

        # Attention의 크기 조정 변수
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 전결합층에서 특징량을 변환
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attention 값을 계산한다
        # 각 값을 덧셈하면 너무 커지므로 root(d_k)로 나누어 조절
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # 여기서 mask를 계산
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmax로 규격화
        normlized_weights = F.softmax(weights, dim=-1)

        # Attention을 Value와 곱하기
        output = torch.matmul(normlized_weights, v)

        # 전결합층에서 특징량을 변환
        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention 층에서 출력을 단순히 전결합층 두 개로 특징량을 변환하는 유닛입니다'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization층
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention층
        self.attn = Attention(d_model)

        # Attention 다음의 전결합층 두 개
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 정규화와 Attention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)

        x2 = x + self.dropout_1(output)

        # 정규화와 전결합층
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


class ClassificationHead(nn.Module):
    '''Transformer_Block의 출력을 사용하여, 마지막에 클래스 분류를 시킨다'''

    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 전결합층
        self.linear = nn.Linear(d_model, output_dim)  # output_dim은 음성, 양성의 두 가지

        # 가중치 초기화
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 각 미니 배치의 각 문장의 선두 단어의 특징량(300차원)을 꺼낸다
        out = self.linear(x0)

        return out


# 최종적인 Transformer 모델의 클래스
class TransformerClassification(nn.Module):
    '''Transformer로 클래스 분류'''

    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        # 모델 구축
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 단어를 벡터로
        x2 = self.net2(x1)  # Positon 정보를 더한다
        x3_1, normlized_weights_1 = self.net3_1(
            x2, mask)  # Self-Attention으로 특징량을 변환
        x3_2, normlized_weights_2 = self.net3_2(
            x3_1, mask)  # Self-Attention으로 특징량을 변환
        x4 = self.net4(x3_2)  # 최종 출력의 0번째 단어를 사용하여, 분류0~1의 스칼라를 출력
        return x4, normlized_weights_1, normlized_weights_2
