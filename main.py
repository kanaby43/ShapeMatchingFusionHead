import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal 위치 인코딩을 생성하는 표준적인 모듈.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, SeqLen, Dim]
        """
        return self.pe[:, :x.size(1), :]


class ShapeMatchingFusionHead(nn.Module):
    """
    CNN 피처의 Shape을 다른 쿼리들과 맞춘 뒤, element-wise 덧셈으로 융합하는 모듈.
    """
    def __init__(self, cnn_in_channels: int, query_dim: int, target_seq_len: int, cnn_h: int, cnn_w: int):
        super().__init__()
        
        # 1. CNN 피처를 (B, SeqLen, Dim) 형태로 변환하기 위한 레이어
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(cnn_in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, query_dim, kernel_size=1)
        )
        self.pos_encoder = PositionalEncoding2D(d_model=query_dim, max_len=cnn_h * cnn_w)

        # 2. 시퀀스 길이를 맞추기 위한 MLP
        self.seq_len_matcher = nn.Linear(cnn_h * cnn_w, target_seq_len)

    def forward(self, driving_query: torch.Tensor, boundary_query: torch.Tensor, cnn_feature: torch.Tensor) -> torch.Tensor:
        
        encoded_cnn = self.cnn_encoder(cnn_feature)
        
        B, C, H, W = encoded_cnn.shape
        cnn_sequence = encoded_cnn.flatten(2).permute(0, 2, 1)
        
        cnn_sequence_with_pe = cnn_sequence + self.pos_encoder(cnn_sequence)
        
        transformed_cnn_feature = self.seq_len_matcher(cnn_sequence_with_pe.transpose(1, 2)).transpose(1, 2)
        
        final_map_query = driving_query + boundary_query + transformed_cnn_feature
        
        return final_map_query


# --- 데모 실행 ---
if __name__ == '__main__':
    # 파라미터 정의
    B = 1      # 배치 사이즈
    N = 1400   # 쿼리 시퀀스 길이
    D = 256    # 쿼리 차원
    C = 3      # CNN 입력 채널
    H = 100    # CNN 입력 높이
    W = 100    # CNN 입력 너비

    # 정의된 파라미터로 모델 초기화
    fusion_head = ShapeMatchingFusionHead(
        cnn_in_channels=C,
        query_dim=D,
        target_seq_len=N,
        cnn_h=H,
        cnn_w=W
    )
    
    # 입력 텐서 생성
    driving_q = torch.randn(B, N, D)
    boundary_q = torch.randn(B, N, D)
    cnn_feat = torch.randn(B, C, H, W)
    
    # 모델 실행
    output = fusion_head(driving_q, boundary_q, cnn_feat)
    
    # 결과 확인
    print(f"입력 driving_query Shape: {driving_q.shape}")
    print(f"입력 cnn_feature Shape:    {cnn_feat.shape}")
    print("-" * 50)
    print(f"최종 출력 Shape:          {output.shape}")
    
    assert output.shape == driving_q.shape
    print("\n최종 출력의 Shape이 driving_query와 동일하게 융합되었습니다.")
