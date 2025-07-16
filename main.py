import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D 공간 정보를 올바르게 보존하는 Sinusoidal 위치 인코딩 모듈.
    (x, y 좌표를 독립적으로 인코딩하여 결합하는 방식)
    """
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model은 4의 배수여야 합니다 (x, y 각각 짝수 차원 필요).")

        pe = torch.zeros(d_model, height, width)
        
        # 각 축(x, y)은 d_model의 절반을 사용
        d_model_half = d_model // 2
        
        # 공유하는 div_term 계산
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))

        # y축 (높이) 위치 인코딩
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        # x축 (너비) 위치 인코딩
        pos_w = torch.arange(0., width).unsqueeze(1)
        pe[d_model_half::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)

        # 학습되지 않는 버퍼로 등록
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, d_model, height, width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서 x에 위치 인코딩을 더합니다.
        x: [B, C, H, W]
        """
        # x와 pe의 채널 수가 같아야 함
        return x + self.pe 


class ShapeMatchingFusionHead(nn.Module):
    """
    CNN 피처의 Shape을 다른 쿼리들과 맞춘 뒤, element-wise 덧셈으로 융합하는 모듈.
    """
    def __init__(self, cnn_in_channels: int, query_dim: int, target_seq_len: int, cnn_h: int, cnn_w: int):
        super().__init__()
        
        # 1. CNN 피처 처리용 레이어
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(cnn_in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, query_dim, kernel_size=1) # 채널 수를 256으로 맞춤
        )
        self.pos_encoder = PositionalEncoding2D(d_model=query_dim, height=cnn_h, width=cnn_w)

        # 2. 시퀀스 길이를 맞추기 위한 MLP
        self.seq_len_matcher = nn.Linear(cnn_h * cnn_w, target_seq_len)

    def forward(self, driving_query: torch.Tensor, boundary_query: torch.Tensor, cnn_feature: torch.Tensor) -> torch.Tensor:
        
        # 1. CNN 피처 의미/차원 변환
        encoded_cnn = self.cnn_encoder(cnn_feature)
        
        # 2. 2D 위치 정보 주입
        cnn_with_pe = self.pos_encoder(encoded_cnn)
        
        # 3. 시퀀스 형태로 변환
        B, C, H, W = cnn_with_pe.shape
        cnn_sequence = cnn_with_pe.flatten(2).permute(0, 2, 1)
        
        # 4. 시퀀스 길이 맞추기
        transformed_cnn_feature = self.seq_len_matcher(cnn_sequence.transpose(1, 2)).transpose(1, 2)
        
        # 5. 최종 융합
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