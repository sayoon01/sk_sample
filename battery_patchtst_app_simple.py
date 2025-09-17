import math
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

st.set_page_config(page_title="간단 PatchTST 예측", layout="centered")
st.title("EV 배터리 SOC 예측 (초간단 PatchTST)")

# --- CSV 업로드 ---
f = st.file_uploader("전처리된 CSV 업로드(예: clean_csv 안 파일)", type=["csv"])
if f is None:
    st.stop()

df = pd.read_csv(f)
st.write("미리보기:", df.head())

# 숫자 컬럼만 타깃 후보
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("숫자 컬럼이 없습니다. SOC 같은 숫자 컬럼이 있어야 합니다.")
    st.stop()
target_col = st.selectbox("타깃 컬럼(SOC 등)", options=num_cols, index=0)

# 하이퍼파라미터(간단)
L = 96     # 입력 길이
H = 12     # 예측 길이
PATCH_LEN = 16
STRIDE = 8
DMODEL = 64
NHEAD = 4
NLAYERS = 2
DFF = 128
DROPOUT = 0.1
EPOCHS = 10
BATCH = 64
LR = 1e-3

# 1D 시계열 + 간단 표준화
y_raw = df[[target_col]].astype(float).values.reshape(-1)
if len(y_raw) < L + H + 5:
    st.warning("데이터가 너무 짧습니다. 더 긴 CSV를 쓰거나 L/H를 줄이세요.")
mu = float(np.nanmean(y_raw))
sigma = float(np.nanstd(y_raw) + 1e-8)
y = ((y_raw - mu) / sigma).astype(np.float32)

# Dataset: (L)->(H)
class SeqDS(Dataset):
    def __init__(self, series, L, H):
        self.s, self.L, self.H = series, L, H
    def __len__(self):
        return max(0, len(self.s) - self.L - self.H + 1)
    def __getitem__(self, i):
        x = self.s[i:i+self.L]               # (L,)
        y = self.s[i+self.L:i+self.L+self.H] # (H,)
        return torch.tensor(x), torch.tensor(y)

# sinusoidal pos-encoding
def sinusoidal_pe(n_tokens, d_model, device):
    pe = torch.zeros(1, n_tokens, d_model, device=device)
    position = torch.arange(0, n_tokens, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0)/d_model))
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe  # (1, n_tokens, d_model)

# PatchTST (단일 채널 초간단 버전)
class PatchTSTSimple(nn.Module):
    def __init__(self, L, H, patch_len=16, stride=8, d_model=64, nhead=4, num_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.L = L
        self.H = H
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)  # patch -> d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, H)  # mean pooled token -> H

    def forward(self, x):              # x: (B, L) or (B, L, 1)
        if x.dim() == 3: x = x.squeeze(-1)    # (B,L,1) -> (B,L)
        B, L = x.shape
        assert L >= self.patch_len, f"L({L}) must be >= patch_len({self.patch_len})"
        # 1D patchify: (B, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, nP, patch_len)
        nP = patches.size(1)
        z = self.proj(patches)  # (B, nP, d_model)
        pe = sinusoidal_pe(nP, z.size(-1), z.device)
        z = z + pe
        z = self.encoder(z)     # (B, nP, d_model)
        h = z.mean(dim=1)       # token 평균 풀링
        y = self.head(h)        # (B, H)
        return y

ds = SeqDS(y, L, H)
if len(ds) == 0:
    st.error("윈도 샘플이 없습니다. L/H를 줄이거나 데이터를 늘리세요.")
    st.stop()
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PatchTSTSimple(L=L, H=H, patch_len=PATCH_LEN, stride=STRIDE,
                       d_model=DMODEL, nhead=NHEAD, num_layers=NLAYERS,
                       d_ff=DFF, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# 패치 수 유효성 체크
n_patches = (L - PATCH_LEN) // STRIDE + 1
if n_patches <= 0:
    st.error("유효한 패치 수가 0입니다. PATCH_LEN/STRIDE 또는 L을 조정하세요.")
    st.stop()

if st.button("학습하고 예측하기 (PatchTST)"):
    # Train
    model.train()
    for ep in range(EPOCHS):
        s, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)         # (B,H)
            loss = loss_fn(yhat, yb) # (B,H) vs (B,H)
            loss.backward()
            opt.step()
            s += loss.item() * len(xb); n += len(xb)
        st.write(f"Epoch {ep+1}/{EPOCHS}  loss={s/max(1,n):.4f}")

    # 마지막 윈도로 H 스텝 예측
    model.eval()
    with torch.no_grad():
        last_x = torch.tensor(y[-L:], dtype=torch.float32).unsqueeze(0).to(device)  # (1,L)
        pred_norm = model(last_x).cpu().numpy()[0]  # (H,)
    pred = (pred_norm * sigma) + mu

    st.success("예측 완료 (마지막 윈도 기준 미래 H스텝):")
    out = pd.DataFrame({"step": np.arange(1, H+1), "pred": pred})
    st.dataframe(out, hide_index=True)
    st.download_button("예측값 CSV 다운로드", out.to_csv(index=False).encode("utf-8"),
                       file_name="pred_patchtst_simple.csv", mime="text/csv")
