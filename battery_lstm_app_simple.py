import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

st.set_page_config(page_title="간단 LSTM 예측", layout="centered")
st.title("EV 배터리 SOC 예측 (초간단 LSTM)")

# --- 업로드 ---
f = st.file_uploader("전처리된 CSV 업로드(예: clean_csv 안 파일)", type=["csv"])
if f is None:
    st.stop()

df = pd.read_csv(f)
st.write("미리보기:", df.head())

# 숫자 컬럼만 타깃 후보로
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("숫자 컬럼이 없습니다. SOC 같은 숫자 컬럼이 있어야 합니다.")
    st.stop()
target_col = st.selectbox("타깃 컬럼(SOC 등)", options=num_cols, index=0)

# 하이퍼파라미터 (간단 고정값)
L = 48   # 입력 길이
H = 6    # 예측 길이
EPOCHS = 10
HID = 64
LR = 1e-3

# 1D 시계열로 만들기 + 간단 표준화
y_raw = df[[target_col]].astype(float).values.reshape(-1)
if len(y_raw) < L + H + 5:
    st.warning("데이터가 너무 짧습니다. 더 긴 CSV를 쓰거나 L/H를 줄이세요.")
mu = float(np.nanmean(y_raw))
sigma = float(np.nanstd(y_raw) + 1e-8)
y = ((y_raw - mu) / sigma).astype(np.float32)

class SeqDS(Dataset):
    def __init__(self, series, L, H):
        self.s, self.L, self.H = series, L, H
    def __len__(self):
        return max(0, len(self.s) - self.L - self.H + 1)
    def __getitem__(self, i):
        x = self.s[i:i+self.L]               # (L,)
        y = self.s[i+self.L:i+self.L+self.H] # (H,)
        return torch.tensor(x), torch.tensor(y)

# (핵심) 입력을 항상 (B,L,1)로 맞춰주는 헬퍼
def to3d(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 1:      # (L,)   -> (1,L,1)
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.dim() == 2:    # (B,L)  -> (B,L,1)
        x = x.unsqueeze(-1)
    return x              # 이미 (B,L,1)이면 그대로

ds = SeqDS(y, L, H)
if len(ds) == 0:
    st.error("윈도 샘플이 없습니다. L/H를 줄여보세요.")
    st.stop()
dl = DataLoader(ds, batch_size=64, shuffle=True)

class TinyLSTM(nn.Module):
    def __init__(self, hid=64, H=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hid, batch_first=True)
        self.fc = nn.Linear(hid, H)
    def forward(self, x):          # x: (B,L,1)
        o,_ = self.lstm(x)         # (B,L,hid)
        h = o[:,-1,:]              # (B,hid)
        return self.fc(h)          # (B,H)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyLSTM(hid=HID, H=H).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

if st.button("학습하고 예측하기"):
    # --- Train ---
    model.train()
    for ep in range(EPOCHS):
        s, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = to3d(xb).to(device), yb.to(device)  # ★ 여기서 항상 3D로
            opt.zero_grad()
            yhat = model(xb)            # (B,H)
            loss = loss_fn(yhat, yb)    # (B,H) vs (B,H)
            loss.backward()
            opt.step()
            s += loss.item() * len(xb); n += len(xb)
        st.write(f"Epoch {ep+1}/{EPOCHS}  loss={s/max(1,n):.4f}")

    # --- 마지막 윈도로 H스텝 예측 ---
    model.eval()
    with torch.no_grad():
        last_x = to3d(y[-L:]).to(device)             # ★ (1,L,1)
        pred_norm = model(last_x).cpu().numpy()[0]   # (H,)
    pred = (pred_norm * sigma) + mu

    st.success("예측 완료 (마지막 윈도 기준 미래 H스텝):")
    out = pd.DataFrame({"step": np.arange(1, H+1), "pred": pred})
    st.dataframe(out, hide_index=True)
    st.download_button("예측값 CSV 다운로드", out.to_csv(index=False).encode("utf-8"),
                       file_name="pred_ultra_simple.csv", mime="text/csv")
