# streamlit_turbine.py

# streamlit run streamlit_turbine.py

import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set font for Japanese characters (Windows compatible)
import platform
if platform.system() == 'Windows':
    # Use Windows fonts that support Japanese characters
    plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
else:
    # Use fonts for other systems
    plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']

st.session_state.clear()

# === 物性定数 ===
GAMMA = 1.4
R_g = 287.0  # J/(kg·K)
Cp = R_g * GAMMA / (GAMMA - 1)

# === ヘルパー関数 ===
def density_from_isentropic(P_static, T0, P0, gamma=GAMMA, Rg=R_g):
    T_static = T0 * (P_static / P0) ** ((gamma - 1) / gamma)
    rho = P_static / (Rg * T_static)
    return rho, T_static

def compute_delta_h_total(T0, P0, P_out, gamma=GAMMA, Cp_local=Cp):
    return Cp_local * T0 * (1 - (P_out / P0) ** ((gamma - 1) / gamma))

def design_iteration_with_solidity_history(
    N_rpm, r_hub,
    c_s, c_r, sigma_s, sigma_r,
    throat_ratio_s, throat_ratio_r,
    T0, P0, P_out, mdot,
    R_init=0.5, tol=1e-4, max_iter=200
):
    R = R_init
    h_s = h_r = 0.1
    history = []
    r_m = None
    U = None
    V2 = Vx = Vtheta = W2 = W3 = None
    delta_h_total = None
    delta_h_s = delta_h_r = None
    rho = T_static_out = None

    for i in range(1, max_iter + 1):
        delta_h_total = compute_delta_h_total(T0, P0, P_out)
        
        # Ensure R is within valid bounds
        R = max(0.0, min(1.0, R))
        
        delta_h_r = R * delta_h_total
        delta_h_s = (1 - R) * delta_h_total

        # Ensure enthalpy drops are non-negative for velocity calculations
        if delta_h_r < 0:
            delta_h_r = 0.0
        if delta_h_s < 0:
            delta_h_s = 0.0

        V2 = math.sqrt(2 * delta_h_r) if delta_h_r > 0 else 1e-8  # rotor側が動翼落差 → stator出口速度はこれを使う場合の解釈もあるが落差配分に合わせる

        # ここでは静翼出口速度は rotor/ statorの落差から同様に V2_stator = sqrt(2*delta_h_s)
        V2_s = math.sqrt(2 * delta_h_s) if delta_h_s > 0 else 1e-8

        # ピッチ逆算
        s_s = c_s / sigma_s if sigma_s != 0 else 1e-8
        s_r = c_r / sigma_r if sigma_r != 0 else 1e-8
        t_s = s_s * throat_ratio_s
        t_r = s_r * throat_ratio_r

        # 流出角（静翼側、周方向基準；ここは静翼のスロート比を用いて）
        if not (0 <= throat_ratio_s <= 1.0):
            alpha2p = 0.0
        else:
            alpha2p = math.asin(throat_ratio_s)
        Vx = V2_s * math.sin(alpha2p)
        Vtheta = V2_s * math.cos(alpha2p)

        # 密度（出口静圧から）
        rho, T_static_out = density_from_isentropic(P_out, T0, P0)

        if Vx <= 0:
            A_th = float("inf")
        else:
            A_th = mdot / (rho * Vx)

        # 翼長
        h_s_new = A_th / t_s if t_s != 0 else h_s
        h_r_new = A_th / t_r if t_r != 0 else h_r

        # 平均半径と周速
        r_m = r_hub + 0.5 * (h_s_new + h_r_new)
        U = 2 * math.pi * r_m * (N_rpm / 60)

        # 動翼入口相対速度（静翼出口の周方向速度 Vtheta と周速 U を使う）
        W2 = math.sqrt((Vtheta - U) ** 2 + Vx ** 2)

        inside = W2 ** 2 - 2 * delta_h_r
        W3 = math.sqrt(inside) if inside > 0 else 1e-8

        # 反動度再推定（動翼側落差を分子に）
        delta_h_r_est = (W2 ** 2 - W3 ** 2) / 2
        delta_h_s_est = V2_s ** 2 / 2
        denom = delta_h_r_est + delta_h_s_est + 1e-16
        R_new = delta_h_r_est / denom

        err_R = abs(R_new - R)
        err_hs = abs(h_s_new - h_s) / (h_s + 1e-16)
        err_hr = abs(h_r_new - h_r) / (h_r + 1e-16)

        history.append({
            "iter": i,
            "R": R,
            "R_new": R_new,
            "h_s": h_s,
            "h_r": h_r,
            "h_s_new": h_s_new,
            "h_r_new": h_r_new,
            "err_R": err_R,
            "err_hs": err_hs,
            "err_hr": err_hr,
            "U": U,
            "V2_stator": V2_s,
            "Vtheta": Vtheta,
            "Vx": Vx,
            "W2": W2,
            "W3": W3,
        })

        if err_R < tol and err_hs < tol and err_hr < tol:
            R = R_new
            h_s, h_r = h_s_new, h_r_new
            break

        R = R_new
        h_s, h_r = h_s_new, h_r_new

    result = {
        "R": R,
        "h_s": h_s,
        "h_r": h_r,
        "r_mean": r_m,
        "U": U,
        "V2_stator": V2_s,
        "Vx": Vx,
        "Vtheta": Vtheta,
        "W2": W2,
        "W3": W3,
        "delta_h_total": delta_h_total,
        "delta_h_s": delta_h_s,
        "delta_h_r": delta_h_r,
        "rho": rho,
        "T_static_out": T_static_out,
        "iterations": i,
    }
    return result, pd.DataFrame(history)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_convergence(history_df):
    fig, ax = plt.subplots()
    ax.plot(history_df["iter"], history_df["R"],     label="R (Reaction ratio)", marker="o")
    ax.plot(history_df["iter"], history_df["h_s"],   label="h_s (Stator span)",  marker="s")
    ax.plot(history_df["iter"], history_df["h_r"],   label="h_r (Rotor span)",   marker="^")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Convergence History (R & Blade Spans)")
    ax.set_xticks(history_df["iter"].astype(int).unique())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig          # ← fig を返すだけにしておく

def plot_errors(history_df):
    fig, ax = plt.subplots()
    ax.plot(history_df["iter"], history_df["err_R"],  label="Error in R")
    ax.plot(history_df["iter"], history_df["err_hs"], label="Error in h_s")
    ax.plot(history_df["iter"], history_df["err_hr"], label="Error in h_r")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Convergence Errors")
    ax.set_yscale("log")
    ax.set_xticks(history_df["iter"].astype(int).unique())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig


# === Streamlit UI ===
st.set_page_config(page_title="1D 軸流タービン設計ビジュアライザ", layout="wide")
st.title("一次元軸流タービン設計（反動度・ソリディティ版）")

with st.sidebar:
    st.header("入力パラメータ")
    N_rpm = st.slider("回転数 N [rpm]", min_value=1000, max_value=30000, value=10000, step=500)
    r_hub = st.number_input("ハブ半径 r_hub [m]", value=0.3, format="%.4f")
    c_s = st.number_input("静翼コード長 c_s [m]", value=0.025, format="%.5f")
    c_r = st.number_input("動翼コード長 c_r [m]", value=0.025, format="%.5f")
    sigma_s = st.slider("静翼ソリディティ σ_s", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    sigma_r = st.slider("動翼ソリディティ σ_r", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    throat_ratio_s = st.slider("静翼スロート比 t_s/s_s", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    throat_ratio_r = st.slider("動翼スロート比 t_r/s_r", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    T0 = st.number_input("入口全温 T0 [K]", value=800.0, format="%.1f")
    P0 = st.number_input("入口全圧 P0 [Pa]", value=2e5, format="%.3e")
    P_out = st.number_input("出口静圧 P_out [Pa]", value=1.0e5, format="%.3e")
    mdot = st.number_input("質量流量 ṁ [kg/s]", value=0.01, format="%.3f")
    R_init = st.slider("初期反動度 R_init", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    tol = st.number_input("収束許容値 tol", value=1e-5, format="%.0e")
    max_iter = st.number_input("最大反復回数", value=200, step=10)

st.subheader("設計反復結果")
result, history_df = design_iteration_with_solidity_history(
    N_rpm=N_rpm, r_hub=r_hub,
    c_s=c_s, c_r=c_r, sigma_s=sigma_s, sigma_r=sigma_r,
    throat_ratio_s=throat_ratio_s, throat_ratio_r=throat_ratio_r,
    T0=T0, P0=P0, P_out=P_out, mdot=mdot,
    R_init=R_init, tol=tol, max_iter=int(max_iter)
)

# 左側：主要結果
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 収束結果サマリ")
    summary = {
        "反動度 R": result["R"],
        "静翼高さ h_s [m]": result["h_s"],
        "動翼高さ h_r [m]": result["h_r"],
        "平均半径 r_m [m]": result["r_mean"],
        "周速 U [m/s]": result["U"],
    }
    st.table(pd.Series(summary).apply(lambda x: f"{x:.5e}"))

with col2:
    st.markdown("### 速度/エネルギー系")
    vtab = {
        "静翼流出速度 V2_stator [m/s]": result["V2_stator"],
        "軸方向成分 Vx [m/s]": result["Vx"],
        "周方向成分 Vθ [m/s]": result["Vtheta"],
        "動翼入口相対速度 W2 [m/s]": result["W2"],
        "動翼出口相対速度 W3 [m/s]": result["W3"],
        "全落差 Δh_total [J/kg]": result["delta_h_total"],
    }
    st.table(pd.Series(vtab).apply(lambda x: f"{x:.5e}"))


# --- Convergence plot ---
st.markdown("### Convergence History")
st.pyplot(plot_convergence(history_df))

# --- Error plot ---
st.markdown("### Error History (log scale)")
st.pyplot(plot_errors(history_df))

st.caption("反動度の定義は動翼（ロータ）側の静エンタルピー落差 / 全体落差。平均半径は r_hub + 0.5*(h_s + h_r) です。")
