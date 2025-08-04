# =====================================================================================
# プログラム仕様書
# =====================================================================================
# プログラム名:    streamlit_turbine.py
# 概要:           一次元軸流タービン設計プログラム（反動度・ソリディティ版）
# 目的:           軸流タービンの予備設計における反動度、翼高さ、速度分布の
#                反復計算による収束設計と可視化
# 
# =====================================================================================
# 主要機能
# =====================================================================================
# 1. 熱力学計算:   等エントロピー関係による状態量計算
# 2. 反復設計:     反動度・翼高さの同時収束計算
# 3. 幾何計算:     ソリディティ、スロート比を考慮した翼列幾何
# 4. 可視化:       収束履歴・誤差履歴の対話的グラフ表示
# 5. WebUI:       Streamlitによる直感的なパラメータ調整界面
# 
# =====================================================================================
# 動作要件
# =====================================================================================
# Python:         3.8以上
# 必須ライブラリ:  streamlit, pandas, matplotlib, math, platform
# 推奨環境:       Windows 10/11, macOS 10.15+, Ubuntu 18.04+
# メモリ:         最小512MB、推奨1GB以上
# 
# =====================================================================================
# 実行例
# =====================================================================================
# コマンドライン実行:
#   streamlit run streamlit_turbine.py
# 
# ブラウザアクセス:
#   http://localhost:8501
# 
# 仮想環境での実行:
#   python -m venv venv
#   venv\Scripts\activate  (Windows)
#   source venv/bin/activate  (Linux/Mac)
#   pip install streamlit pandas matplotlib
#   streamlit run streamlit_turbine.py
# 
# =====================================================================================
# 入力パラメータ範囲
# =====================================================================================
# 回転数 N:           1,000 - 30,000 [rpm]
# ハブ半径 r_hub:     0.001 - 10.0 [m]
# コード長 c_s,c_r:   0.001 - 1.0 [m]
# ソリディティ σ:     0.1 - 3.0 [-]
# スロート比:         0.0 - 1.0 [-]
# 入口全温 T0:        300 - 2000 [K]
# 入口全圧 P0:        1e4 - 1e7 [Pa]
# 出口静圧 P_out:     1e3 - 1e6 [Pa]
# 質量流量 mdot:      1e-6 - 1000 [kg/s]
# 
# =====================================================================================
# 出力結果
# =====================================================================================
# 設計値:    反動度R, 静翼高さh_s, 動翼高さh_r, 平均半径r_mean, 周速U
# 速度分布:  静翼出口速度V2_stator, 軸方向Vx, 周方向Vtheta, 
#           動翼相対速度W2/W3
# エネルギー: 全落差delta_h_total, 静翼/動翼別落差
# 状態量:    密度rho, 出口静温T_static_out
# 収束情報:  反復回数, 収束履歴, 誤差履歴
# 
# =====================================================================================
# 理論背景
# =====================================================================================
# ・反動度定義:     R = 動翼エンタルピー落差 / 全エンタルピー落差
# ・ソリディティ:   σ = 翼弦長c / ピッチs
# ・スロート比:     t/s = 最小流路幅 / ピッチ
# ・等エントロピー:  T/T0 = (P/P0)^((γ-1)/γ)
# ・連続の式:       mdot = ρ * A * V
# ・オイラー式:     ΔH = U * ΔVθ
# 
# =====================================================================================
# 備考
# =====================================================================================
# ・収束判定:       相対誤差 < 1e-5 (デフォルト)
# ・数値安定性:     ゼロ除算回避、平方根負数回避処理実装
# ・座標系:        軸方向x、周方向θ、径方向r (右手系)
# ・速度定義:       絶対速度V、相対速度W、周速U
# ・翼列配置:       静翼→動翼の単段構成
# 
# =====================================================================================
# バージョン履歴
# =====================================================================================
# v1.0 (2025-08-04):  初版リリース、基本機能実装
# 
# =====================================================================================
# 作成者・連絡先
# =====================================================================================
# GitHub:          https://github.com/unikarei/TurbineDesignSystem_1D
# リポジトリ:       TurbineDesignSystem_1D
# ライセンス:       MIT License
# 
# =====================================================================================
# streamlit_turbine.py
# 一次元軸流タービン設計プログラム（反動度・ソリディティ版）
# 実行コマンド: streamlit run streamlit_turbine.py
# GitHub: https://github.com/unikarei/TurbineDesignSystem_1D
# =====================================================================================

import math                         # 数学関数（sqrt, sin, cos, asin等）を使用
import streamlit as st              # Webアプリケーション作成用フレームワーク
import pandas as pd                 # データフレーム操作用ライブラリ
import matplotlib.pyplot as plt     # グラフ作成用ライブラリ

# =====================================================================================
# フォント設定（日本語文字対応、OS別に最適化）
# =====================================================================================
import platform                     # OS判定用モジュール
if platform.system() == 'Windows':  # Windowsの場合
    # Windows用日本語対応フォントを優先順位付きで設定
    plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
else:                               # Windows以外（Mac/Linux）の場合
    # その他OS用フォント設定
    plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']

st.session_state.clear()            # Streamlitセッション状態をクリア（リロード時の状態初期化）

# =====================================================================================
# 物性定数定義部
# 理想気体の熱力学特性を定義
# =====================================================================================
GAMMA = 1.4                         # 比熱比（空気の標準値、無次元）
R_g = 287.0                         # 気体定数 [J/(kg·K)]
Cp = R_g * GAMMA / (GAMMA - 1)      # 定圧比熱 [J/(kg·K)] = 1004.5 J/(kg·K)

# =====================================================================================
# ヘルパー関数群
# 熱力学計算および反復計算に必要な基本関数を定義
# =====================================================================================

# -------------------------------------------------------------------------------------
# 等エントロピー関係から密度と静温度を計算
# 全温・全圧から静圧を指定して静的状態量を求める
# -------------------------------------------------------------------------------------
def density_from_isentropic(P_static, T0, P0, gamma=GAMMA, Rg=R_g):
    """
    等エントロピー関係を用いて密度と静温度を計算
    Args:
        P_static: 静圧 [Pa]
        T0: 全温 [K]
        P0: 全圧 [Pa]
        gamma: 比熱比 [-]
        Rg: 気体定数 [J/(kg·K)]
    Returns:
        rho: 密度 [kg/m³]
        T_static: 静温度 [K]
    """
    T_static = T0 * (P_static / P0) ** ((gamma - 1) / gamma)  # 等エントロピー関係による静温度
    rho = P_static / (Rg * T_static)                          # 理想気体の状態方程式による密度
    return rho, T_static

# -------------------------------------------------------------------------------------
# 全エンタルピー落差計算
# 入口全状態から出口静圧までの理論エンタルピー落差を計算
# -------------------------------------------------------------------------------------
def compute_delta_h_total(T0, P0, P_out, gamma=GAMMA, Cp_local=Cp):
    """
    全エンタルピー落差を計算
    Args:
        T0: 入口全温 [K]
        P0: 入口全圧 [Pa]
        P_out: 出口静圧 [Pa]
        gamma: 比熱比 [-]
        Cp_local: 定圧比熱 [J/(kg·K)]
    Returns:
        delta_h_total: 全エンタルピー落差 [J/kg]
    """
    return Cp_local * T0 * (1 - (P_out / P0) ** ((gamma - 1) / gamma))  # 等エントロピー膨張による落差

# =====================================================================================
# メイン設計反復関数
# 反動度、翼高さ、速度分布を反復的に収束させるコア関数
# ソリディティ（翼弦長/ピッチ比）を考慮した設計手法
# =====================================================================================
def design_iteration_with_solidity_history(
    N_rpm, r_hub,                   # 回転数[rpm]、ハブ半径[m]
    c_s, c_r, sigma_s, sigma_r,     # 静翼・動翼のコード長[m]、ソリディティ[-]
    throat_ratio_s, throat_ratio_r, # 静翼・動翼のスロート比[-]
    T0, P0, P_out, mdot,            # 全温[K]、全圧[Pa]、出口圧[Pa]、質量流量[kg/s]
    R_init=0.5, tol=1e-4, max_iter=200  # 初期反動度、収束判定値、最大反復数
):
    """
    ソリディティを考慮した1Dタービン設計の反復計算
    反動度、翼高さ、速度分布を同時に収束させる
    """
    # -------------------------------------------------------------------------------------
    # 初期値設定
    # -------------------------------------------------------------------------------------
    R = R_init                      # 反動度の初期値（動翼エンタルピー落差/全落差）
    h_s = h_r = 0.1                # 静翼・動翼高さの初期値 [m]
    history = []                    # 反復履歴を保存するリスト
    
    # 計算結果格納用変数の初期化
    r_m = None                      # 平均半径 [m]
    U = None                        # 周速 [m/s]
    V2 = Vx = Vtheta = W2 = W3 = None  # 各種速度成分 [m/s]
    delta_h_total = None            # 全エンタルピー落差 [J/kg]
    delta_h_s = delta_h_r = None    # 静翼・動翼別エンタルピー落差 [J/kg]
    rho = T_static_out = None       # 密度 [kg/m³]、出口静温 [K]

    # -------------------------------------------------------------------------------------
    # 反復計算メインループ
    # -------------------------------------------------------------------------------------
    for i in range(1, max_iter + 1):
        # --- Step 1: 全エンタルピー落差計算 ---
        delta_h_total = compute_delta_h_total(T0, P0, P_out)  # 理論全落差 [J/kg]
        
        # --- Step 2: 反動度の境界値制限 ---
        R = max(0.0, min(1.0, R))   # 反動度を0-1の範囲に制限
        
        # --- Step 3: エンタルピー落差配分 ---
        delta_h_r = R * delta_h_total           # 動翼（ロータ）落差 [J/kg]
        delta_h_s = (1 - R) * delta_h_total     # 静翼（ステータ）落差 [J/kg]

        # --- Step 4: エンタルピー落差の非負制約 ---
        if delta_h_r < 0:           # 動翼落差が負の場合
            delta_h_r = 0.0         # ゼロに設定
        if delta_h_s < 0:           # 静翼落差が負の場合
            delta_h_s = 0.0         # ゼロに設定

        # --- Step 5: 速度計算（安全な平方根計算） ---
        V2 = math.sqrt(2 * delta_h_r) if delta_h_r > 0 else 1e-8     # 動翼基準速度 [m/s]
        V2_s = math.sqrt(2 * delta_h_s) if delta_h_s > 0 else 1e-8   # 静翼出口速度 [m/s]

        # --- Step 6: 翼列幾何パラメータ計算 ---
        s_s = c_s / sigma_s if sigma_s != 0 else 1e-8  # 静翼ピッチ [m]
        s_r = c_r / sigma_r if sigma_r != 0 else 1e-8  # 動翼ピッチ [m]
        t_s = s_s * throat_ratio_s              # 静翼スロート幅 [m]
        t_r = s_r * throat_ratio_r              # 動翼スロート幅 [m]

        # --- Step 7: 流出角・速度成分計算 ---
        if not (0 <= throat_ratio_s <= 1.0):   # スロート比の有効性チェック
            alpha2p = 0.0                       # 無効な場合は軸方向流れ
        else:
            alpha2p = math.asin(throat_ratio_s) # 静翼出口角（スロート比から逆算）[rad]
        Vx = V2_s * math.sin(alpha2p)           # 軸方向速度成分 [m/s]
        Vtheta = V2_s * math.cos(alpha2p)       # 周方向速度成分 [m/s]

        # --- Step 8: 密度・温度計算 ---
        rho, T_static_out = density_from_isentropic(P_out, T0, P0)  # 出口状態量

        # --- Step 9: 流路面積・翼高さ計算 ---
        if Vx <= 0:                            # 軸方向速度がゼロ以下の場合
            A_th = float("inf")                 # 無限大面積（エラー回避）
        else:
            A_th = mdot / (rho * Vx)            # 必要スロート面積 [m²]

        h_s_new = A_th / t_s if t_s != 0 else h_s  # 静翼高さ更新 [m]
        h_r_new = A_th / t_r if t_r != 0 else h_r  # 動翼高さ更新 [m]

        # --- Step 10: 平均半径・周速計算 ---
        r_m = r_hub + 0.5 * (h_s_new + h_r_new)    # 平均半径 [m]
        U = 2 * math.pi * r_m * (N_rpm / 60)       # 周速 [m/s]

        # --- Step 11: 動翼相対速度計算 ---
        W2 = math.sqrt((Vtheta - U) ** 2 + Vx ** 2)  # 動翼入口相対速度 [m/s]

        # --- Step 12: 動翼出口相対速度計算 ---
        inside = W2 ** 2 - 2 * delta_h_r        # 平方根内部の値
        W3 = math.sqrt(inside) if inside > 0 else 1e-8  # 動翼出口相対速度 [m/s]

        # --- Step 13: 反動度再推定 ---
        delta_h_r_est = (W2 ** 2 - W3 ** 2) / 2    # 動翼実際落差 [J/kg]
        delta_h_s_est = V2_s ** 2 / 2               # 静翼実際落差 [J/kg]
        denom = delta_h_r_est + delta_h_s_est + 1e-16  # 分母（ゼロ除算回避）
        R_new = delta_h_r_est / denom               # 新しい反動度 [-]

        # --- Step 14: 収束判定用誤差計算 ---
        err_R = abs(R_new - R)                      # 反動度誤差
        err_hs = abs(h_s_new - h_s) / (h_s + 1e-16) # 静翼高さ相対誤差
        err_hr = abs(h_r_new - h_r) / (h_r + 1e-16) # 動翼高さ相対誤差

        # --- Step 15: 履歴データ保存 ---
        history.append({
            "iter": i,                          # 反復回数
            "R": R,                            # 現在の反動度
            "R_new": R_new,                    # 更新された反動度
            "h_s": h_s,                        # 現在の静翼高さ
            "h_r": h_r,                        # 現在の動翼高さ
            "h_s_new": h_s_new,                # 更新された静翼高さ
            "h_r_new": h_r_new,                # 更新された動翼高さ
            "err_R": err_R,                    # 反動度誤差
            "err_hs": err_hs,                  # 静翼高さ誤差
            "err_hr": err_hr,                  # 動翼高さ誤差
            "U": U,                            # 周速
            "V2_stator": V2_s,                 # 静翼出口速度
            "Vtheta": Vtheta,                  # 周方向速度成分
            "Vx": Vx,                          # 軸方向速度成分
            "W2": W2,                          # 動翼入口相対速度
            "W3": W3,                          # 動翼出口相対速度
        })

        # --- Step 16: 収束判定 ---
        if err_R < tol and err_hs < tol and err_hr < tol:
            R = R_new                          # 反動度更新
            h_s, h_r = h_s_new, h_r_new       # 翼高さ更新
            break                              # 収束したのでループ終了

        # --- Step 17: 次回反復用値更新 ---
        R = R_new                              # 反動度更新
        h_s, h_r = h_s_new, h_r_new           # 翼高さ更新

    # -------------------------------------------------------------------------------------
    # 最終結果まとめ
    # -------------------------------------------------------------------------------------
    result = {
        "R": R,                                # 最終反動度 [-]
        "h_s": h_s,                           # 最終静翼高さ [m]
        "h_r": h_r,                           # 最終動翼高さ [m]
        "r_mean": r_m,                        # 平均半径 [m]
        "U": U,                               # 周速 [m/s]
        "V2_stator": V2_s,                    # 静翼出口速度 [m/s]
        "Vx": Vx,                             # 軸方向速度 [m/s]
        "Vtheta": Vtheta,                     # 周方向速度 [m/s]
        "W2": W2,                             # 動翼入口相対速度 [m/s]
        "W3": W3,                             # 動翼出口相対速度 [m/s]
        "delta_h_total": delta_h_total,       # 全エンタルピー落差 [J/kg]
        "delta_h_s": delta_h_s,               # 静翼エンタルピー落差 [J/kg]
        "delta_h_r": delta_h_r,               # 動翼エンタルピー落差 [J/kg]
        "rho": rho,                           # 密度 [kg/m³]
        "T_static_out": T_static_out,         # 出口静温 [K]
        "iterations": i,                      # 収束に要した反復回数
    }
    return result, pd.DataFrame(history)      # 結果辞書と履歴DataFrameを返す

import matplotlib.pyplot as plt     # matplotlibの重複インポート（削除推奨）

# =====================================================================================
# プロット関数群
# 計算結果の可視化用関数（収束履歴、誤差履歴）
# =====================================================================================
import matplotlib.pyplot as plt     # グラフ作成ライブラリ
from matplotlib.ticker import MaxNLocator  # 軸目盛り制御用

# -------------------------------------------------------------------------------------
# 収束履歴プロット関数
# 反動度と翼高さの反復変化を可視化
# -------------------------------------------------------------------------------------
def plot_convergence(history_df):
    """
    反動度と翼高さの収束過程をプロット
    Args:
        history_df: 反復履歴データフレーム
    Returns:
        fig: matplotlibフィギュアオブジェクト
    """
    fig, ax = plt.subplots()            # 図とサブプロット作成
    
    # 各パラメータの反復変化をプロット
    ax.plot(history_df["iter"], history_df["R"],     label="R (Reaction ratio)", marker="o")  # 反動度
    ax.plot(history_df["iter"], history_df["h_s"],   label="h_s (Stator span)",  marker="s")   # 静翼高さ
    ax.plot(history_df["iter"], history_df["h_r"],   label="h_r (Rotor span)",   marker="^")   # 動翼高さ
    
    # グラフ装飾
    ax.set_xlabel("Iteration")          # X軸ラベル（反復回数）
    ax.set_ylabel("Value")              # Y軸ラベル（値）
    ax.set_title("Convergence History (R & Blade Spans)")  # グラフタイトル
    ax.set_xticks(history_df["iter"].astype(int).unique())  # X軸目盛りを整数に設定
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # 整数目盛り強制
    ax.grid(True, linestyle="--", alpha=0.6)  # グリッド線追加
    ax.legend()                         # 凡例表示
    plt.tight_layout()                  # レイアウト調整
    return fig                          # フィギュアオブジェクトを返す

# -------------------------------------------------------------------------------------
# 誤差履歴プロット関数
# 各パラメータの収束誤差を対数スケールで可視化
# -------------------------------------------------------------------------------------
def plot_errors(history_df):
    """
    収束誤差の履歴を対数スケールでプロット
    Args:
        history_df: 反復履歴データフレーム
    Returns:
        fig: matplotlibフィギュアオブジェクト
    """
    fig, ax = plt.subplots()            # 図とサブプロット作成
    
    # 各誤差の反復変化をプロット
    ax.plot(history_df["iter"], history_df["err_R"],  label="Error in R")      # 反動度誤差
    ax.plot(history_df["iter"], history_df["err_hs"], label="Error in h_s")    # 静翼高さ誤差
    ax.plot(history_df["iter"], history_df["err_hr"], label="Error in h_r")    # 動翼高さ誤差
    
    # グラフ装飾
    ax.set_xlabel("Iteration")          # X軸ラベル（反復回数）
    ax.set_ylabel("Error (log scale)")  # Y軸ラベル（誤差、対数スケール）
    ax.set_title("Convergence Errors") # グラフタイトル
    ax.set_yscale("log")                # Y軸を対数スケールに設定
    ax.set_xticks(history_df["iter"].astype(int).unique())  # X軸目盛りを整数に設定
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # 整数目盛り強制
    ax.grid(True, which="both", linestyle="--", alpha=0.6)  # 主・副グリッド線追加
    ax.legend()                         # 凡例表示
    plt.tight_layout()                  # レイアウト調整
    return fig                          # フィギュアオブジェクトを返す


# =====================================================================================
# Streamlit ユーザーインターフェース部
# Webアプリケーションの画面構成とインタラクティブ要素を定義
# =====================================================================================

# -------------------------------------------------------------------------------------
# ページ設定とタイトル
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="1D 軸流タービン設計ビジュアライザ", layout="wide")  # ページ設定
st.title("一次元軸流タービン設計（反動度・ソリディティ版）")                      # メインタイトル

# -------------------------------------------------------------------------------------
# サイドバー：入力パラメータ設定
# ユーザーが調整可能な設計パラメータをサイドバーに配置
# -------------------------------------------------------------------------------------
with st.sidebar:
    st.header("入力パラメータ")                                               # サイドバーヘッダー
    
    # --- 基本設計パラメータ ---
    N_rpm = st.slider("回転数 N [rpm]", min_value=1000, max_value=30000, value=10000, step=500)    # 回転数スライダー
    r_hub = st.number_input("ハブ半径 r_hub [m]", value=0.3, format="%.4f")                       # ハブ半径入力
    
    # --- 翼形状パラメータ ---
    c_s = st.number_input("静翼コード長 c_s [m]", value=0.025, format="%.5f")                     # 静翼コード長
    c_r = st.number_input("動翼コード長 c_r [m]", value=0.025, format="%.5f")                     # 動翼コード長
    sigma_s = st.slider("静翼ソリディティ σ_s", min_value=0.1, max_value=3.0, value=1.2, step=0.1)  # 静翼ソリディティ
    sigma_r = st.slider("動翼ソリディティ σ_r", min_value=0.1, max_value=3.0, value=1.2, step=0.1)  # 動翼ソリディティ
    
    # --- 流路幾何パラメータ ---
    throat_ratio_s = st.slider("静翼スロート比 t_s/s_s", min_value=0.0, max_value=1.0, value=0.3, step=0.01)  # 静翼スロート比
    throat_ratio_r = st.slider("動翼スロート比 t_r/s_r", min_value=0.0, max_value=1.0, value=0.3, step=0.01)  # 動翼スロート比
    
    # --- 熱力学条件 ---
    T0 = st.number_input("入口全温 T0 [K]", value=800.0, format="%.1f")                          # 入口全温
    P0 = st.number_input("入口全圧 P0 [Pa]", value=2e5, format="%.3e")                           # 入口全圧
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


# --- 収束履歴グラフ ---
st.markdown("### Convergence History")                                                   # セクションタイトル
st.pyplot(plot_convergence(history_df))                                                  # 収束履歴プロット表示

# --- 誤差履歴グラフ ---
st.markdown("### Error History (log scale)")                                             # セクションタイトル
st.pyplot(plot_errors(history_df))                                                       # 誤差履歴プロット表示

# -------------------------------------------------------------------------------------
# 補足説明
# -------------------------------------------------------------------------------------
st.caption("反動度の定義は動翼（ロータ）側の静エンタルピー落差 / 全体落差。平均半径は r_hub + 0.5*(h_s + h_r) です。")  # 補足説明
