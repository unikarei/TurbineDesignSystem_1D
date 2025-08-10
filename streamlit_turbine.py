# =====================================================================================
# プログラム仕様書
# =====================================================================================
# プログラム名:    streamlit_turbine.py (パラメータ感度分析機能付き)
# 概要:           一次元軸流タービン設計プログラム（反動度・ソリディティ版）
#                + パラメータ感度分析モード
# 目的:           軸流タービンの予備設計における反動度、翼高さ、速度分布の
#                反復計算による収束設計と可視化、及びパラメータ感度分析
# 
# =====================================================================================
# 主要機能
# =====================================================================================
# 1. 熱力学計算:   等エントロピー関係による状態量計算
# 2. 反復設計:     反動度・翼高さの同時収束計算
# 3. 幾何計算:     ソリディティ、スロート比を考慮した翼列幾何
# 4. 可視化:       収束履歴・誤差履歴の対話的グラフ表示
# 5. WebUI:       Streamlitによる直感的なパラメータ調整界面
# 6. 感度分析:     単一パラメータ変化に対する設計結果の軌跡分析
# 
# =====================================================================================
# 動作要件
# =====================================================================================
# Python:         3.8以上
# 必須ライブラリ:  streamlit, pandas, matplotlib, math, platform, numpy
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
#   pip install streamlit pandas matplotlib numpy
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
# 感度分析:  パラメータ変化に対する全結果の軌跡グラフ
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
# ・感度分析:       ∂(結果)/∂(パラメータ) の数値的評価
# 
# =====================================================================================
# 備考
# =====================================================================================
# ・収束判定:       相対誤差 < 1e-5 (デフォルト)
# ・数値安定性:     ゼロ除算回避、平方根負数回避処理実装
# ・座標系:        軸方向x、周方向θ、径方向r (右手系)
# ・速度定義:       絶対速度V、相対速度W、周速U
# ・翼列配置:       静翼→動翼の単段構成
# ・感度分析:       10-50点でのパラメータスイープ、線形補間
# 
# =====================================================================================
# バージョン履歴
# =====================================================================================
# v1.0 (2025-08-04):  初版リリース、基本機能実装
# v1.1 (2025-08-04):  パラメータ感度分析モード追加
# 
# =====================================================================================
# 作成者・連絡先
# =====================================================================================
# GitHub:          https://github.com/unikarei/TurbineDesignSystem_1D
# リポジトリ:       TurbineDesignSystem_1D
# ライセンス:       MIT License
# 
# =====================================================================================

import math                         # 数学関数（sqrt, sin, cos, asin等）を使用
import streamlit as st              # Webアプリケーション作成用フレームワーク
import pandas as pd                 # データフレーム操作用ライブラリ
import matplotlib.pyplot as plt     # グラフ作成用ライブラリ
import numpy as np                  # 数値計算用ライブラリ（感度分析で使用）

# =====================================================================================
# フォント設定（日本語文字対応、OS別に最適化）
# =====================================================================================
import platform                     # OS判定用モジュール
try:                                # フォント設定エラー回避
    if platform.system() == 'Windows':  # Windowsの場合
        # Windows用日本語対応フォントを優先順位付きで設定
        plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
    else:                           # Windows以外（Mac/Linux）の場合
        # その他OS用フォント設定
        plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
except:                             # フォント設定失敗時の代替処理
    plt.rcParams['font.family'] = ['DejaVu Sans']  # 基本フォントにフォールバック

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
        delta_h_r_est = (W2 ** 2 - W3 ** 2) / 2    # 動翼側実エンタルピー落差推定値
        delta_h_s_est = V2_s ** 2 / 2               # 静翼側実エンタルピー落差推定値
        denom = delta_h_r_est + delta_h_s_est + 1e-16  # 分母（ゼロ除算回避）
        R_new = delta_h_r_est / denom               # 更新後反動度

        # --- Step 14: 収束判定 ---
        err_R = abs(R_new - R)                      # 反動度誤差
        err_hs = abs(h_s_new - h_s) / (h_s + 1e-16) # 静翼高さ相対誤差
        err_hr = abs(h_r_new - h_r) / (h_r + 1e-16) # 動翼高さ相対誤差

        # --- Step 15: 履歴記録 ---
        history.append({
            "iter": i,              # 反復回数
            "R": R,                 # 現在の反動度
            "R_new": R_new,         # 更新後反動度
            "h_s": h_s,             # 現在の静翼高さ [m]
            "h_r": h_r,             # 現在の動翼高さ [m]
            "h_s_new": h_s_new,     # 更新後静翼高さ [m]
            "h_r_new": h_r_new,     # 更新後動翼高さ [m]
            "err_R": err_R,         # 反動度誤差
            "err_hs": err_hs,       # 静翼高さ誤差
            "err_hr": err_hr,       # 動翼高さ誤差
            "U": U,                 # 周速 [m/s]
            "V2_stator": V2_s,      # 静翼出口速度 [m/s]
            "Vtheta": Vtheta,       # 周方向速度成分 [m/s]
            "Vx": Vx,               # 軸方向速度成分 [m/s]
            "W2": W2,               # 動翼入口相対速度 [m/s]
            "W3": W3,               # 動翼出口相対速度 [m/s]
        })

        # --- Step 16: 収束チェック ---
        if err_R < tol and err_hs < tol and err_hr < tol:  # 全誤差が許容値以下
            R = R_new               # 最終反動度更新
            h_s, h_r = h_s_new, h_r_new  # 最終翼高さ更新
            break                   # 収束により反復終了

        # --- Step 17: 次反復への変数更新 ---
        R = R_new                   # 反動度更新
        h_s, h_r = h_s_new, h_r_new # 翼高さ更新

    # -------------------------------------------------------------------------------------
    # 計算結果の辞書化と返却
    # -------------------------------------------------------------------------------------
    result = {
        "R": R,                     # 最終反動度 [-]
        "h_s": h_s,                 # 最終静翼高さ [m]
        "h_r": h_r,                 # 最終動翼高さ [m]
        "r_mean": r_m,              # 平均半径 [m]
        "U": U,                     # 周速 [m/s]
        "V2_stator": V2_s,          # 静翼出口速度 [m/s]
        "Vx": Vx,                   # 軸方向速度成分 [m/s]
        "Vtheta": Vtheta,           # 周方向速度成分 [m/s]
        "W2": W2,                   # 動翼入口相対速度 [m/s]
        "W3": W3,                   # 動翼出口相対速度 [m/s]
        "delta_h_total": delta_h_total,  # 全エンタルピー落差 [J/kg]
        "delta_h_s": delta_h_s,     # 静翼エンタルピー落差 [J/kg]
        "delta_h_r": delta_h_r,     # 動翼エンタルピー落差 [J/kg]
        "rho": rho,                 # 密度 [kg/m³]
        "T_static_out": T_static_out,  # 出口静温度 [K]
        "iterations": i,            # 実行反復回数
    }
    return result, pd.DataFrame(history)  # 結果辞書と履歴データフレームを返却

# =====================================================================================
# パラメータ感度分析関数
# 指定パラメータを変化させて設計結果の軌跡を計算
# =====================================================================================
def parameter_sensitivity_analysis(
    param_name,                     # 変更対象パラメータ名
    param_range,                    # パラメータ値の範囲（リスト）
    base_params                     # 基準パラメータ辞書
):
    """
    指定パラメータの感度分析を実行
    Args:
        param_name: 変更するパラメータ名（文字列）
        param_range: パラメータ値の範囲（numpy配列またはリスト）
        base_params: 基準となるパラメータ辞書
    Returns:
        sensitivity_df: 感度分析結果のデータフレーム
    """
    results = []                    # 結果格納用リスト
    
    for param_value in param_range: # パラメータ範囲をループ
        # 基準パラメータをコピーして対象パラメータのみ変更
        current_params = base_params.copy()    # 基準パラメータの複製
        current_params[param_name] = param_value  # 対象パラメータ値を更新
        
        try:                        # エラー処理付きで設計計算実行
            # 設計計算を実行
            result, _ = design_iteration_with_solidity_history(
                N_rpm=current_params['N_rpm'],                     # 回転数 [rpm]
                r_hub=current_params['r_hub'],                     # ハブ半径 [m]
                c_s=current_params['c_s'],                         # 静翼コード長 [m]
                c_r=current_params['c_r'],                         # 動翼コード長 [m]
                sigma_s=current_params['sigma_s'],                 # 静翼ソリディティ [-]
                sigma_r=current_params['sigma_r'],                 # 動翼ソリディティ [-]
                throat_ratio_s=current_params['throat_ratio_s'],   # 静翼スロート比 [-]
                throat_ratio_r=current_params['throat_ratio_r'],   # 動翼スロート比 [-]
                T0=current_params['T0'],                           # 入口全温 [K]
                P0=current_params['P0'],                           # 入口全圧 [Pa]
                P_out=current_params['P_out'],                     # 出口静圧 [Pa]
                mdot=current_params['mdot'],                       # 質量流量 [kg/s]
                R_init=current_params.get('R_init', 0.5),          # 初期反動度 [-]
                tol=current_params.get('tol', 1e-5),               # 収束許容値
                max_iter=current_params.get('max_iter', 200)       # 最大反復回数
            )
            
            # パラメータ値と計算結果を記録
            result_row = {param_name: param_value}  # パラメータ値
            result_row.update(result)               # 計算結果を追加
            results.append(result_row)              # 結果リストに追加
            
        except Exception as e:      # 計算エラー時の処理
            # エラー発生時はNaNで埋める
            result_row = {param_name: param_value}  # パラメータ値
            result_row.update({key: float('nan') for key in [  # 結果をNaNで初期化
                'R', 'h_s', 'h_r', 'r_mean', 'U', 'V2_stator', 
                'Vx', 'Vtheta', 'W2', 'W3', 'delta_h_total', 
                'delta_h_s', 'delta_h_r', 'rho', 'T_static_out', 'iterations'
            ]})
            results.append(result_row)              # エラー結果もリストに追加
            print(f"Error at {param_name}={param_value}: {e}")  # エラー情報出力
    
    return pd.DataFrame(results)    # 結果をデータフレームで返却

# =====================================================================================
# 可視化関数群
# 収束履歴、誤差履歴、感度分析結果のグラフ作成
# =====================================================================================

# -------------------------------------------------------------------------------------
# 収束履歴プロット関数
# 反動度と翼高さの反復変化を可視化
# -------------------------------------------------------------------------------------
def plot_convergence(history_df):
    """
    収束履歴をプロット
    Args:
        history_df: 反復履歴のデータフレーム
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    fig, ax = plt.subplots(figsize=(10, 6))     # 図とaxesを作成
    ax.plot(history_df["iter"], history_df["R"],     label="R (反動度)", marker="o")      # 反動度の変化
    ax.plot(history_df["iter"], history_df["h_s"],   label="h_s (静翼高さ)", marker="s")  # 静翼高さの変化
    ax.plot(history_df["iter"], history_df["h_r"],   label="h_r (動翼高さ)", marker="^")  # 動翼高さの変化
    ax.set_xlabel("反復回数")                       # x軸ラベル
    ax.set_ylabel("値")                             # y軸ラベル
    ax.set_title("収束履歴 (反動度 & 翼高さ)")       # グラフタイトル
    ax.grid(True, linestyle="--", alpha=0.6)       # グリッド表示
    ax.legend()                                     # 凡例表示
    plt.tight_layout()                              # レイアウト調整
    return fig                                      # figureオブジェクトを返却

# -------------------------------------------------------------------------------------
# 誤差履歴プロット関数
# 収束誤差の変化を対数スケールで可視化
# -------------------------------------------------------------------------------------
def plot_errors(history_df):
    """
    誤差履歴をプロット
    Args:
        history_df: 反復履歴のデータフレーム
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    fig, ax = plt.subplots(figsize=(10, 6))     # 図とaxesを作成
    ax.plot(history_df["iter"], history_df["err_R"],  label="反動度誤差")        # 反動度誤差
    ax.plot(history_df["iter"], history_df["err_hs"], label="静翼高さ誤差")      # 静翼高さ誤差
    ax.plot(history_df["iter"], history_df["err_hr"], label="動翼高さ誤差")      # 動翼高さ誤差
    ax.set_xlabel("反復回数")                       # x軸ラベル
    ax.set_ylabel("誤差 (対数スケール)")            # y軸ラベル
    ax.set_title("収束誤差履歴")                    # グラフタイトル
    ax.set_yscale("log")                            # y軸を対数スケール
    ax.grid(True, which="both", linestyle="--", alpha=0.6)  # 対数グリッド表示
    ax.legend()                                     # 凡例表示
    plt.tight_layout()                              # レイアウト調整
    return fig                                      # figureオブジェクトを返却

# -------------------------------------------------------------------------------------
# パラメータ感度分析結果プロット関数
# 主要な設計結果のパラメータ依存性を可視化
# -------------------------------------------------------------------------------------
def plot_sensitivity_analysis(sensitivity_df, param_name):
    """
    パラメータ感度分析結果をプロット
    Args:
        sensitivity_df: 感度分析結果のデータフレーム
        param_name: 変更したパラメータ名
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2×3のサブプロット作成
    axes = axes.ravel()                             # 1次元配列に変換
    
    # プロット対象とラベルの定義
    plot_targets = [
        ('R', '反動度 [-]'),                       # 反動度
        ('h_s', '静翼高さ [m]'),                   # 静翼高さ
        ('h_r', '動翼高さ [m]'),                   # 動翼高さ
        ('U', '周速 [m/s]'),                       # 周速
        ('V2_stator', '静翼出口速度 [m/s]'),       # 静翼出口速度
        ('delta_h_total', '全エンタルピー落差 [J/kg]')  # 全エンタルピー落差
    ]
    
    for i, (target, ylabel) in enumerate(plot_targets):  # 各プロット対象をループ
        ax = axes[i]                                # 現在のaxes
        valid_data = sensitivity_df.dropna(subset=[target])  # NaN値を除外
        if not valid_data.empty:                    # 有効データが存在する場合
            ax.plot(valid_data[param_name], valid_data[target], 'bo-', linewidth=2, markersize=4)  # プロット
        ax.set_xlabel(f'{param_name}')              # x軸ラベル
        ax.set_ylabel(ylabel)                       # y軸ラベル
        ax.grid(True, alpha=0.3)                    # グリッド表示
        ax.set_title(f'{ylabel} vs {param_name}')   # サブプロットタイトル
    
    plt.suptitle(f'パラメータ感度分析: {param_name}', fontsize=16)  # 全体タイトル
    plt.tight_layout()                              # レイアウト調整
    return fig                                      # figureオブジェクトを返却

# -------------------------------------------------------------------------------------
# 速度三角形プロット関数
# パラメータ変化に対する速度三角形の変化を可視化
# -------------------------------------------------------------------------------------
def plot_velocity_triangles(sensitivity_df, param_name):
    """
    パラメータ変化に対する速度三角形の変化をプロット
    Args:
        sensitivity_df: 感度分析結果のデータフレーム
        param_name: 変更したパラメータ名
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    # 有効データの抽出（NaN値を除外）
    valid_data = sensitivity_df.dropna(subset=['U', 'Vx', 'Vtheta', 'W2', 'W3']).copy()
    
    if valid_data.empty:
        # データが存在しない場合の空のグラフを返す
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'データが不足しています', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('速度三角形の変化')
        return fig
    
    # パラメータ値でソート
    valid_data = valid_data.sort_values(param_name)
    
    # データを5分割して代表点を選択
    n_triangles = 5
    indices = np.linspace(0, len(valid_data) - 1, n_triangles, dtype=int)
    selected_data = valid_data.iloc[indices]
    
    # 赤系のカラーマップ（ダークからライトへ）
    colors = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A', '#FFB6C1']  # 濃い赤から薄い赤
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側：静翼速度三角形（入口から出口）
    ax1.set_title('静翼速度三角形の変化', fontsize=14, fontweight='bold')
    ax1.set_xlabel('軸方向速度 Vx [m/s]', fontsize=12)
    ax1.set_ylabel('周方向速度 Vθ [m/s]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 右側：動翼相対速度三角形（入口から出口）
    ax2.set_title('動翼相対速度三角形の変化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('軸方向速度 Vx [m/s]', fontsize=12)
    ax2.set_ylabel('相対速度 (U-Vθ) 成分 [m/s]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 各代表点での速度三角形を描画
    for i, (idx, row) in enumerate(selected_data.iterrows()):
        color = colors[i]
        param_val = row[param_name]
        
        # 速度成分の取得
        U = row['U']                    # 周速 [m/s]
        Vx = row['Vx']                  # 軸方向速度成分 [m/s]
        Vtheta = row['Vtheta']          # 周方向速度成分 [m/s]
        W2 = row['W2']                  # 動翼入口相対速度 [m/s]
        W3 = row['W3']                  # 動翼出口相対速度 [m/s]
        V2_stator = row['V2_stator']    # 静翼出口速度 [m/s]
        
        # 静翼速度三角形の描画（左側）
        # 入口：軸方向のみ（Vx=0の場合を想定）
        inlet_x, inlet_y = 0, 0
        # 出口：Vx, Vtheta成分
        outlet_x, outlet_y = Vx, Vtheta
        
        # 静翼内の速度ベクトル
        ax1.arrow(inlet_x, inlet_y, outlet_x - inlet_x, outlet_y - inlet_y,
                 head_width=max(Vx, Vtheta) * 0.05, head_length=max(Vx, Vtheta) * 0.05,
                 fc=color, ec=color, linewidth=2.5, alpha=0.8,
                 label=f'{param_name}={param_val:.2e}')
        
        # 静翼出口速度の大きさを円で表示
        circle1 = plt.Circle((outlet_x, outlet_y), V2_stator * 0.02, 
                           color=color, alpha=0.6, fill=True)
        ax1.add_patch(circle1)
        
        # 座標値の注釈
        ax1.annotate(f'({Vx:.1f}, {Vtheta:.1f})', 
                    xy=(outlet_x, outlet_y), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
        
        # 動翼相対速度三角形の描画（右側）
        # 相対速度成分の計算
        W_theta_in = Vtheta - U         # 動翼入口の相対周方向速度成分
        W_theta_out = -U                # 動翼出口の相対周方向速度成分（理想的に周方向成分ゼロと仮定）
        
        # 動翼入口相対速度ベクトル
        ax2.arrow(0, 0, Vx, W_theta_in,
                 head_width=max(abs(W_theta_in), Vx) * 0.05, 
                 head_length=max(abs(W_theta_in), Vx) * 0.05,
                 fc=color, ec=color, linewidth=2.5, alpha=0.8,
                 label=f'W2: {param_name}={param_val:.2e}')
        
        # 動翼出口相対速度ベクトル（簡略化：軸方向のみと仮定）
        W3_x = Vx  # 軸方向速度は保持
        W3_theta = W_theta_out
        ax2.arrow(0, 0, W3_x, W3_theta,
                 head_width=max(abs(W3_theta), W3_x) * 0.05, 
                 head_length=max(abs(W3_theta), W3_x) * 0.05,
                 fc=color, ec=color, linewidth=1.5, alpha=0.6, linestyle='--',
                 label=f'W3: {param_name}={param_val:.2e}')
        
        # 相対速度の大きさを円で表示
        circle2 = plt.Circle((Vx, W_theta_in), W2 * 0.02, 
                           color=color, alpha=0.6, fill=True)
        ax2.add_patch(circle2)
        
        # 座標値の注釈
        ax2.annotate(f'W2=({Vx:.1f}, {W_theta_in:.1f})', 
                    xy=(Vx, W_theta_in), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    # 凡例の設定
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 軸の範囲を適切に設定
    all_vx = selected_data['Vx'].values
    all_vtheta = selected_data['Vtheta'].values
    all_u = selected_data['U'].values
    
    if len(all_vx) > 0 and len(all_vtheta) > 0:
        # 静翼側の軸範囲
        vx_range = max(all_vx) - min(all_vx) if max(all_vx) != min(all_vx) else max(all_vx) * 0.1
        vtheta_range = max(all_vtheta) - min(all_vtheta) if max(all_vtheta) != min(all_vtheta) else max(all_vtheta) * 0.1
        
        ax1.set_xlim(min(all_vx) - vx_range * 0.1, max(all_vx) + vx_range * 0.1)
        ax1.set_ylim(min(all_vtheta) - vtheta_range * 0.1, max(all_vtheta) + vtheta_range * 0.1)
        
        # 動翼側の軸範囲
        w_theta_values = all_vtheta - all_u
        w_range = max(w_theta_values) - min(w_theta_values) if max(w_theta_values) != min(w_theta_values) else max(abs(w_theta_values)) * 0.1
        
        ax2.set_xlim(min(all_vx) - vx_range * 0.1, max(all_vx) + vx_range * 0.1)
        ax2.set_ylim(min(w_theta_values) - abs(w_range) * 0.1, max(w_theta_values) + abs(w_range) * 0.1)
    
    # 全体のタイトル
    fig.suptitle(f'速度三角形の変化: {param_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------------------------
# タービン速度三角形理論図プロット関数
# 標準的なタービン速度三角形の理論図を描画
# -------------------------------------------------------------------------------------
def plot_turbine_velocity_triangle_theory():
    """
    タービン速度三角形の理論図を描画
    左側：動翼出口（ステーション3）、右側：静翼出口/動翼入口（ステーション2）
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 理論的な速度成分（例示用）
    # ステーション2（静翼出口/動翼入口）
    Ca2 = 100.0          # 軸方向速度 [m/s]
    Ct2 = 80.0           # 周方向速度 [m/s]
    U2 = 120.0           # 周速 [m/s]
    
    # ステーション3（動翼出口）
    Ca3 = 100.0          # 軸方向速度（保持） [m/s]
    Ct3 = 20.0           # 周方向速度（減少） [m/s]
    U3 = 120.0           # 周速（同一半径なら同じ） [m/s]
    
    # 絶対速度の計算
    C2 = (Ca2**2 + Ct2**2)**0.5
    C3 = (Ca3**2 + Ct3**2)**0.5
    
    # 相対速度の計算
    Wt2 = Ct2 - U2       # 相対周方向速度成分
    Wt3 = Ct3 - U3       # 相対周方向速度成分
    W2 = (Ca2**2 + Wt2**2)**0.5
    W3 = (Ca3**2 + Wt3**2)**0.5
    
    # 角度の計算（度）
    alpha2 = np.degrees(np.arctan(Ct2/Ca2))  # 絶対速度角
    alpha3 = np.degrees(np.arctan(Ct3/Ca3))
    beta2 = np.degrees(np.arctan(Wt2/Ca2))   # 相対速度角
    beta3 = np.degrees(np.arctan(Wt3/Ca3))
    
    # 右側：ステーション2（静翼出口/動翼入口）
    ax2.set_title('Turbine Stator Outlet (Rotor Inlet)\nStation 2', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Axial Direction [m/s]', fontsize=12)
    ax2.set_ylabel('Tangential Direction [m/s]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 軸方向基準線
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # 絶対速度ベクトル C2
    ax2.arrow(0, 0, Ca2, Ct2, head_width=8, head_length=8, 
             fc='blue', ec='blue', linewidth=3, label='C2 (Absolute velocity)')
    ax2.text(Ca2/2, Ct2/2 + 10, f'C2={C2:.1f}m/s', fontsize=10, fontweight='bold', color='blue')
    
    # 周速ベクトル U2
    ax2.arrow(0, 0, 0, U2, head_width=8, head_length=8, 
             fc='red', ec='red', linewidth=3, label='U2 (Blade speed)')
    ax2.text(-15, U2/2, f'U2={U2:.1f}m/s', fontsize=10, fontweight='bold', color='red', rotation=90)
    
    # 相対速度ベクトル W2
    ax2.arrow(0, U2, Ca2, Wt2, head_width=8, head_length=8, 
             fc='green', ec='green', linewidth=3, label='W2 (Relative velocity)')
    ax2.text(Ca2/2, U2 + Wt2/2 - 15, f'W2={W2:.1f}m/s', fontsize=10, fontweight='bold', color='green')
    
    # 軸方向速度成分の表示
    ax2.arrow(0, -20, Ca2, 0, head_width=5, head_length=5, 
             fc='orange', ec='orange', linewidth=2, linestyle='--')
    ax2.text(Ca2/2, -35, f'Ca2={Ca2:.1f}m/s', fontsize=10, fontweight='bold', color='orange')
    
    # 角度の表示
    # α2角度
    arc_alpha2 = plt.Circle((0, 0), 30, fill=False, color='blue', linestyle='--', alpha=0.7)
    ax2.add_patch(arc_alpha2)
    ax2.text(35, 10, f'α2={alpha2:.1f}°', fontsize=10, color='blue')
    
    # β2角度
    arc_beta2 = plt.Circle((0, U2), 25, fill=False, color='green', linestyle='--', alpha=0.7)
    ax2.add_patch(arc_beta2)
    ax2.text(30, U2-25, f'β2={beta2:.1f}°', fontsize=10, color='green')
    
    ax2.set_xlim(-30, 140)
    ax2.set_ylim(-50, 150)
    ax2.legend(loc='upper right', fontsize=10)
    
    # 左側：ステーション3（動翼出口）
    ax1.set_title('Turbine Rotor Outlet\nStation 3', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Axial Direction [m/s]', fontsize=12)
    ax1.set_ylabel('Tangential Direction [m/s]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 軸方向基準線
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # 絶対速度ベクトル C3
    ax1.arrow(0, 0, Ca3, Ct3, head_width=8, head_length=8, 
             fc='blue', ec='blue', linewidth=3, label='C3 (Absolute velocity)')
    ax1.text(Ca3/2, Ct3/2 + 10, f'C3={C3:.1f}m/s', 
             fontsize=10, fontweight='bold', color='blue')
    
    # 周速ベクトル U3
    ax1.arrow(0, 0, 0, U3, head_width=8, head_length=8, 
             fc='red', ec='red', linewidth=3, label='U3 (Blade speed)')
    ax1.text(-15, U3/2, f'U3={U3:.1f}m/s', fontsize=10, fontweight='bold', color='red', rotation=90)
    
    # 相対速度ベクトル W3
    ax1.arrow(0, U3, Ca3, Wt3, head_width=8, head_length=8, 
             fc='green', ec='green', linewidth=3, label='W3 (Relative velocity)')
    ax1.text(Ca3/2, U3 + Wt3/2 - 15, f'W3={W3:.1f}m/s', 
             fontsize=10, fontweight='bold', color='green')
    
    # 軸方向速度成分の表示
    ax1.arrow(0, -max(Ca3, U3)*0.2, Ca3, 0, head_width=max(Ca3, U3)*0.05, head_length=max(Ca3, U3)*0.05, 
             fc='orange', ec='orange', linewidth=2, linestyle='--')
    ax1.text(Ca3/2, -max(Ca3, U3)*0.35, f'Ca3={Ca3:.1f}m/s', fontsize=10, fontweight='bold', color='orange')
    
    # 角度の表示
    if alpha3 > 0:
        ax1.text(max(Ca3, U3)*0.3, max(Ca3, U3)*0.08, f'α3={alpha3:.1f}°', 
                fontsize=10, color='blue')
    
    if beta3 != 0:
        ax1.text(max(Ca3, U3)*0.25, U3-max(Ca3, U3)*0.3, f'β3={beta3:.1f}°', 
                fontsize=10, color='green')
    
    # ステーション3の軸範囲調整（改善版）
    # Y軸範囲の計算
    y_values_3 = [0, Ct3, U3, U3 + Wt3, -max(Ca3, U3)*0.35]  # 全ての重要なY座標
    y_min_3 = min(y_values_3)
    y_max_3 = max(y_values_3)
    y_range_3 = y_max_3 - y_min_3
    y_margin_3 = y_range_3 * 0.15  # 15%のマージン
    
    # X軸範囲の計算
    x_values_3 = [0, Ca3, -max(Ca3, U3)*0.15]  # 全ての重要なX座標
    x_min_3 = min(x_values_3)
    x_max_3 = max(x_values_3)
    x_range_3 = x_max_3 - x_min_3
    x_margin_3 = x_range_3 * 0.15  # 15%のマージン
    
    ax1.set_xlim(x_min_3 - x_margin_3, x_max_3 + x_margin_3)
    ax1.set_ylim(y_min_3 - y_margin_3, y_max_3 + y_margin_3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 周方向速度変化の表示（DCt を使用）
    fig.suptitle(f'タービン速度三角形（計算結果）\n周方向速度変化: ΔCt = {DCt:.1f} m/s', 
                 fontsize=16, fontweight='bold')
    
    # 説明テキストの追加
    textstr = '\n'.join([
        '速度三角形構成要素:',
        '• C: 絶対速度 (青色)',
        '• W: 相対速度 (緑色)', 
        '• U: 周速 (赤色)',
        '• Ca: 軸方向速度成分 (オレンジ色)',
        '• α: 絶対流れ角',
        '• β: 相対流れ角'
    ])
    
    # テキストボックスを図の下部に配置
    fig.text(0.5, 0.02, textstr, fontsize=11, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.93])
    return fig

# -------------------------------------------------------------------------------------
# 実際の計算結果に基づく速度三角形描画関数
# 通常設計モード用の速度三角形可視化
# -------------------------------------------------------------------------------------
def plot_actual_velocity_triangles(result):
    """
    実際の計算結果に基づくタービン速度三角形を描画
    Args:
        result: 設計計算結果の辞書
    Returns:
        fig: matplotlibのfigureオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 計算結果から速度成分を取得
    U = result['U']                    # 周速 [m/s]
    Vx = result['Vx']                  # 軸方向速度成分 [m/s]
    Vtheta = result['Vtheta']          # 周方向速度成分 [m/s]
    W2 = result['W2']                  # 動翼入口相対速度 [m/s]
    W3 = result['W3']                  # 動翼出口相対速度 [m/s]
    V2_stator = result['V2_stator']    # 静翼出口速度 [m/s]
    
    # ステーション2（静翼出口/動翼入口）
    Ca2 = Vx                          # 軸方向速度 [m/s]
    Ct2 = Vtheta                      # 周方向速度 [m/s]
    U2 = U                            # 周速 [m/s]
    C2 = V2_stator                    # 絶対速度 [m/s]
    
    # ステーション3（動翼出口）
    Ca3 = Vx                          # 軸方向速度（保持） [m/s]
    Ct3 = Vtheta * 0.2                # 周方向速度（大幅減少） [m/s]
    U3 = U                            # 周速（同一半径なら同じ） [m/s]
    C3 = (Ca3**2 + Ct3**2)**0.5       # 絶対速度 [m/s]
    
    # 相対速度成分の計算
    Wt2 = Ct2 - U2                    # 相対周方向速度成分（ステーション2）
    Wt3 = Ct3 - U3                    # 相対周方向速度成分（ステーション3）
    W2_calc = (Ca2**2 + Wt2**2)**0.5  # 相対速度計算値
    W3_calc = (Ca3**2 + Wt3**2)**0.5  # 相対速度計算値
    
    # 周方向速度変化の計算（ここで DCt を定義）
    DCt = Ct2 - Ct3                   # 周方向速度変化 [m/s]
    
    # 角度の計算（度）
    alpha2 = np.degrees(np.arctan(abs(Ct2)/abs(Ca2))) if Ca2 != 0 else 0  # 絶対速度角
    alpha3 = np.degrees(np.arctan(abs(Ct3)/abs(Ca3))) if Ca3 != 0 else 0
    beta2 = np.degrees(np.arctan(abs(Wt2)/abs(Ca2))) if Ca2 != 0 else 0   # 相対速度角
    beta3 = np.degrees(np.arctan(abs(Wt3)/abs(Ca3))) if Ca3 != 0 else 0
    
    # 右側：ステーション2（静翼出口/動翼入口）
    ax2.set_title('Turbine Stator Outlet (Rotor Inlet)\nStation 2', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Axial Direction [m/s]', fontsize=12)
    ax2.set_ylabel('Tangential Direction [m/s]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 軸方向基準線
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # 絶対速度ベクトル C2
    ax2.arrow(0, 0, Ca2, Ct2, head_width=8, head_length=8, 
             fc='blue', ec='blue', linewidth=3, label='C2 (Absolute velocity)')
    ax2.text(Ca2/2, Ct2/2 + 10, f'C2={C2:.1f}m/s', fontsize=10, fontweight='bold', color='blue')
    
    # 周速ベクトル U2
    ax2.arrow(0, 0, 0, U2, head_width=8, head_length=8, 
             fc='red', ec='red', linewidth=3, label='U2 (Blade speed)')
    ax2.text(-15, U2/2, f'U2={U2:.1f}m/s', fontsize=10, fontweight='bold', color='red', rotation=90)
    
    # 相対速度ベクトル W2
    ax2.arrow(0, U2, Ca2, Wt2, head_width=8, head_length=8, 
             fc='green', ec='green', linewidth=3, label='W2 (Relative velocity)')
    ax2.text(Ca2/2, U2 + Wt2/2 - 15, f'W2={W2:.1f}m/s', fontsize=10, fontweight='bold', color='green')
    
    # 軸方向速度成分の表示
    ax2.arrow(0, -20, Ca2, 0, head_width=5, head_length=5, 
             fc='orange', ec='orange', linewidth=2, linestyle='--')
    ax2.text(Ca2/2, -35, f'Ca2={Ca2:.1f}m/s', fontsize=10, fontweight='bold', color='orange')
    
    # 角度の表示
    # α2角度
    arc_alpha2 = plt.Circle((0, 0), 30, fill=False, color='blue', linestyle='--', alpha=0.7)
    ax2.add_patch(arc_alpha2)
    ax2.text(35, 10, f'α2={alpha2:.1f}°', fontsize=10, color='blue')
    
    # β2角度
    arc_beta2 = plt.Circle((0, U2), 25, fill=False, color='green', linestyle='--', alpha=0.7)
    ax2.add_patch(arc_beta2)
    ax2.text(30, U2-25, f'β2={beta2:.1f}°', fontsize=10, color='green')
    
    ax2.set_xlim(-30, 140)
    ax2.set_ylim(-50, 150)
    ax2.legend(loc='upper right', fontsize=10)
    
    # 左側：ステーション3（動翼出口）
    ax1.set_title('Turbine Rotor Outlet\nStation 3', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Axial Direction [m/s]', fontsize=12)
    ax1.set_ylabel('Tangential Direction [m/s]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 軸方向基準線
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # 絶対速度ベクトル C3
    ax1.arrow(0, 0, Ca3, Ct3, head_width=8, head_length=8, 
             fc='blue', ec='blue', linewidth=3, label='C3 (Absolute velocity)')
    ax1.text(Ca3/2, Ct3/2 + 10, f'C3={C3:.1f}m/s', 
             fontsize=10, fontweight='bold', color='blue')
    
    # 周速ベクトル U3
    ax1.arrow(0, 0, 0, U3, head_width=8, head_length=8, 
             fc='red', ec='red', linewidth=3, label='U3 (Blade speed)')
    ax1.text(-15, U3/2, f'U3={U3:.1f}m/s', fontsize=10, fontweight='bold', color='red', rotation=90)
    
    # 相対速度ベクトル W3
    ax1.arrow(0, U3, Ca3, Wt3, head_width=8, head_length=8, 
             fc='green', ec='green', linewidth=3, label='W3 (Relative velocity)')
    ax1.text(Ca3/2, U3 + Wt3/2 - 15, f'W3={W3:.1f}m/s', 
             fontsize=10, fontweight='bold', color='green')
    
    # 軸方向速度成分の表示
    ax1.arrow(0, -max(Ca3, U3)*0.2, Ca3, 0, head_width=max(Ca3, U3)*0.05, head_length=max(Ca3, U3)*0.05, 
             fc='orange', ec='orange', linewidth=2, linestyle='--')
    ax1.text(Ca3/2, -max(Ca3, U3)*0.35, f'Ca3={Ca3:.1f}m/s', fontsize=10, fontweight='bold', color='orange')
    
    # 角度の表示
    if alpha3 > 0:
        ax1.text(max(Ca3, U3)*0.3, max(Ca3, U3)*0.08, f'α3={alpha3:.1f}°', 
                fontsize=10, color='blue')
    
    if beta3 != 0:
        ax1.text(max(Ca3, U3)*0.25, U3-max(Ca3, U3)*0.3, f'β3={beta3:.1f}°', 
                fontsize=10, color='green')
    
    # ステーション3の軸範囲調整（改善版）
    # Y軸範囲の計算
    y_values_3 = [0, Ct3, U3, U3 + Wt3, -max(Ca3, U3)*0.35]  # 全ての重要なY座標
    y_min_3 = min(y_values_3)
    y_max_3 = max(y_values_3)
    y_range_3 = y_max_3 - y_min_3
    y_margin_3 = y_range_3 * 0.15  # 15%のマージン
    
    # X軸範囲の計算
    x_values_3 = [0, Ca3, -max(Ca3, U3)*0.15]  # 全ての重要なX座標
    x_min_3 = min(x_values_3)
    x_max_3 = max(x_values_3)
    x_range_3 = x_max_3 - x_min_3
    x_margin_3 = x_range_3 * 0.15  # 15%のマージン
    
    ax1.set_xlim(x_min_3 - x_margin_3, x_max_3 + x_margin_3)
    ax1.set_ylim(y_min_3 - y_margin_3, y_max_3 + y_margin_3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 周方向速度変化の表示（DCt を使用）
    fig.suptitle(f'タービン速度三角形（計算結果）\n周方向速度変化: ΔCt = {DCt:.1f} m/s', 
                 fontsize=16, fontweight='bold')
    
    # 説明テキストの追加
    textstr = '\n'.join([
        '速度三角形構成要素:',
        '• C: 絶対速度 (青色)',
        '• W: 相対速度 (緑色)', 
        '• U: 周速 (赤色)',
        '• Ca: 軸方向速度成分 (オレンジ色)',
        '• α: 絶対流れ角',
        '• β: 相対流れ角'
    ])
    
    # テキストボックスを図の下部に配置
    fig.text(0.5, 0.02, textstr, fontsize=11, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.93])
    return fig

# =====================================================================================
# Streamlit WebUI 部分
# ユーザーインターフェース定義、パラメータ入力、結果表示
# =====================================================================================

# -------------------------------------------------------------------------------------
# Streamlitページ設定
# ページタイトル、レイアウト、初期設定
# -------------------------------------------------------------------------------------
st.set_page_config(
    page_title="1D 軸流タービン設計ツール",       # ブラウザタブのタイトル
    layout="wide",                              # 広いレイアウト使用
    initial_sidebar_state="expanded"            # サイドバーを最初から展開
)

# -------------------------------------------------------------------------------------
# メインタイトル表示
# -------------------------------------------------------------------------------------
st.title("一次元軸流タービン設計（反動度・ソリディティ版）+ パラメータ感度分析")

# -------------------------------------------------------------------------------------
# モード選択
# 通常設計モードか感度分析モードかを選択
# -------------------------------------------------------------------------------------
mode = st.radio(
    "動作モード選択",                           # ラジオボタンのタイトル
    ["通常設計モード", "パラメータ感度分析モード"],  # 選択肢
    index=0                                     # デフォルト選択（通常設計モード）
)

# -------------------------------------------------------------------------------------
# サイドバーでの共通パラメータ入力
# 両モードで使用する基本パラメータの入力UI
# -------------------------------------------------------------------------------------
with st.sidebar:
    st.header("入力パラメータ")                 # サイドバーヘッダー
    
    # 運転条件パラメータ群
    st.subheader("運転条件")                    # サブヘッダー
    N_rpm = st.slider("回転数 N [rpm]", min_value=1000, max_value=30000, value=10000, step=500)
    r_hub = st.number_input("ハブ半径 r_hub [m]", value=0.3, format="%.4f", min_value=0.001, max_value=10.0)
    
    # 翼形状パラメータ群
    st.subheader("翼形状")                      # サブヘッダー
    c_s = st.number_input("静翼コード長 c_s [m]", value=0.025, format="%.5f", min_value=0.001, max_value=1.0)
    c_r = st.number_input("動翼コード長 c_r [m]", value=0.025, format="%.5f", min_value=0.001, max_value=1.0)
    sigma_s = st.slider("静翼ソリディティ σ_s", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    sigma_r = st.slider("動翼ソリディティ σ_r", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
    throat_ratio_s = st.slider("静翼スロート比 t_s/s_s", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    throat_ratio_r = st.slider("動翼スロート比 t_r/s_r", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    
    # 熱力学条件パラメータ群
    st.subheader("熱力学条件")                  # サブヘッダー
    T0 = st.number_input("入口全温 T0 [K]", value=800.0, format="%.1f", min_value=300.0, max_value=2000.0)
    P0 = st.number_input("入口全圧 P0 [Pa]", value=2e5, format="%.3e", min_value=1e4, max_value=1e7)
    P_out = st.number_input("出口静圧 P_out [Pa]", value=1.0e5, format="%.3e", min_value=1e3, max_value=1e6)
    mdot = st.number_input("質量流量 ṁ [kg/s]", value=0.01, format="%.3f", min_value=1e-6, max_value=1000.0)
    
    # 計算制御パラメータ群
    st.subheader("計算制御")                    # サブヘッダー
    R_init = st.slider("初期反動度 R_init", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    tol = st.number_input("収束許容値 tol", value=1e-5, format="%.0e", min_value=1e-8, max_value=1e-2)
    max_iter = st.number_input("最大反復回数", value=200, step=10, min_value=10, max_value=1000)

# -------------------------------------------------------------------------------------
# 基準パラメータ辞書の作成
# 感度分析で使用する基準値を辞書形式で格納
# -------------------------------------------------------------------------------------
base_params = {
    'N_rpm': N_rpm,                             # 回転数 [rpm]
    'r_hub': r_hub,                             # ハブ半径 [m]
    'c_s': c_s,                                 # 静翼コード長 [m]
    'c_r': c_r,                                 # 動翼コード長 [m]
    'sigma_s': sigma_s,                         # 静翼ソリディティ [-]
    'sigma_r': sigma_r,                         # 動翼ソリディティ [-]
    'throat_ratio_s': throat_ratio_s,           # 静翼スロート比 [-]
    'throat_ratio_r': throat_ratio_r,           # 動翼スロート比 [-]
    'T0': T0,                                   # 入口全温 [K]
    'P0': P0,                                   # 入口全圧 [Pa]
    'P_out': P_out,                             # 出口静圧 [Pa]
    'mdot': mdot,                               # 質量流量 [kg/s]
    'R_init': R_init,                           # 初期反動度 [-]
    'tol': tol,                                 # 収束許容値
    'max_iter': int(max_iter)                   # 最大反復回数
}

# =====================================================================================
# モード別処理分岐
# 通常設計モードまたは感度分析モードの処理
# =====================================================================================

if mode == "通常設計モード":
    # ---------------------------------------------------------------------------------
    # 通常設計モード処理
    # 単一条件での設計計算と結果表示
    # ---------------------------------------------------------------------------------
    
    st.subheader("設計反復結果")               # セクションヘッダー
    
    # 設計計算実行
    result, history_df = design_iteration_with_solidity_history(
        N_rpm=N_rpm, r_hub=r_hub,              # 運転条件
        c_s=c_s, c_r=c_r, sigma_s=sigma_s, sigma_r=sigma_r,  # 翼形状
        throat_ratio_s=throat_ratio_s, throat_ratio_r=throat_ratio_r,  # スロート比
        T0=T0, P0=P0, P_out=P_out, mdot=mdot,   # 熱力学条件
        R_init=R_init, tol=tol, max_iter=int(max_iter)  # 計算制御
    )
    
    # 実際の速度三角形の表示（結果表示の最初に配置）
    st.markdown("### タービン速度三角形（実計算結果）")
    try:
        fig_actual = plot_actual_velocity_triangles(result)
        st.pyplot(fig_actual, clear_figure=True)
        plt.close(fig_actual)
        
        st.info("""
        **実計算結果による速度三角形:**
        - **左側**: 動翼出口（ステーション3）での速度分布
        - **右側**: 静翼出口/動翼入口（ステーション2）での速度分布
        - **色分け**: 絶対速度（青）、相対速度（緑）、周速（赤）、軸方向成分（オレンジ）
        - **角度**: α（絶対流れ角）、β（相対流れ角）
        - **エネルギー抽出**: ΔCt（周方向速度変化）で仕事を取り出す
        """)
    except Exception as e:
        st.error(f"速度三角形作成エラー: {e}")

    # 結果表示用カラムレイアウト
    col1, col2 = st.columns(2)                 # 2カラムレイアウト作成
    
    # 左カラム：主要設計結果
    with col1:
        st.markdown("### 収束結果サマリ")       # マークダウンヘッダー
        summary = {                             # サマリ辞書作成
            "反動度 R": result["R"],            # 反動度
            "静翼高さ h_s [m]": result["h_s"],  # 静翼高さ
            "動翼高さ h_r [m]": result["h_r"],  # 動翼高さ
            "平均半径 r_m [m]": result["r_mean"],  # 平均半径
            "周速 U [m/s]": result["U"],        # 周速
        }
        # サマリテーブル表示（科学記法フォーマット）
        st.table(pd.Series(summary).apply(lambda x: f"{x:.5e}"))
    
    # 右カラム：速度・エネルギー系結果
    with col2:
        st.markdown("### 速度/エネルギー系")    # マークダウンヘッダー
        vtab = {                                # 速度・エネルギー系辞書
            "静翼流出速度 V2_stator [m/s]": result["V2_stator"],  # 静翼出口速度
            "軸方向成分 Vx [m/s]": result["Vx"],                  # 軸方向速度成分
            "周方向成分 Vθ [m/s]": result["Vtheta"],              # 周方向速度成分
            "動翼入口相対速度 W2 [m/s]": result["W2"],            # 動翼入口相対速度
            "動翼出口相対速度 W3 [m/s]": result["W3"],            # 動翼出口相対速度
            "全落差 Δh_total [J/kg]": result["delta_h_total"],    # 全エンタルピー落差
        }
        # 速度・エネルギーテーブル表示（科学記法フォーマット）
        st.table(pd.Series(vtab).apply(lambda x: f"{x:.5e}"))
    
    # 収束履歴グラフ表示
    st.markdown("### 収束履歴")                # セクションヘッダー
    st.pyplot(plot_convergence(history_df))    # 収束履歴プロット表示
    
    # 誤差履歴グラフ表示
    st.markdown("### 誤差履歴 (対数スケール)")  # セクションヘッダー
    st.pyplot(plot_errors(history_df))         # 誤差履歴プロット表示
    
    # フッター情報
    st.caption("反動度の定義は動翼（ロータ）側の静エンタルピー落差 / 全体落差。平均半径は r_hub + 0.5*(h_s + h_r) です。")

else:
    # ---------------------------------------------------------------------------------
    # パラメータ感度分析モード処理
    # 指定パラメータを変化させて設計結果の軌跡を分析
    # ---------------------------------------------------------------------------------
    
    st.subheader("パラメータ感度分析")          # セクションヘッダー
    
    # 理論速度三角形の表示
    st.markdown("### タービン速度三角形（理論図）")
    try:
        fig_theory = plot_turbine_velocity_triangle_theory()
        st.pyplot(fig_theory, clear_figure=True)
        plt.close(fig_theory)
        
        st.info("""
        **速度三角形の説明:**
        - **絶対速度 C**: 静止座標系での流体速度（青色）
        - **相対速度 W**: 動翼に対する流体の相対速度（緑色）
        - **周速 U**: 動翼の周方向速度（赤色）
        - **軸方向速度 Ca**: 軸方向成分（オレンジ色）
        - **角度 α**: 絶対流れ角、**角度 β**: 相対流れ角
        - **エネルギー抽出**: ΔCt（周方向速度変化）により仕事を取り出す
        """)
    except Exception as e:
        st.error(f"理論速度三角形作成エラー: {e}")
    
    # 感度分析対象パラメータ選択UI
    col1, col2, col3 = st.columns(3)           # 3カラムレイアウト作成
    
    with col1:
        # 感度分析対象パラメータ選択
        param_options = {                       # パラメータ選択肢辞書
            "回転数 N_rpm [rpm]": "N_rpm",      # 回転数
            "ハブ半径 r_hub [m]": "r_hub",      # ハブ半径
            "静翼コード長 c_s [m]": "c_s",      # 静翼コード長
            "動翼コード長 c_r [m]": "c_r",      # 動翼コード長
            "静翼ソリディティ σ_s": "sigma_s",  # 静翼ソリディティ
            "動翼ソリディティ σ_r": "sigma_r",  # 動翼ソリディティ
            "静翼スロート比": "throat_ratio_s",  # 静翼スロート比
            "動翼スロート比": "throat_ratio_r",  # 動翼スロート比
            "入口全温 T0 [K]": "T0",            # 入口全温
        }
        selected_param_display = st.selectbox(  # セレクトボックス
            "感度分析対象パラメータ",           # ラベル
            list(param_options.keys())          # 選択肢
        )
        selected_param = param_options[selected_param_display]  # 選択パラメータ名取得
    
    with col2:
        # パラメータ変化範囲設定
        st.write("変化範囲設定")                # 説明テキスト
        base_value = base_params[selected_param]  # 基準値取得
        min_factor = st.slider("最小値倍率", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        max_factor = st.slider("最大値倍率", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
        n_points = st.slider("計算点数", min_value=5, max_value=50, value=20, step=5)
    
    with col3:
        # 計算範囲表示
        min_val = base_value * min_factor       # 最小値計算
        max_val = base_value * max_factor       # 最大値計算
        st.write("計算範囲")                    # 説明テキスト
        st.write(f"基準値: {base_value:.3e}")   # 基準値表示
        st.write(f"最小値: {min_val:.3e}")      # 最小値表示
        st.write(f"最大値: {max_val:.3e}")      # 最大値表示
        st.write(f"計算点数: {n_points}")       # 計算点数表示
    
    # 感度分析実行ボタン
    if st.button("感度分析実行", type="primary"):  # 実行ボタン
        
        # パラメータ範囲生成
        param_range = np.linspace(min_val, max_val, n_points)  # 等間隔点列生成
        
        # 進捗表示
        progress_bar = st.progress(0)           # プログレスバー初期化
        status_text = st.empty()                # ステータステキスト領域確保
        
        # 感度分析実行
        sensitivity_results = []                # 結果格納リスト
        for i, param_value in enumerate(param_range):  # パラメータ範囲をループ
            
            # 進捗表示更新
            progress = (i + 1) / len(param_range)  # 進捗率計算
            progress_bar.progress(progress)     # プログレスバー更新
            status_text.text(f"計算中... {i+1}/{len(param_range)} ({param_value:.3e})")  # ステータス表示
            
            # 現在パラメータで設計計算実行
            current_params = base_params.copy() # 基準パラメータ複製
            current_params[selected_param] = param_value  # 対象パラメータ更新
            
            try:                                # エラー処理付き計算
                result, _ = design_iteration_with_solidity_history(**current_params)  # 設計計算実行
                result[selected_param] = param_value  # パラメータ値追加
                sensitivity_results.append(result)  # 結果リストに追加
            except Exception as e:              # エラー発生時
                st.warning(f"計算エラー (パラメータ値: {param_value:.3e}): {e}")  # 警告表示
                # エラー時もNaNデータを追加して軌跡を継続
                error_result = {selected_param: param_value}  # パラメータ値
                error_result.update({key: float('nan') for key in [  # 結果をNaNで初期化
                    'R', 'h_s', 'h_r', 'r_mean', 'U', 'V2_stator', 
                    'Vx', 'Vtheta', 'W2', 'W3', 'delta_h_total', 
                    'delta_h_s', 'delta_h_r', 'rho', 'T_static_out', 'iterations'
                ]})
                sensitivity_results.append(error_result)  # エラー結果もリストに追加
        
        # 進捗表示クリア
        progress_bar.empty()                    # プログレスバー削除
        status_text.text("計算完了!")            # 完了メッセージ
        
        # 結果データフレーム作成
        if sensitivity_results:                 # 結果が存在する場合
            sensitivity_df = pd.DataFrame(sensitivity_results)  # データフレーム作成
            
            # 感度分析結果グラフ表示
            st.markdown("### 感度分析結果")      # セクションヘッダー
            try:                                # グラフ作成エラー処理
                fig_sensitivity = plot_sensitivity_analysis(sensitivity_df, selected_param)  # 感度分析プロット作成
                st.pyplot(fig_sensitivity, clear_figure=True)  # グラフ表示（メモリリーク回避）
                plt.close(fig_sensitivity)      # 明示的にfigureを閉じる
            except Exception as e:              # グラフ作成エラー時
                st.error(f"グラフ作成エラー: {e}")  # エラー表示
                st.write("データの詳細:")          # デバッグ情報
                st.write(f"データフレーム形状: {sensitivity_df.shape}")  # データ形状
                st.write(f"カラム: {sensitivity_df.columns.tolist()}")   # カラム名
                st.write("先頭5行:")               # 先頭データ
                st.dataframe(sensitivity_df.head())  # 先頭データ表示
            
            # 速度三角形の変化グラフ追加
            st.markdown("### 速度三角形の変化")  # セクションヘッダー
            try:                                # グラフ作成エラー処理
                fig_velocity = plot_velocity_triangles(sensitivity_df, selected_param)  # 速度三角形プロット作成
                st.pyplot(fig_velocity, clear_figure=True)  # グラフ表示（メモリリーク回避）
                plt.close(fig_velocity)         # 明示的にfigureを閉じる
                
                # 速度三角形の説明
                st.info("""
                **速度三角形の説明:**
                - **左側（静翼）**: 入口から出口への絶対速度変化。矢印は速度ベクトル、円は速度の大きさを表示
                - **右側（動翼）**: 相対速度の変化。実線は入口相対速度(W2)、破線は出口相対速度(W3)
                - **色の変化**: 濃い赤（パラメータ最小値）→ 薄い赤（パラメータ最大値）
                - **座標**: X軸は軸方向速度、Y軸は周方向速度成分
                """)
            except Exception as e:              # グラフ作成エラー時
                st.error(f"速度三角形グラフ作成エラー: {e}")  # エラー表示
            
            # 追加の個別グラフ表示
            st.markdown("### 個別グラフ")        # セクションヘッダー
            
            # 2列レイアウトで個別グラフを表示
            col_graph1, col_graph2 = st.columns(2)  # 2カラムレイアウト
            
            # 反動度と翼高さのグラフ
            with col_graph1:
                try:
                    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2行1列のサブプロット
                    
                    # 反動度のプロット
                    valid_data = sensitivity_df.dropna(subset=['R'])  # NaN値除外
                    if not valid_data.empty:
                        ax1.plot(valid_data[selected_param], valid_data['R'], 'ro-', linewidth=2, markersize=5)
                    ax1.set_xlabel(f'{selected_param}')
                    ax1.set_ylabel('反動度 [-]')
                    ax1.set_title(f'反動度 vs {selected_param}')
                    ax1.grid(True, alpha=0.3)
                    
                    # 静翼高さのプロット
                    valid_data = sensitivity_df.dropna(subset=['h_s'])  # NaN値除外
                    if not valid_data.empty:
                        ax2.plot(valid_data[selected_param], valid_data['h_s'], 'bo-', linewidth=2, markersize=5)
                    ax2.set_xlabel(f'{selected_param}')
                    ax2.set_ylabel('静翼高さ [m]')
                    ax2.set_title(f'静翼高さ vs {selected_param}')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig1, clear_figure=True)
                    plt.close(fig1)
                except Exception as e:
                    st.error(f"個別グラフ1作成エラー: {e}")
            
            # 周速と速度のグラフ
            with col_graph2:
                try:
                    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2行1列のサブプロット
                    
                    # 周速のプロット
                    valid_data = sensitivity_df.dropna(subset=['U'])  # NaN値除外
                    if not valid_data.empty:
                        ax1.plot(valid_data[selected_param], valid_data['U'], 'go-', linewidth=2, markersize=5)
                    ax1.set_xlabel(f'{selected_param}')
                    ax1.set_ylabel('周速 [m/s]')
                    ax1.set_title(f'周速 vs {selected_param}')
                    ax1.grid(True, alpha=0.3)
                    
                    # 静翼出口速度のプロット
                    valid_data = sensitivity_df.dropna(subset=['V2_stator'])  # NaN値除外
                    if not valid_data.empty:
                        ax2.plot(valid_data[selected_param], valid_data['V2_stator'], 'mo-', linewidth=2, markersize=5)
                    ax2.set_xlabel(f'{selected_param}')
                    ax2.set_ylabel('静翼出口速度 [m/s]')
                    ax2.set_title(f'静翼出口速度 vs {selected_param}')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig2, clear_figure=True)
                    plt.close(fig2)
                except Exception as e:
                    st.error(f"個別グラフ2作成エラー: {e}")
            
            # 詳細結果テーブル表示
            st.markdown("### 詳細結果テーブル")  # セクションヘッダー
            # 主要結果のみ抽出して表示
            display_columns = [selected_param, 'R', 'h_s', 'h_r', 'U', 'V2_stator', 'delta_h_total']
            
            # 存在するカラムのみを選択
            available_columns = [col for col in display_columns if col in sensitivity_df.columns]
            
            if available_columns:
                st.dataframe(sensitivity_df[available_columns].round(6))  # 小数点6桁で表示
            else:
                st.error("表示可能なデータがありません")
            
            # CSV ダウンロード機能
            csv = sensitivity_df.to_csv(index=False)  # CSV形式変換
            st.download_button(                 # ダウンロードボタン
                label="結果をCSVでダウンロード",  # ボタンラベル
                data=csv,                       # ダウンロードデータ
                file_name=f'sensitivity_analysis_{selected_param}.csv',  # ファイル名
                mime='text/csv'                 # MIMEタイプ
            )
        else:
            st.error("すべての計算でエラーが発生しました。パラメータ範囲を確認してください。")  # エラーメッセージ
