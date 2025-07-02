#!/usr/bin/env python
"""
交互式 2D‑COS 绘图工具
----------------------
* 运行：
    streamlit run 2d_cos_interactive_app.py
* 功能：用户上传任意同格式的波长‑强度矩阵 .xlsx 文件，即刻得到同步/异步二维相关光谱。
* 依赖：streamlit, pandas, numpy, matplotlib, scipy
"""
import re
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
import streamlit as st
from pathlib import Path
import matplotlib as mpl

# ---- ① 指定字体文件路径 ----
FONT_PATH = Path(__file__).parent / "fonts" / "NotoSansCJKsc-Regular.otf"
# 如果你用 apt 安装系统字体（packages.txt → fonts-noto-cjk），
# 可以把这一行改成系统路径，或者直接省略 addfont()。

# ---- ② 动态注册 & 设置默认字体 ----
if FONT_PATH.exists():
    mpl.font_manager.fontManager.addfont(str(FONT_PATH))
    mpl.rcParams["font.family"] = "Noto Sans CJK SC"   # 与字体内部 family 同名
# 正负号正常显示
mpl.rcParams["axes.unicode_minus"] = False



# ================= 自定义绿‑白‑红渐变色 ================= #
cmap = LinearSegmentedColormap.from_list(
    "GreenWhiteRed",
    ["#005700", "#66cc66", "#ffffff", "#ff9999", "#7f0000"],
    N=256,
)

# ================= 辅助函数 ================= #

def natural_key(s: str):
    """提取标签中的数字用于自然排序。"""
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else float("inf")


@st.cache_data(show_spinner=False)
def process_file(uploaded_file, header_row: int, use_std: bool, sigma: int):
    """读取 Excel → 同步 / 异步谱矩阵等中间结果。"""
    df = pd.read_excel(uploaded_file, header=header_row)
    df.rename(columns={df.columns[0]: "wavelength"}, inplace=True)
    df = (
        df.dropna(subset=["wavelength"])  # 去除空行
        .assign(wavelength=lambda x: pd.to_numeric(x["wavelength"], errors="coerce"))
        .sort_values("wavelength")
        .reset_index(drop=True)
    )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # 去掉空列
    df.columns = df.columns.str.strip()

    wls = df["wavelength"].to_numpy()
    spectra = df.drop(columns="wavelength")

    # ---- 同标签多列求均值 ---- #
    prefix = spectra.columns.str.replace(r"\.\d+$", "", regex=True)
    tags = sorted(prefix.unique(), key=natural_key)
    Y = np.vstack([spectra.loc[:, prefix == tag].mean(axis=1).to_numpy() for tag in tags])

    if np.isnan(Y).any():
        raise ValueError("数据包含 NaN，可能列标签分组有误。")

    # ---- 中心化或 Z‑score ---- #
    if use_std:
        std = Y.std(axis=0, ddof=1)
        std[std == 0] = 1
        Yp = (Y - Y.mean(0)) / std
        cbar_lbl = "相关系数 ρ"
    else:
        Yp = Y - Y.mean(0)
        cbar_lbl = "协方差 (arb. u.)"

    # ---- 计算同步 / 异步矩阵 ---- #
    m = Yp.shape[0]
    sync = (Yp.T @ Yp) / (m - 1)
    async_ = (Yp.T @ np.imag(hilbert(Yp, axis=0))) / (m - 1)
    async_ = 0.5 * (async_ - async_.T)

    if sigma > 0:
        sync = gaussian_filter(sync, sigma)
        async_ = gaussian_filter(async_, sigma)

    # ---- 等值面等级 ---- #
    vmax = np.percentile(np.abs(async_), 99)
    pos = np.linspace(0.05 * vmax, vmax, 6)
    levels = np.concatenate([-pos[::-1], [0], pos])
    norm = BoundaryNorm(levels, 256, clip=True)

    return wls, sync, async_, levels, norm, cbar_lbl, tags


def plot_matrix(wls, mat, title, levels, norm, cbar_lbl):
    """返回 matplotlib Figure 方便 Streamlit 渲染和下载。"""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cf = ax.contourf(wls, wls, mat, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.contour(wls, wls, mat, levels=levels, colors="k", linewidths=0.4)
    ax.set_xlabel("发射波长 / nm")
    ax.set_ylabel("发射波长 / nm")
    ax.set_title(title)
    fig.colorbar(cf, ticks=np.unique(levels), label=cbar_lbl)
    fig.tight_layout()
    return fig


# ================= Streamlit 页面设置 ================= #
st.set_page_config(page_title="2D‑COS 绘图工具", layout="centered")
st.title("交互式二维相关光谱 (2D‑COS) 绘图工具")

with st.sidebar:
    st.header("参数设置")
    header_row = st.number_input("Excel 头行索引 (波长正上方)", min_value=0, value=1, step=1)
    use_std = st.checkbox("使用相关系数 (Z‑score)", value=False)
    sigma = st.slider("高斯平滑 σ", 0, 5, value=0)
    enable_download = st.checkbox("启用 PNG 下载按钮", value=True)
    st.markdown("""**说明**\n- 波长列应位于首列，标题行索引从 0 开始计。\n- 同标签多列自动求平均并按数字顺序排序。""")

uploaded_file = st.file_uploader("上传 .xlsx 文件", type=["xlsx"])

if uploaded_file is not None:
    try:
        wls, sync, async_, levels, norm, cbar_lbl, tags = process_file(
            uploaded_file, header_row, use_std, sigma
        )
        st.success(f"文件解析成功！检测到 {len(tags)} 个实验标签：{', '.join(tags)}")

        fig_sync = plot_matrix(wls, sync, "同步二维相关光谱", levels, norm, cbar_lbl)
        st.pyplot(fig_sync)

        fig_async = plot_matrix(wls, async_, "异步二维相关光谱", levels, norm, cbar_lbl)
        st.pyplot(fig_async)

        if enable_download:
            buf1, buf2 = BytesIO(), BytesIO()
            fig_sync.savefig(buf1, format="png", dpi=300)
            fig_async.savefig(buf2, format="png", dpi=300)
            st.download_button("下载同步谱 PNG", buf1.getvalue(), "sync_2dcos.png", "image/png")
            st.download_button("下载异步谱 PNG", buf2.getvalue(), "async_2dcos.png", "image/png")
    except Exception as e:
        st.error(f"解析或绘图出错：{e}")
else:
    st.info("请在上方选择或拖拽 Excel 数据文件 (.xlsx)")
