import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Lettuce Growth Modeling Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DAYS = np.arange(1, 31, dtype=float)
BIOMASS = np.array(
    [3.1, 7.8, 9.5, 10.2, 12.5, 15.3, 18.5, 21.8, 26.2, 31.0,
     35.6, 40.5, 45.8, 52.3, 60.1, 69.2, 77.8, 84.5, 93.1, 100.5,
     108.4, 114.9, 120.7, 126.8, 134.1, 141.2, 146.3, 141.8, 148.2, 152.6],
    dtype=float,
)


def logistic_model(t, L, k, t0):
    """y = L / (1 + exp(-k * (t - t0)))."""
    return L / (1 + np.exp(-k * (t - t0)))


def gompertz_model(t, L, k, t0):
    """y = L * exp(-exp(-k * (t - t0)))."""
    return L * np.exp(-np.exp(-k * (t - t0)))


def style_chart(fig, ax, mode):
    """Apply a cohesive research-dashboard style to the Matplotlib plot."""
    dark = mode == "Dark"
    background = "#111827" if dark else "#FFFFFF"
    panel = "#182234" if dark else "#F8FAFC"
    text = "#E5E7EB" if dark else "#162A1D"
    muted = "#AAB7C4" if dark else "#51615A"
    grid = "#334155" if dark else "#D8E0DA"

    fig.patch.set_facecolor(background)
    ax.set_facecolor(panel)
    ax.grid(True, color=grid, linewidth=0.85, alpha=0.78)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(grid)
        spine.set_linewidth(1.15)
    ax.tick_params(colors=text, labelsize=10)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)
    return text, muted


def make_figure(model_name, L, k, t0, theme):
    """Create the responsive chart and return it with predictions/metrics."""
    model_function = logistic_model if model_name == "Logistic Model" else gompertz_model
    line_color = "#35C16F" if model_name == "Logistic Model" else "#F59E0B"
    predicted_observed = model_function(DAYS, L, k, t0)
    dense_days = np.linspace(0, 30, 600)
    dense_prediction = model_function(dense_days, L, k, t0)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    text, muted = style_chart(fig, ax, theme)

    ax.scatter(
        DAYS, BIOMASS, s=56, color="#E84B4B", edgecolor="#FFFFFF",
        linewidth=0.9, alpha=0.96, zorder=4, label="Observed biomass",
    )
    ax.plot(
        dense_days, dense_prediction, color=line_color, linewidth=3.5,
        zorder=3, label=f"{model_name} prediction",
    )
    ax.axvline(t0, color=muted, linestyle="--", linewidth=1.25, alpha=0.82)
    ax.annotate(
        f"Inflection point\nDay {t0:.1f}", xy=(t0, model_function(t0, L, k, t0)),
        xytext=(8, -42), textcoords="offset points", color=text, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": muted, "lw": 1},
    )
    ax.axhline(L, color=line_color, linestyle=":", linewidth=1.4, alpha=0.85)
    ax.text(29.7, L + 2, f"L = {L:.1f} g", color=line_color, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold")

    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(200, L * 1.08, BIOMASS.max() * 1.12))
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(0, 201, 20))
    ax.set_xlabel("Days after transplanting (DAT)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_ylabel("Average biomass (g)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_title(f"{model_name}: interactive parameter fit", fontsize=15, fontweight="bold", pad=14)
    legend = ax.legend(loc="upper left", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#1F2937" if theme == "Dark" else "#FFFFFF")
    legend.get_frame().set_edgecolor("#475569" if theme == "Dark" else "#C8D3CB")
    for legend_text in legend.get_texts():
        legend_text.set_color(text)

    return fig, predicted_observed

with st.sidebar:
    st.header("Model controls")
    st.caption("Adjust the values to manually fit the curve to the observed data.")
    selected_model = st.selectbox("Growth model", ["Logistic Model", "Gompertz Model"])
    st.divider()
    L = st.slider("Carrying capacity, L (g)", 100.0, 200.0, 155.0, 0.5)
    k = st.slider("Growth coefficient, k", 0.05, 0.50, 0.20, 0.01)
    t0 = st.slider("Inflection point, t₀ (day)", 5.0, 25.0, 15.0, 0.5)
    st.divider()
    theme = st.radio("Chart appearance", ["Light", "Dark"], horizontal=True)
    st.caption("Tip: a high R² and low RMSE indicate a closer fit.")

st.title("🌱 Lettuce Growth Modeling Dashboard")
st.write("Use the sidebar sliders to explore how growth-model parameters change the predicted biomass curve.")

figure, predictions = make_figure(selected_model, L, k, t0, theme)
r_squared = r2_score(BIOMASS, predictions)
rmse = float(np.sqrt(mean_squared_error(BIOMASS, predictions)))

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Selected model", selected_model.replace(" Model", ""))
metric_b.metric("R²", f"{r_squared:.4f}")
metric_c.metric("RMSE", f"{rmse:.2f} g")
metric_d.metric("Predicted biomass on day 30", f"{predictions[-1]:.1f} g")

st.pyplot(figure, use_container_width=True)
plt.close(figure)

with st.expander("Current parameter values and model equations"):
    left, right = st.columns(2)
    with left:
        st.markdown(f"""
        **Current settings**

        - Carrying capacity, **L** = {L:.1f} g
        - Growth coefficient, **k** = {k:.2f}
        - Inflection point, **t₀** = {t0:.1f} days
        """)
    with right:
        st.latex(r"y = \frac{L}{1 + e^{-k(t-t_0)}}")
        st.caption("Logistic model")
        st.latex(r"y = L\,e^{-e^{-k(t-t_0)}}")
        st.caption("Gompertz model")

st.caption("Observed values: lettuce biomass measured daily over a 30-day timeline.")
