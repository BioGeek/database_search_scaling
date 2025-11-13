import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""# A Comparative Framework for Database Search""")
    return


@app.cell
def _(mo):
    mo.md("""### Interactive Scaling Analysis""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    Use the sliders and input boxes below to see how changes in hardware 
        performance and search parameters affect the estimated runtimes for 
        different proteomics search strategies. Figures 3, 4, and 5 will update
        automatically.
    """
    )
    return


@app.cell
def _(mo):
    # For CPU-based methods, we model a multi-core server processor with vectorised
    # scoring kernels sustaining ∼ 2 × 10^6 spectrum–peptide comparisons per second.
    cpu_slider = mo.ui.slider(
        start=1e5, stop=1e8, step=1e5, value=2e6, label="CPU spectrum–peptide comparisons per second/s", full_width=True, show_value=True
    )
    # GPU acceleration is modeled on a datacenter-class device (e.g., NVIDIA A100) sustaining
    #  ∼ 2 × 10^7 dot-product evaluations per second. 
    gpu_slider = mo.ui.slider(
        start=1e6, stop=1e9, step=1e6, value=2e7, label="GPU dot-product evaluations per second/s", full_width=True, show_value=True
    )
    # cross-encoder inference is assumed at ∼ 10^4 spectrum–peptide pairs per second
    cross_encoder_slider = mo.ui.slider(
        start=1e3, stop=1e5, step=100, value=1e4, label="Cross-encoder inferences/s", full_width=True, show_value=True
    )
    # encoder–decoder de novo decoding achieves ∼ 2×10^3 spectra per second
    de_novo_slider = mo.ui.slider(
        start=1e2, stop=1e4, step=100, value=2e3, label="De novo decodes/s", full_width=True, show_value=True
    )
    # ANN retrieval is modeled as logarithmic in P , with ∼ 0.2 ms per query and an
    # additional cost for re-ranking
    ann_slider = mo.ui.slider(
        start=0.01, stop=5.0, step=0.01, value=0.2, label="ANN retrieval (ms/query)", full_width=True, show_value=True
    )
    rho_slider = mo.ui.slider(
        start=1e-6, stop=1e-3, step=1e-6, value=2e-5, label="ρ (precursor tolerance fraction)", full_width=True, show_value=True
    )
    k_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=50, label="K (candidates to rescore)", full_width=True, show_value=True
    )
    # S: number of spectra
    s_slider = mo.ui.slider(
        start=1e3, stop=1e8, step=1e4, value=1e6, label="S (number of spectra)", full_width=True, show_value=True
    )
    # P: database size
    p_slider = mo.ui.slider(
        start=1e5, stop=1e10, step=1e6, value=4e8, label="P (database size)", full_width=True, show_value=True
    )
    return (
        ann_slider,
        cpu_slider,
        cross_encoder_slider,
        de_novo_slider,
        gpu_slider,
        k_slider,
        p_slider,
        rho_slider,
        s_slider,
    )


@app.cell
def _(
    ann_slider,
    cpu_slider,
    cross_encoder_slider,
    de_novo_slider,
    gpu_slider,
    k_slider,
    mo,
    p_slider,
    rho_slider,
    s_slider,
):
    # Arrange them into a layout with range labels under each slider
    ui_layout = mo.vstack([
                mo.md("**Performance Parameters**"),
                cpu_slider,
                mo.md("_Range: 100,000 to 100 million_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                gpu_slider,
                mo.md("_Range: 1 million to 1 billion_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                cross_encoder_slider,
                mo.md("_Range: 1,000 to 100,000_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                de_novo_slider,
                mo.md("_Range: 100 to 10,000_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                ann_slider,
                mo.md("_Range: 0.01 to 5.0 ms_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                mo.md("**Search Space Parameters**"),
                s_slider,
                mo.md("_Range: 1,000 to 100 million_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                p_slider,
                mo.md("_Range: 100,000 to 10 billion_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                rho_slider,
                mo.md("_Range: 0.000001 to 0.001_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"}),
                k_slider,
                mo.md("_Range: 10 to 500_").style({"font-size": "0.85em", "color": "#666", "margin-top": "-10px"})],
        align='center'
    )

    # Display the final layout
    ui_layout
    return


# @app.cell
# def _(mo):
#     mo.md("""## Figure 2: Asymptotic Scaling""")
#     return


# @app.cell
# def _(np, plt):
#     def plot_figure2():
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 
#         # --- Figure 2A: Scaling vs Database Size P ---
#         S_fixed = 1e6
#         P_range = np.logspace(2, 10, 200)
# 
#         # Simplified Big-O for illustration
#         classical_p = P_range
#         cross_encoder_p = P_range * 100 # Larger constant factor
#         bi_encoder_p = np.log(P_range)
#         fragment_indexed_p = np.full_like(P_range, 10) # Constant
#         denovo_p = np.full_like(P_range, 30) # Constant, higher than fragment
# 
#         ax1.loglog(P_range, classical_p, label="Classical CPU ~O(Sρ*P)")
#         ax1.loglog(P_range, cross_encoder_p, label="Cross-encoder ~ O(Sρ*P)")
#         ax1.loglog(P_range, bi_encoder_p, label="Bi-encoder+ANN ~O(S(log P+K))")
#         ax1.loglog(P_range, fragment_indexed_p, label="Fragment-indexed ~ O(S(L+M))")
#         ax1.loglog(P_range, denovo_p, label="De novo→CE ~O(S(C_dn+K))")
# 
#         ax1.set_title("Big-O scaling vs database size")
#         ax1.set_xlabel("Database size P")
#         ax1.set_ylabel("Relative runtime (arb. units)")
#         ax1.legend()
#         ax1.grid(True, which="major", ls="--", alpha=0.5)
# 
#         # --- Figure 2B: Scaling vs Spectra S ---
#         S_range = np.logspace(3, 8, 100)
# 
#         # All are O(S), so they are parallel. Differences are constant factors.
#         ax2.loglog(S_range, S_range, label="Classical CPU~O(S)")
#         ax2.loglog(S_range, S_range * 10, label="Cross-encoder ~ O(S)")
#         ax2.loglog(S_range, S_range * 0.1, label="Bi-encoder+ANN ~ O(S)")
#         ax2.loglog(S_range, S_range * 0.05, label="Fragment-indexed ~ O(S)")
#         ax2.loglog(S_range, S_range * 0.5, label="De novo→CE ~O(S)")
# 
#         ax2.set_title("Big-O scaling vs spectra")
#         ax2.set_xlabel("Spectra S")
#         ax2.set_ylabel("Relative runtime (arb. units)")
#         ax2.legend()
#         ax2.grid(True, which="major", ls="--", alpha=0.5)
# 
#         fig.tight_layout()
#         return fig
# 
#     plot_figure2()
#     return


@app.cell
def _(mo):
    mo.md("""## Figure 3: Memory and Rescoring Overhead""")
    return


@app.cell
def _(np):
    # --- Data Generation Functions ---

    def calculate_runtimes(S, P, K, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho):
        """Calculates runtimes using parameters from the interactive UI elements."""
        S = np.asarray(S) # The number of experimental tandem mass spectra
        P = np.asarray(P) # The number of peptides in the search database
        K = np.asarray(K) # The number of top candidates to rescore

        ann_perf_s = ann_perf_ms / 1000.0

        # Create a helper array of ones with the correct broadcasted shape.
        # Adding the arrays forces numpy to determine the final shape.
        ones = np.ones_like(S + P + K)

        # rho is the proportion of peptides falling within the precursor tolerance window
        time_classical_cpu = (S * rho * P) / cpu_perf
        time_classical_gpu = (S * rho * P) / gpu_perf
        time_cross_encoder = (S * rho * P) / cross_enc_perf

        # Multiply by `ones` to ensure correct output shape.
        C_frag_index_cpu = (rho * 1e5) / cpu_perf
        time_fragment_indexed_cpu = S * C_frag_index_cpu * ones

        C_frag_index_gpu = (rho * 1e5) / gpu_perf
        time_fragment_indexed_gpu = S * C_frag_index_gpu * ones

        # S is already an array, so subsequent operations will broadcast correctly.
        time_denovo_decode = S / de_novo_perf
        time_denovo_rescore = (S * K) / cross_enc_perf
        time_denovo_ce = time_denovo_decode + time_denovo_rescore * ones

        time_ann_retrieval = S * (np.log(P) * ann_perf_s)
        time_bi_encoder_rescore = (S * K) / cross_enc_perf
        time_bi_encoder_ann = time_ann_retrieval + time_bi_encoder_rescore

        time_classical_rescoring = (S * K) / cross_enc_perf
        time_classical_ce = time_classical_cpu + time_classical_rescoring

        return {
            "Classical (CPU)": time_classical_cpu,
            "Classical (GPU)": time_classical_gpu,
            "Pure cross-encoder": time_cross_encoder,
            "Fragment-indexed (CPU)": time_fragment_indexed_cpu,
            "Fragment-indexed (GPU)": time_fragment_indexed_gpu,
            "De novo+CE/ED": time_denovo_ce,
            "Bi-encoder+ANN": time_bi_encoder_ann,
            "Classical+CE": time_classical_ce,
        }

    def calculate_memory(P):
        """Calculates memory requirements."""
        P = np.asarray(P)
        bytes_to_gib = 1 / (1024**3)
        mem_sequence = P * 25 * bytes_to_gib
        mem_ann = P * 400 * bytes_to_gib
        mem_fragment = P * 1200 * bytes_to_gib
        return {
            "Sequence store": mem_sequence, "ANN embedding index": mem_ann, "Fragment-ion index": mem_fragment
        }
    return calculate_memory, calculate_runtimes


@app.cell
def _(
    ann_slider,
    calculate_memory,
    calculate_runtimes,
    cpu_slider,
    cross_encoder_slider,
    de_novo_slider,
    k_slider,
    np,
    p_slider,
    plt,
    rho_slider,
    s_slider,
):
    def plot_figure3(s_value, p_value, k_value, cpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot A: Runtime vs K
        S_fixed, P_fixed = s_value, p_value
        K_range = np.arange(5,401,5)
        runtimes_k = calculate_runtimes(S_fixed, P_fixed, K_range, cpu_perf, 1, cross_enc_perf, de_novo_perf, ann_perf_ms, rho)

        ax1.plot(K_range, runtimes_k["Classical+CE"], label="Classical+CE")
        ax1.plot(K_range, runtimes_k["Bi-encoder+ANN"], label="Bi-encoder+ANN")
        ax1.plot(K_range, runtimes_k["De novo+CE/ED"], label="De novo→CE")

        ax1.set_title("Runtime vs K")
        ax1.set_xlabel("K candidates per spectrum")
        ax1.set_ylabel(f"Wall time (s) for S={S_fixed:.0e}, P={P_fixed:.0e}")
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # Subplot B: RAM vs Database Size
        P_range_mem = np.logspace(6, 10, 100)
        memory_data = calculate_memory(P_range_mem)

        ax2.loglog(P_range_mem, memory_data["Sequence store"], label="Sequence store")
        ax2.loglog(P_range_mem, memory_data["ANN embedding index"], label="ANN embedding index")
        ax2.loglog(P_range_mem, memory_data["Fragment-ion index"], label="Fragment-ion index")

        ax2.set_title("RAM vs database size")
        ax2.set_xlabel("Database size P")
        ax2.set_ylabel("Memory (GiB)")
        ax2.legend()
        ax2.grid(True, which="major", ls="--", alpha=0.5)

        fig.tight_layout()
        return fig

    plot_figure3(s_slider.value, p_slider.value, k_slider.value, cpu_slider.value, cross_encoder_slider.value, de_novo_slider.value, ann_slider.value, rho_slider.value)
    return


@app.cell
def _(mo):
    mo.md("""## Figure 4: Estimated Wall Times and Practical Limits""")
    return


@app.cell
def _(
    ann_slider,
    calculate_runtimes,
    cpu_slider,
    cross_encoder_slider,
    de_novo_slider,
    gpu_slider,
    k_slider,
    np,
    p_slider,
    plt,
    rho_slider,
    s_slider,
):
    def plot_figure4(s_value, p_value, k_value, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot A: Wall-time vs Database size P
        S_fixed, P_range = s_value, np.logspace(5, 10, 100)
        runtimes_p = calculate_runtimes(S_fixed, P_range, k_value, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho)
        for label, data in runtimes_p.items():
            ax1.loglog(P_range, data, label=label)

        ax1.set_title(f"A. Wall-time vs database size at fixed S = {S_fixed:.0e}")
        ax1.set_xlabel("Database size P")
        ax1.set_ylabel(f"Wall time (s) @ S={S_fixed:.0e}")
        ax1.set_ylim(1e1, 1e12)
        ax1.legend(fontsize='small')
        ax1.grid(True, which="major", ls="--", alpha=0.5)

        # Subplot B: Wall-time vs Spectra S
        P_fixed, S_range = p_value, np.logspace(5, 7, 100)
        runtimes_s = calculate_runtimes(S_range, P_fixed, k_value, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho)
        for label, data in runtimes_s.items():
            ax2.loglog(S_range, data, label=label)

        ax2.set_title(f"B. Wall-time vs spectra at fixed P = {P_fixed:.0e}")
        ax2.set_xlabel("Spectra S")
        ax2.set_ylabel(f"Wall time (s) @ P={P_fixed:.0e}")
        ax2.set_ylim(1e1, 1e12)
        ax2.legend(fontsize='small')
        ax2.grid(True, which="major", ls="--", alpha=0.5)

        fig.tight_layout()
        return fig

    plot_figure4(s_slider.value, p_slider.value, k_slider.value, cpu_slider.value, gpu_slider.value, cross_encoder_slider.value, de_novo_slider.value, ann_slider.value, rho_slider.value)
    return


@app.cell
def _(mo):
    mo.md("""## Figure 5: Runtime Scaling Heatmaps""")
    return


@app.cell
def _(
    ann_slider,
    calculate_runtimes,
    cpu_slider,
    cross_encoder_slider,
    de_novo_slider,
    gpu_slider,
    k_slider,
    np,
    p_slider,
    plt,
    rho_slider,
    s_slider,
):
    from matplotlib.colors import LogNorm

    def plot_figure5(s_value, p_value, k_value, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho):
        # --- Grid --
        S_range = np.logspace(5, 8, 70)
        P_range = np.logspace(7, 10, 70)
        S_grid, P_grid = np.meshgrid(S_range, P_range, indexing='ij')

        # --- Runtimes Dictionary ---
        heatmaps = calculate_runtimes(S_grid, P_grid, k_value, cpu_perf, gpu_perf, cross_enc_perf, de_novo_perf, ann_perf_ms, rho)
    
        # --- Methods to Plot ---
        methods_to_plot = {
            "Classical (CPU)": heatmaps["Classical (CPU)"],
            "Fragment-indexed (CPU)": heatmaps["Fragment-indexed (CPU)"],
            "Classical+CE": heatmaps["Classical+CE"],
            "De novo+CE": heatmaps["De novo+CE/ED"],
            "Bi-encoder+ANN": heatmaps["Bi-encoder+ANN"],
            "Cross-encoder": heatmaps["Pure cross-encoder"]
        }
    
        # --- Plot Setup ---
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    
        # --- Color Normalization ---
        # To ensure a consistent color scale, find global min/max percentiles
        all_data_flat = np.concatenate([Z.ravel() for Z in methods_to_plot.values()])
        vmin = np.percentile(all_data_flat, 20)
        vmax = np.percentile(all_data_flat, 80)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    
        # --- Plotting Loop ---
        im = None  # Initialize im to be accessible for the colorbar
        for ax, (name, Z) in zip(axes.ravel(), methods_to_plot.items()):
            im = ax.imshow(Z, origin='lower', aspect='auto',
                           extent=[P_range.min(), P_range.max(), S_range.min(), S_range.max()],
                           norm=norm,
                           cmap='viridis')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(name)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Database size P")
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Spectra S")

        # --- Colorbar ---
        if im:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right', shrink=0.95, pad=0.02)
            cbar.set_label("Runtime (s), log scale")

        fig.suptitle("Heatmaps of runtime scaling as a function of both S and P", fontsize=16)
    
        return fig

    plot_figure5(s_slider.value, p_slider.value, k_slider.value, cpu_slider.value, gpu_slider.value, cross_encoder_slider.value, de_novo_slider.value, ann_slider.value, rho_slider.value)
    return


if __name__ == "__main__":
    app.run()
