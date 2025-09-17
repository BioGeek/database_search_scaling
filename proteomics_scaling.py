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
    mo.md("""# Reproduction of Figures 2, 3, 4, and 5""")
    return


@app.cell
def _():
    # --- Constants and Assumptions from the paper ---
    # These values are based on the "Comparative scaling analysis" section.

    # Performance estimates
    CPU_SPEC_PEPTIDE_PER_S = 2e6  # Spectrum-peptide comparisons per second on CPU
    GPU_DOT_PRODUCT_PER_S = 2e7  # Dot-product evaluations per second on GPU
    CROSS_ENCODER_INFERENCE_PER_S = 1e4  # Spectrum-peptide pairs per second for cross-encoder
    ENCODER_DECODER_DECODING_PER_S = 2e3  # Spectra per second for de novo decoding
    ANN_RETRIEVAL_MS = 0.2  # milliseconds per query
    ANN_RETRIEVAL_S = ANN_RETRIEVAL_MS / 1000.0  # seconds per query

    # Parameters for scaling laws
    rho = 1e-4  # Proportion of peptides in precursor mass window (assumed, typical value)
    return (
        ANN_RETRIEVAL_S,
        CPU_SPEC_PEPTIDE_PER_S,
        CROSS_ENCODER_INFERENCE_PER_S,
        ENCODER_DECODER_DECODING_PER_S,
        GPU_DOT_PRODUCT_PER_S,
        rho,
    )


@app.cell
def _(
    ANN_RETRIEVAL_S,
    CPU_SPEC_PEPTIDE_PER_S,
    CROSS_ENCODER_INFERENCE_PER_S,
    ENCODER_DECODER_DECODING_PER_S,
    GPU_DOT_PRODUCT_PER_S,
    np,
    rho,
):
    # --- Data Generation Functions ---

    def calculate_runtimes(S, P, K):
        """
        Calculates estimated wall-clock runtimes for all methods.
        This version is corrected to handle NumPy broadcasting properly.
        """
        S = np.asarray(S)
        P = np.asarray(P)

        # This creates a zero-array with the correct broadcast shape from S and P.
        # It ensures that calculations that don't depend on one of the variables
        # still produce an array of the correct dimension.
        broadcast_zeros = S * 0 + P * 0

        # Calculations that depend on both S and P are naturally broadcast correctly.
        time_classical_cpu = (S * rho * P) / CPU_SPEC_PEPTIDE_PER_S
        time_classical_gpu = (S * rho * P) / GPU_DOT_PRODUCT_PER_S
        time_cross_encoder = (S * rho * P) / CROSS_ENCODER_INFERENCE_PER_S

        # For calculations that are independent of P, we add broadcast_zeros
        # to enforce the correct array shape when P is varied.
        C_frag_index_cpu = (rho * 1e5) / CPU_SPEC_PEPTIDE_PER_S
        time_fragment_indexed_cpu = S * C_frag_index_cpu + broadcast_zeros

        C_frag_index_gpu = (rho * 1e5) / GPU_DOT_PRODUCT_PER_S
        time_fragment_indexed_gpu = S * C_frag_index_gpu + broadcast_zeros

        time_denovo_decode = S / ENCODER_DECODER_DECODING_PER_S + broadcast_zeros
        time_denovo_rescore = (S * K) / CROSS_ENCODER_INFERENCE_PER_S + broadcast_zeros
        time_denovo_ce = time_denovo_decode + time_denovo_rescore

        # Hybrid calculations
        time_ann_retrieval = S * (np.log(P) * ANN_RETRIEVAL_S)
        time_bi_encoder_rescore = (S * K) / CROSS_ENCODER_INFERENCE_PER_S + broadcast_zeros
        time_bi_encoder_ann = time_ann_retrieval + time_bi_encoder_rescore

        time_classical_rescoring = (S * K) / CROSS_ENCODER_INFERENCE_PER_S + broadcast_zeros
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
        # Assumptions for memory calculation
        # Sequence store: ~25 bytes per peptide on average (e.g., for 10^8 peptides -> 2.5 GiB)
        bytes_per_peptide = 25

        # ANN embedding index: ~30-50 GiB at 10^8 peptides
        # This implies roughly 400 bytes per peptide
        bytes_per_embedding = 400

        # Fragment-ion index: >100 GiB at 10^8 peptides (with decoys)
        # This implies roughly 1200 bytes per peptide (including decoys)
        bytes_per_fragment_entry = 1200

        # Convert bytes to GiB
        bytes_to_gib = 1 / (1024**3)

        mem_sequence = P * bytes_per_peptide * bytes_to_gib
        mem_ann = P * bytes_per_embedding * bytes_to_gib
        mem_fragment = P * bytes_per_fragment_entry * bytes_to_gib

        return {
            "Sequence store": mem_sequence,
            "ANN embedding index": mem_ann,
            "Fragment-ion index": mem_fragment,
        }
    return calculate_memory, calculate_runtimes


@app.cell
def _(mo):
    mo.md("## Figure 2: Asymptotic Scaling")
    return


@app.cell
def _(np, plt):
    def plot_figure2():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Figure 2A: Scaling vs Database Size P ---
        S_fixed = 1e6
        P_range = np.logspace(2, 10, 100)

        # Simplified Big-O for illustration
        classical_p = P_range
        cross_encoder_p = P_range * 100 # Larger constant factor
        bi_encoder_p = np.log(P_range) * 1e6 # Scaled to be visible
        fragment_indexed_p = np.full_like(P_range, 1e5) # Constant
        denovo_p = np.full_like(P_range, 2e5) # Constant, higher than fragment

        ax1.loglog(P_range, classical_p, label="Classical CPU ~O(SρP)")
        ax1.loglog(P_range, cross_encoder_p, label="Cross-encoder ~ O(SρP)")
        ax1.loglog(P_range, bi_encoder_p, label="Bi-encoder+ANN ~O(S(log P+K))")
        ax1.loglog(P_range, fragment_indexed_p, label="Fragment-indexed ~ O(S(L+M))")
        ax1.loglog(P_range, denovo_p, label="De novo→CE ~O(S(C_dn+K))")

        ax1.set_title("A. Big-O scaling vs database size")
        ax1.set_xlabel("Database size P")
        ax1.set_ylabel("Relative runtime (arb. units)")
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # --- Figure 2B: Scaling vs Spectra S ---
        S_range = np.logspace(3, 8, 100)

        # All are O(S), so they are parallel. Differences are constant factors.
        ax2.loglog(S_range, S_range * 0.1, label="Classical CPU~O(S)")
        ax2.loglog(S_range, S_range * 10, label="Cross-encoder ~ O(S)")
        ax2.loglog(S_range, S_range * 0.05, label="Bi-encoder+ANN ~ O(S)")
        ax2.loglog(S_range, S_range * 0.02, label="Fragment-indexed ~ O(S)")
        ax2.loglog(S_range, S_range * 0.5, label="De novo→CE ~O(S)")

        ax2.set_title("B. Big-O scaling vs spectra")
        ax2.set_xlabel("Spectra S")
        ax2.set_ylabel("Relative runtime (arb. units)")
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        fig.tight_layout()
        plt.show()

    plot_figure2()
    return


@app.cell
def _(mo):
    mo.md("## Figure 3: Memory and Rescoring Overhead")
    return


@app.cell
def _(
    CROSS_ENCODER_INFERENCE_PER_S,
    calculate_memory,
    calculate_runtimes,
    np,
    plt,
):
    def plot_figure3():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Figure 3A: Runtime vs K ---
        S_fixed = 1e6
        P_fixed = 4e8
        K_range = np.linspace(0, 400, 100)

        # Calculate baseline costs (at K=0)
        runtimes_at_K0 = calculate_runtimes(S_fixed, P_fixed, K=0)

        # Calculate full runtimes
        classical_ce_k = runtimes_at_K0["Classical+CE"] + (S_fixed * K_range) / CROSS_ENCODER_INFERENCE_PER_S
        bi_encoder_k = runtimes_at_K0["Bi-encoder+ANN"] + (S_fixed * K_range) / CROSS_ENCODER_INFERENCE_PER_S
        denovo_ce_k = runtimes_at_K0["De novo+CE/ED"] + (S_fixed * K_range) / CROSS_ENCODER_INFERENCE_PER_S

        ax1.plot(K_range, classical_ce_k, label="Classical+CE")
        ax1.plot(K_range, bi_encoder_k, label="Bi-encoder+ANN")
        ax1.plot(K_range, denovo_ce_k, label="De novo→CE")

        ax1.set_title("A. Runtime vs K")
        ax1.set_xlabel("K candidates per spectrum")
        ax1.set_ylabel(f"Wall time (s) for S={S_fixed:.0e}, P={P_fixed:.0e}")
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # --- Figure 3B: RAM vs Database Size ---
        P_range = np.logspace(6, 10, 100)
        memory_data = calculate_memory(P_range)

        ax2.loglog(P_range, memory_data["Sequence store"], label="Sequence store")
        ax2.loglog(P_range, memory_data["ANN embedding index"], label="ANN embedding index")
        ax2.loglog(P_range, memory_data["Fragment-ion index"], label="Fragment-ion index")

        ax2.set_title("B. RAM vs database size")
        ax2.set_xlabel("Database size P")
        ax2.set_ylabel("Memory (GiB)")
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        fig.tight_layout()
        return fig

    plot_figure3()
    return


@app.cell
def _(mo):
    mo.md("## Figure 4: Estimated Wall Times and Practical Limits")
    return


@app.cell
def _(calculate_runtimes, np, plt):
    def plot_figure4():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        K_fixed = 100

        # --- Figure 4A: Wall-time vs Database size P ---
        S_fixed = 1e6
        P_range = np.logspace(5, 10, 100)
        runtimes_p = calculate_runtimes(S_fixed, P_range, K_fixed)

        for label, data in runtimes_p.items():
            ax1.loglog(P_range, data, label=label)

        ax1.set_title(f"A. Wall-time vs database size at fixed S = {S_fixed:.0e}")
        ax1.set_xlabel("Database size P")
        ax1.set_ylabel(f"Wall time (s) @ S={S_fixed:.0e}")
        ax1.set_ylim(1e1, 1e12)
        ax1.legend(fontsize='small')
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # --- Figure 4B: Wall-time vs Spectra S ---
        P_fixed = 4e8
        S_range = np.logspace(5, 7, 100)
        runtimes_s = calculate_runtimes(S_range, P_fixed, K_fixed)

        for label, data in runtimes_s.items():
            ax2.loglog(S_range, data, label=label)

        ax2.set_title(f"B. Wall-time vs spectra at fixed P = {P_fixed:.0e}")
        ax2.set_xlabel("Spectra S")
        ax2.set_ylabel(f"Wall time (s) @ P={P_fixed:.0e}")
        ax2.set_ylim(1e1, 1e12)
        ax2.legend(fontsize='small')
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        fig.tight_layout()
        return fig

    plot_figure4()
    return


@app.cell
def _(mo):
    mo.md("## Figure 5: Runtime Scaling Heatmaps")
    return


@app.cell
def _(calculate_runtimes, np, plt):


    def plot_figure5():
        S_range = np.logspace(5, 8, 50)
        P_range = np.logspace(7, 10, 50)
        S_grid, P_grid = np.meshgrid(S_range, P_range, indexing='ij')
        K_fixed = 100

        all_runtimes = calculate_runtimes(S_grid, P_grid, K_fixed)

        methods_to_plot = [
            "Classical (CPU)", "Fragment-indexed (CPU)", "Classical+CE",
            "De novo+CE/ED", "Bi-encoder+ANN", "Pure cross-encoder"
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
        axes = axes.ravel()

        with np.errstate(divide='ignore'):
            log_runtimes = {label: np.log10(all_runtimes[label]) for label in methods_to_plot}

        valid_runtimes = [rt[np.isfinite(rt)] for rt in log_runtimes.values() if np.any(np.isfinite(rt))]
        vmin = np.min([np.min(rt) for rt in valid_runtimes]) if valid_runtimes else 0
        vmax = np.max([np.max(rt) for rt in valid_runtimes]) if valid_runtimes else 10

        for i, method in enumerate(methods_to_plot):
            ax = axes[i]
            pcm = ax.pcolormesh(P_range, S_range, log_runtimes[method], 
                                vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(method)
            if i >= 3:
                ax.set_xlabel("Database size P")
            if i % 3 == 0:
                ax.set_ylabel("Spectra S")

        fig.subplots_adjust(right=0.85, top=0.92)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label('Runtime (s) log scale')

        fig.suptitle("Heatmaps of runtime scaling as a function of both S and P", fontsize=16)
        return fig

    plot_figure5()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
