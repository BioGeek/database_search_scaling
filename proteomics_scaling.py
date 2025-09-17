import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

mo.md("# Reproduction of Figures 2, 3, 4, and 5")

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
rho = 1e-5  # Proportion of peptides in precursor mass window (adjusted to match graph scales)

# --- Data Generation Functions ---

def calculate_runtimes(S, P, K):
    """Calculates estimated wall-clock runtimes for all methods."""
    S = np.asarray(S)
    P = np.asarray(P)
    
    # Classical Search (CPU)
    time_classical_cpu = (S * rho * P) / CPU_SPEC_PEPTIDE_PER_S
    
    # Classical Search (GPU) - Assumed to be similar scaling but faster constant
    time_classical_gpu = (S * rho * P) / GPU_DOT_PRODUCT_PER_S

    # Pure Cross-encoder
    time_cross_encoder = (S * rho * P) / CROSS_ENCODER_INFERENCE_PER_S
    
    # Fragment-ion indexing (CPU) - Query time is independent of P.
    # We'll estimate its constant per-spectrum cost to be equivalent to a
    # classical search against a small, fixed-size database (e.g., 1e5 peptides).
    C_frag_index_cpu = (rho * 1e5) / CPU_SPEC_PEPTIDE_PER_S
    time_fragment_indexed_cpu = S * C_frag_index_cpu
    
    # Fragment-ion indexing (GPU)
    C_frag_index_gpu = (rho * 1e5) / GPU_DOT_PRODUCT_PER_S
    time_fragment_indexed_gpu = S * C_frag_index_gpu

    # De novo sequencing with rescoring
    time_denovo_decode = S / ENCODER_DECODER_DECODING_PER_S
    time_denovo_rescore = (S * K) / CROSS_ENCODER_INFERENCE_PER_S
    time_denovo_ce = time_denovo_decode + time_denovo_rescore

    # Bi-encoder + ANN retrieval
    time_ann_retrieval = S * (np.log(P) * ANN_RETRIEVAL_S) # A slightly more realistic log scaling for retrieval
    time_bi_encoder_rescore = (S * K) / CROSS_ENCODER_INFERENCE_PER_S # Rescoring top K
    time_bi_encoder_ann = time_ann_retrieval + time_bi_encoder_rescore
    
    # Classical search with neural rescoring
    time_classical_ce = time_classical_cpu + (S * K) / CROSS_ENCODER_INFERENCE_PER_S

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
    """Calculates estimated memory requirements in GiB."""
    P = np.asarray(P)
    bytes_to_gib = 1 / (1024**3)
    
    # Sequence store: ~2.5 GiB for 10^8 peptides -> 25 bytes/peptide
    bytes_per_peptide_seq = 25
    
    # ANN index: ~30-50 GiB for 10^8 peptides -> ~400 bytes/peptide
    bytes_per_embedding = 400

    # Fragment-ion index: >100 GiB for 10^8 peptides -> ~1200 bytes/peptide
    bytes_per_fragment_entry = 1200
    
    mem_sequence = P * bytes_per_peptide_seq * bytes_to_gib
    mem_ann = P * bytes_per_embedding * bytes_to_gib
    mem_fragment = P * bytes_per_fragment_entry * bytes_to_gib
    
    return {
        "Sequence store": mem_sequence,
        "ANN embedding index": mem_ann,
        "Fragment-ion index": mem_fragment,
    }

# --- Figure 2 ---
mo.md("## Figure 2: Asymptotic Scaling")

@mo.capture
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

# --- Figure 3 ---
mo.md("## Figure 3: Memory and Rescoring Overhead")

@mo.capture
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
    plt.show()

plot_figure3()


# --- Figure 4 ---
mo.md("## Figure 4: Estimated Wall Times and Practical Limits")

@mo.capture
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
    ax1.legend()
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
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    plt.show()

plot_figure4()

# --- Figure 5 ---
mo.md("## Figure 5: Runtime Scaling Heatmaps")

@mo.capture
def plot_figure5():
    S_range = np.logspace(5, 8, 50)
    P_range = np.logspace(7, 10, 50)
    S_grid, P_grid = np.meshgrid(S_range, P_range)
    K_fixed = 100
    
    # Calculate runtimes for the grid
    all_runtimes = calculate_runtimes(S_grid, P_grid, K_fixed)
    
    methods_to_plot = [
        "Classical (CPU)", "Fragment-indexed (CPU)", "Classical+CE",
        "De novo+CE/ED", "Bi-encoder+ANN", "Pure cross-encoder"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    
    # Use a common color scale
    log_runtimes = {label: np.log10(all_runtimes[label]) for label in methods_to_plot}
    vmin = np.min([np.min(rt) for rt in log_runtimes.values()])
    vmax = np.max([np.max(rt) for rt in log_runtimes.values() if not np.isinf(np.max(rt))]) # Avoid inf from cross-encoder

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
            
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label('Runtime (s) log scale')
    
    plt.suptitle("Heatmaps of runtime scaling as a function of both S and P", fontsize=16)
    plt.show()

plot_figure5()