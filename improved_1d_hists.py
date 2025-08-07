#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ─── Official binning from Vlada's specifications ─────────────────────────────
COSZ_BINS = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# e-like events (*_nueselec.root) - 16 bins with overflow at 70 GeV
NUE_ENERGY_BINS = np.array([0.0, 0.07, 0.1, 0.15, 0.22, 0.33, 0.47, 0.7, 
                           1.1, 1.7, 2.9, 5, 9, 16, 32, 48, 70, 10000.0])

# non e-like events (*_numuselec.root) - 10 bins with overflow at 70 GeV
NUMU_ENERGY_BINS = np.array([0.0, 0.12, 0.22, 0.37, 0.63, 1.5, 4, 11, 19, 32, 70, 10000.0])

def find_selection_trees(rootfile):
    """Find all selection TTree paths in the ROOT file."""
    with uproot.open(rootfile) as f:
        keys = list(f.keys(recursive=True))
        
    # Look for events_TTree under Selec directories - handle spacing variations
    trees = []
    for key in keys:
        if (key.startswith("FitterEngine/preFit/model/") and 
            "Selec" in key and
            "events_TTree;1" in key):
            trees.append(key)
    
    print(f"Found {len(trees)} selection trees:")
    for tree in sorted(trees):
        print(f"  {tree}")
    
    return sorted(trees)

def load_event_data(rootfile, tree_path, selection_type, use_weights=True):
    """Load energy and weight data from a TTree."""
    with uproot.open(rootfile) as f:
        tree = f[tree_path]
        
        # Check if tree has any entries
        num_entries = tree.num_entries
        print(f"  Tree has {num_entries} entries")
        
        if num_entries == 0:
            print(f"  Warning: Tree is empty, skipping...")
            return np.array([]), np.array([]), np.array([])
        
        # Load a small sample to see the data structure
        try:
            sample_data = tree.arrays(library="np", entry_stop=1)
            print(f"  Tree structure: {list(sample_data.keys())}")
            
            # Check the Leaves structure
            if 'Leaves' in sample_data:
                leaves_sample = sample_data['Leaves']
                if hasattr(leaves_sample, 'dtype') and leaves_sample.dtype.names:
                    available_branches = list(leaves_sample.dtype.names)
                    print(f"  Available branches in Leaves: {available_branches[:10]}...")
                else:
                    print(f"  Leaves structure: {type(leaves_sample)}")
                    return np.array([]), np.array([]), np.array([])
            else:
                print(f"  No Leaves found in tree")
                return np.array([]), np.array([]), np.array([])
                
        except Exception as e:
            print(f"  Error reading tree structure: {e}")
            return np.array([]), np.array([]), np.array([])
        
        # Check for required branches
        if "mcEnu" not in available_branches:
            print(f"  mcEnu not found in branches: {available_branches}")
            return np.array([]), np.array([]), np.array([])
        
        # Choose reco energy branch based on selection type
        if selection_type == "nue":
            reco_candidates = ["recoEnuEcalo", "recoEnuCalo", "recoEnuLepCalo"]
        else:  # numu
            reco_candidates = ["recoEnuLepCalo", "recoEnuCalo", "recoEnuEcalo"]
        
        # Find which reco energy branch exists
        reco_branch = None
        for candidate in reco_candidates:
            if candidate in available_branches:
                reco_branch = candidate
                break
        
        if reco_branch is None:
            reco_available = [b for b in available_branches if 'reco' in b.lower() and 'enu' in b.lower()]
            print(f"  No preferred reco branch found. Available reco branches: {reco_available}")
            # Take the first available reco energy branch
            if reco_available:
                reco_branch = reco_available[0]
                print(f"  Using {reco_branch}")
            else:
                print(f"  No reco energy branches found!")
                return np.array([]), np.array([]), np.array([])
        
        # Find weight branch
        weight_branch = None
        if use_weights:
            weight_candidates = ["mcOscW", "mcGenWeight"]
            for candidate in weight_candidates:
                if candidate in available_branches:
                    weight_branch = candidate
                    break
        
        print(f"  Loading: mcEnu, {reco_branch}" + (f", {weight_branch}" if weight_branch else ""))
        
        # Load all the data at once
        try:
            all_data = tree.arrays(library="np")
            leaves_data = all_data['Leaves']
            
            # Extract the arrays we need
            E_true = leaves_data['mcEnu']
            E_reco = leaves_data[reco_branch]
            
            if use_weights and weight_branch:
                weights = leaves_data[weight_branch]
                print(f"  Using weights from {weight_branch}")
            else:
                # Try to get weights from Event structure
                if use_weights and 'Event' in all_data:
                    event_data = all_data['Event']
                    if hasattr(event_data, 'dtype') and event_data.dtype.names:
                        if 'eventWeight' in event_data.dtype.names:
                            weights = event_data['eventWeight']
                            print(f"  Using weights from Event.eventWeight")
                        else:
                            weights = np.ones(len(E_true))
                            print(f"  No eventWeight found, using unit weights")
                    else:
                        weights = np.ones(len(E_true))
                        print(f"  No Event structure found, using unit weights")
                else:
                    weights = np.ones(len(E_true))
                    print(f"  Using unit weights")
            
            print(f"  Loaded {len(E_true)} events")
            return E_reco, E_true, weights
            
        except Exception as e:
            print(f"  Error loading data: {e}")
            return np.array([]), np.array([]), np.array([])

def determine_selection_type(tree_path):
    """Determine selection type based on the third neutrino (detection type) in the path."""
    if "Selec" in tree_path:
        channel_part = tree_path.split("Selec")[0].strip()
        parts = [p.strip() for p in channel_part.split(" x ")]
        if len(parts) >= 3:
            selection_part = parts[2].strip()
            if "#nu_{e}" in selection_part:
                return "nue"
            elif "#nu_{#mu}" in selection_part:
                return "numu"
            elif "#nu_{#tau}" in selection_part:
                return "nue"
        if " #nu_{e} " in channel_part or channel_part.endswith("#nu_{e}"):
            return "nue"
        elif " #nu_{#mu} " in channel_part or channel_part.endswith("#nu_{#mu}"):
            return "numu"
    print(f"Warning: Could not determine selection type for {tree_path}, defaulting to numu")
    return "numu"

def get_energy_bins(selection_type):
    """Get appropriate energy bins for selection type."""
    if selection_type == "nue":
        return NUE_ENERGY_BINS
    else:
        return NUMU_ENERGY_BINS

def clean_channel_name(tree_path):
    """Extract and clean channel name from tree path."""
    if "model/" in tree_path and "Selec" in tree_path:
        channel_raw = tree_path.split("model/")[1].split("Selec")[0].strip()
    else:
        channel_raw = tree_path.split("/")[-2]
    clean_name = (channel_raw
                 .replace("#bar{#nu_{e}}", "antinue")
                 .replace("#bar{#nu_{#mu}}", "antinumu") 
                 .replace("#bar{#nu_{#tau}}", "antinutau")
                 .replace("#nu_{e}", "nue")
                 .replace("#nu_{#mu}", "numu")
                 .replace("#nu_{#tau}", "nutau")
                 .replace("{", "").replace("}", "")
                 .replace("#", "")
                 .replace(" x ", "_to_")
                 .replace(" ", "_"))
    if "_nue" in clean_name:
        clean_name = clean_name.replace("_nue", "_eselec")
    elif "_numu" in clean_name:
        clean_name = clean_name.replace("_numu", "_muselec")
    return clean_name

def plot_1d_comparison(hist_osc, hist_unosc, bins, channel_name, selection_type, xlabel, output_path):
    """Create 1D comparison plot with improved styling."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Bin centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Calculate statistics
    total_osc = hist_osc.sum()
    total_unosc = hist_unosc.sum()
    weighted_mean_osc = np.sum(bin_centers * hist_osc) / total_osc if total_osc > 0 else 0
    weighted_mean_unosc = np.sum(bin_centers * hist_unosc) / total_unosc if total_unosc > 0 else 0
    std_osc = np.sqrt(np.sum(hist_osc * (bin_centers - weighted_mean_osc)**2) / total_osc) if total_osc > 0 else 0
    std_unosc = np.sqrt(np.sum(hist_unosc * (bin_centers - weighted_mean_unosc)**2) / total_unosc) if total_unosc > 0 else 0
    
    # Main plot - overlay histograms
    ax.step(bin_centers, hist_unosc, where='mid', label='Unoscillated', linewidth=2, color='blue')
    ax.step(bin_centers, hist_osc, where='mid', label='Oscillated', linewidth=2, color='red')
    
    # Title and labels
    sel_name = "Electron-like" if selection_type == "nue" else "Muon-like"
    ax.set_title(f"{channel_name} ({sel_name} Selection)", fontsize=14, fontweight='bold')
    ax.set_xlabel('log₁₀(E/GeV)', fontsize=12)
    ax.set_ylabel('Event Rate', fontsize=12)
    
    # Set log scale and limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Show full overflow bin up to bins[-1] (e.g. 10000 GeV)
    min_energy = bins[bins > 0].min()
    ax.set_xlim(min_energy, bins[-1])
    
    # Tick marks at every decade out to the overflow edge
    ax.set_xticks([0.1, 1, 10, 100, 1000, bins[-1]])
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    
    # Create statistics box
    legend_text = f"""Unoscillated:
Entries: {total_unosc:.0f}
Mean: {weighted_mean_unosc:.2f} GeV
Std: {std_unosc:.2f} GeV

Oscillated:
Entries: {total_osc:.1f}
Mean: {weighted_mean_osc:.2f} GeV  
Std: {std_osc:.2f} GeV"""
    line_legend = ax.legend(loc='upper right')
    ax.add_artist(line_legend)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create 1D neutrino energy histograms")
    parser.add_argument("--osc", required=True, help="Oscillated ROOT file")
    parser.add_argument("--unosc", required=True, help="Unoscillated ROOT file")
    parser.add_argument("--outdir", default="1d_histograms", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Just list trees, don't plot")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    trees = find_selection_trees(args.osc)
    if not trees:
        print("No selection trees found!")
        return
    if args.dry_run:
        print("Dry run complete - trees listed above")
        return
    
    # Storage for aggregate histograms
    agg_nue_reco_osc = agg_nue_reco_unosc = None
    agg_nue_true_osc = agg_nue_true_unosc = None
    agg_numu_reco_osc = agg_numu_reco_unosc = None 
    agg_numu_true_osc = agg_numu_true_unosc = None
    
    for tree_path in trees:
        sel_type = determine_selection_type(tree_path)
        energy_bins = get_energy_bins(sel_type)
        channel_name = clean_channel_name(tree_path)
        
        E_reco_osc, E_true_osc, weights_osc = load_event_data(
            args.osc, tree_path, sel_type, use_weights=True)
        E_reco_unosc, E_true_unosc, weights_unosc = load_event_data(
            args.unosc, tree_path, sel_type, use_weights=False)
        
        hist_reco_osc, _ = np.histogram(E_reco_osc, bins=energy_bins, weights=weights_osc)
        hist_reco_unosc, _ = np.histogram(E_reco_unosc, bins=energy_bins, weights=weights_unosc)
        hist_true_osc, _ = np.histogram(E_true_osc, bins=energy_bins, weights=weights_osc)
        hist_true_unosc, _ = np.histogram(E_true_unosc, bins=energy_bins, weights=weights_unosc)
        
        plot_1d_comparison(hist_reco_osc, hist_reco_unosc, energy_bins, channel_name, sel_type,
                          "Reconstructed Energy [GeV]",
                          os.path.join(args.outdir, f"{channel_name}_reco.png"))
        plot_1d_comparison(hist_true_osc, hist_true_unosc, energy_bins, channel_name, sel_type,
                          "True Energy [GeV]",
                          os.path.join(args.outdir, f"{channel_name}_true.png"))
        
        if sel_type == "nue":
            if agg_nue_reco_osc is None:
                agg_nue_reco_osc = hist_reco_osc.copy()
                agg_nue_reco_unosc = hist_reco_unosc.copy()
                agg_nue_true_osc = hist_true_osc.copy()
                agg_nue_true_unosc = hist_true_unosc.copy()
            else:
                agg_nue_reco_osc += hist_reco_osc
                agg_nue_reco_unosc += hist_reco_unosc
                agg_nue_true_osc += hist_true_osc
                agg_nue_true_unosc += hist_true_unosc
        else:
            if agg_numu_reco_osc is None:
                agg_numu_reco_osc = hist_reco_osc.copy()
                agg_numu_reco_unosc = hist_reco_unosc.copy()
                agg_numu_true_osc = hist_true_osc.copy()
                agg_numu_true_unosc = hist_true_unosc.copy()
            else:
                agg_numu_reco_osc += hist_reco_osc
                agg_numu_reco_unosc += hist_reco_unosc
                agg_numu_true_osc += hist_true_osc
                agg_numu_true_unosc += hist_true_unosc
    
    # Create aggregate plots
    if agg_nue_reco_osc is not None:
        plot_1d_comparison(agg_nue_reco_osc, agg_nue_reco_unosc, NUE_ENERGY_BINS,
                          "All Channels", "nue", "Reconstructed Energy [GeV]", 
                          os.path.join(args.outdir, "aggregate_nue_reco.png"))
        plot_1d_comparison(agg_nue_true_osc, agg_nue_true_unosc, NUE_ENERGY_BINS,
                          "All Channels", "nue", "True Energy [GeV]",
                          os.path.join(args.outdir, "aggregate_nue_true.png"))
    if agg_numu_reco_osc is not None:
        plot_1d_comparison(agg_numu_reco_osc, agg_numu_reco_unosc, NUMU_ENERGY_BINS,
                          "All Channels", "numu", "Reconstructed Energy [GeV]",
                          os.path.join(args.outdir, "aggregate_numu_reco.png"))
        plot_1d_comparison(agg_numu_true_osc, agg_numu_true_unosc, NUMU_ENERGY_BINS,
                          "All Channels", "numu", "True Energy [GeV]", 
                          os.path.join(args.outdir, "aggregate_numu_true.png"))
    
    print(f"\nAll plots saved to: {args.outdir}")

if __name__ == "__main__":
    main()
