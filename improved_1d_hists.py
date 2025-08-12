#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

# ─── Official binning from Vlada's specifications ─────────────────────────────
COSZ_BINS = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# e-like events (*_nueselec.root) - 17 bins with overflow at 70 GeV
NUE_ENERGY_BINS = np.array([0.0, 0.07, 0.1, 0.15, 0.22, 0.33, 0.47, 0.7, 
                           1.1, 1.7, 2.9, 5, 9, 16, 32, 48, 70, 10000.0])

# non e-like events (*_numuselec.root) - 11 bins with overflow at 70 GeV  
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
    for t in sorted(trees):
        print(f"  {t}")
    return sorted(trees)

def load_event_data(rootfile, tree_path, selection_type, use_weights=True):
    """Load energy and weight data from a TTree."""
    try:
        with uproot.open(rootfile) as f:
            tree = f[tree_path]
            
            # Check if tree has any entries
            if tree.num_entries == 0:
                print(f"  WARNING: Tree {tree_path} has 0 entries")
                return np.array([]), np.array([]), np.array([])
            
            print(f"  Loading {tree.num_entries:,} entries from {tree_path.split('/')[-2]}")
            
            # Load all data at once
            data = tree.arrays(['Event', 'Leaves'], library='np')
            
            # Access structured arrays
            leaves_data = data['Leaves']
            event_data = data['Event']
            
            # Get available branches
            leaves_branches = list(leaves_data.dtype.names)
            event_branches = list(event_data.dtype.names)
            
            print(f"  Available Leaves branches: {leaves_branches}")
            print(f"  Available Event branches: {event_branches}")
            
            # True energy (always mcEnu)
            if 'mcEnu' not in leaves_branches:
                print(f"  ERROR: mcEnu branch not found in {tree_path}")
                return np.array([]), np.array([]), np.array([])
            E_true = leaves_data['mcEnu']
            
            # Reco energy selection based on selection type
            reco_candidates = []
            if selection_type == 'nue':  # electron-like selection
                reco_candidates = ['recoEnuEcalo', 'recoEnuCalo', 'recoEnuLepCalo']
            else:  # muon-like selection
                reco_candidates = ['recoEnuLepCalo', 'recoEnuCalo', 'recoEnuEcalo']
            
            reco_branch = None
            for candidate in reco_candidates:
                if candidate in leaves_branches:
                    reco_branch = candidate
                    break
            
            if reco_branch is None:
                print(f"  ERROR: No suitable reco energy branch found in {tree_path}")
                print(f"  Tried: {reco_candidates}")
                return np.array([]), np.array([]), np.array([])
            
            E_reco = leaves_data[reco_branch]
            print(f"  Using reco energy branch: {reco_branch}")
            
            # Weight selection
            weights = np.ones(len(E_true))  # Default unit weights
            if use_weights:
                # Try different weight branches in order of preference
                weight_candidates = ['eventWeight']  # Event structure
                if 'eventWeight' in event_branches:
                    weights = event_data['eventWeight']
                    print(f"  Using weights from Event.eventWeight")
                else:
                    # Try Leaves structure for backup
                    leaves_weight_candidates = ['mcOscW', 'mcGenWeight']
                    for candidate in leaves_weight_candidates:
                        if candidate in leaves_branches:
                            weights = leaves_data[candidate]
                            print(f"  Using weights from Leaves.{candidate}")
                            break
                    else:
                        print(f"  WARNING: No weight branch found, using unit weights")
            else:
                print(f"  Using unit weights (unoscillated)")
            
            print(f"  Energy ranges - Reco: {E_reco.min():.3f}-{E_reco.max():.3f} GeV, True: {E_true.min():.3f}-{E_true.max():.3f} GeV")
            print(f"  Weight range: {weights.min():.6f}-{weights.max():.6f}, Total weighted events: {weights.sum():.2f}")
            
            return E_reco, E_true, weights
            
    except Exception as e:
        print(f"  ERROR loading data from {tree_path}: {e}")
        return np.array([]), np.array([]), np.array([])

def determine_selection_type(tree_path):
    """Determine selection type based on the third neutrino (detection type) in the path."""
    # Path format: #nu_{initial} x #nu_{final} #nu_{selection} Selec
    # The selection type is determined by the third neutrino (what was detected/reconstructed)
    
    # Clean up the path and look for the selection neutrino
    path_part = tree_path.split('Selec')[0]
    
    # Look for patterns indicating electron or muon selection
    if '#nu_{e}' in path_part.split(' x ')[-1]:  # Last neutrino determines selection
        return 'nue'
    elif '#nu_{#mu}' in path_part.split(' x ')[-1]:
        return 'numu'
    else:
        # Fallback: look for any indication in the path
        if 'nu_{e}' in path_part:
            return 'nue'
        else:
            return 'numu'

def get_energy_bins(selection_type):
    """Get energy bins for the given selection type."""
    return NUE_ENERGY_BINS if selection_type == 'nue' else NUMU_ENERGY_BINS

def clean_channel_name(tree_path):
    """Extract and clean channel name from tree path for nice display."""
    # Extract the channel part between "model/" and "Selec"
    if "model/" in tree_path and "Selec" in tree_path:
        channel_raw = tree_path.split("model/")[1].split("Selec")[0].strip()
        
        # Clean up the channel name
        channel_clean = (channel_raw
                        .replace("#bar{#nu_{e}}", "antinue")
                        .replace("#bar{#nu_{#mu}}", "antinumu") 
                        .replace("#bar{#nu_{#tau}}", "antinutau")
                        .replace("#nu_{e}", "nue")
                        .replace("#nu_{#mu}", "numu")
                        .replace("#nu_{#tau}", "nutau")
                        .replace("{", "").replace("}", "")
                        .replace("#", "")
                        .replace(" x ", "_to_")
                        .replace("  ", "_")
                        .replace(" ", "_"))
        
        # Add selection type suffix
        if channel_clean.endswith("_nue"):
            return channel_clean + "_eselec"
        elif channel_clean.endswith("_numu"):
            return channel_clean + "_muselec"
        else:
            return channel_clean
    
    return "unknown_channel"

def plot_1d_comparison(hist_osc, hist_unosc, bins, channel_name, selection_type, xlabel, output_path):
    """Create 1D comparison plot with improved styling."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Bin centers for plotting (convert to log10 space)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    log_centers = np.log10(bin_centers)
    
    # Create step plots
    ax.step(log_centers, hist_unosc, where='mid', label='Unoscillated', 
            color='blue', linewidth=2, alpha=0.8)
    ax.step(log_centers, hist_osc, where='mid', label='Oscillated', 
            color='red', linewidth=2, alpha=0.8)
    
    # Set x-axis as log10(E/GeV)
    ax.set_xlim(-1, 2)  # 0.1 GeV to 100 GeV
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels(['-1', '0', '1', '2'])
    ax.set_xlabel(r'$\log_{10}(E/\mathrm{GeV})$', fontsize=12)
    
    # Set y-axis to log scale to show both distributions
    ax.set_yscale('log')
    ax.set_ylabel('Event Rate', fontsize=12)
    
    # Clean up the title
    nice_title = (channel_name.replace("_to_", r" $\to$ ")
                             .replace("antinue", r"$\bar{\nu}_e$")
                             .replace("antinumu", r"$\bar{\nu}_\mu$")
                             .replace("antinutau", r"$\bar{\nu}_\tau$")
                             .replace("nue", r"$\nu_e$") 
                             .replace("numu", r"$\nu_\mu$")
                             .replace("nutau", r"$\nu_\tau$")
                             .replace("eselec", "Electron-like Selection")
                             .replace("muselec", "Muon-like Selection"))
    
    ax.set_title(f'{nice_title}\n{xlabel} Energy', fontsize=12, pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Statistics box
    total_unosc = hist_unosc.sum()
    total_osc = hist_osc.sum()
    
    if total_unosc > 0:
        mean_unosc = np.average(bin_centers, weights=hist_unosc)
        std_unosc = np.sqrt(np.average((bin_centers - mean_unosc)**2, weights=hist_unosc))
    else:
        mean_unosc = 0
        std_unosc = 0
        
    if total_osc > 0:
        mean_osc = np.average(bin_centers, weights=hist_osc)
        std_osc = np.sqrt(np.average((bin_centers - mean_osc)**2, weights=hist_osc))
    else:
        mean_osc = 0
        std_osc = 0
    
    stats_text = (f"Unoscillated:\n"
                 f" Entries: {total_unosc:.0f}\n"
                 f" Mean:    {mean_unosc:.2f} GeV\n"
                 f" Std:     {std_unosc:.2f} GeV\n\n"
                 f"Oscillated:\n"
                 f" Entries: {total_osc:.0f}\n"
                 f" Mean:    {mean_osc:.2f} GeV\n"
                 f" Std:     {std_osc:.2f} GeV")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create 1D neutrino energy histograms")
    parser.add_argument("--osc", required=True, help="Oscillated ROOT file")
    parser.add_argument("--unosc", required=True, help="Unoscillated ROOT file")
    parser.add_argument("--outdir", default="1d_histograms", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Just list channels and exit")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Get date stamp for file naming
    date_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Find all channels
    found_trees = find_selection_trees(args.osc)
    
    if args.dry_run:
        print("\nDry run complete - trees listed above")
        return
    
    if not found_trees:
        print("No trees found!")
        return
    
    print(f"\nProcessing {len(found_trees)} channels...")
    
    # Initialize aggregate containers
    agg_nue_reco_osc = None
    agg_nue_reco_unosc = None
    agg_nue_true_osc = None  
    agg_nue_true_unosc = None
    agg_numu_reco_osc = None
    agg_numu_reco_unosc = None
    agg_numu_true_osc = None
    agg_numu_true_unosc = None
    
    processed_count = 0
    
    for tree_path in found_trees:
        print(f"\n--- Processing channel {processed_count + 1}/{len(found_trees)} ---")
        
        # Determine selection type and energy bins
        sel_type = determine_selection_type(tree_path)
        energy_bins = get_energy_bins(sel_type)
        channel_name = clean_channel_name(tree_path)
        
        print(f"Channel: {channel_name}")
        print(f"Selection type: {sel_type}")
        print(f"Energy bins: {len(energy_bins)-1} bins")
        
        try:
            # Load data (weights only for oscillated)
            E_reco_osc, E_true_osc, weights_osc = load_event_data(
                args.osc, tree_path, sel_type, use_weights=True)
            E_reco_unosc, E_true_unosc, weights_unosc = load_event_data(
                args.unosc, tree_path, sel_type, use_weights=False)
            
            # Check if we got valid data
            if len(E_reco_osc) == 0 or len(E_reco_unosc) == 0:
                print(f"  SKIPPING: No valid data loaded for {channel_name}")
                continue
            
            # Create histograms
            hist_reco_osc, _ = np.histogram(E_reco_osc, bins=energy_bins, weights=weights_osc)
            hist_reco_unosc, _ = np.histogram(E_reco_unosc, bins=energy_bins, weights=weights_unosc)
            hist_true_osc, _ = np.histogram(E_true_osc, bins=energy_bins, weights=weights_osc)
            hist_true_unosc, _ = np.histogram(E_true_unosc, bins=energy_bins, weights=weights_unosc)
            
            # Check for empty histograms
            if hist_reco_osc.sum() == 0 and hist_reco_unosc.sum() == 0:
                print(f"  WARNING: Both reco histograms empty for {channel_name}")
            if hist_true_osc.sum() == 0 and hist_true_unosc.sum() == 0:
                print(f"  WARNING: Both true histograms empty for {channel_name}")
            
            # Individual channel plots with date stamp
            reco_filename = f"{channel_name}_reco_{date_stamp}.png"
            true_filename = f"{channel_name}_true_{date_stamp}.png"
            
            plot_1d_comparison(hist_reco_osc, hist_reco_unosc, energy_bins, channel_name, sel_type,
                             "Reconstructed", os.path.join(args.outdir, reco_filename))
            plot_1d_comparison(hist_true_osc, hist_true_unosc, energy_bins, channel_name, sel_type,
                             "True", os.path.join(args.outdir, true_filename))
            
            # Add to aggregates
            if sel_type == 'nue':
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
            else:  # numu
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
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ERROR processing {channel_name}: {e}")
            continue
    
    # Create aggregate plots
    print(f"\nCreating aggregate plots...")
    
    if agg_nue_reco_osc is not None:
        plot_1d_comparison(agg_nue_reco_osc, agg_nue_reco_unosc, NUE_ENERGY_BINS,
                         "All νₑ-like Selection Channels", "nue", "Reconstructed",
                         os.path.join(args.outdir, f"aggregate_nue_reco_{date_stamp}.png"))
        plot_1d_comparison(agg_nue_true_osc, agg_nue_true_unosc, NUE_ENERGY_BINS,
                         "All νₑ-like Selection Channels", "nue", "True",
                         os.path.join(args.outdir, f"aggregate_nue_true_{date_stamp}.png"))
    
    if agg_numu_reco_osc is not None:
        plot_1d_comparison(agg_numu_reco_osc, agg_numu_reco_unosc, NUMU_ENERGY_BINS,
                         "All νμ-like Selection Channels", "numu", "Reconstructed", 
                         os.path.join(args.outdir, f"aggregate_numu_reco_{date_stamp}.png"))
        plot_1d_comparison(agg_numu_true_osc, agg_numu_true_unosc, NUMU_ENERGY_BINS,
                         "All νμ-like Selection Channels", "numu", "True",
                         os.path.join(args.outdir, f"aggregate_numu_true_{date_stamp}.png"))
    
    print(f"\nAnalysis complete!")
    print(f"Processed {processed_count}/{len(found_trees)} channels successfully")
    print(f"Output files saved in: {args.outdir}")
    print(f"Date stamp: {date_stamp}")

if __name__ == "__main__":
    main()