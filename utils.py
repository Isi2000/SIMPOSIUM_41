import numpy as np
import tensorly as tl
import json
import imageio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

###Questo utils e' a dir poco vergognoso, lo faccio cosi' posso commentare facilmente la roba nei notebook
# Se qualcuno che non sono io (isacco) dovra' usare sto file le mie piu' sentite scuse e buona fortuna

#-------------------------------------------------------------------
def print_statistics(tensors: dict, component_names):
    for dataset_path, tensor in tensors.items():
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_path}")
        print("=" * 80)
        print(f"{'Component':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)        
        for c_idx, comp_name in enumerate(component_names):
            component_data = tensor[:, :, c_idx, :]
            mean_val = np.mean(component_data)
            std_val = np.std(component_data)
            min_val = np.min(component_data)
            max_val = np.max(component_data)
            
            print(f"{comp_name:<15} {mean_val:<12.6e} {std_val:<12.6e} {min_val:<12.6e} {max_val:<12.6e}")
        
        print("-" * 80)
        
        # Overall tensor stats
        print(f"Overall Tensor Statistics:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Total elements: {tensor.size:,}")
        print(f"  Memory size: {tensor.nbytes / (1024**2):.2f} MB")
        print(f"  Global mean: {np.mean(tensor):.6e}")
        print(f"  Global std:  {np.std(tensor):.6e}")
        print(f"  Global min:  {np.min(tensor):.6e}")
        print(f"  Global max:  {np.max(tensor):.6e}")
        print("=" * 80)

#-------------------------------------------------------------------
def load_dataset(data_path, component_names, file_key_map, Ny, Nx, n_snapshots, molar_masses, subsample_x, subsample_y):
    with open(f"{data_path}" + '/info.json', 'r') as f:
        metadata = json.load(f)

    # Sta roba da quando facevo il caso non reattivo
    available_components = []
    available_indices = []
    for c_idx, comp_name in enumerate(component_names):
        filename_key = file_key_map[comp_name]
        if filename_key in metadata['local'][0]:
            available_components.append(comp_name)
            available_indices.append(c_idx)

    print(f"  Available components in dataset: {available_components}")
    n_available = len(available_components)

    tensor = np.zeros((Ny//subsample_y, Nx//subsample_x, n_available, n_snapshots))
    for t_idx in range(n_snapshots):
        for new_idx, (comp_name, orig_idx) in enumerate(zip(available_components, available_indices)):
            filename_key = file_key_map[comp_name]
            filename = metadata['local'][t_idx][filename_key]
            data = np.fromfile(f"{data_path}/{filename}", dtype='<f4').reshape(Ny, Nx)
            molar_data = data / molar_masses[comp_name]
            tensor[:, :, new_idx, t_idx] = molar_data[::subsample_x, ::subsample_y]
    return tensor
#-------------------------------------------------------------------
def scale_and_center_tensors(tensors, component_names, log_scale=True, epsilon=1e-12):
    tensors_scaled = {}
    
    for dataset_path, tensor in tensors.items():
        tensor_scaled = tensor.copy()
        
        for c_idx, comp_name in enumerate(component_names):
            component_data = tensor_scaled[:, :, c_idx, :]
            
            if log_scale:
                component_data = np.log10(np.maximum(component_data, epsilon))
            
            mean_val = component_data.mean()
            std_val = component_data.std()
            
            if std_val < epsilon:
                std_val = epsilon  # prevent divide-by-zero
            
            component_data_scaled = (component_data - mean_val) / std_val
            
            tensor_scaled[:, :, c_idx, :] = component_data_scaled
            temporal_mean = tensor_scaled.mean(axis=3, keepdims=True)
            tensor_scaled = tensor_scaled - temporal_mean
        
        
        tensors_scaled[dataset_path] = tensor_scaled
    
    print("\nLog scaling and standardization complete.\n")
    return tensors_scaled

#-------------------------------------------------------------------
def create_hosvd_reconstruction_gif(
    decomposition_results,
    multi_mode_dot,
    output_path='README_PLOTS/hosvd_reconstruction_chemical_modes_singular_cbars.gif',
    Lx=12.5,
    Ly=15.6,
    time_conversion_factor=5e-06,
    fps=10,
    dpi=150,
    figsize=(20, 10)
):
    """
    Create a gif showing HOSVD reconstruction across all time steps for all chemical modes.
    
    Parameters:
    -----------
    decomposition_results : dict
        Dictionary containing decomposition results with 'factors' and 'core'
    multi_mode_dot : function
        Function to perform multi-mode dot product (from tensorly)
    output_path : str
        Path where the gif will be saved
    Lx, Ly : float
        Domain size in D units
    time_conversion_factor : float
        Conversion factor for time
    fps : int
        Frames per second for the gif
    dpi : int
        DPI for saving frames
    figsize : tuple
        Figure size (width, height)
    """

    
    for dataset_path, result in decomposition_results.items():
        factors = result['factors']
        core = result['core']
        U_y = factors[0]      # Spatial Y factor (Ny_sub, Ny_sub)
        U_x = factors[1]      # Spatial X factor (Nx_sub, Nx_sub)
        U_chem = factors[2]   # Chemical dimension factor (8, 8)
        U_time = factors[3]   # Time factor (200, 200)
        
        n_time_steps = U_time.shape[0]
        n_chem_modes = U_chem.shape[1]
        
        print("Reconstructing tensors for all chemical modes...")
        reconstructed_tensors = [
            multi_mode_dot(core, [U_y, U_x, U_chem[:, mode], U_time], modes=[0, 1, 2, 3]) 
            for mode in tqdm(range(n_chem_modes), desc="Chemical modes")
        ]
        
        # Compute min/max for each tensor across all time steps
        print("Computing min/max values for each chemical mode...")
        vmin_per_mode = [tensor.min() for tensor in reconstructed_tensors]
        vmax_per_mode = [tensor.max() for tensor in reconstructed_tensors]
        
        # Compute global min/max across all modes
        vmin = min(vmin_per_mode)
        vmax = max(vmax_per_mode)
        print(f"Global value range: [{vmin:.2e}, {vmax:.2e}]")
        
        # Create temporary directory for frames
        temp_dir = 'temp_frames'
        os.makedirs(temp_dir, exist_ok=True)
        frames = []
        
        print(f"Generating frames for gif...")
        for t in tqdm(range(n_time_steps)):  # Iterate through all time points
            fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"HOSVD Reconstruction at t={t * time_conversion_factor:.2e}s using Chemical Modes", 
                         fontsize=18, fontweight='bold')
            axes = axes.flatten()
            
            for mode in range(n_chem_modes):
                reconstructed_t = reconstructed_tensors[mode][:, :, t]
                ax = axes[mode]
                im = ax.imshow(
                    reconstructed_t,
                    cmap='inferno',
                    origin='lower',
                    extent=[0, Lx, 0, Ly],
                    aspect='auto',
                    vmin=vmin,  # Set global minimum
                    vmax=vmax   # Set global maximum
                )
                ax.set_title(f'Chemical Mode {mode + 1}', fontsize=14, fontweight='bold')
                ax.set_xlabel("X / D", fontsize=12)
                ax.set_ylabel("Y / D", fontsize=12)
                ax.tick_params(labelsize=10)
                
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
                cbar.ax.tick_params(labelsize=10)
            
            frame_path = f'{temp_dir}/frame_{t:03d}.png'
            plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            frames.append(imageio.imread(frame_path))
            plt.close(fig)
        
        print(f"Creating gif at {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        
        print("Cleaning up temporary files...")
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        print(f"Gif saved to {output_path}")

