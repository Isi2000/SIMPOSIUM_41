import numpy as np
import tensorly as tl
import json
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

    # Initialize tensor with only available components: (x, y, components, time)
    tensor = np.zeros((Ny//subsample_y, Nx//subsample_x, n_available, n_snapshots))
    for t_idx in range(n_snapshots):
        for new_idx, (comp_name, orig_idx) in enumerate(zip(available_components, available_indices)):
            filename_key = file_key_map[comp_name]
            filename = metadata['local'][t_idx][filename_key]
            data = np.fromfile(f"{data_path}/{filename}", dtype='<f4').reshape(Ny, Nx)
            molar_data = data / molar_masses[comp_name]
            tensor[:, :, new_idx, t_idx] = molar_data[::subsample_x, ::subsample_y]
    return tensor