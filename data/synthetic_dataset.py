import numpy as np

def generate_dataset(num_samples=50, input_size=64):
    """
    Generates a synthetic dataset of binary patterns.
    Returns:
        data: (num_samples, input_size) NumPy array
    """
    data = []
    
    # 1. Random Binary Patterns
    for _ in range(num_samples // 3):
        pattern = np.random.randint(0, 2, size=input_size)
        data.append(pattern)
        
    # 2. Geometric Shapes (Simple Lines/Blocks for 8x8 grid)
    # Assuming input_size=64 corresponds to 8x8
    if input_size == 64:
        for _ in range(num_samples // 3):
            grid = np.zeros((8, 8))
            # Horizontal line
            row = np.random.randint(0, 8)
            grid[row, :] = 1
            data.append(grid.flatten())
            
            grid = np.zeros((8, 8))
            # Vertical line
            col = np.random.randint(0, 8)
            grid[:, col] = 1
            data.append(grid.flatten())

    # 3. Clusterable Low-Dimensional Vectors (Random + Noise)
    # Fill the rest with these
    remaining = num_samples - len(data)
    base_patterns = [np.random.randint(0, 2, size=input_size) for _ in range(3)]
    
    for _ in range(remaining):
        base = base_patterns[np.random.randint(0, 3)]
        # Flip a few bits to add noise
        noise_mask = np.random.rand(input_size) < 0.1
        noisy_pattern = np.abs(base - noise_mask) # XOR-like flip
        data.append(noisy_pattern)

    return np.array(data)
