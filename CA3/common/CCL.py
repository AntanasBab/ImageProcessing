from matplotlib import pyplot as plt
import numpy as np

def connected_component_labeling(binary_image):
    rows, cols = binary_image.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    label = 1
    equivalences = {}

    # First pass: assign provisional labels and track equivalences
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                neighbors = []

                for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and labels[ni, nj] > 0:
                        neighbors.append(labels[ni, nj])

                if not neighbors:
                    labels[i, j] = label
                    equivalences[label] = label
                    label += 1
                else:
                    min_label = min(neighbors)
                    labels[i, j] = min_label

                    # Update equivalence table
                    for neighbor_label in neighbors:
                        if neighbor_label != min_label:
                            equivalences[max(min_label, neighbor_label)] = min(min_label, neighbor_label)

    # Resolve equivalences
    for key in sorted(equivalences.keys(), reverse=True):
        root = equivalences[key]
        while root != equivalences[root]:
            root = equivalences[root]
        equivalences[key] = root

    # Second pass: relabel pixels with resolved labels
    resolved_labels = np.zeros_like(labels)
    new_label = 1
    label_map = {}

    for i in range(rows):
        for j in range(cols):
            if labels[i, j] > 0:
                resolved_label = equivalences[labels[i, j]]
                if resolved_label not in label_map:
                    label_map[resolved_label] = new_label
                    new_label += 1
                resolved_labels[i, j] = label_map[resolved_label]

    return resolved_labels, new_label - 1

def plot_labeled_image(labels, num_labels):
    plt.figure(figsize=(10, 8))
    labeled_image = np.zeros_like(labels, dtype=np.float32)
    
    for label in range(1, num_labels + 1):
        labeled_image[labels == label] = label
    
    # Custom colormap for indexes
    from matplotlib.colors import ListedColormap
    cmap = plt.cm.nipy_spectral
    cmap_colors = cmap(np.linspace(0, 1, num_labels + 1))
    cmap_colors[0] = [0, 0, 0, 1.0]
    custom_cmap = ListedColormap(cmap_colors)
    
    plt.imshow(labeled_image, cmap=custom_cmap)
    plt.colorbar(ticks=np.arange(num_labels + 1), label="Cell Index")
    plt.title("Labeled Cells with Indices")
    plt.show()