import numpy as np;

def otsu_threshold(image):
    pixel_counts = np.bincount(image.astype(int).flatten(), minlength=256)
    total_pixels = np.sum(pixel_counts)
    total_sum = np.sum(np.arange(256) * pixel_counts)
    
    max_variance = 0
    threshold = 0
    weight_background = 0
    sum_background = 0
    
    for t in range(256):
        weight_background += pixel_counts[t]
        weight_foreground = total_pixels - weight_background
        
        if weight_background == 0 or weight_foreground == 0:
            continue
        
        sum_background += t * pixel_counts[t]
        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground
        
        # Calculate between class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    
    return threshold

def threshold_image(image, threshold):
    return (image > threshold).astype(np.uint8)

def apply_dual_otsu_threshold(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size

    # Compute cumulative sums and means
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    total_mean = cumulative_mean[-1]

    max_variance = 0
    threshold1 = 0
    threshold2 = 0

    for t1 in range(1, 256):
        for t2 in range(t1 + 1, 256):

            # Class 1: [0, t1), Class 2: [t1, t2), Class 3: [t2, 256)
            weight1 = cumulative_sum[t1]
            weight2 = cumulative_sum[t2] - cumulative_sum[t1]
            weight3 = total_pixels - cumulative_sum[t2]

            if weight1 == 0 or weight2 == 0 or weight3 == 0:
                continue

            mean1 = cumulative_mean[t1] / weight1
            mean2 = (cumulative_mean[t2] - cumulative_mean[t1]) / weight2
            mean3 = (total_mean - cumulative_mean[t2]) / weight3

            variance_between = (
                weight1 * (mean1 - total_mean) ** 2
                + weight2 * (mean2 - total_mean) ** 2
                + weight3 * (mean3 - total_mean) ** 2
            )

            if variance_between > max_variance:
                max_variance = variance_between
                threshold1 = t1
                threshold2 = t2
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image < threshold1] = 0
    binary_image[(image >= threshold1) & (image < threshold2)] = 1 
    binary_image[image >= threshold2] = 2

    return binary_image, threshold1, threshold2