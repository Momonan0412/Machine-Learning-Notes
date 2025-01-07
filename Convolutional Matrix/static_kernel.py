import numpy as np
from PIL import Image
class StaticKernels:
    
    @staticmethod
    def _gaussian_blur():
        return np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=float) / 16 + 0 # Placeholder
        
    @staticmethod
    def _sharpen():
        return np.array([
            [0,  -2, 0],
            [-2, 11,-2],
            [0,  -2, 0]
        ], dtype=float) / 3 + 0 # Placeholder
        
    @staticmethod
    def _mean_removal():
        return np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=float) / 1 + 0 # Placeholder
    
    # The +127 in the embosses adjusts the convolution result to fit in a typical image value range.
    @staticmethod
    def _emboss_laplascian():
        return np.array([
            [-1, 0,-1],
            [0,  4, 0],
            [-1, 0,-1]
        ], dtype=float) / 1 + 127
        
    @staticmethod
    def _emboss_horz_vertical():
        return np.array([
            [0, -1, 0],
            [-1, 4,-1],
            [0, -1, 0]
        ], dtype=float) / 1 + 127
    
    @staticmethod
    def _emboss_all_directions():
        return np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=float) / 1 + 127
        
    @staticmethod
    def _emboss_lossy():
        return np.array([
            [ 1, -2,  1],
            [-2,  4, -2],
            [-2,  1, -2]
        ], dtype=float) / 1 + 127
        
    @staticmethod
    def _emboss_horizontal():
        return np.array([
            [ 0,  0,  0],
            [-1,  2, -1],
            [ 0,  0,  0]
        ], dtype=float) / 1 + 127
    
    @staticmethod
    def _emboss_vertical():
        return np.array([
            [ 0, -1,  0],
            [ 0,  0,  0],
            [ 0, -1,  0]
        ], dtype=float) / 1 + 127
        
# channels refer to the individual color components or layers that make up an image.
def _zero_padding(image_array, pad_size=1):
    if(image_array.shape == (5,5)):
        pad_size = 2
    return np.pad(image_array,
                  # ((ROW),(COL),(CHANNELS!))
                  ((pad_size, pad_size), (pad_size, pad_size), (0, 0,)),
                  mode="constant", constant_values=0  # Padding with zero values
                  )
def apply_kernel(image_array, kernel):
    height, width, channels = image_array.shape
    output_image = np.zeros(image_array.shape, dtype=np.uint8)
    
    for i in range(1, height-1):
        print("prcessing #", i)
        for j in range(1, width-1):
            red = image_array[i-1:i+2, j-1:j+2, 0] * kernel
            green = image_array[i-1:i+2, j-1:j+2, 1] * kernel
            blue = image_array[i-1:i+2, j-1:j+2, 2] * kernel
            output_image[i, j, 0] = np.sum(red)  # Red channel
            output_image[i, j, 1] = np.sum(green)  # Green channel
            output_image[i, j, 2] = np.sum(blue)  # Blue channel
    return output_image

if __name__ == "__main__":
    image = Image.open(r"Convolutional Neural Networks\\Convolutional Matrix\\Subaru_and_Beako.jpg")
    image = image.convert("RGB")
    image_array = np.array(image)
    print(image_array.shape)
    
    padded_image = _zero_padding(image_array)
    new_image = apply_kernel(padded_image, StaticKernels._emboss_all_directions())
    new_image = Image.fromarray(new_image)
    new_image.show()
    
    # Visualize
    
    # arr = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #             [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    #             [[255, 255, 255], [0, 0, 0], [127, 127, 127]]])
    # print(arr.shape)
    
    # pad_arr = np.pad(arr, ((1,1),(1,1), (0,0)), mode="constant", constant_values=0)
    
    # print(pad_arr)
    
    # # Example 3D array (RGB image), size 5x5 for simplicity
    # image = np.array([[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
    #                 [[255, 0, 0], [1, 0, 0], [4, 0, 0], [7, 0, 0], [255, 0, 0]],
    #                 [[255, 0, 0], [2, 0, 0], [5, 0, 0], [8, 0, 0], [255, 0, 0]],
    #                 [[255, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0], [255, 0, 0]],
    #                 [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]], dtype=int)

    # # Define a simple 3x3 kernel (e.g., box blur kernel)
    # kernel = np.array([[1, 1, 1],
    #                 [1, 1, 1],
    #                 [1, 1, 1]], dtype=float)

    # # Normalize kernel (sum = 9, so divide by 9)
    # kernel = kernel / 9.0

    # # Select a region from the image (let's select region around pixel (2,2))
    # i, j = 2, 2  # Center of the region
    # region = image[i-1:i+2, j-1:j+2, 0]  # Take the Red channel (i-1:i+2, j-1:j+2, 0)

    # Apply the kernel (element-wise multiplication followed by summing)
    # output_value = np.sum(region * kernel)
    # print(region * kernel)
    # print(output_value)
    # Print the original region and the result
    # print("Region (Red Channel):")
    # print(region)
    # print("\nKernel:")
    # print(kernel)
    # print("\nOutput after applying kernel:")
    # print(output_value)