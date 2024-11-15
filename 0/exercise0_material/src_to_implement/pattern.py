import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be evenly divisible by 2 times tile size.")
        
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # Create the tile patterns
        black_tile = np.zeros((self.tile_size, self.tile_size))
        white_tile = np.ones((self.tile_size, self.tile_size))
        first_row = np.concatenate([black_tile, white_tile] * (self.resolution // (2 * self.tile_size)), axis=1)
        second_row = np.concatenate([white_tile, black_tile] * (self.resolution // (2 * self.tile_size)), axis=1)
        pattern = np.concatenate([first_row,second_row] * (self.resolution // (2 * self.tile_size)), axis=0)
        self.output = pattern
        return pattern.copy()

    def show(self):
        if self.output is None:
            self.draw()
        
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        xx, yy = np.meshgrid(x, y)
        circle_eq = (xx - self.position[0])**2 + (yy - self.position[1])**2 <= self.radius**2
        self.output = np.zeros((self.resolution, self.resolution))
        self.output[circle_eq] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title('Circle Pattern')
        plt.axis('off')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        grid=np.zeros([self.resolution,self.resolution, 3])
        horizontal = np.linspace(0, 1, self.resolution)
        grid[:, :, 0] = horizontal
        grid[:, :, 2] = horizontal[::-1]
        grid[:, :, 1] = horizontal[:, np.newaxis]
        self.output = grid
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.title('RGB Color Spectrum')
        plt.axis('off')
        plt.show()
