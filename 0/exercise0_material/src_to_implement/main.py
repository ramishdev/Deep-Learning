import pattern
import generator

if __name__ == "__main__":
    spectrum = pattern.Spectrum(1024)
    spectrum.draw()
    spectrum.show()
    
    gen = generator.ImageGenerator('./exercise_data', './Labels.json', 12, (128, 128, 3), rotation=True, mirroring=True, shuffle=True)
    gen.show()