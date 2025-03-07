from swift import (
    DispatchQueue,
    DispatchQueueAttributes,
    DispatchWorkItem,
)
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time
from pathlib import Path
from multiprocessing import Manager


class ImageProcessor:
    def __init__(self, urls, effects, output_dir):
        self.urls = urls
        self.effects = effects
        self.output_dir = Path(output_dir)
        self.total = len(urls) * len(effects)

        # Create a manager for shared state
        manager = Manager()
        self.state = manager.dict()
        self.state["completed"] = 0

        # Create output directory
        if self.output_dir.exists():
            print("🧹 Cleaning up old output files...")
            for file in self.output_dir.glob("*.png"):
                file.unlink()
        else:
            self.output_dir.mkdir()

    def update_progress(self):
        self.state["completed"] += 1
        current = self.state["completed"]

        progress = current / self.total * 100
        bar_length = 20
        filled_length = int(bar_length * current // self.total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(
            f"\rProgress: [{bar}] {progress:.1f}% ({current}/{self.total} images) 🔥",
            end="",
            flush=True,
        )
        if current == self.total:
            print()  # New line after completion

    @staticmethod
    def download_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content))

    @staticmethod
    def apply_effect(image, effect_name, intensity=1.0):
        # Convert to numpy array for fast processing
        img_array = np.array(image)

        if effect_name == "cyberpunk":
            # Boost neon colors
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.5, 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.3, 0, 255)  # Blue

            # Add scanlines
            scanlines = np.zeros_like(img_array)
            scanlines[::2, :] = [0, 0, 50]  # Dark blue scanlines
            img_array = np.clip(img_array + scanlines, 0, 255)

        elif effect_name == "vaporwave":
            # Pink/Purple tint
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.4, 0, 255)  # Red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.2, 0, 255)  # Blue

            # Add noise
            noise = np.random.randint(0, 30, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)

        elif effect_name == "matrix":
            # Enhance greens
            img_array = img_array.astype(
                "float64"
            )  # Convert to float for multiplication
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.5, 0, 255)
            img_array[:, :, [0, 2]] = (
                img_array[:, :, [0, 2]] * 0.5
            )  # Reduce red and blue
            img_array = img_array.astype("uint8")  # Convert back to uint8

            # Add digital rain effect
            rain = (
                np.random.randint(0, 2, (img_array.shape[0], img_array.shape[1])) * 255
            )
            rain_mask = np.random.random(img_array.shape[:2]) > 0.99
            img_array[rain_mask] = [0, 255, 0]  # Green digital rain

        return Image.fromarray(img_array.astype("uint8"))

    def process_image(self, url, idx, effect):
        img = self.download_image(url)
        processed = self.apply_effect(img, effect)
        output_path = self.output_dir / f"img_{idx}_{effect}.png"
        processed.save(output_path)
        self.update_progress()
        return output_path

    def process_all(self):
        """Process all effects in parallel."""
        start = time.time()

        # Create a queue for all work
        queue = DispatchQueue(
            "com.example.imageprocessor",
            attributes=DispatchQueueAttributes.CONCURRENT_WITH_MULTIPROCESSING,
        )

        # Store work items for synchronization
        work_items = []

        # Create work items for each image and effect combination
        for effect in self.effects:
            print(f"\n🎨 Applying {effect} effect to all images...")
            for i, url in enumerate(self.urls):
                work_item = DispatchWorkItem(
                    block=self.process_image, args=(url, i, effect)
                )
                queue.async_(work_item)
                work_items.append(work_item)

        # Wait for all processing to complete
        for item in work_items:
            item.wait()

        end = time.time()
        print(f"\n✨ All effects applied in {end - start:.2f} seconds!")

        # List the output files
        output_files = list(self.output_dir.glob("*.png"))
        print(f"\n🖼️  Generated {len(output_files)} images in {self.output_dir}:")
        for file in sorted(output_files):
            print(f"  - {file.name}")


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting parallel image processor with swift.py\n")

    # Some example images
    urls = [
        "https://picsum.photos/800/600",  # Random images from picsum
        "https://picsum.photos/800/601",  # Different URLs to avoid caching
        "https://picsum.photos/800/602",
        "https://picsum.photos/800/603",
        "https://picsum.photos/800/604",
    ]

    effects = ["cyberpunk", "vaporwave", "matrix"]

    # Create and run processor
    processor = ImageProcessor(urls, effects, "output")
    processor.process_all()
