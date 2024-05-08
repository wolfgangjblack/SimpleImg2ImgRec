import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms as tfms
import faiss
import json
import torch
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

metadata_file = '../data/metadata.json'
faiss_index = faiss.read_index("../data/image_index_vit.index")

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Load the pre-trained ViT model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# Define preprocessing pipeline for images
transform = tfms.Compose([
    tfms.Resize(256),
    tfms.CenterCrop(224),
    tfms.ToTensor(),
    tfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageApp:
    """
    Simple image viewer application using Tkinter.
    """

    def __init__(self, master):
        """
        Initialize the ImageApp.

        Parameters:
            master (tk.Tk): The root window of the application.
        """
        self.master = master
        self.master.title("Image Viewer")

        # Initialize variables
        self.image_paths = []  # List to store image paths
        self.current_index = 0  # Index of currently displayed image
        self.model = model
        self.metadata = metadata
        self.transform = transform
        self.faiss_index = faiss_index
        self.img = None  # Placeholder for currently displayed image

        # Create and pack widgets
        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        # Entry widget for typing image path
        self.entry = tk.Entry(self.master)
        self.entry.pack(fill=tk.X)

        # Button to browse for an image file
        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        # Button for previous image
        self.prev_button = tk.Button(self.master, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        # Button for next image
        self.next_button = tk.Button(self.master, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        # Load initial image
        self.load_image()

        # Bind arrow key events
        self.master.bind("<Left>", lambda event: self.prev_image())
        self.master.bind("<Right>", lambda event: self.next_image())

    def load_image(self, path=None):
        """
        Load and display an image.

        Parameters:
            path (str): Path to the image file. If None, use the value from the entry widget.
        """
        # Check if path is provided, otherwise use entry value
        if path is None:
            path = self.entry.get()

        if path:
            try:
                self.img = Image.open(path)
                photo = ImageTk.PhotoImage(self.img.resize((400, 400)))
                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep reference to prevent garbage collection
                self.generate_recommended_images()  # Generate recommended images immediately after loading
                return self.img
            except Exception as e:
                self.image_label.config(text="Error loading image: " + str(e))
        else:
            self.image_label.config(text="No image to display")

    def prev_image(self):
        """
        Display the previous image in the list.
        """
        print("user hit prev")
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image(self.image_paths[self.current_index])

    def next_image(self):
        """
        Display the next image in the list.
        """
        print("user hit next")
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image(self.image_paths[self.current_index])

    def generate_recommended_images(self):
        """
        Generate recommended images based on the currently displayed image.
        """
        print('Retrieving recommended images...')
        if self.img is not None:
            print("self IMG is not None")
            start_time = time.time()
            with torch.no_grad():
                encoded_input = self.model(self.transform(self.img).unsqueeze(0)).squeeze().reshape(1, -1).cpu()
                _, indices = self.faiss_index.search(encoded_input, 100)
            end_time = time.time()
            runtime = end_time - start_time
            indices = list(indices[0])
            self.image_paths.extend([os.path.join('../data/imgs',str(self.metadata[str(i)]['image_id']))+'.jpg' for i in indices])
            print(f"Runtime: {runtime} seconds for retrieving {len(indices)} recommendations")
            

    def update_ui(self):
        """
        Update the UI after generating recommendations.
        """
        # Display the first image
        self.current_index = 0
        self.load_image(self.image_paths[self.current_index])

    def browse_image(self):
        """
        Open a file dialog to browse for an image file and display it.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            self.load_image(file_path)


def main():
    """
    Main function to run the application.
    """
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
