import os
from PIL import Image

def make_gif(output_dir, frame_folder_name, num_frames, gif_name, duration=33):
    frames = [Image.open(os.path.join(output_dir, frame_folder_name, f"{i}.jpg")) for i in range(num_frames)]
    frame_one = frames[0]
    frame_one.save(os.path.join(output_dir, gif_name), format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    
if __name__ == "__main__":
    pass