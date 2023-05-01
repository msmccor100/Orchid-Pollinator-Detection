"""
This script takes a path to vgg weights, a directory of (orchid pollinator) images,
an output directory, a batch size, a device name (optional, "cpu" is default), and a 
gpu minimum memory (optional)

The batch size should be suitable for the cpu/gpu memory constraints. The script
uses ResNet to classify the images working in batches. For each image it determines
the maximum probability class and outputs

    <best class index> <best class probability>

to a text file in the output directory with filename of the form

    <image file base name>.txt

If the gpu minimum memory is specified, this routine verifies this memory is
available before processing each batch. If it is unavailable, it performs a python 
garbage collection, frees the gpu cache, and polls until the minimum memory is 
available.
"""

import os
import sys
import torch
from torchvision.models import vgg16
from PIL import Image
from torchvision import transforms
import torch
import pynvml 
import time
import gc 
import numpy as np

NUM_CLASSES = 7
NVML_INITIALIZED = False

def verify_paths(paths_list, exit_code):
    for path in paths_list:
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist")
            sys.exit(exit_code)

def get_image_filepaths(dir):
    fp_list = []

    for fn in os.listdir(dir):
        split = os.path.splitext(fn)
        if len(split) < 2:
            continue
        ext = split[1]
        if ext not in [".jpg",".png"]:
            continue
        fp = os.path.join(dir, fn)
        if not os.path.isfile(fp):
            continue
        fp_list.append(fp)
    return fp_list


def poll_for_gpu_memory_availability(device_index, amount_needed, timeout=30,poll_period=0.2):
    """This routine polls until the required amount of GPU
    memory is available. If no GPU itself is available a
    RuntimeError is raised. So this routine should only be called
    if torch.cuda.is_available(). If the required amount of memory
    does not become available before the timeout (seconds) this routine
    raises a RuntimeError.
    """

    global NVML_INITIALIZED
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")
    if not NVML_INITIALIZED:
        pynvml.nvmlInit()
        NVML_INITIALIZED = True

    hDevice = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(hDevice)
    if info.free > amount_needed:
        return
    
    gc.collect()
    torch.cuda.empty_cache()

    time_waited = 0.0    
    while(True):
        time.sleep(poll_period)
        time_waited += poll_period
        info = pynvml.nvmlDeviceGetMemoryInfo(hDevice)
        if info.free > amount_needed:
            return
        elif time_waited >= timeout:
            raise RuntimeError("Timed out waiting for GPU memory")    


def main():
    if len(sys.argv) not in [5,6,7]:
        print(f"Call is: {sys.argv[0]} <weights path> <images path> <output dir> <batch size> [<device>] [<min gpu mem>]")
        sys.exit(-1)

    start = time.time()
 
    paths = sys.argv[1:4]
    [weights_path, images_path, output_dir] = paths

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    verify_paths(paths, -2)

    batch_size = int(sys.argv[4])

    device = "cpu"
    min_gpu_mem = None
    if len(sys.argv) >= 6:
        device = sys.argv[5]
    if device == "cuda" and len(sys.argv) == 7:
        min_gpu_mem = int(sys.argv[6])

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ERROR: Cuda is not available.")

    model = vgg16(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
    if device == 'cpu':
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    model = model.to(device)

    image_fps = get_image_filepaths(images_path)
    print(f"Processing {len(image_fps)} images.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    num_images = len(image_fps)
    count = 0
    for i, fp in enumerate(image_fps):
        if i % batch_size == 0:
            batch_fns = []
            images = torch.empty(1, 3, 640, 640)

        batch_fns.append(os.path.split(fp)[-1])

        image = Image.open(fp).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(dim=0)
        images = torch.concat([images, image])

        if i % batch_size == batch_size - 1 or i == num_images - 1:
            images = images[1:]
            images = images.to(device)

            if min_gpu_mem is not None:
                poll_for_gpu_memory_availability(0, min_gpu_mem)

            with torch.no_grad():  # no updation of gradient based on the validation data
                out = model(images)
         
            classes = torch.argmax(out, dim=1).tolist()
            probs = torch.nn.functional.softmax(out, dim=1)

            if device == "cuda":
                probs = probs.detach().cpu().numpy()
            else:
                probs = probs.detach().numpy()
            max_probs = np.max(probs,axis=1)

            for fn, cls, prob in zip(batch_fns,classes,max_probs):
                [base,ext] = os.path.splitext(fn)
                label_fn =  base + ".txt"
                new_fp = os.path.join(output_dir, label_fn)
                with open(new_fp, "w") as fd:
                    fd.write(f"{cls} {prob}\n")

            print(f"{i+1} in {round(time.time() - start)} seconds")
                


if __name__ == "__main__":
    main()