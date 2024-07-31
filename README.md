# optiraw
Image processing from dng to jpeg using an image quality assessment model

## Features

- Uses pytorch and GPU if available
- Tunes white balance and brightness using IQA model (clipiqa+)
- Has 'fast' batch mode, and 'high quality' mode

## Installation

```bash
pip install optiraw
```

## Usage

### 'Fast' batch mode

This works on dngs in a directory and makes jpegs at half native resolution

```python
import optiraw

optiraw.fast_dng_to_jpg(input_dir, output_dir)

```

### 'High quality' mode

```python
import optiraw

output_size = (height, width) # or None to use native resolution
optiraw.hq_dng_to_jpg(input_file, output_file, output_size)

```


