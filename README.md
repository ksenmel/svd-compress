# svd-compress
Comressing/Decompressing .BMP images using Singular value decomposition (SVD) to .PNG format

## Compressed format

The compressed image has the following structure:

Part| Size (bytes) | Description                                                                                                        |
--|------------|--------------------------------------------------------------------------------------------------------------------
Header | 16         | Contains general information about the compressed image
Body |            | Contains an image                                                                                 

### Header

Offset (hex) |Size (bytes)| Description|
--|--|--|
00|4| File format signature (ASCII) 
04|4| n - (width of image)  
08|4| m - (height of image in pixels)                                                                           
0C|4| k - number of computed singular values

### Body
There are tree part for each RBG's channel. All values are written sequentially, without any alignments or separators.

Name |Size (bytes)| Description|
--|-|--|
U|4 * width * k| Matrix containing left singular vectors
S|4 * k| Vector containing singular numbers
Vt|4 * height * k| Matrix containing right singular vector

# Usage
```
$ python3 main.py --mode compress --method numpy --compression 2 --input_file /path/to/dir/tree.bmp --output_file /path/to/dir/tree.CSVD
$ python3 main.py --mode decompress --input_file /path/to/dir/tree.CSVD --output_file /path/to/dir/tree.png
```