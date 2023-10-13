# hamrgb

Generate pretty images with all RGB colors.

![image with all RGB colors](https://i.imgur.com/j2n46iz.png)

The images are generated by forming a hamiltonian cycle through the pixel locations of the canvas, and another hamiltonian cycle through the RGB locations of the color cube. These cycles are walked simultaneously to produce an extraordinarily "smooth" output image.

# usage

```
usage: hamrgb [-q] [-x NUM] [-y NUM] [-r NUM] [-g NUM] [-b NUM] [-h] [-v] FILE

build images with all possible colors and positions in respective hamiltonian cycles

positional arguments:
  FILE
    output .pbm file (use -- for stdout)

optional arguments:
  -x NUM,--width NUM
    width of output image, must be a power of 2 (default 4096)
  -y NUM,--height NUM
    height of output image, must be a power of 2 (default 4096)
  -r NUM,--red NUM
    number of discrete red values, must be a power of 2 (default 256)
  -g NUM,--green NUM
    number of discrete green values, must be a power of 2 (default 256)
  -b NUM,--blue NUM
    number of discrete blue values, must be a power of 2 (default 256)
  -q,--quiet
    don't output diagnostic messages
  -h,--help
    show this help text and exit
  -v,--version
    show version text and exit

This program, given parameters about the dimensions of an image and RGB cube, will find a hamiltonian cycle through the positions of the output image and a hamiltonian cycle through the positions of the RGB cube, and subsequently walk through the two at the same time in order to create an aesthetically pleasing image.
Copyright (c)2023 Max Nurzia
```
