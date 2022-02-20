This repository is based on https://github.com/ruchikaverma-iitg/Nuclei-Segmentation. [J.W. Johnson](https://gitlab.com/TimSchmittmann/dl-based-image-cell-segmentation-with-mask-rcnn/-/blob/master/72.pdf) used a very similar approach to this one and can be consulted for more information on the approach itself and the used frameworks.  

We adjusted the code to be compatible with our own image examples and added some pre- and postprocessing steps to handle images of different sizes and extract whole cells instead of only nuclei. The [Notebook](https://gitlab.com/TimSchmittmann/dl-based-image-cell-segmentation-with-mask-rcnn/-/blob/master/Nuclei_Segmentation_mask_generator.ipynb) should be able to run inside colab without any further actions. 

If you want to recreate the environment and run the notebook locally try to use 

    pip install -r requirements.txt
    
which was created from [pipreqs](https://pypi.org/project/pipreqs/). Otherwise [pipfreeze.requirements.txt](https://gitlab.com/TimSchmittmann/dl-based-image-cell-segmentation-with-mask-rcnn/-/blob/master/pipfreeze.requirements.txt) contains all the installed packages from the colab environment. 
