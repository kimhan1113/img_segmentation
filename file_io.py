import numpy as np
from skimage import color, img_as_ubyte, img_as_float
import os

SAVEFILE_EXTENSION = '.imglabtag'
SAVEFILE_DESCRIPTION = 'Labeled and Tagged Image File(s)'

def openSavefile(path, force_open=False):
    # Attempt to read file from path and load its data
    # If path has extension SAVEFILE_EXTENSION, attempt to read as a SAVEFILE_EXTENSION file
    # If else, attempt to read ONLY IF force_open is True
    extension = os.path.splitext(path)[1]
    if (extension == SAVEFILE_EXTENSION or force_open):
        with open(path, 'rb') as f:
            try:
                loaded_data = np.load(f, allow_pickle=True)
                mat_label = loaded_data['mat_label']
                img_original = loaded_data['img_original']
                img_original = img_as_float(color.gray2rgb(img_original))
                tags = loaded_data['tags'].item()
                labels = sorted(tags.keys())
                return mat_label, img_original, labels, tags
            except Exception as e:
                raise e
    else:
        raise IOError('cannot open files without extension {} unless force_open is set to True'.format(SAVEFILE_EXTENSION))

def saveToFile(path, mat_label, img_original, tags):
    # Compress-save parameters into a single SAVEFILE_EXTENSION file at path
    if path:
        # File name cannot be empty
        if (os.path.splitext(path)[1] != SAVEFILE_EXTENSION):
            path += SAVEFILE_EXTENSION
        try:
            with open(path, 'wb') as f:
                np.savez_compressed(
                    f,
                    mat_label = mat_label,
                    img_original = img_as_ubyte(color.rgb2gray(img_original)),
                    tags = tags,
                )
        except Exception as e:
            raise e
    else:
        return IOError('file name cannot be empty')