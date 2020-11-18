import numpy as np
from file_io import *
import cv2

path = './dest/FovB001.imglabtag'

colors_label={
    #'Black':    (0,0,0),
 	'Red':	(255,0,0),
 	'Lime':	(0,255,0),
 	'Blue':	(0,0,255),
 	'Yellow':	(255,255,0),
 	'Cyan':	(0,255,255),
 	'Magenta':	(255,0,255),
 	#'Silver':	(192,192,192),
 	#'Gray':	(128,128,128),
    'maroon': (176,48,96),
 	#'Olive':	(128,128,0),
 	'Green':	(0,128,0),
 	'Purple':	(128,0,128),
 	#'Teal':	(0,128,128),
    'CadetBlue': (95,158,160),
 	#'Navy':	(0,0,128),
    #'Indigo':(75,0,130),
    'GreenYellow': (173,255,47),
    'light pink': (255,182,193),
    'khaki': (240,230,140),
    'MidnightBlue': (25,25,112),
    'MediumVioletRed': (199,21,133),
    #'turquoise': (64,224,208)
}

preferences = dict()
preferences['alpha'] = 0.2

mat_label, img_original, labels, tags = openSavefile(path)
height, width = mat_label.shape
img_grayscale = img_as_float(color.rgb2gray(img_original))


x1,y1,x2,y2 =  0,0,width,height
mat_label_temp = mat_label[y1:y2,x1:x2].copy()
label_ = np.sort(np.unique(mat_label))

img_temp = np.pad(
            img_grayscale[y1:y2, x1:x2],
            ((len(label_),0),(0,0)),
            'constant',
            constant_values = 0,
        )

for label in label_:
    mat_label_temp = np.pad(
        mat_label_temp,
        ((1, 0), (0, 0)),
        'constant',
        constant_values=label,
    )

img = color.label2rgb(
    mat_label_temp,
    image = 255*img_temp,
    colors=list(colors_label.values()),
    alpha=preferences['alpha'],
    kind='overlay',
    bg_label=-1
)

img = np.asarray(img, dtype=np.uint8)[len(label_):]
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()