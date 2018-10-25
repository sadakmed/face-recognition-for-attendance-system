import numpy as np
import os 
from collections import OrderedDict

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_IDXS = OrderedDict ( [
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
] )

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict ( [
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
] )


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros ( (68, 2), dtype=dtype )

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range ( 0, 68 ):
        coords[i] = (shape.part ( i ).x, shape.part ( i ).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left ()
    y = rect.top ()
    w = rect.right () - x
    h = rect.bottom () - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)



def shuffle( img, label, sed=None):
    #     if sed is not None:
    #     np.random.seed(2)
    permit = np.random.permutation ( img.size )
    img = img[permit]
    label = label[permit]
    return img, label

def flow_from_dataframe( img_data_gen, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname ( path_col[0] )
    print ( '## Ignore next message from keras, values are replaced anyways' )
    df_gen = img_data_gen.flow_from_directory ( base_dir,
                                                class_mode='sparse',
                                                **dflow_args )
    df_gen.filenames = path_col
    df_gen.classes = y_col
    df_gen.samples = path_col.shape[0]
    df_gen.n = path_col.shape[0]
    df_gen._set_index_array ()
    df_gen.directory = ''  # since we have the full path
    print ( 'Reinserting dataframe: {} images'.format ( path_col.shape[0] ) )
    return df_gen
def bring_data():
        img = np.array ( [] )
        label = np.array ( [] )
        for i, cl in enumerate ( os.listdir('dataset') ):
            imgcl = np.array ( [x.path for x in os.scandir ( cl )] )
            if card == 'all' :
                card = imgcl.size
            labelcl = np.array ( [i for j in range ( imgcl.size )] )
            img = np.concatenate ( (img, imgcl[:card]), axis=0 )
            label = np.concatenate ( (label, labelcl[:card]), axis=0 )
        return img, label
def data():
    images=np.array ([])
    label=np.array ([],dtype=np.uint8)
    for i,dirs in enumerate(os.scandir('dataset')):
        for di in (os.scandir(dirs)):
              images=np.append(images,np.array([di.path]),axis=0)
              label=np.append(label,np.array([i]),axis=0) 
    return images,label
