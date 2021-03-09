import numpy as np
import cv2
from skimage.util import random_noise

def generate_noisy_patterns(dpath, imlist, noisyp, rate):
    """ Create a salt and pepper noisy version of the patters, with a give noise rate """
    white, black = 255, 0
    for i in imlist:
        img = cv2.imread(dpath + i)
        nimg = random_noise(img, mode='s&p', amount=rate)
        nimg = np.array(255 * nimg, dtype='uint8')

        cv2.imwrite(noisyp+i, nimg)


def generate_rotated_patterns(dpath, imlist, rpath):
    """ Create 180 degree rotated patterns and save them in rpath. """
    for i in imlist:
        img = cv2.imread(dpath+i)
        img_rot = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(rpath+i, img_rot)


def generate_rectangled_patterns(dpath, imlist, rectpath):
    """ Put a black rectangle in the middle of image"""

    filling, color = -1, (0, 0, 0) 
    spoint, epoint = (140, 140), (350, 250)
    for i in imlist:
        img = cv2.imread(dpath+i)
        rimg = cv2.rectangle(img, spoint, epoint, color, filling)
        cv2.imwrite(rectpath+i, rimg)


def create_green_bordered_imgs(pspath, green_spath, nof_patterns):
    """ Create red-bordered version of scene patterns.
        Later, they will be used for creating switching scene
    """
    h, w = 512, 512
    font_color = (0, 0, 0)
    for i in range(nof_patterns):
        img = pspath + str(i)+'.png'
        rimg = cv2.imread(img)
        bimg = rimg
        cv2.rectangle(bimg, (0, 0), (w, h), (0, 255, 0), 13)
        cv2.imwrite(green_spath+str(i)+'.png', bimg)

  
def create_red_bordered_imgs(pspath, red_spath, nof_patterns):
    """ Create red-bordered version of scene patterns.
        Later, they will be used for selecting an action.
    """
    h, w = 512, 512
    for i in range(nof_patterns):
        img = pspath + str(i)+'.png' 
        rimg = cv2.imread(img)
        bimg = rimg
        cv2.rectangle(bimg, (0, 0), (w, h), (0, 0, 255), 13)
        cv2.imwrite(red_spath+str(i)+'.png', bimg)


def create_scene(patterns, pspath):
    """ Create a scene by randomly permuting the pattern order.
         Then adapt one of the configurations.
    """

    #perm = np.random.permutation(len(patterns)).reshape((4, 5))
    perm = np.asarray([[0, 1, 12, 17, 15],
                       [10, 16, 6, 7, 4],
                       [14, 3, 11, 2, 8],
                       [19, 18, 5, 13, 9]])
    srow, scol = 4, 5
    tmps = []
    font_color = (0, 0, 0)
    font_ind = 0
    # TODO: this loop can be modularized later.
    for i in range(srow):
        img = cv2.imread(pspath + str(perm[i,0]) + '.png')
        cv2.putText(img, str(font_ind), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA, False)
        font_ind += 1
        img2 = cv2.imread(pspath + str(perm[i,1]) + '.png')
        cv2.putText(img2, str(font_ind), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA, False)
        font_ind += 1
        img3 = cv2.imread(pspath + str(perm[i,2]) + '.png')
        cv2.putText(img3, str(font_ind), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA, False)
        font_ind += 1
        img4 = cv2.imread(pspath + str(perm[i,3]) + '.png')
        cv2.putText(img4, str(font_ind), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA, False)
        font_ind += 1
        img5 = cv2.imread(pspath + str(perm[i,4]) + '.png')
        cv2.putText(img5, str(font_ind), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA, False)
        font_ind += 1
        conc = np.concatenate((img, img2, img3, img4, img5), axis=1)
        tmps.append(conc)
        conc = []

    scene = tmps[0]
    for i in range(1, srow):
        scene = np.concatenate((scene, tmps[i]), axis=0)

    cv2.imwrite("assets/monitor/green_scene.png", scene)
    cv2.imshow('scene.png', scene)
    cv2.waitKey(0)