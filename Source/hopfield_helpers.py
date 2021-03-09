import cv2
import pickle
import numpy as np
import seaborn as sns
import os
from naoqi import ALProxy

# The SARSA algorithm can be reached via the below link.
# https://github.com/muratkirtay/ADAPTIVE2019/tree/master/Source

cv2.CV_LOAD_IMAGE_GRAYSCALE = 0


def show_monitor_img(img):
    """ Show the image in fullscreen ont experiment monitor"""
    show_img = 'eog --fullscreen ' + img + ' &'
    #time.sleep(0.1)
    os.system(show_img)


def kill_scene():
    """"Kill the eye of the gnome window/s"""
    os.system('killall eog')

def capture_robot_camera(IP_PEPPER, PORT):
    """ Capture images from the robot TOP camera.
        Remember you need to subscribe and unsubscribe respectively
        see, https://ai-coordinator.jp/pepper-ssd#i-3
    """
    SubID = "Pepper"
    videoDevice = ALProxy('ALVideoDevice', PI_PEPPER, PORT)

    # subscribe top camera, get an image with the size of 640x480
    AL_kTopCamera, AL_kQVGA, Frame_Rates  = 0, 2, 10 
    AL_kBGRColorSpace = 13  # Buffer contains triplet on the format 0xRRGGBB, equivalent to three unsigned char
    captureDevice = videoDevice.subscribeCamera(SubID, AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, Frame_Rates)

    width, height = 640, 480
    image = np.zeros((height, width, 3), np.uint8)
    result = videoDevice.getImageRemote(captureDevice)

    if result == None:
        print "Camera problem."
    elif result[6] == None:
        print "No image was captured. "
    else:
        # translate value to mat
        values = map(ord, list(result[6]))
        i = 0
        for y in range(0, height):
            for x in range(0, width):
                image.itemset((y, x, 0), values[i + 0])
                image.itemset((y, x, 1), values[i + 1])
                image.itemset((y, x, 2), values[i + 2])
                i += 3

        # uncomment below lines to see the camera image
        #cv2.imwrite("assets/monitor/robocam.png", image)
        #cv2.imshow("Camera image", image)
        #cv2.waitKey(1)

    # unsubscribe from the camera.Otherwise, the camera image
    # might be corrupted. To be absoulutely sure, perform 
    # a null check on result[6]
    videoDevice.unsubscribe(captureDevice)

    return result[6], image

def extract_convergence_rate(test, original, num_of_neurons):
    """ Extract contamination rate for converged pattern. The 
    	high rate for convergence rate indicates how well 
    	pattern recalled by HN dynamics
    """

    return np.sum(test == original) * 100/num_of_neurons

def bipolarize_pattern_robot(pattern_name, rsize):
    """ Convert percieved patterns images into Bipolarized (-1, 1) inputs. """

    # the ROI coordinates of the percieved patterns.
    crop_y1, crop_y2 = 232, 452 
    crop_x1, crop_x2 = 238, 464 

    gimg = cv2.imread(pattern_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    roi_image = gimg[crop_y1:crop_y2, crop_x1:crop_x2]
    rimg = cv2.resize(roi_image, rsize)
    bimg = cv2.threshold(rimg, 125, 255, cv2.THRESH_BINARY)[1]

    # uncomment the below lines to see the binary images 
    # cv2.imshow("bin robo", bimg)
    # cv2.imshow("gray robot", gimg)
    # cv2.imshow("grsize", rimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # convert 255 to -1 and 0 to 1
    bimg = bimg.astype('int64')
    nonz_inds = bimg.nonzero()
    bimg[nonz_inds], bimg[bimg == 0] = -1, 1 # convert 255 to -1 and 0 to 1

    return bimg.flatten()

def construct_weight_matrix(bipolar_inps, nof_neurons):
    """ Construct weight matrix by using the patterns to be stored in the memory. """

    # initialize weights for the patterns.
    total_w = np.zeros((nof_neurons, nof_neurons), dtype=np.int64)
    pattern_w = np.zeros((nof_neurons, nof_neurons), dtype=np.int64)
    for pat_ind in range(len(bipolar_inps)):
        for i in range(nof_neurons):
            for j in range(nof_neurons):
                if i == j:  # additional operation eliminated
                    pattern_w[i, i] = 0
                else:
                    bpattern = bipolar_inps[pat_ind] # for debuging
                    temp = bpattern[i] * bpattern[j]
                    pattern_w[i, j], pattern_w[j, i] = temp, temp
        total_w += pattern_w
        pattern_w.fill(0)

    return total_w

def run_hopfield(test_pattern, orig_test_pattern, nof_neurons, weight_mat, unchanged_states, max_iters):
	""" Run hopfield network to perform visual recalling. 
        Note that the number of flipped bit used as an energy value.
	"""
 
    nof_flipped_bits = 0
    st_change_counter, iteration = 0, 0
    while st_change_counter < unchanged_states and iteration < max_iters:
        # generate rand_int from discrete uniform distribution
        rand_ind = np.random.randint(nof_neurons - 1)
        sum = 0
        for it in range(nof_neurons):
            if rand_ind != it:
                sum += test_pattern[it] * weight_mat[it, rand_ind]
                #print sum, test_pattern[it], weight_mat[it, rand_ind]

        # perform sgn function
        if sum >= 0:
            test_pattern[rand_ind] = 1
        else:
            test_pattern[rand_ind] = -1

        # trace number of flipped bits.
        if test_pattern[rand_ind] != orig_test_pattern[rand_ind]:
            nof_flipped_bits += 1

        # check network state
        state_change_flag = False

        for i in range(len(test_pattern)):
            if test_pattern[i] != orig_test_pattern[i]:
                state_change_flag = True
                break

        if state_change_flag:
            st_change_counter += 1
        else:
            st_change_counter = 0

        iteration += 1

    return test_pattern, st_change_counter, nof_flipped_bits


def extract_reward(srew, arew):
    """ Extract reward value as a function of energy.
		Note that in real robot experiment the = part of >= might
		not be achieve due to noise in the environment.
    """
    if srew >= arew:
        reward = 1
    else:
        reward = -1
    return reward

