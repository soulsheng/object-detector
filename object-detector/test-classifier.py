# Import the required modules
from skimage.transform import pyramid_gaussian,resize
from skimage.io import imread, imshow
from skimage.feature import hog
from sklearn.externals import joblib
import cv2, time, os, glob
import argparse as ap
from nms import nms
from config import *
import numpy as np

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    # Parse the command line arguments
  parser = ap.ArgumentParser()
  parser.add_argument('-i', "--image", help="Path to the test image", required=True)
  parser.add_argument('-o', "--output", help="Path to the output image", required=True)
  parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=float)
  parser.add_argument('-t', '--threshold', help="threshold of confidence score", default=0.6,
            type=float)
  parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
  parser.add_argument('-s', '--show', help="Show result image",
            action="store_true")
  parser.add_argument('-w', '--write', help="Write result image",
            action="store_true")
             
  args = vars(parser.parse_args())
  thresh_score = args["threshold"]
  output_path = args["output"]
  img_path = args["image"]
  
  img_files =[]
  
  if os.path.isfile(img_path) :
   img_files.append(img_path)
  else : # path
   img_files = glob.glob(img_path + '/*')

  dicNeg = {}
  dicPos = {}
  dicMinMax = {"min": 5, "max": -5}

  for img_file in img_files :
    # Read the image
    filepath,fullflname = os.path.split(img_file)
    fname,ext = os.path.splitext(fullflname)

    im_color = cv2.imread(img_file)
    im = imread(img_file, as_gray=False)
    min_wdw_sz = (96, 96)
    #imshow(im)
    im = resize(im, min_wdw_sz)
    #imshow(im)
    #print( img_file )
    #exit(10)
    step_size = (10, 10)
    downscale = args['downscale']
    visualize_det = args['visualize']
    bShow = args['show']
    bWrite = args['write']

    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            st = time.clock()
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, block_norm='L1')
            print("HOG Time=%.3f"%(time.clock()-st))
            
            st = time.clock()
            fd_new = np.array(fd).reshape(1, -1)
            pred = clf.predict(fd_new)
            print("predict Time=%.3f"%(time.clock()-st))
            
            if 1: # pred == 1:
                print(  "Detection:: Location -> ({}, {})".format(x, y) )
                print( "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd_new)) )
                detections.append((x, y, clf.decision_function(fd_new),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
            # If visualize is set to true, display the working
            # of the sliding window
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _, _  in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                        im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(30)
        # Move the the next scale
        scale+=1

    # Display the results before performing NMS
    clone = im_color.copy()
    for (x_tl, y_tl, _, w, h) in detections:
      if bShow:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)

        cv2.imshow("Raw Detections before NMS", clone)
        cv2.waitKey()

    # Perform Non Maxima Suppression
    detections = nms(detections, threshold)

    # Display the results after performing NMS
    clone = im_color.copy()
    for (x_tl, y_tl, score, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
        if score > thresh_score:
            type_str = 'mask'
            type_clr = (0, 255, 0)
            dicPos[fname] = score
        else:
            type_str = 'no mask'
            type_clr = (0, 0, 255)
            dicNeg[fname] = score

        cv2.putText(clone, type_str, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, type_clr, 2)
        cv2.putText(clone, '%.2f'%score, (60,140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        if score > dicMinMax["max"]:
            dicMinMax["max"] = score
        if score < dicMinMax["min"]:
            dicMinMax["min"] = score

    if bShow:
        cv2.imshow("Final Detections after applying NMS", clone)
        cv2.waitKey()
    if bWrite:
        out_name = output_path + fname + ext
        print( out_name )
        cv2.imwrite(out_name, clone)

  if len(dicNeg) > len(dicPos)  :
      dic = dicPos
  else:
      dic = dicNeg

  for key in dic:
     print("%s:%.2f"%(key, dic[key] ) )
  print(" miss count = %d"%( len(dic) ) )

  for key in dicMinMax:
     print("%s:%.2f"%(key, dicMinMax[key] ) )