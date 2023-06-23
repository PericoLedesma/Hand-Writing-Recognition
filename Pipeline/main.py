print('Start of the script.')
# Functions from other py files
from read_data import *
from line_segmentation import *
# from segment_letter0 import *
from get_bbox1 import *
from drop_fall import *

# Libraries
import tensorflow as tf
import cv2 as cv

# -------------------------------------------------------------------
plt.close('all')

if len(sys.argv) == 1:
    print("Please give a path in the system arguments")
    quit()
else:
    path = sys.argv[1]

print(path)

# READ DATA
data_SCROLLS = Read_inputs_pipeline(path) # From the file Pipeline_inputs

# Label character classification
labels, char_map = Model_data()
model = tf.keras.models.load_model('task2_model/CR_adam_monk1.hdf5')
doc_path='./result_docs/'
txt_path='./result_txts/'
if not os.path.isdir(txt_path):
    print("Make directory 'result_txts'")
    os.mkdir(txt_path)
if not os.path.isdir(doc_path):
    print("Make directory 'result_docs'")
    os.mkdir(doc_path)
# -------------------------------------------------------------------
counter = 0
for image_i in data_SCROLLS['number']:
    print('*'*10, ' IMAGE ', counter, ' ', '*'*10)
    print('\tFile name: ', data_SCROLLS['filename'][image_i])

    # IMAGE FOLDER
    Create_directory(counter)

    # LINE SEGMENTATION
    print('\n           LINE SEGMENTATION')
    image_evaluated = Class_Image_Analyzed(data_SCROLLS['image'][image_i], counter)
    counter += 1


    # # CHARACTER SEGMENTATION
    # print('\n           CHARACTER SEGMENTATION AND CLASSIFICATION')
    # print('Processing...')
    # text = []
    #
    # #loop over the lines
    # for idx1 in range(0, len(image_evaluated.line)):
    #     line = ""
    #
    #     #format the image correctly
    #     img=255-image_evaluated.line[idx1].image
    #     img = img.astype(np.uint8)
    #     img = np.dstack([img, img, img])
    #     img=cv.resize(img, None,fx=0.5, fy=0.5)
    #     img_d=img.copy()
    #     img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     _, binary = cv.threshold(img_grey, 50, 255, cv.THRESH_BINARY)
    #
    #     #extract the bounding boxes in the line image
    #     rects=[]
    #     rects=bbox(img)
    #
    #     #Check the type of the bounding boxes that are returned by the bbox Function:
    #     #type 0 represents correctly segmented characters
    #     #type 1 represents multi-character bounding boxes without convex split points
    #     #type 2 represents multi-character bounding boxes with convex split points
    #     #
    #     #if the type is 1 or 2, split the characters with the acid drop fall method appropriately
    #     characters = []
    #     line_characters = []
    #     if rects is not None:
    #         rects.sort(key=takeSecond, reverse=False)
    #         if rects !=['#']:
    #             for char in rects:
    #                 if char[4] == 0:
    #                     solo_char = 255 - binary[char[1]:char[1]+char[3], char[0]:char[0]+char[2]]
    #                     solo_char = np.dstack((solo_char, solo_char, solo_char))
    #                     line_characters.append(solo_char)
    #                 elif char[4] == 1:
    #                     char[5].sort(key=takeSecond, reverse=False)
    #                     try:
    #                         characters = split(char, binary)
    #                     except:
    #                         continue
    #                     for charz in characters:
    #                         line_characters.append(np.array(charz))
    #                 elif char[4] == 2:
    #                     try:
    #                         characters = split(char, binary)
    #                     except:
    #                         continue
    #                     for charz in characters:
    #                         line_characters.append(np.array(charz))
    #
    #             #CHECK CHARACTERS OF LINE WITH PLOT
    #             '''
    #             for character_ in line_characters:
    #                 cv.imshow("char", character_)
    #                 cv.waitKey(0)
    #                 cv2.destroyWindow('char')
    #             cv2.destroyAllWindows()
    #             '''
    #             # CHARACTER CLASSIFICATION
    #             for idx, char in enumerate(line_characters):
    #                 #format input
    #                 char = np.array(char) / 255
    #                 x = cv2.resize(char, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    #                 x = np.expand_dims(x, 0)
    #
    #                 #predict label
    #                 y_pred = model.predict(x, verbose=0)
    #                 y_pred = labels[np.argmax(y_pred)]
    #                 line += char_map[y_pred]
    #             text.append([line])
    #         else:
    #             text.append(['#'])
    #
    # output_doc(text, doc_path+data_SCROLLS['filename'][image_i][:-4] + "_characters")
    # output_txt(text, txt_path+data_SCROLLS['filename'][image_i][:-4] + "_characters")
print('End of the script')
