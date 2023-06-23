import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelmin, argrelmax

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import random
import pandas as pd

# ---------------------------------------------------------------
class Class_Image_Analyzed:
    def __init__(self, image, counter):
        THRESHOLD = 5000
        # ------- PARAMETERS -------
        self.ID = counter
        # Image split
        self.SIGMA = 13    # For blurring the histogram at the initial steps
        self.SLICE_THRESHOLD = THRESHOLD  # If the sum of the histogram of a slice has less than this, we declare that there is no text
        self.horizontal_split = 100 # Number of pixel for each vertical split
        # Cell boundaries
        self.CELL_THRESHOLD = 100  # If there is no gap line in this distance, we create one
        # Cell optimization
        self.CELL_PIXEL_THRESHOLD = 100  # Resizing boundaries. We push down the gap line if there is less than this threshold
        self.TEXT_LINE_PIXEL_THRESHOLD = THRESHOLD # If a cell text line has less than this pixels around, we delete it(histogram sum)
        self.SEP_LINE_PIXEL_THRESHOLD = THRESHOLD # If separation lines hace less than this, we delete them(histogram sum)
        self.THRESHOLD_BETWEEN_LINES = 10 # Pixels. If cells have less space than this, we delete them
        # Lines matching.
        self.LINES_MATCHING_THRESHOLD = 90  # Distance between text line to match
        self.SEPARATION_THRESHOLD = 1000  # How many slices to declare a new line. Put value really big if not.

        directory_name = 'Image_{}'.format(counter)
        self.directory = 'Results/' + directory_name + '/' + directory_name

        # ------- IMAGE ATTRIBUTES -------
        np_img = np.asarray(image)
        self.image = np.flip(np_img, axis=1)  # Reversing. They write from right to left. We will undo in the line image

        self.height, self.base = self.image.shape

        mask = np.full(self.image.shape, 255)
        mod_img = mask - self.image
        # self.img_inverse = mod_img.astype(np.int64)
        self.img_inverse = mod_img.astype(np.uint8)

        # Lets blur the image
        ksize = (30, 10)
        self.img_blur = cv.blur(self.img_inverse, ksize)

        # ------- FUNCTIONS -------
        # SPLIT vertical image
        self.Split_image(image_used=self.img_blur)  # Splitting horizontally anc compute the horizontal projection
        self.Plot_image(image_used=self.image, save_name='_1_Initial_layout')

        # Counting cells where there is a maxima in the projection histogram
        self.Cell_counting()

        # Cell boundaries
        self.Cell_boundaries()
        self.Plot_cells('_2_Plot_cells(no optimize).png')

        # Cell boundaries optimization
        self.Cell_optimization(image_used=self.img_blur)
        self.Plot_cells('_3_Plot_cells.png')

        # Once we have optimize the cell, we take a image of the cell isolated
        self.Cell_image()

        # Search for LINES.
        self.Lines_matching()
        self.Plotting_lines(directory_name)

    # ---------------------------------------------------------------------------
    def Split_image(self, image_used):
        print('Splitting the image ...')
        cont = 0  # To count the slice number
        dict_slices = {}  # To store the slices instances created

        # Loop over all the x-axis
        for pixel_index in range(1, self.base):
            if pixel_index % self.horizontal_split == 0:
                # The vertical lines of the slides
                horizontal_index = np.array([pixel_index - self.horizontal_split, pixel_index])

                # We create the name of the slice
                name = 'Slice_{}'.format(cont)

                # We create the class instance and store it in the dict
                dict_slices[cont] = dict_slices.get(cont, self.Class_Slices(name, cont, horizontal_index, self, image_used))
                cont = cont + 1

            elif pixel_index == (self.base - 1):  # We do a slice with the left overs of pixels
                # The vertical lines of the slides
                horizontal_index = np.array([horizontal_index[1], pixel_index])

                # We create the name of the slice
                name = 'Slice_{}'.format(cont)

                # We create the class instance and store it in the dict
                dict_slices[cont] = dict_slices.get(cont, self.Class_Slices(name, cont, horizontal_index, self, image_used))
                cont = cont + 1

        self.slices = dict_slices
        print('*Done*')

    def Cell_counting(self):
        print('Counting cells...')
        # Creating all cells
        self.cells = {}
        counter = 0
        for SLICE in self.slices.keys():
            if self.slices[SLICE].text == True:
                for LINES in self.slices[SLICE].text_line:
                    self.cells[counter] = self.Class_Cell(self, SLICE, counter, LINES)
                    self.slices[SLICE].cells = np.append(self.slices[SLICE].cells, counter)
                    counter = counter + 1
        print('*Done*')

    def Cell_boundaries(self):
        print('Searching cell boundaries...')
        '''
        CELL boundaries. If there is too much distance of there is no boundaries for a text line, we create it
        '''
        # ABOVE
        for CEll_ID in self.cells.keys():
            SLICE = self.cells[CEll_ID].slice

            separation_line_array = self.slices[SLICE].separation_lines
            separation_line_array = separation_line_array[separation_line_array < self.cells[CEll_ID].text_line]

            if separation_line_array.any():
                distance = np.absolute(self.cells[CEll_ID].text_line - separation_line_array[-1])
                if distance < self.CELL_THRESHOLD:
                    self.cells[CEll_ID].y_1 = separation_line_array[-1]

                elif distance >= self.CELL_THRESHOLD:  # If there is no separation line, we create another one
                    y_1 = self.cells[CEll_ID].text_line - self.CELL_THRESHOLD
                    if y_1 > 0:
                        self.cells[CEll_ID].y_1 = y_1
                    else:
                        self.cells[CEll_ID].y_1 = 0

            elif not separation_line_array.any():
                y_1 = self.cells[CEll_ID].text_line - self.CELL_THRESHOLD
                if y_1 > 0:
                    self.cells[CEll_ID].y_1 = y_1
                else:
                    self.cells[CEll_ID].y_1 = 0

        # UNDER
        for CEll_ID in self.cells.keys():
            SLICE = self.cells[CEll_ID].slice

            separation_line_array = self.slices[SLICE].separation_lines
            separation_line_array = separation_line_array[separation_line_array > self.cells[CEll_ID].text_line]

            if separation_line_array.any():
                distance = np.absolute(self.cells[CEll_ID].text_line - separation_line_array[0])
                if distance <= self.CELL_THRESHOLD:
                    self.cells[CEll_ID].y_2 = separation_line_array[0]

                elif distance >= self.CELL_THRESHOLD:  # If there is no separation line, we create another one
                    y_2 = self.cells[CEll_ID].text_line + self.CELL_THRESHOLD
                    if y_2 > self.height:
                        self.cells[CEll_ID].y_2 = self.height - 1
                    else:
                        self.cells[CEll_ID].y_2 = y_2

            elif not separation_line_array.any():
                y_2 = self.cells[CEll_ID].text_line + self.CELL_THRESHOLD
                if y_2 > self.height:
                    self.cells[CEll_ID].y_2 = self.height - 1
                else:
                    self.cells[CEll_ID].y_2 = y_2
        print('*Done*')

    def Cell_optimization(self,image_used):
        print('Optimizing cell boundaries....')
        '''
        We push down and up the gap lines in the extremes
        '''
        for CELL_ID in self.cells.keys():
            # Pushing down the TOP boundaries if there is no text
            histogram = []
            for row in range(self.cells[CELL_ID].y_1, self.cells[CELL_ID].y_2):
                pixel_density = sum(image_used[row,
                                    self.cells[CELL_ID].horizontal_index[0]:self.cells[CELL_ID].horizontal_index[1]])

                if pixel_density < self.CELL_PIXEL_THRESHOLD:  # If there is no pixel, we push down the line
                    self.cells[CELL_ID].y_1 = self.cells[CELL_ID].y_1 + 1
                else:
                    break
            # Pushing up the button boundaries if there is no text
            histogram = []
            for row in range(self.cells[CELL_ID].y_2, self.cells[CELL_ID].y_1, -1):
                pixel_density = sum(image_used[row,
                                    self.cells[CELL_ID].horizontal_index[0]:self.cells[CELL_ID].horizontal_index[1]])

                if pixel_density < self.CELL_PIXEL_THRESHOLD:  # If there is no pixel, we push down the line
                    self.cells[CELL_ID].y_2 = self.cells[CELL_ID].y_2 - 1
                else:
                    break

        # Deleting small cells
        for CELL_ID in list(self.cells.keys()):
            if self.cells[CELL_ID].y_2 - self.cells[CELL_ID].y_1 < self.THRESHOLD_BETWEEN_LINES:
                self.slices[self.cells[CELL_ID].slice].cells = self.slices[self.cells[CELL_ID].slice].cells[self.slices[self.cells[CELL_ID].slice].cells!=CELL_ID]
                del self.cells[CELL_ID]
        print('*Done*')

    def Cell_image(self):
        for CELL_ID in self.cells.keys():  # For each cell
            image = []

            for row in range(self.cells[CELL_ID].y_1, self.cells[CELL_ID].y_2):  # Up and down boundaries
                image.append(
                    self.image[row, self.cells[CELL_ID].horizontal_index[0]:self.cells[CELL_ID].horizontal_index[1]])

            image = np.array(image)
            self.cells[CELL_ID].image = image

    def Lines_matching(self):
        print('Matching cells to create lines....')

        # SORT BY SCORE
        CELL_IDs = np.array([])
        altitude = np.array([])
        latitude = np.array([])
        for CELL_ID in self.cells:
            CELL_IDs = np.append(CELL_IDs, self.cells[CELL_ID].ID)
            altitude = np.append(altitude, self.cells[CELL_ID].text_line)
            latitude = np.append(latitude, self.cells[CELL_ID].horizontal_index[1])

        numpy_data = np.vstack((CELL_IDs, altitude, latitude))
        numpy_data = numpy_data.T
        numpy_data = numpy_data.astype('int64')

        df = pd.DataFrame(data=numpy_data, index=CELL_IDs.astype('int32'),
                          columns=["cell_id", "altitude", 'latitude'])
        df['score'] = df['altitude'] ** 2 * df['latitude']
        df_sort = df.sort_values(by=['score'], ascending=True)

        # LINE MATCHING
        LINE_ID = 0
        ref_text_line = np.array([])
        np_line_id = np.array([])
        np_averague = np.array([])

        for index, row in df_sort.iterrows():
            CELL_ID = row['cell_id']
            if self.cells[CELL_ID].line is not None:
                continue
            if self.cells[CELL_ID].line is None:
                self.cells[CELL_ID].line = LINE_ID
            #
            # print('\tCreating line', LINE_ID)
            ref_text_line = np.array([self.cells[CELL_ID].text_line])
            ref_cell = CELL_ID

            slice_list = np.array(list(self.slices.keys()))
            slice_list = slice_list[slice_list >= self.cells[CELL_ID].slice]

            for SLICE in slice_list:
                if (SLICE - self.cells[ref_cell].slice) > self.SEPARATION_THRESHOLD:
                    break
                if self.slices[SLICE].text is not True:
                    continue

                for CELL_ID_match in self.slices[SLICE].cells:
                    if self.cells[CELL_ID_match].line is None and CELL_ID_match != CELL_ID:
                        if len(ref_text_line) < 6:
                            avg_altitude = np.average(ref_text_line)
                        else:
                            avg_altitude = np.average(ref_text_line[-6:])

                        distance = np.absolute(avg_altitude - self.cells[CELL_ID_match].text_line)
                        if distance < self.LINES_MATCHING_THRESHOLD:
                            self.cells[CELL_ID_match].line = self.cells[CELL_ID].line
                            ref_text_line = np.append(ref_text_line, self.cells[CELL_ID_match].text_line)
                            ref_cell = CELL_ID_match
            np_line_id = np.append(np_line_id, LINE_ID)
            np_averague = np.append(np_averague, np.average(ref_text_line))
            LINE_ID += 1

        # SORT
        numpy_data = np.vstack((np_line_id, np_averague)).T
        df_order = pd.DataFrame(data=numpy_data, index=np_line_id,
                          columns=["LINE_id", 'altitude_avg'])
        df_order = df_order.sort_values(by=['altitude_avg'], ascending=True)

        self.number_lines = LINE_ID
        print('\tOverview')
        print('\t\tNumber of cells:', list(self.cells.keys())[-1])
        print('\t\tNumber of lines: ', LINE_ID - 1)

        self.line = {}
        NEW_LINE = 0
        print('\tSorting...')
        for index, row in df_order.iterrows():
            self.line[NEW_LINE] = self.Class_Lines(self, row['LINE_id'], NEW_LINE)
            NEW_LINE += 1
        print('*Done*')

    # PLOTTING FUNCTIONS
    def Plotting_lines(self, directory_name):
        print('Plotting LINES...')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(self.image, cmap='gray')

        for LINE_ID in range(self.number_lines):
            r = random.random()
            b = random.random()
            g = random.random()
            line_color = (r, g, b)

            for CELL_ID in self.cells.keys():
                if self.cells[CELL_ID].new_line != LINE_ID:
                    continue

                h_1, h_2 = self.cells[CELL_ID].horizontal_index
                y_1 = self.cells[CELL_ID].y_1
                y_2 = self.cells[CELL_ID].y_2

                # Cell rectangle
                rect = Rectangle((h_1, y_1), (h_2 - h_1), (y_2 - y_1), edgecolor=line_color, linewidth=0.5,
                                 linestyle='--', facecolor='none')
                ax.add_patch(rect)

                # Text line
                y = [self.cells[CELL_ID].text_line, self.cells[CELL_ID].text_line]
                X = self.cells[CELL_ID].horizontal_index
                ax.plot(X, y, color=line_color, linestyle='-', linewidth=2)
                first_cell = CELL_ID

            # ax.text(h_2, self.cells[CELL_ID].y_1, r'an equation: $E=mc^2$', fontsize=15)
            text_kwargs = dict(ha='center', va='center', fontsize=10, weight="bold", color=line_color)
            plt.text(h_2, y_1, LINE_ID, **text_kwargs)

        plt.savefig(self.directory + '_4_Lines_image.png')
        plt.close(fig)
        print('*Done*')

    def Plot_image(self, image_used, save_name):
        print('Plotting layout...')
        # Plot initial image
        fig = plt.figure(figsize=(15, 7))
        plt.imshow(image_used, cmap='gray')

        for number in self.slices.keys():
            index = self.slices[number].horizontal_index
            for X in index:
                plt.axvline(x=X, color='r', linestyle='--', linewidth=0.5)
            if self.slices[number].text:
                for i in self.slices[number].separation_lines:
                    y = [i, i]
                    X = self.slices[number].horizontal_index
                    plt.plot(X, y, color='b')

                for i in self.slices[number].text_line:
                    y = [i, i]
                    X = self.slices[number].horizontal_index
                    plt.plot(X, y, color='green', linestyle='--', linewidth=0.5)
        plt.savefig(self.directory + save_name)
        plt.close(fig)

    def Plot_cells(self, save_name):
        print('Plotting....')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(self.image, cmap='gray')

        colors = ["r", "b", "g"]  # List
        color_index = 0

        for CELL_ID in self.cells.keys():
            h_1, h_2 = self.cells[CELL_ID].horizontal_index
            y_1 = self.cells[CELL_ID].y_1
            y_2 = self.cells[CELL_ID].y_2

            rect = Rectangle((h_1, y_1), (h_2 - h_1), (y_2 - y_1), edgecolor=colors[color_index], linewidth=0.5,
                             linestyle='--', facecolor='none')
            ax.add_patch(rect)

            # Text line
            y = [self.cells[CELL_ID].text_line, self.cells[CELL_ID].text_line]
            X = self.cells[CELL_ID].horizontal_index
            ax.plot(X, y, color='green', linestyle='--')
            color_index = color_index + 1
            if color_index == 3:
                color_index = 0
        plt.savefig(self.directory + save_name)
        plt.close(fig)

    # ---------------------------------------------------------------------------
    # CLASSES INSIDE IMAGE CLASS
    class Class_Lines:
        def __init__(self, image_class, line_number, NEW_LINE_number):
            self.ID = NEW_LINE_number
            self.old_line_number = line_number
            self.y_1 = None
            self.y_2 = None
            self.cells = np.array([])
            self.image = np.full((image_class.image.shape), 255)

            self.Assembling_line(image_class)
            self.Image_resizing()
            self.Save_line(image_class)

        def __str__(self):  # print(image_test.slices[1])
            return '\t\tLine {} from {} to {}'.format(self.ID, self.y_1, self.y_2)

        def Assembling_line(self, image_class):
            y_2_boundary = 0
            y_1_boundary = image_class.height

            for CELL_ID in image_class.cells.keys():
                if image_class.cells[CELL_ID].line != self.old_line_number:
                    continue
                self.cells = np.append(self.cells, CELL_ID)
                image_class.cells[CELL_ID].new_line = self.ID
                # Vertical boundaries of the cell
                y_1 = image_class.cells[CELL_ID].y_1
                y_2 = image_class.cells[CELL_ID].y_2

                # Horizontal boundaries of the cell
                h_1 = image_class.cells[CELL_ID].horizontal_index[0]
                h_2 = image_class.cells[CELL_ID].horizontal_index[1]

                self.image[y_1:y_2, h_1:h_2] = image_class.image[y_1:y_2, h_1:h_2]

                if y_1 < y_1_boundary:
                    y_1_boundary = y_1
                if y_2 > y_2_boundary:
                    y_2_boundary = y_2
            self.y_1 = y_1_boundary
            self.y_2 = y_2_boundary

        def Save_line(self, image_class):
            image = np.uint8(self.image)
            im = Image.fromarray(image)

            im.save('Results/Image_' + str(image_class.ID) + '/Image_' + str(image_class.ID) + '_lines/Line_' + str(self.ID) + '.jpeg')

        def Image_resizing(self):
            np_img = self.image[self.y_1:self.y_2, :]
            self.image = np.flip(np_img, axis=1)

        def Plot_line(self):
            print('Plotting line..')
            plt.figure(figsize=(15, 7))
            plt.imshow(self.image, cmap='gray')
            plt.show()

    class Class_Cell:
        def __init__(self, image_class, SLICE, counter, text_line):
            # Attributes
            self.image = None
            self.y_1 = None
            self.y_2 = None
            self.ID = counter
            self.slice = SLICE
            self.text_line = text_line
            self.horizontal_index = image_class.slices[SLICE].horizontal_index
            self.line = None

        def __str__(self):  # print(image_test.slices[1])
            return 'Cell {} (slice {}) from h_{} to h_{}. Text line in {}'.format(self.ID, self.slice, self.horizontal_index[0], self.horizontal_index[1], self.text_line)


        def Plot_cell(self):
            print('Plotting cell ', self.ID, '.')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(self.image, cmap='gray')

            # # Text line
            # y = [self.text_line, self.text_line]
            # X = self.horizontal_index
            # ax.plot(X, y, color='green', linestyle='--')

            plt.savefig('3.Cell.png')
            plt.show()

    class Class_Slices:
        # PARAMETERS
        def __init__(self, name, number, horizontal_index, image_data, image_used):
            self.name = name
            self.image_used = image_used
            self.number = number
            self.horizontal_index = horizontal_index
            self.cells = np.array([])

            self.Vertical_split(image_data)  # Split by the histogram max and min

        def __str__(self):  # print(image_test.slices[1])
            return '{} from {} to {}'.format(self.name, self.horizontal_index[0], self.horizontal_index[1])

        def Vertical_split(self, image_data):
            histogram = []
            # HISTOGRAM OF SLICE
            for row in range(image_data.height):
                histogram.append(np.asarray(sum(self.image_used[row, self.horizontal_index[0]:self.horizontal_index[1]])))

            # Blur histogram
            self.histogram = gaussian_filter(histogram, sigma=image_data.SIGMA)

            # MIN --> Separation lines
            local_min_array = argrelmin(self.histogram)[0]  # Is a tuple, we want only the array
            # MAX --> Where the lines are
            local_max_array = argrelmax(self.histogram)[0]  # Middle of the line

            # CLEANING text lines
            for index, line in enumerate(local_max_array):
                line_1 = local_max_array[index] - 100  # Other variable can be set here
                line_2 = local_max_array[index] + 100
                # CHECK
                if line_1 == line_2:
                    line_1 = line_1 - 1
                    line_2 += 1
                if line_1 < 0:
                    line_1 = 0
                if line_2 > image_data.height:
                    line_2 = image_data.height - 1

                # HISTOGRAM
                histogram_2 = np.asarray(sum(self.image_used[line_1:line_2, self.horizontal_index[0]:self.horizontal_index[1]]))

                if sum(histogram_2) < image_data.TEXT_LINE_PIXEL_THRESHOLD:  # Threshold, put variable same as up
                    local_max_array[index] = 0
            local_max_array = local_max_array[local_max_array != 0]

            # Lets CHECK if it is empty --> No text, OUT function
            if local_max_array.shape[0] == 0 or sum(self.histogram) < image_data.SLICE_THRESHOLD:
                self.text = False
                return  # If there is no text, we will not store lines
            self.text = True

            # Lets make another separation to the first and last one
            first_line_separation = 0  # local_min_array[0] - 100  # Make a variable for this value
            last_line_separation = image_data.height - 1

            local_min_array = np.append(first_line_separation, local_min_array)
            local_min_array = np.append(local_min_array, last_line_separation)

            # CLEANING line separation --> FROM UP TO DOWN
            for index, line in enumerate(local_min_array):
                if index == len(local_min_array) - 1:
                    break
                line_1 = local_min_array[index]
                line_2 = local_min_array[index + 1]
                # CHECK
                if line_1 == line_2:
                    line_1 = line_1 - 1
                    line_2 += 1
                if line_1 < 0:
                    line_1 = 0
                if line_2 > image_data.height:
                    line_2 = image_data.height - 1

                # HISTOGRAM
                histogram_2 = np.asarray(sum(
                    self.image_used[line_1:line_2, self.horizontal_index[0]:self.horizontal_index[1]]))

                if sum(histogram_2) < image_data.SEP_LINE_PIXEL_THRESHOLD:
                    local_min_array[index] = 0
                else:
                    break
            local_min_array = local_min_array[local_min_array != 0]

            # FROM DOWN TO UP
            if local_min_array.any():
                for index in range(local_min_array.shape[0] - 1, 0, -1):
                    if index == 0:
                        break

                    line_1 = local_min_array[index - 1]
                    line_2 = local_min_array[index]
                    # CHECK
                    if line_1 == line_2:
                        line_1 = line_1 - 1
                        line_2 += 1
                    if line_1 < 0:
                        line_1 = 0
                    if line_2 > image_data.height:
                        line_2 = image_data.height - 1

                    # HISTOGRAM
                    histogram_2 = np.asarray(sum(
                        self.image_used[line_1:line_2, self.horizontal_index[0]:self.horizontal_index[1]]))

                    if sum(histogram_2) < image_data.SEP_LINE_PIXEL_THRESHOLD:  # Threshold, put variable same as up
                        local_min_array[index] = 0
                    else:
                        break
                local_min_array = local_min_array[local_min_array != 0]

            # If there is no separation line between text line, we create it.
            if len(local_max_array) > 1:
                for index, line in enumerate(local_max_array):
                    if index == len(local_max_array) - 1:
                        break
                    line_1 = local_max_array[index]
                    line_2 = local_max_array[index + 1]

                    any_line = local_min_array[local_min_array > line_1]
                    any_line = any_line[any_line < line_2]

                    if any_line.any():
                        continue
                    min_value = (line_1 + line_2) / 2
                    min_value = min_value.astype('int64')

                    local_min_array_1 = local_min_array[local_min_array < min_value]
                    local_min_array_2 = local_min_array[local_min_array > min_value]

                    local_min_array = np.append(local_min_array_1, min_value)
                    local_min_array = np.append(local_min_array, local_min_array_2)

            # STORING
            self.separation_lines = local_min_array
            self.text_line = local_max_array
