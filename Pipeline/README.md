# HandWriting recognition project: Automatic Handwriting Recognition of Ancient Hebrew in the Dead Sea Scrolls

This project aims to classify character from binarized scrolls text images by segmenting the text in lines, segmenting lines in characters and last but not least, classifying characters.

The project was coded using python 3.8. We cannot say whether it works for other versions.
The pipeline has been tested on Ubuntu 18.04.6.
--
Setup
--

1. Download the zip file named <em>DSS.zip</em>
2. Extract the zip to some directory <em>project_dir</em>
3. In <em>project_dir</em>, run the command <strong>pip install -r requirements.txt</strong> to install the required packages:

Predict labels of test images
--

1.To predict the labels of the test images, follow the steps in <strong>Setup</strong><br>
2.Run the file <em>main.py</em> in <em>project_dir/</em> with the following command:<br>
- <strong>python main.py full/length/path/to/test/images/</strong> <br>(there should be a '/' at the end of the path)<br>

The output will be a txt file and a docx file for each image with the name <em>img_name_characters.txt/docx</em><br>
The text files will be stored in the directory <em>project_dir/result_txts</em><br>
The docx files will be stored in the directory <em>project_dir/result_docs</em><br>
The line sigmentation result will be stored in the directory <em>project_dir/Results</em><br>
The input image examples are provided in the directory <em>project_dir/Pipeline_inputs</em><br>
