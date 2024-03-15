Project Instructions
SETTING UP:

I use anaconda3 for my setup, but you could use any thing, but make sure you create an environment with streamlit installed with the following command and all of the library:

pip install streamlit

Then, move into the repository and install the requirements with:

pip install -r requirements.txt

After that you should change all the data path in the code to suit your set up.


RUNNING THE PROJECT:

But if you do not have the model files, MAKE SURE YOU DOWNLOADED AND UNZIP THE DATASET IN THE SAME DIRECTORY OR CHANGE THE DATAPATH TO THE DATASET, after that run the file trainEfficientnetV2, trainSelfv1, trainSelfv2, trainSelfv3, each will have instruction in the code, and it automatically save the model.

After that you if you have the h5 files of all the models or the directory to the folders of the models, you just change the data path, run the file in your IDE, you will get a prompt like this 

Warning: to view this Streamlit app on a browser, run it with the following
  command:

    streamlit run C:\Users\Admin\PycharmProjects\pythonProject1\appmain.py

Then open up the environment and run the app script:

streamlit run appmain.py

Or just run the command they suggest you.

(The zip files will also have some images for you to test with the GUI)
(This also has some past training logs)
(The train files will take around 2 to 4 hours or more depending on your setup and hardware)
(I run these training file on RTX 3070)
