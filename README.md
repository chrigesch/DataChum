# DataChum
DataChum is a web application that streamlines data analysis, facilitates result interpretation, and effectively communicates findings through visually engaging graphics. This tool empowers users to easily navigate complex datasets and gain valuable insights, promoting efficient decision-making and enhancing data-driven communication. DataChum encompasses four modules: Exploratory Data Analysis, Classification and Regression, Cluster Analysis, and Deployment.
There are several ways to run this app: (1) on streamlit webpage, (2) on Google Colab, (3) on your local machine, running an .exe file, or (4) on your local machine using this repository.
### (1) Run on streamlit webpage
Click [here](https://datachum.streamlit.app/) to go to the web app hosted on Streamlit Community Cloud. **Pros:** it is online. **Cons:** it runs on an external server with 1 GB of RAM, so it is quite slow.
### (2) Run on Google Colab
Run the "run_on_colab-ipynb" Notebook on [Google Colab](https://colab.google/). **Pros:** it is online and a GPU can be accessed. **Cons:** it runs on an external server. Furthermore, requirements.txt must be installed (by running the first cell of the Notebook) and *localtunnel* must be installed to serve the Streamlit app (by running the second cell of the Notebook). 
#### (a) Copy the ID that appears right before "npx: installed" and (b) click on the link that appears under "npx: installed".
![image](https://github.com/chrigesch/DataChum/assets/117320400/ee3432c4-48ce-4dd8-a577-00f9e6bea3f2)
#### (c) Once the page of step (b) opens, paste the ID you copied in step (a) in "Endpoint ID".
![image](https://github.com/chrigesch/DataChum/assets/117320400/405c554f-a768-4ed5-9d47-7c0a3f76cfeb)
### (3) Run on local machine with an .exe file
Click [here](https://psiubaar-my.sharepoint.com/:f:/g/personal/christianschetsche_psi_uba_ar/EkoVGLDHYnpCngdXvhIaxBYBqB9xeHC1c-na195ncmX-Tg?e=afykLq) to download the .exe file and double click to run it. A console will open (which should not be closed) but the web applitation will be launched in a new tab of your default browser. **Pros:** it runs without internet connection, no need to install Python or other dependencies, it runs on the local machine and a local GPU could be accessed. **Cons:** it takes some time to open the the .exe. 
### (4) Run on local machine using this repository
Download this repository, (create a virtual environment in Python 3.10), activate the virtual environment, install requirements.txt and, within the DataChum folder, run "streamlit run index.py". **Pros:** it runs without internet connection, it runs on the local machine and a local GPU could be accessed, opens immediately. **Cons:** needs Python 3.10 and other dependencies to be installed.
