# Smart Sales Helper Bot

> ⚠️ To facilitate the implementation, a reverse proxy tool (ngrok) is used. This tool is only suitable for the development and testing phase and cannot be used in the production environment. Before using it, you must confirm whether it complies with the company's network security policy.

## Features and Functionality

A lark bot that detects to-be-avoided words and the formality of language during meeting calls.

When to-be-avoided words or informal sentences are detected, the bot will send a reminder message promptly in private chat with the user. Informal languages are detected using our trained machine learning model, and the to-be-avoided words can be found and added in the file avoid_phrases.csv.

The user can send `ENROLL` to the bot to receive reminders or send `STOP` to stop receiving future reminders. 

## Credits

This project is built on top of the [**Quick Starts>Develop a Bot App**](https://open.larksuite.com/document/home/develop-a-bot-in-5-minutes/create-an-app) from the Lark official documentation. 

The dataset used to train the model is the 
[pavlick-formality-scores](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) from Hugging Face. 

## Dependencies

#### API used
- [Google Speech-to-text](https://cloud.google.com/speech-to-text/docs/quickstart)

#### Libraries used
- [Python sounddevice](https://python-sounddevice.readthedocs.io/en/0.4.7/index.html)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)
- [PyMongo](https://pymongo.readthedocs.io/en/stable/)
- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

#### Word Embedding Model
- [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/)

## Runtime environment
- [Python 3](https://www.python.org/)
- [ngrok](https://ngrok.com/download) (intranet penetration tool)


## Run the Bot
### Prep work

1. Create the custom app

   In [Developer Console](https://open.feishu.cn/app/), click **Create custom app**, then click the app name to go to the app details page.

2. On the left navigation bar, click to enter the **Test Companies and Users** page, click **Create Test Company**, fill in details then click **Confirm**. 

3. Click **Install this app**, then on the left of the navigation bar, click the toggle icon to the right of the app name and select **Test version**.

4. Go to the **Add Features** page to add **Bot**.

5. Pull the latest code to local and enter the corresponding directory.
   ```
   git clone https://github.com/stacey0331/smart_sales_helper_bot.git
   cd smart_sales_helper_bot
   ```

6. Edit environment variables

   Edit the app credential data in the `.env` file to real data.
   ```
   APP_ID=
   APP_SECRET=
   VERIFICATION_TOKEN=
   ENCRYPT_KEY=
   LARK_HOST=https://open.feishu.cn
   MONGO_URI=
   ```

   Mongo URI can be empty for now.

   The above parameters can be viewed in [Developer Console](https://open.feishu.cn/app/). 

   Go to **Credentials & Basic Info** to obtain the `App ID` and `App Secret`, and then go to **Event Subscriptions** to
   obtain the
      `Encrypt Key` and `Verification Token`.

### Setup Database
   Create a database called **sales-helper** however you would like.
   The easiest way is to use [MongoDB Atlas](https://www.mongodb.com/) to create a new free cluster. 

   After creating the database, paste the connection string into the `MONGO_URI` field in file `.env`.

### Download Pre-trained Word Vectors

Download the word vectors you want to use at [GloVe](https://nlp.stanford.edu/projects/glove/). After unzip, put the file (e.g. glove.6B.300d.txt) in the `script` folder. 

Make sure to change the `glove_file` variable in `svm.py` to match the word vectors file name. 

### Generate Model

Run the following to generate the SVM model (currently svm works the best in this case compared to linear and logistic)
```
python3 script/svm.py
```

### Run Server

You can choose to run your code either
with Docker or locally.

#### Option 1: Running with Docker

 Ensure that [Docker](https://www.docker.com/) has been installed before running.

   **Mac/Linux**

   ```
   sh exec.sh
   ```

   **Windows**

   ```
   .\exec.ps1
   ```

#### Option 2: Running Locally

1. Create and activate a new virtual environment.

   **Mac/Linux**
   ```
   python3 -m venv venv 
   . venv/bin/activate
   ```

      **Windows**
   ```
   python3 -m venv venv 
   venv\Scripts\activate
   ```

   Once activated, the terminal will display the virtual environment's name.
   ```
   (venv) **** python %
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

3. Run

   ```
   python3 server.py
   ```

### Complete the Configuration and Experience the Bot

1. Use the tool to expose the public network access portal for the local server. ngrok is used as an example here.Register and install [ngrok](https://ngrok.com/download) according to the official guidelines.

2. On the personal dashboard page, get the Authtoken. Then run: 
   ```
   ngrok authtoken <token> // <token> needs to be replaced
   ngrok http 3000
   ```


3. Go to the **Event & Callbacks** page to configure the **Request URL**. Paste in the url in the Forwarding field in ngrok (e.g. https://742b-136-49-109-67.ngrok-free.app)
 ![image.png](https://sf3-cn.feishucdn.com/obj/open-platform-opendoc/0ce38ea653e636accbd6d268b69360f9_Osy22NvNOK.png)
   **Note**: Configuring the request URL and sending messages to the bot will send requests to the backend server. So please make sure your server is running when you paste the URL and use the bot. 

4. Select the events listened to by the bot.

   On the **Event Subscriptions** page, click **Add event** and select and subscribe to the following events:
      
      `Message received (im.message.receive_v1)`

      `Corporate meeting started (vc.meeting.all_meeting_started_v1)`

      `Corporate meeting ended (vc.meeting.all_meeting_ended_v1)`

   Add the scope required for all these events before continuing. 

5. Add More Scopes

   On the **Permissions & Scopes** page, search for the scopes below and add them to the bot: 
      - Read and send messages in private and group chats

6. Open **Lark** or **Feishu** and search for the **Bot name** to begin experiencing the bot's auto replies.

## Future Improvements
- Add a bot menu for enrolling and unenrolling for better front-end experience. 
- Currently the bot only monitors users that starts the meeting. Although the sales person are usually the ones starts the call, ideally the bot should be able to handle other cases too. 
- The dataset we use to train our model are from the news, blogs, etc. The accuracy of the model might increase if it's trained on conversational/speech datasets.  


## Release
On the **Version Management & Release** page, click **Create a version** > **Submit for release**.

   **Note**: The release involves scopes that need to be manually approved. You can use Test companies and users
 function to generate a test version and complete the test. Note: After release, you can check whether users are
 within the bot's availability range based on whether they can find the bot.
