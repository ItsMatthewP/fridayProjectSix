# Project Guide: Apple Vision Pro Feedback Analysis

## 1. Goal of the Program

The primary goal of this program is to automatically analyze customer feedback for the Apple Vision Pro. It reads reviews from a SQLite database (`feedback.db`), uses the OpenAI language model to perform two key tasks on each review:

1.  **Sentiment Analysis:** Categorizes each review as **Positive**, **Negative**, or **Neutral**.
2.  **Aspect Extraction:** Identifies specific product features (e.g., "display quality", "battery life", "price") mentioned in the review and determines the sentiment associated with each specific aspect.

The program then aggregates all this data to generate a final report with visualizations, highlighting customer likes, dislikes, and overall sentiment.

## 2. Where to Find the Results

Once the script (`analyze_feedback.py`) finishes running, it will generate several new files in the same directory. Your primary results can be found in:

* **`analysis_report.md`**: This is the main summary report. Open this file to see a high-level overview of the findings, including sentiment percentages, top positive and negative aspects, and all the visualizations.
* **`sentiment_distribution.png`**: A bar chart showing the total count of Positive, Negative, and Neutral reviews.
* **`aspect_frequency.png`**: A horizontal bar chart showing the most frequently mentioned product aspects (both good and bad).
* **`positive_aspects_wordcloud.png`**: A word cloud visualizing the most common *positive* aspects.
* **`negative_aspects_wordcloud.png`**: A word cloud visualizing the most common *negative* aspects.
* **`reviews_with_sentiment.csv`**: A spreadsheet containing all your original reviews with their new sentiment category appended.
* **`extracted_aspects.csv`**: A spreadsheet listing every single aspect extracted from all reviews, along with its specific sentiment.

## 3. Description of Project Files

### Input Files (You Provide These)

* **`production.env`**: This is a critical configuration file. It **must** contain your secret OpenAI API key. The program reads this file to get permission to use the AI model.
* **`feedback.db`**: This is your SQLite database that contains the customer feedback. The script assumes this file has a table named `reviews` and a column named `review_text`.

### Program Files (Core Logic)

* **`analyze_feedback.py`**: This is the main Python script that runs the entire process. It connects to the database, calls the OpenAI API, processes the data, and generates all the output files.
* **`requirements.txt`**: A plain text file that lists all the Python libraries this project depends on.
* **`.gitignore`**: A configuration file for Git (version control). It tells Git to ignore sensitive files like `production.env` and `feedback.db` so you don't accidentally upload them to the internet.

### Output Files (The Script Creates These)

* **`analysis_report.md`**: The final, human-readable summary of the analysis.
* **`*.png`**: All the visualization images (charts and word clouds).
* **`*.csv`**: Raw data output, useful for further analysis in Excel or other tools.

4\. Dependencies & Installation
-------------------------------

To run this program, you need Python and several external libraries. It is highly recommended to use a Python virtual environment.

### Step 1: Create a Virtual Environment (Recommended)

```
# Create a new environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows (cmd.exe):
venv\Scripts\activate
# On macOS/Linux (bash):
source venv/bin/activate

```

### Step 2: Install All Dependencies

You can install all required libraries at once using the `requirements.txt` file:

```
pip install -r requirements.txt

```

Alternatively, you can install them one by one:

```
pip install openai
pip install pandas
pip install matplotlib
pip install wordcloud
pip install python-dotenv
pip install sqlite3

```

*(Note: `sqlite3` is often included with Python by default, but installing it ensures it's available.)*

5\. OpenAI API Key Setup
------------------------

The program **cannot** run without a valid OpenAI API key.

1.  Create a new file in the project directory named exactly `production.env`

2.  Open this file in a text editor.

3.  Add a single line to this file, replacing `sk-YourKeyHere` with your actual secret key:

    ```
    OPENAI_API_KEY=sk-YourKeyHere

    ```

4.  Save and close the file. The `.gitignore` file will prevent this from being uploaded if you use Git.

Once you have set up your `production.env` file and installed the dependencies, you can run the analysis by typing `python analyze_feedback.py` in your terminal.