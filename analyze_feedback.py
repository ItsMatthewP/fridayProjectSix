import sqlite3
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration and Setup ---

def load_config():
    """
    Loads configuration from .env file and sets up the OpenAI API key.
    Assumes production.env file is in the same directory.
    """
    # Load the .env file named 'production.env'
    load_dotenv(dotenv_path='production.env')
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in production.env file.")
        print("Please ensure the file exists and contains: OPENAI_API_KEY=your_key_here")
        return None
        
    # Initialize and return the OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a simple model list
        client.models.list() 
        print("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("This might be due to an invalid API key or network issues.")
        return None

def connect_db(db_file="feedback.db"):
    """
    Connects to the specified SQLite database file.
    
    Args:
        db_file (str): The path to the SQLite database file.

    Returns:
        sqlite3.Connection: A connection object or None if connection fails.
    """
    try:
        conn = sqlite3.connect(db_file)
        print(f"Successfully connected to database: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {db_file}: {e}")
        return None

def fetch_reviews(conn):
    """
    Fetches all reviews from the 'reviews' table.
    
    Args:
        conn (sqlite3.Connection): The database connection object.

    Returns:
        pd.DataFrame: A DataFrame containing the reviews with 'review_id' and 'text' columns.
                      Returns an empty DataFrame if the table or column is not found.
    """
    # This query assumes your table is 'reviews' and the text column is 'review_text'
    # It also assumes a unique id column, here aliased as 'review_id'
    # !! ADJUST THIS QUERY if your schema is different !!
    query = "SELECT rowid as review_id, review_text as text FROM reviews"
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("Warning: No reviews found. The 'reviews' table might be empty or the query is incorrect.")
        else:
            print(f"Successfully fetched {len(df)} reviews from the database.")
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error fetching reviews: {e}")
        print("Please check if a table named 'reviews' with a column 'review_text' exists in your database.")
        return pd.DataFrame(columns=['review_id', 'text']) # Return empty DF on error

# --- OpenAI API Functions ---

def get_sentiment(client, review_text):
    """
    Analyzes the sentiment of a single review text.
    
    Args:
        client (OpenAI): The initialized OpenAI client.
        review_text (str): The customer review text.

    Returns:
        str: 'Positive', 'Negative', or 'Neutral'. Returns 'Error' on failure.
    """
    system_prompt = (
        "You are a sentiment analysis expert for tech products. "
        "Classify the following customer review as 'Positive', 'Negative', or 'Neutral'. "
        "Respond with only one of these three words and nothing else."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # You can swap this for "gpt-4" for higher accuracy
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text}
            ],
            temperature=0, # Low temperature for deterministic classification
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip().replace("'", "").replace('"', '')
        
        # Validate response
        if sentiment in ['Positive', 'Negative', 'Neutral']:
            return sentiment
        else:
            print(f"Warning: Received unexpected sentiment value: {sentiment}. Defaulting to Neutral.")
            return 'Neutral' # Default fallback
            
    except Exception as e:
        print(f"Error during sentiment analysis API call: {e}")
        return "Error"

def get_aspects(client, review_text):
    """
    Extracts specific aspects and their associated sentiment from a review.
    
    Args:
        client (OpenAI): The initialized OpenAI client.
        review_text (str): The customer review text.

    Returns:
        list: A list of dictionaries, e.g., 
              [{"aspect": "screen resolution", "sentiment": "Positive"}].
              Returns empty list on failure or if no aspects are found.
    """
    system_prompt = (
        "You are a product feedback analyst for the Apple Vision Pro. "
        "Extract the specific features, product attributes, or aspects (e.g., 'eye tracking', 'battery life', 'comfort', 'price', 'display quality') mentioned in the review. "
        "For each aspect, determine the associated sentiment (Positive, Negative, or Neutral). "
        "Return your answer as a valid JSON array of objects. Each object must have two keys: 'aspect' and 'sentiment'. "
        "Example: [{\"aspect\": \"display quality\", \"sentiment\": \"Positive\"}, {\"aspect\": \"headband comfort\", \"sentiment\": \"Negative\"}] "
        "If no specific aspects are mentioned, or if the review is too generic, return an empty array []."
    )
    
    try:
        # Using a model that is good at following JSON instructions
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview", # GPT-4 is highly recommended for structured JSON output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text}
            ],
            response_format={"type": "json_object"}, # Enforce JSON output
            temperature=0.1
        )
        
        # The API response with json_object type might be nested.
        # Let's find the JSON part.
        json_string = response.choices[0].message.content
        
        # The prompt asks for an array, but the model might wrap it in an object
        # e.g., {"aspects": [...]}. We need to handle this.
        parsed_json = json.loads(json_string)
        
        # Try to find the list of aspects
        if isinstance(parsed_json, list):
            aspect_list = parsed_json
        elif isinstance(parsed_json, dict):
            # Find the first value in the dict that is a list
            found_list = None
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    found_list = value
                    break
            if found_list is not None:
                aspect_list = found_list
            else:
                print(f"Warning: JSON response was a dict but contained no list: {json_string}")
                return []
        else:
            print(f"Warning: Received unexpected JSON structure: {json_string}")
            return []

        # Validate the structure of the list
        validated_list = []
        for item in aspect_list:
            if isinstance(item, dict) and 'aspect' in item and 'sentiment' in item:
                # Standardize aspect text
                item['aspect'] = item['aspect'].strip().lower()
                # Validate sentiment
                if item['sentiment'] not in ['Positive', 'Negative', 'Neutral']:
                    item['sentiment'] = 'Neutral' # Default if invalid
                validated_list.append(item)
        
        return validated_list

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from API response: {json_string}")
        return []
    except Exception as e:
        print(f"Error during aspect extraction API call: {e}")
        return []

# --- Data Processing and Visualization ---

def create_sentiment_chart(df):
    """
    Creates and saves a bar chart for sentiment distribution.
    """
    if 'sentiment' not in df.columns:
        print("Error: 'sentiment' column not found for plotting.")
        return
        
    sentiment_counts = df['sentiment'].value_counts()
    
    # Ensure all categories are present
    for cat in ['Positive', 'Negative', 'Neutral', 'Error']:
        if cat not in sentiment_counts:
            sentiment_counts[cat] = 0
            
    sentiment_counts = sentiment_counts.reindex(['Positive', 'Negative', 'Neutral', 'Error'])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=['#4CAF50', '#F44336', '#FFEB3B', '#9E9E9E'])
    plt.title('Sentiment Analysis of Customer Reviews', fontsize=16)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xlabel('Sentiment', fontsize=12)
    plt.xticks(fontsize=11)
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    print("Sentiment distribution bar chart saved as 'sentiment_distribution.png'")

def create_aspect_charts(aspects_df):
    """
    Creates and saves a bar chart for top mentioned aspects.
    """
    if aspects_df.empty:
        print("No aspects were extracted. Skipping aspect frequency chart.")
        return

    top_n = 20
    aspect_counts = aspects_df['aspect'].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    aspect_counts.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title(f'Top {top_n} Mentioned Aspects', fontsize=16)
    plt.xlabel('Number of Mentions', fontsize=12)
    plt.ylabel('Aspect', fontsize=12)
    plt.tight_layout()
    plt.savefig('aspect_frequency.png')
    print("Aspect frequency bar chart saved as 'aspect_frequency.png'")

def create_word_clouds(aspects_df):
    """
    Creates and saves word clouds for positive and negative aspects.
    """
    if aspects_df.empty:
        print("No aspects were extracted. Skipping word clouds.")
        return

    try:
        # Positive aspects
        pos_aspects = aspects_df[aspects_df['sentiment'] == 'Positive']['aspect']
        if not pos_aspects.empty:
            pos_text = ' '.join(pos_aspects)
            wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.axis('off')
            plt.title('Top Positive Aspects', fontsize=16)
            plt.savefig('positive_aspects_wordcloud.png')
            print("Positive aspects word cloud saved as 'positive_aspects_wordcloud.png'")
        else:
            print("No positive aspects found for word cloud.")

        # Negative aspects
        neg_aspects = aspects_df[aspects_df['sentiment'] == 'Negative']['aspect']
        if not neg_aspects.empty:
            neg_text = ' '.join(neg_aspects)
            wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.axis('off')
            plt.title('Top Negative Aspects', fontsize=16)
            plt.savefig('negative_aspects_wordcloud.png')
            print("Negative aspects word cloud saved as 'negative_aspects_wordcloud.png'")
        else:
            print("No negative aspects found for word cloud.")
            
    except ValueError as e:
        print(f"Error creating word cloud (might be due to no words): {e}")


# --- Reporting ---

def generate_report(reviews_df, aspects_df):
    """
    Generates a summary report as a text file.
    """
    if 'sentiment' not in reviews_df.columns:
        print("Cannot generate report: Sentiment data is missing.")
        return
        
    sentiment_counts = reviews_df['sentiment'].value_counts()
    total_reviews = len(reviews_df)
    
    with open('analysis_report.md', 'w') as f:
        f.write("# Apple Vision Pro Feedback Analysis Report\n\n")
        f.write("This report summarizes the sentiment and aspect analysis of customer feedback.\n\n")
        
        f.write("## 1. Overall Sentiment Distribution\n\n")
        f.write(f"Total reviews analyzed: {total_reviews}\n\n")
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_reviews) * 100
            f.write(f"- **{sentiment}**: {count} reviews ({percentage:.1f}%)\n")
        
        f.write("\n![Sentiment Distribution](sentiment_distribution.png)\n\n")
        
        if aspects_df.empty:
            f.write("## 2. Aspect Analysis\n\n")
            f.write("No specific aspects were successfully extracted from the reviews.\n")
            print("Generated basic report as 'analysis_report.md'")
            return

        f.write("## 2. Key Aspects Analysis\n\n")
        f.write("### Top Mentioned Aspects\n")
        f.write("This chart shows the most frequently discussed features overall.\n\n")
        f.write("![Top Aspects](aspect_frequency.png)\n\n")
        
        # Get top 10 positive aspects
        pos_aspects_counts = aspects_df[aspects_df['sentiment'] == 'Positive']['aspect'].value_counts().head(10)
        f.write("### Key Strengths (Top Positive Aspects)\n\n")
        if pos_aspects_counts.empty:
            f.write("No significant positive aspects were identified.\n\n")
        else:
            for aspect, count in pos_aspects_counts.items():
                f.write(f"- **{aspect.capitalize()}**: Mentioned positively {count} times\n")
            f.write("\n![Positive Aspects Word Cloud](positive_aspects_wordcloud.png)\n\n")
            
        # Get top 10 negative aspects
        neg_aspects_counts = aspects_df[aspects_df['sentiment'] == 'Negative']['aspect'].value_counts().head(10)
        f.write("### Key Weaknesses (Top Negative Aspects)\n\n")
        if neg_aspects_counts.empty:
            f.write("No significant negative aspects were identified.\n\n")
        else:
            for aspect, count in neg_aspects_counts.items():
                f.write(f"- **{aspect.capitalize()}**: Mentioned negatively {count} times\n")
            f.write("\n![Negative Aspects Word Cloud](negative_aspects_wordcloud.png)\n\n")
            
        f.write("## 3. Actionable Recommendations (Example)\n\n")
        f.write("Based on the analysis, consider the following:\n\n")
        
        if not neg_aspects_counts.empty:
            top_negative = neg_aspects_counts.index[0]
            f.write(f"- **Priority Issue**: Investigate and improve '{top_negative.capitalize()}'. This was the most frequent complaint.\n")
        
        if not pos_aspects_counts.empty:
            top_positive = pos_aspects_counts.index[0]
            f.write(f"- **Marketing Focus**: Double down on marketing '{top_positive.capitalize()}'. Customers love this feature.\n")
            
        if 'price' in neg_aspects_counts.index:
             f.write("- **Pricing Strategy**: The 'price' is a common negative point. Explore cost-reduction possibilities or emphasize value-for-money in messaging.\n")
        
        if 'comfort' in neg_aspects_counts.index or 'weight' in neg_aspects_counts.index:
            f.write("- **Ergonomics**: 'Comfort' and 'weight' appear to be concerns. Prioritize R&D for lighter materials and improved weight distribution for the next model.\n")

    print("Analysis report saved as 'analysis_report.md'")


# --- Main Execution ---

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("Starting feedback analysis process...")
    
    # 1. Setup
    client = load_config()
    if not client:
        return # Stop if API key is not loaded

    conn = connect_db()
    if not conn:
        return # Stop if DB connection fails

    # 2. Fetch Data
    reviews_df = fetch_reviews(conn)
    conn.close() # Close connection after fetching
    
    if reviews_df.empty:
        print("No reviews to analyze. Exiting.")
        return

    # 3. Analyze Data (Loop through reviews)
    sentiments = []
    all_aspects = []
    total_reviews = len(reviews_df)

    print(f"\nAnalyzing {total_reviews} reviews...")
    
    for index, row in reviews_df.iterrows():
        print(f"  Processing review {index + 1} of {total_reviews} (ID: {row['review_id']})...")
        
        # --- Rate Limiting ---
        # Add a short delay to avoid hitting API rate limits
        # Free tier is often 3 requests/min. Paid tier is much higher.
        # Adjust this as needed based on your API plan.
        time.sleep(1) # 1 second delay between full analyses
        
        # Get sentiment
        sentiment = get_sentiment(client, row['text'])
        sentiments.append(sentiment)
        
        # Get aspects (only if sentiment analysis was successful)
        if sentiment != "Error":
            # We can use a slightly faster delay for the second call
            time.sleep(0.5) 
            aspects = get_aspects(client, row['text'])
            # Add review_id to each aspect for easier tracking
            for aspect_dict in aspects:
                aspect_dict['review_id'] = row['review_id']
                all_aspects.append(aspect_dict)

    print("\nAnalysis complete.")

    # 4. Process Results
    reviews_df['sentiment'] = sentiments
    aspects_df = pd.DataFrame(all_aspects)
    
    # Save raw results to CSV for inspection
    reviews_df.to_csv('reviews_with_sentiment.csv', index=False)
    print("Saved reviews with sentiment to 'reviews_with_sentiment.csv'")
    
    if not aspects_df.empty:
        aspects_df.to_csv('extracted_aspects.csv', index=False)
        print("Saved extracted aspects to 'extracted_aspects.csv'")
    else:
        print("No aspects were extracted to save.")

    # 5. Visualize
    print("Generating visualizations...")
    create_sentiment_chart(reviews_df)
    create_aspect_charts(aspects_df)
    create_word_clouds(aspects_df)
    
    # 6. Report
    print("Generating final report...")
    generate_report(reviews_df, aspects_df)
    
    print("\nProcess finished successfully!")


if __name__ == "__main__":
    main()
