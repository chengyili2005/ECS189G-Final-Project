"""
This script is to make the dataset used for finetuning.

requirements.txt:
- nltk
- pandas
- transformers
- tf-keras

Setup:
1) Download the Wikipedia dataset: wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
2) In this script, ensure that INPUT_PATH and OUTPUT_PATH (lines 19-20) are defined properly
3) Ensure requirements.txt are downloaded
4) python3 make_data.py

"""

# Change me if necessary!
INPUT_PATH = 'psgs_w100.tsv'
OUTPUT_PATH = 'dataset.csv'

import csv
from nltk.tokenize import sent_tokenize
import pandas as pd
import random
from transformers import pipeline

# For logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# For NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
logger.info("Named Entity Recognition model downloaded")

# Read in data
subset = 500000 # 50,000 gives us 20k training examples.

# Iterate through a subset of the .tsv file
logger.info("Creating Wikipedia subset")
with open(INPUT_PATH, 'r', encoding='utf-8-sig') as r:
  reader = csv.reader(r, delimiter="\t")
  corpus_list = []
  load_num = 0
  for row in reader:
      corpus_list.append(row)
      load_num += 1
      if load_num > subset:
          break

  # Format for dataframe
  header = corpus_list[0]
  sub_corpus = corpus_list[1:subset]

# Initialize dataframe
df = pd.DataFrame(sub_corpus)
df.columns = header
df.drop(columns='id', inplace=True) # 'id' just tells you what index it got that piece of text from in the .tsv file. It's not useful in anyway.
logger.info("Wikipedia subset created\n")

# Shuffle and group the dataframe by articles
df = df.sample(frac=1, random_state=0)
grouped_df = list(df.groupby('title', sort=False))

# Initialize for iterating through the scenarios
scenarios = ['Extraction', 'Correction', 'Stimulation']
third_length = len(grouped_df) // 3
start_idx = 0
end_idx = third_length

# Parameters
mask_token = ''

# Initialize for collecting data
final_df = []
seen = []

# Iterate through each scenario
for scenario_index, scenario in enumerate(scenarios):

   logger.info(f"Creating {scenario} partition")

   # Grab a unique third of the dataset
   third_df = grouped_df[start_idx:end_idx]

   # Iterate through each unique article in this subset
   for title, group in third_df:

      # Skip seen titles
      if title in seen:
         continue

      logger.info(f"Current Article = {title}")

      # Ensure no duplicate and na articles
      group.drop_duplicates(inplace=True)
      group.dropna(inplace=True)

      # Create a list of sentences in this article
      curr_article = ""
      for index, row in group.iterrows():
         curr_article += row['text']
      curr_sentences = sent_tokenize(curr_article)

      # Task specific creation
      if scenario == 'Extraction':

         # Input is all k-sentences. Output is just a randomly selected sentence
         input_context = ' '.join(curr_sentences)
         output_target = random.choice(curr_sentences)

      elif scenario == 'Correction':

         # Randomly mask/replace on informative tokens
         output_target = random.choice(curr_sentences)
         masked_sentences = []
         sentences_entitied = ner_pipeline(curr_sentences)
         for sentence, entities in zip(curr_sentences, sentences_entitied):
            masked_sentence = sentence
            offset = 0
            for entity in entities:
               masked_word = sentence[entity['start']:entity['end']]
               masked_sentence = masked_sentence[:entity['start'] + offset] + mask_token + masked_sentence[entity['end'] + offset:]
               masked_sentences.append(masked_sentence)
               offset += len(mask_token) - len(masked_word)
         input_context = ' '.join(masked_sentences)

      elif scenario == 'Stimulation':

         # Input is all k-sentences without the target. Output is the target sentence
         output_target = random.choice(curr_sentences)
         not_target = []
         for sentence in curr_sentences:
            if sentence != output_target:
               not_target.append(sentence)
         input_context = ' '.join(not_target)

      # Split the output target sentence somewhere between 1/3 and 2/3 of the sentence randomly.
      tokens = output_target.split()
      total_tokens = len(tokens)
      min_split = total_tokens // 3
      max_split = (2 * total_tokens) // 3
      token_point = random.randint(min_split, max_split)
      input_prefix = ' '.join(tokens[:token_point])
      output_suffix = ' '.join(tokens[token_point:])

      # Append into our data
      final_df.append({
         'scenario': scenario,
         'title': title,
         'context': input_context,
         'target': output_target,
         'target_suffix': input_prefix,
         'target_prefix': output_suffix
      })
      seen.append(scenario)

   # Continue for iteration
   start_idx = third_length
   end_idx += third_length

   logger.info(f"{scenario} partition created")

final_df = pd.DataFrame(final_df)

# Check for NA values
logger.info(f"Original dataset size: {len(final_df)}")
logger.info("\n=== Data Quality Check ===")
logger.info("Data types:")
logger.info(final_df[['context', 'target_prefix', 'target_suffix']].dtypes)
logger.info("\nNull values:")
logger.info(final_df[['context', 'target_prefix', 'target_suffix']].isnull().sum())
logger.info("\nEmpty strings:")
logger.info((final_df['context'] == '').sum(), "empty contexts")
logger.info((final_df['target_prefix'] == '').sum(), "empty prefixes")
logger.info((final_df['target_suffix'] == '').sum(), "empty suffixes")

# Apply cleaning
def clean_text(text):
    """Clean and validate text"""
    if pd.isna(text):
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) == 0:
        return None
    return text
final_df['context_clean'] = final_df['context'].apply(clean_text)
final_df['prefix_clean'] = final_df['target_prefix'].apply(clean_text)
final_df['suffix_clean'] = final_df['target_suffix'].apply(clean_text)
final_df_clean = final_df[
    (final_df['context_clean'].notna()) &
    (final_df['prefix_clean'].notna()) &
    (final_df['suffix_clean'].notna())
].copy()
logger.info(f"\nDataset size after cleaning: {len(final_df_clean)}")
logger.info(f"Removed {len(final_df) - len(final_df_clean)} bad rows")

# Sample to fit distribution
# TODO

# Output dataset
final_df_clean.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Finetuning dataset created, {len(final_df_clean)} examples")