"""
This script is to make the dataset used for finetuning.

requirements.txt:
- nltk
- pandas
- transformers
- tf-keras

Setup:
- Download above libraries
- Download the Wikipedia dataset: wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

"""

# Default paths!
INPUT_PATH = 'psgs_w100.tsv'
OUTPUT_PATH = 'dataset.csv'

import csv
from nltk.tokenize import sent_tokenize
import pandas as pd
import random
from transformers import pipeline
from tqdm import tqdm
import logging

def clean_text(text):
    if pd.isna(text):
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) == 0:
        return None
    return text

def create_info_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH, subset=500000):

    # Log
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # For NER
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    logger.info("Named Entity Recognition model downloaded")
    
    # Read in data
    subset = subset # 50,000 gives us 20k training examples.
    
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
    mask_token = '[MASK]'
    k = 15
    
    # Initialize for collecting data
    final_df = []
    seen = []
    
    # Iterate through each scenario
    for scenario_index, scenario in enumerate(scenarios):
    
       logger.info(f"Creating {scenario} partition")
    
       # Grab a unique third of the dataset
       third_df = grouped_df[start_idx:end_idx]
    
       # Iterate through each unique article in this subset
       for title, group in tqdm(third_df):
    
          # Skip seen titles
          if title in seen:
             continue
    
          # logger.info(f"Current Article = {title}")
    
          # Ensure no duplicate and na articles
          group.drop_duplicates(inplace=True)
          group.dropna(inplace=True)
    
          # Create a list of sentences in this article
          curr_article = ""
          for index, row in group.iterrows():
             curr_article += row['text']
          curr_sentences = sent_tokenize(curr_article)
          if len(curr_sentences) > k:
              start = random.randint(0, len(curr_sentences) - k)
              curr_sentences = curr_sentences[start:start + k]
    
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
                   offset += len(mask_token) - len(masked_word)
                masked_sentences.append(masked_sentence)
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
             'target_prefix': input_prefix,
             'target_suffix': output_suffix
          })
          seen.append(title)
    
       # Continue for iteration
       start_idx += third_length
       end_idx += third_length
    
       logger.info(f"{scenario} partition created")
    
    final_df = pd.DataFrame(final_df)

    # Cleaning
    final_df['context_clean'] = final_df['context'].apply(clean_text)
    final_df['prefix_clean'] = final_df['target_prefix'].apply(clean_text)
    final_df['suffix_clean'] = final_df['target_suffix'].apply(clean_text)
    final_df_clean = final_df[
        (final_df['context_clean'].notna()) &
        (final_df['prefix_clean'].notna()) &
        (final_df['suffix_clean'].notna())
    ].copy()
    final_df_clean.drop(columns=['context', 'target', 'target_suffix','target_prefix'], inplace=True)
    logger.info(f"\nDataset size after cleaning: {len(final_df_clean)}")
    logger.info(f"Removed {len(final_df) - len(final_df_clean)} bad rows")
    
    # Sample to fit distribution
    scenario_counts = final_df_clean['scenario'].value_counts()
    total_size = len(final_df_clean)
    target_counts = {
        'Extraction': int(total_size * 0.20),
        'Correction': int(total_size * 0.40),
        'Stimulation': int(total_size * 0.40)
    }
    sampled_dfs = []
    for scenario, target_count in target_counts.items():
        scenario_df = final_df_clean[final_df_clean['scenario'] == scenario]
        replace = len(scenario_df) < target_count
        sampled = scenario_df.sample(n=target_count, replace=replace, random_state=42)
        sampled_dfs.append(sampled)
    final_df_clean = pd.concat(sampled_dfs, ignore_index=True)
    final_df_clean = final_df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    # Shuffle
    final_df_shuffled = final_df_clean.sample(frac=1)
    
    # Output dataset
    final_df_shuffled.to_csv(output_path, index=False)
    logger.info(f"Finetuning dataset created, {len(final_df_shuffled)} examples")
    return final_df_shuffled