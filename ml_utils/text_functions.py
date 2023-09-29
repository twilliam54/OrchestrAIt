import pandas as pd
import nltk

from .pdf_formatter import split_pdf_template

nltk.download('punkt')  # Download the required resource

def doc_to_dataframe(source_path, output_path):
    doc_list=split_pdf_template(source_path, output_path)
    doc_list_df=pd.DataFrame(doc_list, columns=['doc_path', 'doc_name','doc_content'])
    doc_list_df=doc_list_df.dropna()
    return doc_list_df

def segment_text(text, max_length):
    word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') # tokenize words
    tokens = word_tokenizer.tokenize(text)
    segments = []
    current_segment = []
    for token in tokens:
        current_segment.append(token)
        if len(current_segment) == max_length:
            segments.append(current_segment)
            current_segment = []
    # If there are remaining tokens at the end, add them as the last segment
    if current_segment:
        segments.append(current_segment)
    return segments

# Create a new DataFrame to store segmented text
def word_segmenter(df, seg_size):
    # Set the sequence length
    max_sequence_length = seg_size
    # Initialize an empty DataFrame to store the segmented text
    doc_segments_df = pd.DataFrame(columns=['doc_path', 'doc_name', 'doc_content', 'doc_segments'])

    # Iterate over each row in the doc_df DataFrame
    for index, row in df.iterrows():
        segments = segment_text(row['doc_content'], max_sequence_length)

        # Create a new DataFrame for the segments of the current document
        single_doc_segment_df = pd.DataFrame({'doc_path': row['doc_path'],
                                              'doc_name': row['doc_name'],
                                              'doc_content': row['doc_content'],
                                              'doc_segments': [' '.join(segment) for segment in segments]})

        # Append the segmented text of the current document to the doc_segments_df DataFrame
        doc_segments_df = pd.concat([doc_segments_df, single_doc_segment_df], ignore_index=True)
    return doc_segments_df