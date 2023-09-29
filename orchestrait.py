import ml_utils
import tensorflow as tf

if __name__ == "__main__":
    model=tf.keras.models.load_model('mortgage_doc_identifier_v5')
    sample_file = input('Enter the folder location of your source packet files: ')
    output_path = input('Enter the folder location where you want to store your processed packet files: ')

    doc_df=ml_utils.doc_to_dataframe(sample_file, output_path)
    ml_utils.ml_sort_data(doc_df, 512, model)