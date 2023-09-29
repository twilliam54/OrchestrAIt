import os
import pandas as pd

doc_type_df=pd.DataFrame({'doc_type': ['Foreclosure', 'General','Mortgage','Note','Origination','Title']})

def subdir_maker(full_file_path, final_output_path):

    # Extract the file name from the path
    file_name = os.path.basename(full_file_path)

    # Remove the file extension (if any)
    file_name = os.path.splitext(file_name)[0]

    # Create a folder with the file name
    folder_path = os.path.join(final_output_path, file_name)
    os.makedirs(folder_path)
    
    
    # create a list of these folders and iterate through the list to create new subdirs
    folder_list=doc_type_df.doc_type.tolist()
    for folder in folder_list:
        new_subfolder_path = os.path.join(folder_path, folder)
        os.makedirs(new_subfolder_path)
    
    # return the root folder path so that the split_pdf function can put the files that need to be sorted here.
    return folder_path, file_name