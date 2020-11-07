# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Imports
import pickle
import pandas as pd
import os
import re
i = 0

# %%
#file and directory paths formats. 

#dir_path is used in function extract_business_text.
#dir_path = r"D:\College\Study\IRE\Project\data\10k"   #Be careful here as 10k needs to be preceeded by 2 backslashs i.e. \\ for proper rendering when passed as cmd argument

#dataframe_load_path is the pickle file i.e. dataframe shared by Vivek
#dataframe_load_path = r"D:\College\Study\IRE\Project\data\df_10k_1900_org.pkl"

#dataframe_dump_path is newly edited pickle file which will be generated 
#dataframe_dump_path = r"D:\College\Study\IRE\Project\data\Task 1\dataframe.pickle

# %%
def extract_business_text(file_name, dir_path):
    #This method extracts the Item 1 Business section from file(argument 1) by applying regex
    #counter for number of files processed. Used in function extract_business_text
    global i
    i = i + 1
    if(not i%1000):
        print(i, " files processed")
    file_path = dir_path + "\\" + file_name
    
    f = open(file_path)
    text = f.read()
    matches = list(re.finditer(r'(Item|ITEM)\s*[0-9]+[A-Z]*\.', text)) #find all occurrences of this pattern
    #print(len(matches))
    #print(matches)
    if(len(matches)):
        start_index_list = [i for i in range(len(matches))if re.match(r'(Item|ITEM)\s*1\.', matches[i][0])] #occurrences of pattern 'Item 1.' and 'ITEM 1.'
        if(len(start_index_list)):
            if len(start_index_list) >= 2:
                start_index = start_index_list[1]
            else:
                start_index = start_index_list[0]
            end_index = start_index + 1
            #print(start_index, end_index)
            #print(matches[start_index], matches[end_index])
            start = matches[start_index].span()[1]
            if(end_index >= len(matches)):
                end = -1    
            else:    
                end = matches[end_index].span()[0]
            #print(start, end)
            text = text[start:end]
        else:
            text = "No Business text found" #indicates no Item 1. 
    else:
        text = "No text found" #indicates no Item found.
    f.close()
    
    return text


# %%
def run(dir_path, dataframe_load_path, dataframe_dump_path):
    print("In run() method")
    dataframe = pickle.load(open(dataframe_load_path, 'rb'))
    print("Loaded the original dataframe")
    dataframe['f_path'] = dataframe['f_path'].apply(lambda file_name: re.sub('/media/ssd/vivek.a/10k/10k_1900_org/', '', file_name))
    dataframe['text'] = dataframe['f_path'].apply(extract_business_text, args=(dir_path, ))
    print("Done with preprocessing. Now storing the new pickle file") 
    pickle.dump(dataframe, open(dataframe_dump_path, 'wb')) 
    print("Stored the pickle file. Returning the dataframe.")  
    return dataframe

# %%


# %%
#No text found: 1335 rows
#No Business text found: 3836 rows


# %%



