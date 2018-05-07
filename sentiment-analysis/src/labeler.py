import pandas as pd
import numpy as np
import os
import json
import re
import gc
import pickle

pd.set_option('display.max_colwidth', -1)

data_path =  "/home/juan/Desktop/Text_Mining/Om_Project"
# reader_ = pd.read_csv ( os.path.join( data_path,"db_tweets.csv" ) , sep = "|", lineterminator = '\n')
# data_ = reader_.sample(5000)
# data_.to_csv(os.path.join(data_path,"data_labeling.csv"),index=False,sep='|')
# del reader_

data_labels = pd.read_csv ( os.path.join( data_path,"data_labeling.csv" ) ,
                            sep = "|",
                            lineterminator = '\n' )

# sentiment_label = []
# tweet_id = []
#
# with open(os.path.join(data_path,"sentiment_labels"), 'wb') as fp:
#     pickle.dump(sentiment_label, fp)
# fp.close()
# with open(os.path.join(data_path,"tweet_id"), 'wb') as fp:
#     pickle.dump(tweet_id, fp)
# fp.close()

with open(os.path.join(data_path,"sentiment_labels"), 'rb') as fp:
    sentiment_label = pickle.load(fp)
fp.close()

with open(os.path.join(data_path,"tweet_id"), 'rb') as fp:
    tweet_id = pickle.load(fp)
fp.close()

if len(tweet_id)!=0:
    start = data_labels.index[ data_labels.tweet_id == tweet_id[-1]].tolist()[0]
else:
    start = 0

data_labels = data_labels.iloc [ start: , : ]
data_labels.columns

for i in range( data_labels.shape[0] ):

    print(data_labels.iloc[i,:][["text_tweet","hashtags","user_description"]])
    label = input ( "label (END): " )

    if label == "END":
        break
    else:
        sentiment_label.append( int( label ) )
        tweet_id.append( data_labels.tweet_id[i] )

    print ( "__________" )

with open(os.path.join(data_path,"sentiment_labels"), 'wb') as fp:
    pickle.dump(sentiment_label, fp)
fp.close()
with open(os.path.join(data_path,"tweet_id"), 'wb') as fp:
    pickle.dump(tweet_id, fp)
fp.close()



# i = np.random.randint(low=0,high = 6)
# j = 0
# for chunk in reader_:
#     if j == i:
#         break
#     j=+1
#
# my_df_chunk = chunk.sample(2000)[["tweet_id","text_tweet","user_description"]]
