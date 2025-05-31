
import os
import numpy as np
import pandas as pd
import pickle as pkl

discussions = os.listdir('./slashdot_conversations/')

indx = 0
for g in discussions:
    print('Processing', indx, g)
    discuss_df = pd.read_csv('./slashdot_conversations/' + g)
    discuss_df['isNan'] = discuss_df['Parent ID'].isnull()
    indx += 1

    nodes = {}
    edges = {}

    for i in range(len(discuss_df)):
        row = discuss_df.iloc[i]

        nodes[row['Comment ID']] = {
            'timestamp': row['Timestamp'],
            'title': row['Title'],
            'text': row['Text'],
            'author_id': row['Author ID'],
            'score': row['Score'],
            'category': row['Category']
        }
        if row['isNan']:
            edges[row['Comment ID']] = row['Parent ID']
        else:
            edges[row['Comment ID']] = int(row['Parent ID'])

    discuss_id = g.split('-')[1].split('.')[0]

    with open('./slashdot_graphs/' + discuss_id + '.pkl', 'wb') as f:
        pkl.dump({'nodes': nodes, 'edges': edges}, f)
