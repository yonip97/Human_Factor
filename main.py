import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import numpy as np
from functools import reduce


def reduce_to_int(response):
    response_list = response.split(',')
    scores = [int(x.replace(" ", "")[0]) for x in response_list]
    return sum(scores) / len(scores)


def main():
    traits = ['friendliness', 'charisma', 'confidence', 'trustworthiness']
    g_1_r = pd.read_csv('group1_responses.csv')
    g_1_r = g_1_r.rename(columns={'What is your prolific ID?': 'ID',
                                  'How many people do you encounter on an average day?': 'daily encounters',
                                  'Do you consider yourself a good judge of character?': 'judgment scores'})
    g_1_r = g_1_r.drop(columns=[' [Charisma]', ' [Friendliness]', ' [Trustworthiness]', ' [Confidence]'])
    g_1_p = pd.read_csv('group_1_participants.csv')
    g_1_p = g_1_p.rename(columns={'participant_id': 'ID'})
    g_2_r = pd.read_csv('group2_responses.csv')
    g_2_r = g_2_r.drop(columns=[' [Charisma]', ' [Friendliness]', ' [Trustworthiness]', ' [Confidence]'])

    g_2_r = g_2_r.rename(columns={'What is your prolific ID?': 'ID',
                                  'How many people do you encounter on an average day?': 'daily encounters',
                                  'Do you consider yourself a good judge of character?': 'judgment scores'})
    g_2_p = pd.read_csv('group_2_participants.csv')
    g_2_p = g_2_p.rename(columns={'participant_id': 'ID'})
    g_1 = g_1_p.merge(g_1_r, on='ID')
    g_2 = g_2_p.merge(g_2_r, on='ID')
    g_1 = g_1[g_1['status'] == 'APPROVED']
    g_2 = g_2[g_2['status'] == 'APPROVED']
    g_1.columns = map(str.lower, g_1.columns)
    g_2.columns = map(str.lower, g_2.columns)
    rel_cols_1 = [x for x in g_1.columns if any(
        word in x for word in traits)]
    rel_cols_2 = [x for x in g_2.columns if any(
        word in x for word in traits)]

    g_1[rel_cols_1] = g_1[rel_cols_1].astype(str).applymap(reduce_to_int)
    g_2[rel_cols_2] = g_2[rel_cols_2].astype(str).applymap(reduce_to_int)
    rel_cols_1 = [x for x in rel_cols_1 if 'rate' not in x]
    rel_cols_2 = [x for x in rel_cols_2 if 'rate' not in x]
    g_1['var'] = g_1[rel_cols_1].std(axis=1)
    g_2['var'] = g_2[rel_cols_2].std(axis=1)
    times = g_1['time_taken'].tolist()
    times += g_2['time_taken'].tolist()

    print(f"Average time for {sum(times) / len(times) / 60:.3f} Minutes")
    print(f"Median time for {np.median(np.array(times)) / 60:.3f} Minutes")
    columns_by_trait = {}
    for trait in traits:
        columns_by_trait[trait] = {}
        columns_by_trait[trait]['1'] = [x for x in g_1.columns if trait in x and 'rate' not in x]
        columns_by_trait[trait]['2'] = [x for x in g_2.columns if trait in x and 'rate' not in x]
        columns_by_trait[trait]['self_1'] = [x for x in g_1.columns if trait in x and 'rate' in x]
        columns_by_trait[trait]['self_2'] = [x for x in g_2.columns if trait in x and 'rate' in x]



    # temp =[(x,y) for x,y in pd.concat((g_1['daily encounters'],g_2['daily encounters'])).value_counts().iteritems()]
    # order = [2,0,1,3]
    # temp = [temp[i] for i in order]
    # x = [b[0] for b in temp]
    # y = [b[1] for b in temp]
    # plt.bar(x, y)
    # plt.title('Daily encounters')
    # plt.show()
    temp = pd.concat((g_1['judgment scores'],g_2['judgment scores']))
    temp.hist()
    plt.grid(False)
    plt.title('Judgment scores')
    plt.show()


    columns_by_group = {'1': {}, '2': {}}
    # for trait,info in columns_by_trait.items():
    #     x =g_1[info['self_1']].copy()
    #     x['key'] = g_1[info['self_1']].copy()
    #     y = g_2[info['self_2']].copy()
    #     y['key'] = g_2[info['self_2']].copy()
    #     temp = pd.concat((x['key'],y['key']),axis = 0)
    #     print(f"Mean of the {trait} scores of the users is {temp.mean():.3f}")
    #     print(f"STD of the {trait} scores of the users is {temp.std():.3f}")
    #     print(f"Median of the {trait} scores of the users is {temp.median():.3f}")
    #     temp.hist()
    #     plt.grid(False)
    #     plt.title(trait.capitalize())
    #     plt.show()
    # for trait,info in columns_by_trait.items():
    #     x =g_1[info['1']].copy()
    #     x['key'] = g_1[info['1']].copy().mean(axis = 1)
    #     y = g_2[info['2']].copy()
    #     y['key'] = g_2[info['2']].copy().mean(axis=1)
    #     temp = pd.concat((x['key'],y['key']),axis = 0)
    #     print(f"Mean of the {trait} scores of the users is {temp.mean():.3f}")
    #     print(f"STD of the {trait} scores of the users is {temp.std():.3f}")
    #     print(f"Median of the {trait} scores of the users is {temp.median():.3f}")
    #     temp.hist()
    #     plt.grid(False)
    #     plt.title(trait.capitalize())
    #     plt.show()

    for trait in traits:
        columns_by_group['1'][trait] = [x for x in g_1.columns if trait in x and 'rate' not in x]
        columns_by_group['2'][trait] = [x for x in g_2.columns if trait in x and 'rate' not in x]

    for trait, cols_data in columns_by_trait.items():
        wilcoxon_test(trait, g_1, g_2, cols_data['1'], cols_data['2'])
    # test(g_1, g_2, columns_by_group['1'], columns_by_group['2'])
    # printable(g_1,g_2,columns_by_group['1'], columns_by_group['2'])

    # for trait in traits:
    #     self_rate_1 = [x for x in g_1.columns if 'rate' in x and trait in x]
    #     self_rate_2 = [x for x in g_2.columns if 'rate' in x and trait in x]
    #
    #     self_vs_scores(g_1, self_rate_1, columns_by_trait[trait]['1'], g_2, self_rate_2, columns_by_trait[trait]['2'],
    #                    trait)


def wilcoxon_test(trait, data_1, data_2, cols_1, cols_2):
    scores_1 = data_1[cols_1].mean()
    scores_2 = data_2[cols_2].mean()
    p_value = wilcoxon(scores_1, scores_2)[1]
    print(f"p value for {trait} is {p_value}")


def self_vs_scores(data_1, self_col_1, rate_cols_1, data_2, self_col_2, rate_cols_2, trait):
    data_1 = procces(data_1.copy(), self_col_1, rate_cols_1)
    data_1['group'] = 'Group 1'
    data_2 = procces(data_2.copy(), self_col_2, rate_cols_2)
    data_2['group'] = 'Group 2'
    data = pd.concat((data_1, data_2))
    rel = ['self_rating', 'group', 'pictures_rating']

    pd.plotting.parallel_coordinates(data[rel], class_column='group', color=('red', 'green'))
    plt.title(trait.capitalize())
    plt.show()


def procces(data, self_col, rate_cols):
    data = data.copy()
    data['self_rating'] = data[self_col]
    cols = ['self_rating']
    cols += rate_cols
    data = data[cols]
    data['pictures_rating'] = data[rate_cols].mean(axis=1)
    data = data[['self_rating', 'pictures_rating']]
    return data


def test(data_1, data_2, cols_1, cols_2):
    all = []
    tables_1 = test_process(data_1, cols_1)
    tables_2 = test_process(data_2, cols_2)
    for trait in tables_1.keys():
        df = pd.concat((tables_1[trait], tables_2[trait]), axis=0)
        df_1 = df.groupby(by='judgment scores', as_index=False).mean()
        df = df.groupby(by='judgment scores', as_index=False).count()
        temp = [x for x in df.columns]
        df[trait] = df_1[cols_1[trait]].mean(axis=1)
        df['count'] = df[temp[1]]
        df = df[['judgment scores', 'count', trait]]
        df = df.round(3)
        all.append(df)
    total = reduce(lambda left, right: pd.merge(left, right, on=['judgment scores', 'count'],
                                                how='outer'), all)
    total.columns = map(str.capitalize, total.columns)
    plt.axis('off')
    plt.axis('tight')
    table = plt.table(cellText=total.values, colLabels=total.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.title("Evaluation by judgment score")
    plt.tight_layout()
    plt.show()


def test_process(data, cols):
    tables = {}
    for i, (trait, cols) in enumerate(cols.items()):
        temp = ['judgment scores']
        temp += cols
        df = data[temp]
        tables[trait] = df
    return tables


def printable(data_1, data_2, cols_1, cols_2):
    all = []
    tables_1 = printable_process(data_1, cols_1)
    tables_2 = printable_process(data_2, cols_2)
    order = [0, 2, 3, 1]
    for trait in tables_1.keys():
        df = pd.concat((tables_1[trait], tables_2[trait]), axis=0)
        df_1 = df.groupby(by='daily encounters', as_index=False).mean()
        df = df.groupby(by='daily encounters', as_index=False).count()
        temp = [x for x in df.columns]
        df[trait] = df_1[cols_1[trait]].mean(axis=1)
        df['count'] = df[temp[1]]
        df = df[['daily encounters', 'count', trait]].reindex(order)
        df = df.round(3)
        all.append(df)
    total = reduce(lambda left, right: pd.merge(left, right, on=['daily encounters', 'count'],
                                                how='outer'), all)
    total.columns = map(str.capitalize, total.columns)
    plt.axis('off')
    plt.axis('tight')
    table = plt.table(cellText=total.values, colLabels=total.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.title("Evaluation by daily people encounter quantity")
    plt.tight_layout()
    plt.show()


def printable_process(data, cols):
    tables = {}
    for i, (trait, cols) in enumerate(cols.items()):
        temp = ['daily encounters']
        temp += cols
        df = data[temp]
        tables[trait] = df
    return tables


if __name__ == '__main__':
    main()
